# evaluators/prompt_adherence.py
#
# Evaluates whether the generated image matches the text prompt.
#
# Method:
#   1. Split the prompt into overlapping word-level chunks, each sized to fit
#      within CLIP_MAX_TOKENS (77 for standard CLIP, 248 for LongCLIP).
#   2. Encode each chunk via OpenCLIP's text encoder.
#   3. Encode the generated image via OpenCLIP's image encoder.
#   4. Compute cosine similarity for every chunk → take the MAX score.
#
# Why max-pooling?
#   A generated image may faithfully render ONE part of a long prompt (e.g. the
#   character design) even if other parts (e.g. background details) are absent.
#   Max-pooling finds the chunk that best matches the image, which gives a more
#   useful signal than averaging across chunks that describe things not visible.
#
# LongCLIP upgrade path:
#   Change CLIP_MAX_TOKENS = 248, CLIP_MODEL_NAME, CLIP_PRETRAINED in config.py.
#   The chunking word budget is derived from CLIP_MAX_TOKENS automatically —
#   no code changes needed here.
#
# Important notes on CLIP score interpretation:
#   - Raw CLIP cosine similarity is NOT on a 0–1 scale intuitively.
#   - Good matches typically score 0.25–0.35, not 0.80+.
#   - Thresholds in config.py are calibrated to this raw range.
#
# Edge cases handled:
#   - Prompt longer than CLIP_MAX_TOKENS → chunked automatically
#   - Single short prompt              → no chunking overhead (fast path)
#   - Model load failure               → raises RuntimeError at init time

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class PromptAdherenceResult:
    score: Optional[float]      # raw CLIP cosine similarity (max across chunks)
    status: str                 # pass | marginal | fail | error
    model: str
    model_version: str
    clip_truncated: bool        # True if prompt was split into multiple chunks
    token_count: Optional[int]  # estimated token count of the full prompt
    chunks_used: int            # number of chunks evaluated (1 = no chunking)
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class PromptAdherenceEvaluator:
    """
    Wraps OpenCLIP for text-image similarity scoring with automatic chunking
    for prompts longer than CLIP's token limit.

    Instantiate once and reuse across a batch — model loading (~1–3s on CPU).
    """

    # Word budget per chunk: leave room for SOT + EOT (2 tokens).
    # Average English word ≈ 1.35 tokens; use 1.4 to be conservative.
    _WORDS_PER_CHUNK: int = math.floor((config.CLIP_MAX_TOKENS - 2) / 1.4)

    def __init__(self) -> None:
        self._model, self._preprocess, self._tokenizer = self._load_model()
        # Expose for sharing with artifact/safety evaluators — avoids loading CLIP 3x
        self.model      = self._model
        self.preprocess = self._preprocess
        self.tokenizer  = self._tokenizer
        logger.info(
            "PromptAdherenceEvaluator ready — chunk size: %d words, overlap: %d words",
            self._WORDS_PER_CHUNK, config.CLIP_CHUNK_OVERLAP_WORDS,
        )

    def _load_model(self):
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                config.CLIP_MODEL_NAME,
                pretrained=config.CLIP_PRETRAINED,
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
            logger.info(
                "OpenCLIP model loaded: %s / %s",
                config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED,
            )
            return model, preprocess, tokenizer
        except ImportError as e:
            raise RuntimeError(
                "open_clip_torch is not installed. Run: pip install open_clip_torch"
            ) from e

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Split text into overlapping word-level chunks, each sized to fit within
        CLIP_MAX_TOKENS. Overlap = CLIP_CHUNK_OVERLAP_WORDS.

        Short prompts that fit in one chunk are returned as a single-element list.
        """
        words = text.split()
        if len(words) <= self._WORDS_PER_CHUNK:
            return [text]

        chunks = []
        step = max(1, self._WORDS_PER_CHUNK - config.CLIP_CHUNK_OVERLAP_WORDS)
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + self._WORDS_PER_CHUNK])
            chunks.append(chunk)
            i += step

        logger.debug(
            "Prompt split into %d chunks (%d words, step %d, overlap %d)",
            len(chunks), len(words), step, config.CLIP_CHUNK_OVERLAP_WORDS,
        )
        return chunks

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for the full text.
        Uses the tokenizer output length; if the prompt was truncated to 77 tokens
        by the tokenizer, the real count is estimated from word count instead.
        """
        tokens = self._tokenizer([text])[0]
        non_pad = int((tokens != 0).sum())
        if non_pad >= config.CLIP_MAX_TOKENS:
            # Tokenizer hit its limit — estimate from word count
            return round(len(text.split()) * 1.35)
        return non_pad

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_text_chunk(self, text: str) -> np.ndarray:
        """Encode a single text chunk (must fit within CLIP_MAX_TOKENS)."""
        tokens = self._tokenizer([text])
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    def _encode_image(self, image_path: Path) -> np.ndarray:
        # Context-manager open so the underlying file handle is released as
        # soon as convert() returns the in-memory RGB copy.
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    @staticmethod
    def _score_to_status(score: float) -> str:
        if score >= config.CLIP_THRESHOLD_PASS:
            return "pass"
        if score >= config.CLIP_THRESHOLD_FAIL:
            return "marginal"
        return "fail"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prompt: str,
        generated_image_path: str | Path,
    ) -> PromptAdherenceResult:
        """
        Score how well the generated image matches the text prompt.

        Long prompts are automatically split into overlapping chunks;
        the maximum chunk score is returned.

        Parameters
        ----------
        prompt                : the original text prompt used for generation
        generated_image_path  : path to the generated image being evaluated

        Returns
        -------
        PromptAdherenceResult
        """
        generated_path = Path(generated_image_path)
        base = dict(
            model=f"openclip/{config.CLIP_MODEL_NAME}",
            model_version=config.CLIP_MODEL_VERSION,
        )

        try:
            image_embedding = self._encode_image(generated_path)
            token_count     = self._estimate_token_count(prompt)
            chunks          = self._split_into_chunks(prompt)
            chunked         = len(chunks) > 1

            if chunked:
                logger.info(
                    "Long prompt (%d est. tokens) → %d chunks (max %d words each)",
                    token_count, len(chunks), self._WORDS_PER_CHUNK,
                )

            # Score every chunk; keep the maximum
            chunk_scores = []
            for i, chunk in enumerate(chunks):
                text_emb = self._encode_text_chunk(chunk)
                s = self._cosine(image_embedding, text_emb)
                chunk_scores.append(s)
                logger.debug("  chunk %d/%d score: %.4f", i + 1, len(chunks), s)

            best_score = max(chunk_scores)

            if chunked:
                logger.info(
                    "Chunk scores: %s → best=%.4f",
                    [round(s, 4) for s in chunk_scores], best_score,
                )

            return PromptAdherenceResult(
                score=round(best_score, 4),
                status=self._score_to_status(best_score),
                clip_truncated=chunked,
                token_count=token_count,
                chunks_used=len(chunks),
                error=None,
                **base,
            )

        except Exception as exc:
            logger.exception("Unexpected error in PromptAdherenceEvaluator: %s", exc)
            return PromptAdherenceResult(
                score=None,
                status="error",
                clip_truncated=False,
                token_count=None,
                chunks_used=0,
                error=str(exc),
                **base,
            )
