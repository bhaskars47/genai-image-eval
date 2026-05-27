# pipeline/moondream_loader.py
#
# Loads the VQA model once and returns it for sharing across artifact,
# safety, and style evaluators.
#
# Current model: Salesforce/blip-vqa-base (~450MB)
#   - Standard transformers integration, no trust_remote_code
#   - Designed specifically for binary yes/no VQA
#   - Compatible with transformers >= 4.15, Python 3.8+
#
# Why BLIP-VQA instead of Moondream2?
#   Moondream2 (2024-07-23) bundles a custom Phi text model that is
#   incompatible with transformers > 4.41 and Python 3.13. BLIP-VQA-base
#   is purpose-built for VQA (~10x smaller), natively integrated in
#   transformers, and has no Python/transformers version constraints.
#
# To upgrade to a stronger model:
#   Change VQA_MODEL_ID in config.py to "Salesforce/blip-vqa-large"
#   or "Salesforce/blip2-opt-2.7b" — no code changes needed here.
#
# Usage:
#   from pipeline.moondream_loader import load_moondream
#   vqa_model, vqa_processor = load_moondream()
#   artifact_eval = ArtifactEvaluator(model=vqa_model, tokenizer=vqa_processor)
#   safety_eval   = SafetyEvaluator(model=vqa_model, tokenizer=vqa_processor)
#   style_eval    = StyleEvaluator(model=vqa_model, tokenizer=vqa_processor)

from __future__ import annotations

import logging
from typing import Tuple, Any

import config

logger = logging.getLogger(__name__)


def load_moondream() -> Tuple[Any, Any]:
    """
    Load BLIP-VQA model and processor via transformers.

    Returns
    -------
    (model, processor)
        model     : BlipForQuestionAnswering (eval mode, CPU)
        processor : BlipProcessor (handles image + text jointly)
    """
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
    except ImportError as e:
        raise RuntimeError(
            "transformers is not installed. "
            "Run: pip install transformers --break-system-packages"
        ) from e

    logger.info("Loading VQA model: %s …", config.VQA_MODEL_ID)

    processor = BlipProcessor.from_pretrained(config.VQA_MODEL_ID)

    model = BlipForQuestionAnswering.from_pretrained(
        config.VQA_MODEL_ID,
        torch_dtype=None,   # default float32, CPU-safe
    )
    model.eval()

    logger.info("VQA model ready: %s", config.VQA_MODEL_ID)
    return model, processor
