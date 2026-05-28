# pipeline/vqa_loader.py
#
# Loads the BLIP-VQA model once and returns it for sharing across the
# artifact, safety, and style evaluators.
#
# Current model: Salesforce/blip-vqa-base (~450 MB)
#   - Standard transformers integration, no trust_remote_code
#   - Designed specifically for binary yes/no VQA
#   - Compatible with transformers >= 4.15, Python 3.8+
#
# History:
#   This module was originally named `moondream_loader.py` and loaded
#   Moondream2 (vikhyatk/moondream2 revision 2024-07-23). Moondream2 bundles
#   a custom Phi text model that is incompatible with transformers > 4.41
#   and Python 3.13. We switched to BLIP-VQA-base, which is purpose-built
#   for VQA, ~10x smaller, natively integrated in transformers, and has
#   no Python/transformers version constraints. The Moondream2-era RoPE
#   monkey-patch (`patch_moondream_rope.py`) is no longer needed.
#
# To swap in a larger model later, change `VQA_MODEL_ID` in config.py to:
#   "Salesforce/blip-vqa-large"     (~900 MB, better accuracy)
#   "Salesforce/blip2-opt-2.7b"     (~5.5 GB, much stronger but slower)
# No code change required here.
#
# Usage:
#   from pipeline.vqa_loader import load_vqa
#   vqa_model, vqa_processor = load_vqa()
#   artifact_eval = ArtifactEvaluator(model=vqa_model, tokenizer=vqa_processor)
#   safety_eval   = SafetyEvaluator(model=vqa_model, tokenizer=vqa_processor)
#   style_eval    = StyleEvaluator(model=vqa_model, tokenizer=vqa_processor)
#
# Note: the `tokenizer=` keyword on the evaluator constructors is preserved
# for backwards compatibility with existing evaluator code. The value passed
# is a `BlipProcessor`, not a tokenizer in the HuggingFace sense.

from __future__ import annotations

import logging
from typing import Tuple, Any

import config

logger = logging.getLogger(__name__)


def load_vqa() -> Tuple[Any, Any]:
    """
    Load the BLIP-VQA model and processor via transformers.

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
