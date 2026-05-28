# pipeline/image_generator.py
#
# Image generators for identity-preserving image generation.
#
# Two providers are supported:
#   GeminiImageGenerator  — uses google-genai SDK, model: gemini-2.5-flash-image
#   OpenAIImageGenerator  — uses openai SDK, model: gpt-image-1 (images.edit)
#
# Both follow the same pattern:
#   1. Load the reference identity image
#   2. Send it + the prompt to the provider API
#   3. Save the returned image to outputimage/
#
# The API key is NEVER stored in this file or config.py.
# Pass it via --api-key CLI flag or GEMINI_API_KEY / OPENAI_API_KEY env vars.

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema (shared by both generators)
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    success: bool
    output_path: Optional[Path]   # where the image was saved
    model: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Gemini generator
# ---------------------------------------------------------------------------

class GeminiImageGenerator:
    """
    Wraps the Gemini API for identity-preserving image generation.

    Instantiate once and reuse across a batch.
    Uses google-genai SDK (replaces deprecated google-generativeai).
    """

    def __init__(self, api_key: str) -> None:
        self._client = self._load_client(api_key)

    def _load_client(self, api_key: str):
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client ready. Model: %s", config.GENERATION_MODEL)
            return client
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        identity_image_path: str | Path,
        prompt: str,
        output_path: str | Path,
    ) -> GenerationResult:
        """
        Generate an image of the person in identity_image_path using the prompt.

        Parameters
        ----------
        identity_image_path : reference identity image (the person to preserve)
        prompt              : scene/style description from the CSV
        output_path         : where to save the generated image

        Returns
        -------
        GenerationResult
        """
        identity_path = Path(identity_image_path)
        output_path = Path(output_path)

        try:
            from google import genai
            from google.genai import types

            full_prompt = f"{config.GENERATION_PROMPT_PREFIX}{prompt}"
            logger.info("Generating — model: %s | prompt: %s", config.GENERATION_MODEL, full_prompt)

            # Convert reference image to bytes for the API.
            # Context-manager open so the source file handle is closed
            # before the API call (which may run for tens of seconds).
            with Image.open(identity_path) as im:
                ref_pil = im.convert("RGB")
            buf = io.BytesIO()
            ref_pil.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            response = self._client.models.generate_content(
                model=config.GENERATION_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    full_prompt,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            # Extract image bytes from response
            image_data = self._extract_image(response)

            if image_data is None:
                return GenerationResult(
                    success=False,
                    output_path=None,
                    model=config.GENERATION_MODEL,
                    error="Gemini returned no image. The model may have declined "
                          "the request or returned text only.",
                )

            # Save to disk
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)

            logger.info("Image saved → %s", output_path)
            return GenerationResult(
                success=True,
                output_path=output_path,
                model=config.GENERATION_MODEL,
                error=None,
            )

        except Exception as exc:
            logger.exception("Gemini generation failed: %s", exc)
            return GenerationResult(
                success=False,
                output_path=None,
                model=config.GENERATION_MODEL,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(response) -> Optional[bytes]:
        """
        Pull image bytes from a Gemini response.
        Returns None if no image part is found.
        """
        try:
            parts = response.candidates[0].content.parts
        except (IndexError, AttributeError):
            return None

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if data is None:
                continue
            # SDK returns bytes directly
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            # Fallback: base64 string
            if isinstance(data, str):
                return base64.b64decode(data)

        return None


# ---------------------------------------------------------------------------
# OpenAI generator
# ---------------------------------------------------------------------------

class OpenAIImageGenerator:
    """
    Wraps the OpenAI API for identity-preserving image generation.

    Uses client.images.edit() with the reference image + prompt.
    gpt-image-1 understands the person in the reference image and
    applies the requested style/scene while preserving identity.

    Instantiate once and reuse across a batch.
    Requires: pip install openai
    """

    def __init__(self, api_key: str) -> None:
        self._client = self._load_client(api_key)

    def _load_client(self, api_key: str):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client ready. Model: %s", config.OPENAI_GENERATION_MODEL)
            return client
        except ImportError as e:
            raise RuntimeError(
                "openai is not installed. "
                "Run: pip install openai"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        identity_image_path: str | Path,
        prompt: str,
        output_path: str | Path,
    ) -> GenerationResult:
        """
        Generate an image of the person in identity_image_path using the prompt.

        Uses images.edit() — the reference image is passed as-is, and the
        prompt instructs the model to preserve identity while applying the
        requested style/scene.

        Parameters
        ----------
        identity_image_path : reference identity image (the person to preserve)
        prompt              : scene/style description from the CSV
        output_path         : where to save the generated image

        Returns
        -------
        GenerationResult
        """
        identity_path = Path(identity_image_path)
        output_path = Path(output_path)

        try:
            full_prompt = f"{config.GENERATION_PROMPT_PREFIX}{prompt}"
            logger.info(
                "Generating — model: %s | prompt: %s",
                config.OPENAI_GENERATION_MODEL, full_prompt
            )

            # OpenAI images.edit requires a PNG file object.
            # Convert to PNG in memory if the source is a JPEG. Use a context
            # manager so the source file descriptor is released before the API
            # call (which can run for tens of seconds).
            with Image.open(identity_path) as im:
                ref_pil = im.convert("RGBA")
            buf = io.BytesIO()
            ref_pil.save(buf, format="PNG")
            buf.seek(0)
            buf.name = "reference.png"   # OpenAI SDK reads .name for MIME type

            response = self._client.images.edit(
                model=config.OPENAI_GENERATION_MODEL,
                image=buf,
                prompt=full_prompt,
                n=1,
                size=config.OPENAI_IMAGE_SIZE,
                quality=config.OPENAI_IMAGE_QUALITY,
            )

            # gpt-image-1 returns b64_json (not URLs)
            b64 = response.data[0].b64_json
            if not b64:
                return GenerationResult(
                    success=False,
                    output_path=None,
                    model=config.OPENAI_GENERATION_MODEL,
                    error="OpenAI returned no image data (b64_json was empty).",
                )

            image_data = base64.b64decode(b64)

            # Save to disk
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)

            logger.info("Image saved → %s", output_path)
            return GenerationResult(
                success=True,
                output_path=output_path,
                model=config.OPENAI_GENERATION_MODEL,
                error=None,
            )

        except Exception as exc:
            logger.exception("OpenAI generation failed: %s", exc)
            return GenerationResult(
                success=False,
                output_path=None,
                model=config.OPENAI_GENERATION_MODEL,
                error=str(exc),
            )
