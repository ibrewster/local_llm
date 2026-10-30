import asyncio
import gc
import io
import json
import logging
import random
import threading
from typing import Any, cast

from . import config, utils, worker

_flux = None
_generation_lock = threading.Lock()
logger = logging.getLogger(__name__)


class ImageModelUnavailableError(Exception):
    """Raised when the configured prompt-formatting model is unavailable."""


def _get_flux():
    global _flux
    if _flux is None:
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

        mx.metal.set_cache_limit(512 * 1024 * 1024)
        _flux = Flux1(quantize=6, model_config=ModelConfig.schnell())
    return _flux


def _generate_flux_image(
    prompt: str,
    *,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """Generate an image using the cached quantized FLUX.1 Schnell model."""
    import mlx.core as mx

    if seed is None:
        seed = random.randrange(0, 2**32)

    with _generation_lock:
        try:
            image = _get_flux().generate_image(
                prompt=prompt,
                seed=seed,
                num_inference_steps=4,
                width=width,
                height=height,
            )
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
        finally:
            gc.collect()
            mx.metal.clear_cache()


async def generate_image(
    prompt: str,
    *,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """Format a user prompt with the configured LLM and generate a FLUX image."""
    model_info = utils.loaded_models.get(config.MAIN_MODEL)
    if model_info is None:
        raise ImageModelUnavailableError(
            f"model '{config.MAIN_MODEL}' not found"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's image request as one optimized FLUX.1 image-generation prompt. "
                "Preserve the requested subject, intent, and important details. Describe a clear "
                "composition, visual style, lighting, colors, and relevant materials or atmosphere. "
                "Be concrete and visually specific without adding unrelated elements. Return only "
                "the finished prompt, with no explanation, labels, or quotation marks."
            ),
        },
        {"role": "user", "content": prompt.strip()},
    ]

    final_chunk: dict[str, Any] | None = None
    async for chunk_json in worker.generate_stream(
        False,
        cast(dict[str, Any], model_info),
        messages,
        {"temp": 0.4, "max_tokens": 300},
    ):
        final_chunk = cast(dict[str, Any], json.loads(chunk_json))
    if final_chunk is None:
        raise RuntimeError("model returned no response")

    formatted_prompt = (final_chunk.get("message") or {}).get("content")
    if not isinstance(formatted_prompt, str) or not formatted_prompt.strip():
        raise RuntimeError("model returned an empty image prompt")

    return await asyncio.to_thread(
        _generate_flux_image,
        formatted_prompt.strip(),
        seed=seed,
        width=width,
        height=height,
    )
