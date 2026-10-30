import gc
import mlx.core as mx

# The exact class path in mflux
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.common.config import ModelConfig

# Cap Metal scratchpad cache
mx.metal.set_cache_limit(512 * 1024 * 1024)

# 1. Initialize FLUX.1 Schnell with 6-bit quantization
# quantize=6 compresses the Transformer, T5, and CLIP models
flux = Flux1(
    quantize=6,
    model_config=ModelConfig.schnell()
)

# 2. Run generation
image = flux.generate_image(
    prompt="An Alaskan mountain valley at sunrise, a serene glacial lake reflecting distant craggy peaks, photorealistic 35mm photography",
    seed=42,
    num_inference_steps=4,  # Schnell is distilled for 4 steps
    width=1024,
    height=1024,
)

# 3. Save
image.save("output.png")

# 4. Clean up Metal cache
gc.collect()
mx.metal.clear_cache()