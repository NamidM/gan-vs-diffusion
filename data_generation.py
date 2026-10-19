import numpy as np
from noise import pnoise2

NOISE_SCALE = 16.0
NOISE_OCTAVES = 3
NOISE_PERSISTENCE = 0.5
CONTRAST = 3.0
THRESHOLD = 0.5
DOWNSCALE_FACTOR = 4

def downscale_noise(image, factor=DOWNSCALE_FACTOR):
    """Average each factor x factor block and repeat it over the block (same size, lower resolution)."""
    h, w = image.shape
    assert h % factor == 0 and w % factor == 0, "image size must be divisible by the downscale factor"
    low_res = image.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return np.repeat(np.repeat(low_res, factor, axis=0), factor, axis=1)


def perlin_noise(width, height, seed):
    offset_x, offset_y = np.random.default_rng(seed).uniform(0.0, 1000.0, size=2)
    field = np.array([
        [
            pnoise2(
                x / NOISE_SCALE + offset_x,
                y / NOISE_SCALE + offset_y,
                octaves=NOISE_OCTAVES,
                persistence=NOISE_PERSISTENCE,
            )
            for x in range(width)
        ]
        for y in range(height)
    ], dtype=np.float32)

    field = (field - field.min()) / (field.max() - field.min())
    field = np.clip(0.5 + CONTRAST * (field - THRESHOLD), 0.0, 1.0).astype(np.float32)
    return field, downscale_noise(field)


def generate_data(num_data, width=64, height=64, save_path="perlin_data.npz", seed=0):
    samples = [perlin_noise(width, height, seed + i) for i in range(num_data)]
    original = np.stack([high_res for high_res, _ in samples])
    downscaled = np.stack([low_res for _, low_res in samples])
    np.savez_compressed(save_path, original=original, downscaled=downscaled)
    return original, downscaled