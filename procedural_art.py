import numpy as np
from PIL import Image
import perlin_numpy as pn


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2, seed=0):
    """
    Generate 2D fractal Perlin noise using perlin_numpy with strict integer casting.
    """
    np.random.seed(seed)

    # Ensure all resolution parameters are integers
    res_x = max(1, int(round(res[0])))
    res_y = max(1, int(round(res[1])))

    noise = np.zeros(shape, dtype=np.float32)
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(int(octaves)):
        # Compute integer resolution for each octave
        octave_res_x = max(1, int(round(res_x * frequency)))
        octave_res_y = max(1, int(round(res_y * frequency)))

        # Generate perlin noise with strictly integer resolution
        perlin = pn.generate_perlin_noise_2d(shape, (octave_res_x, octave_res_y), tileable=(False, False))
        noise += amplitude * perlin

        max_amplitude += amplitude
        amplitude *= float(persistence)
        frequency *= float(lacunarity)

    return noise / max_amplitude


def generate_procedural_image(width: int, height: int, octave: int = 4, seed: int = 1):
    """
    Generate a procedural texture image using Perlin-based fractal noise.
    """
    # 🔒 Force all input values to strict integers
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    octave = max(1, int(round(octave)))
    seed = int(seed)

    # Define initial integer resolution
    base_res = (int(width // 64) or 1, int(height // 64) or 1)

    noise = generate_fractal_noise_2d(
        (height, width),
        base_res,
        octaves=octave,
        persistence=0.5,
        lacunarity=2.0,
        seed=seed
    )

    # Normalize and convert to image
    normalized = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode='L').convert('RGB')


if __name__ == "__main__":
    img = generate_procedural_image(512, 512, octave=4, seed=1)
    img.save("procedural_texture.png")
    img.show()
    print("✅ Image saved as procedural_texture.png")
