# simulate_degrade.py

import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm  # Progress bar for large datasets
import yaml  # For reading config file

# List of valid image extensions
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

def downscale_image(image, scale_factor=0.2):
    """Reduce image size to simulate distance"""
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, resample=Image.Resampling.NEAREST)

def apply_gaussian_blur(image, radius=2):
    """Apply Gaussian blur to simulate atmospheric distortion"""
    return image.filter(ImageFilter.GaussianBlur(radius))

def add_gaussian_noise(image, mean=0, std=20):
    """Add Gaussian noise to simulate sensor/environmental noise"""
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.float32)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def apply_jpeg_compression(image, quality=20):
    """Reduce JPEG quality to simulate compression artifacts"""
    temp_path = "temp.jpg"
    image.save(temp_path, "JPEG", quality=quality)
    compressed_image = Image.open(temp_path)
    os.remove(temp_path)
    return compressed_image

def add_haze(image, alpha=0.7):
    """Simulate haze/fog by overlaying a white layer"""
    haze = Image.new("RGB", image.size, (255, 255, 255))
    return Image.blend(image, haze, alpha)

def darken_image(image, factor=0.75):
    """Simulate low-light conditions by reducing brightness"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def simulate_distant_view(image, config):
    """Apply all transformations to simulate a far-away view based on config."""

    if config.get("downscale", False):
        image = downscale_image(image, config.get("downscale_factor", 0.2))

    if config.get("gaussian_blur", False):
        image = apply_gaussian_blur(image, config.get("gaussian_blur_radius", 2))

    if config.get("gaussian_noise", False):
        image = add_gaussian_noise(image, config.get("gaussian_noise_mean", 0), config.get("gaussian_noise_std", 20))

    if config.get("jpeg_compression", False):
        image = apply_jpeg_compression(image, config.get("jpeg_compression_quality", 20))

    if config.get("haze", False):
        image = add_haze(image, config.get("haze_alpha", 0.7))

    if config.get("darken", False):
        image = darken_image(image, config.get("darken_factor", 0.75))

    return image

def process_dataset(input_path, output_path, config):
    """
    Process all images in a dataset, applying transformations and saving results.

    :param input_path: Path to original dataset (expects subfolders for each class)
    :param output_path: Path where transformed dataset will be saved
    :param config: Dictionary containing the desired image transformations.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loop through all class subdirectories
    for class_name in os.listdir(input_path):
        class_input_dir = os.path.join(input_path, class_name)
        class_output_dir = os.path.join(output_path, class_name)

        if not os.path.isdir(class_input_dir):
            continue  # Skip if not a directory

        # Ensure the output class directory exists
        os.makedirs(class_output_dir, exist_ok=True)

        # Process each image in the class directory
        for image_name in tqdm(os.listdir(class_input_dir), desc=f"Processing {class_name}"):
            input_image_path = os.path.join(class_input_dir, image_name)
            output_image_path = os.path.join(class_output_dir, image_name)

            # Validate image file extension
            if not any(image_name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                continue  # Skip non-image files

            # Load the image
            try:
                image = Image.open(input_image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Unable to load {input_image_path}, error: {e}")
                continue

            # Apply transformations
            transformed_image = simulate_distant_view(image, config)

            # Save the transformed image
            transformed_image.save(output_image_path)

    print("Processing complete! Transformed images saved.")

if __name__ == "__main__":
    input_directory = "/projects/dsci410_510/Levin_MAED/data/split_aug/test"
    output_directory = "/projects/dsci410_510/Levin_MAED/data/test_degraded"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load configuration from YAML file
    config_path = "config.yaml"  # Replace with your config file path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    process_dataset(input_directory, output_directory, config['image_degradation']) #pass in the correct part of the config file.

    print(f"Augmented images saved to {output_directory}")