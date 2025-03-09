# simulate_degrade.py

import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm  # Progress bar for large datasets
import yaml  # For reading config file

# List of valid image extensions
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

def downscale_image(image, config):
    """Reduce image size to simulate distance"""
    scale_factor = config.get("downscale_factor", 0.2)
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, resample=Image.Resampling.NEAREST)

def apply_gaussian_blur(image, config):
    """Apply Gaussian blur to simulate atmospheric distortion"""
    radius = config.get("gaussian_blur_radius", 2)
    return image.filter(ImageFilter.GaussianBlur(radius))

def add_gaussian_noise(image, config):
    """Add Gaussian noise to simulate sensor/environmental noise"""
    mean = config.get("gaussian_noise_mean", 0)
    std = config.get("gaussian_noise_std", 20)
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.float32)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def apply_jpeg_compression(image, config):
    """Reduce JPEG quality to simulate compression artifacts"""
    quality = config.get("jpeg_compression_quality", 20)
    temp_path = "temp.jpg"
    image.save(temp_path, "JPEG", quality=int(quality)) #make sure quality is an int.
    compressed_image = Image.open(temp_path)
    os.remove(temp_path)
    return compressed_image

def add_haze(image, config):
    """Simulate haze/fog by overlaying a white layer"""
    alpha = config.get("haze_alpha", 0.7)
    haze = Image.new("RGB", image.size, (255, 255, 255))
    return Image.blend(image, haze, alpha)

def darken_image(image, config):
    """Simulate low-light conditions by reducing brightness"""
    factor = config.get("darken_factor", 0.75)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def simulate_distant_view(image, config):
    """Apply all transformations to simulate a far-away view based on config."""

    if config.get("downscale", False):
        image = downscale_image(image, config)

    if config.get("gaussian_blur", False):
        image = apply_gaussian_blur(image, config)

    if config.get("gaussian_noise", False):
        image = add_gaussian_noise(image, config)

    if config.get("jpeg_compression", False):
        image = apply_jpeg_compression(image, config)

    if config.get("haze", False):
        image = add_haze(image, config)

    if config.get("darken", False):
        image = darken_image(image, config)

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
        image_name_to_return = None
        for image_name in tqdm(os.listdir(class_input_dir), desc=f"Processing {class_name}"):
            input_image_path = os.path.join(class_input_dir, image_name)
            output_image_path = os.path.join(class_output_dir, image_name)
            if image_name_to_return is None:
                image_name_to_return = output_image_path

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

    return image_name_to_return