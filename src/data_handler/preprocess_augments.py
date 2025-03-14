import os
from PIL import Image, ImageOps
import argparse
import yaml

def augment_images(input_dir, output_dir):
    """
    Augments images in the input directory with class subdirectories.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)

        if os.path.isdir(class_input_dir):  # Ensure it's a directory
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)

            for filename in os.listdir(class_input_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                    input_path = os.path.join(class_input_dir, filename)
                    image = Image.open(input_path).convert('RGB')

                    # Save original image
                    original_output_path = os.path.join(class_output_dir, filename)
                    image.save(original_output_path)

                    # Create black and white duplicate
                    bw_image = ImageOps.grayscale(image)
                    bw_output_path = os.path.join(class_output_dir, f"bw_{filename}")
                    bw_image.save(bw_output_path)

                    # Create horizontally flipped duplicate
                    flipped_image = ImageOps.mirror(image)
                    flipped_output_path = os.path.join(class_output_dir, f"flipped_{filename}")
                    flipped_image.save(flipped_output_path)

                    # Create vertically flipped duplicate
                    v_flipped_image = ImageOps.flip(image)
                    v_flipped_output_path = os.path.join(class_output_dir, f"v_flipped_{filename}")
                    v_flipped_image.save(v_flipped_output_path)

                    # Create rotated image.
                    rotated_image = image.rotate(90)
                    rotated_output_path = os.path.join(class_output_dir, f"rotated_{filename}")
                    rotated_image.save(rotated_output_path)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment images using configuration file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = config['data']["original_data_path"]
    output_directory = config['data']["original_data_path_aug"]

    augment_images(input_directory, output_directory)
    print(f"Augmented images saved to {output_directory}")