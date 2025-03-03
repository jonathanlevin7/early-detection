# src/data/split_data.py

import os
import shutil
import random
import numpy as np

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

def split_data(original_data_path, split_data_path, split_ratio, seed):
    """
    Splits image data into train, validation, and test sets with class-specific counts.

    Args:
        original_data_path (str): Path to the directory containing class subdirectories.
        split_data_path (str): Path to store the train/val/test split data.
        split_ratio (list): List of split ratios [train, validation, test].
        seed (int): Random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Get class names (subdirectories)
    class_names = [d for d in os.listdir(original_data_path) if os.path.isdir(os.path.join(original_data_path, d))]

    # Create train, validation, and test directories
    for folder in ['train', 'validation', 'test']:
        folder_path = os.path.join(split_data_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Split and copy images
    for class_name in class_names:
        class_path = os.path.join(original_data_path, class_name)
        images = [f for f in os.listdir(class_path)
                  if os.path.isfile(os.path.join(class_path, f))
                  and f.lower().endswith(tuple(VALID_EXTENSIONS))
                  and '?' not in f]  # Exclude files with '?' in name

        random.shuffle(images)

        num_images = len(images)
        num_train = int(split_ratio[0] * num_images)
        num_validation = int(split_ratio[1] * num_images)
        num_test = num_images - num_train - num_validation

        for i, image in enumerate(images):
            image_path = os.path.join(class_path, image)
            if i < num_train:
                dest_path = os.path.join(split_data_path, 'train', class_name)
            elif i < num_train + num_validation:
                dest_path = os.path.join(split_data_path, 'validation', class_name)
            else:
                dest_path = os.path.join(split_data_path, 'test', class_name)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.copy(image_path, dest_path)

    # Print set sizes with class-specific counts
    for folder in ['train', 'validation', 'test']:
        folder_path = os.path.join(split_data_path, folder)
        print(f'\n{folder} set:')
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):
                num_files = len([f for f in os.listdir(class_path) if f.lower().endswith(tuple(VALID_EXTENSIONS))])
                print(f'  Class {class_name}: {num_files} files')

if __name__ == "__main__":
    # Example usage (adjust data_path as needed)
    # original_data_path = "data/raw"
    original_data_path = "/projects/dsci410_510/Levin_MAED/data/raw_aug"
    # split_data_path = "data/split"
    split_data_path = "/projects/dsci410_510/Levin_MAED/data/split_aug"
    split_ratio = [0.7, 0.15, 0.15]
    seed = 42
    split_data(original_data_path, split_data_path, split_ratio, seed)