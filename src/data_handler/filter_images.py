# src/data/dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import yaml

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

class AircraftDatasetFilter(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.targets = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # create a class to index mapping.
        with open("config.yaml", 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.crop_size = config["transforms"]["crop_size"]

        photos_loaded = {}
        photos_skipped = {}
        # Traverse class subdirectories
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):  # Ensure it's a directory
                for file_name in os.listdir(class_path):
                    if any(file_name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                        # only add pictures of size at least crop_size x crop_size
                        with Image.open(os.path.join(class_path, file_name)) as img:
                            if img.size[0] < self.crop_size and img.size[1] < self.crop_size:
                                if class_name not in photos_skipped:
                                    photos_skipped[class_name] = 0
                                photos_skipped[class_name] += 1

                                os.remove(os.path.join(class_path, file_name))
                                continue
                        if class_name not in photos_loaded:
                            photos_loaded[class_name] = 0
                        photos_loaded[class_name] += 1
                        self.image_files.append(os.path.join(class_path, file_name))
                        self.targets.append(self.class_to_idx[class_name]) #use the class to index mapping.

        print(f"Found {len(self.image_files)} files in {data_dir}")
        print(f"Files found: {self.image_files}")
        print(f"Photos loaded: {photos_loaded}")
        print(f"Photos skipped: {photos_skipped}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.targets[idx] #get the label.
        return image, label #return image and label.

if __name__ == "__main__":
    data_dir = "data/raw"
    filtered_dataset = AircraftDatasetFilter(data_dir)