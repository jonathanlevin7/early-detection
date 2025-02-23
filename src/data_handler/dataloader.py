# src/data/dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

class AircraftDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.targets = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # create a class to index mapping.

        # Traverse class subdirectories
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):  # Ensure it's a directory
                for file_name in os.listdir(class_path):
                    if any(file_name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                        self.image_files.append(os.path.join(class_path, file_name))
                        self.targets.append(self.class_to_idx[class_name]) #use the class to index mapping.

        print(f"Found {len(self.image_files)} files in {data_dir}")
        print(f"Files found: {self.image_files}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.targets[idx] #get the label.
        return image, label #return image and label.

def calculate_normalization_values(data_dir, crop_size, batch_size):
    """Calculates normalization mean and std from raw images."""

    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform,
        is_valid_file=lambda x: any(x.lower().endswith(ext) for ext in VALID_EXTENSIONS)
    )

    # Manually exclude any directories that don't contain valid images
    valid_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != '.ipynb_checkpoints']
    dataset.samples = [(path, target) for path, target in dataset.samples if path.split(os.sep)[-2] in [os.path.basename(folder) for folder in valid_folders]]

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

# Example usage (in your main.py or notebook)
if __name__ == "__main__":
    # Example usage.
    mean, std = calculate_normalization_values("data/split/train", crop_size=256, batch_size=16)
    print(f"Calculated mean: {mean}")
    print(f"Calculated std: {std}")