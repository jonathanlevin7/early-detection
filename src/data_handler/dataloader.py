# src/data/dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import yaml

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
        transforms.Resize(crop_size),
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

def get_data_loaders(split_data_dir, crop_size, batch_size, full_transform = True):
    """Create DataLoaders for train, validation, and test sets."""

    if full_transform:
        mean, std = calculate_normalization_values(os.path.join(split_data_dir, 'train'), crop_size, batch_size)

        final_transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)  # Use calculated mean and std
        ])
    else:
        mean = torch.zeros(3)
        std = torch.ones(3)
        final_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)  # Use calculated
        ])

    train_dir = os.path.join(split_data_dir, 'train')
    val_dir = os.path.join(split_data_dir, 'validation')
    test_dir = os.path.join(split_data_dir, 'test')

    train_dataset = AircraftDataset(train_dir, transform=final_transform)
    val_dataset = AircraftDataset(val_dir, transform=final_transform)
    test_dataset = AircraftDataset(test_dir, transform=final_transform)

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, mean, std

# Example usage (in your main.py or notebook)
if __name__ == "__main__":
    # Example usage.
    mean, std = calculate_normalization_values("data/split/train", crop_size=256, batch_size=16)
    print(f"Calculated mean: {mean}")
    print(f"Calculated std: {std}")