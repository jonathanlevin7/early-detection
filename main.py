# src/main.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml

from src.data_handler.split_data import split_data
from src.data_handler.dataloader import AircraftDataset, calculate_normalization_values
from src.models.architectures import get_resnet50

def load_config(config_path):
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config):
    """Main function to run the aircraft detection pipeline."""

    # 1. Split the data
    split_data(config['data']['original_data_path'],
               config['data']['split_data_path'],
               [config['data']['train_split'], config['data']['val_split'], config['data']['test_split']],
               config['data']['seed'])

    # 2. Calculate normalization values
    train_dir = os.path.join(config['data']['split_data_path'], 'train')
    mean, std = calculate_normalization_values(train_dir, config['transforms']['crop_size'], config['training']['batch_size'])

    # 3. Define transforms with calculated mean and std
    final_transform = transforms.Compose([
        transforms.CenterCrop(config['transforms']['crop_size']),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Use calculated mean and std
    ])

    # 4. Create datasets
    train_dataset = AircraftDataset(train_dir, transform=final_transform)
    val_dataset = AircraftDataset(os.path.join(config['data']['split_data_path'], 'validation'), transform=final_transform)
    test_dataset = AircraftDataset(os.path.join(config['data']['split_data_path'], 'test'), transform=final_transform)

    # Debugging: Print dataset sizes
    print(f"Training data directory: {train_dir}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 6. Load the Model
    num_classes = len(train_dataset.classes)
    if config['model']['architecture'] == "get_resnet50":
        model = get_resnet50(num_classes)
    # elif config['model']['architecture'] == "get_custom_cnn":
    #     model = get_custom_cnn(num_classes)
    # elif config['model']['architecture'] == "get_efficientnet":
    #     model = get_efficientnet(num_classes)
    else:
        raise ValueError(f"Invalid model architecture: {config['model']['architecture']}")

    # 7. Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config['training']['learning_rate'])

    # 8. Basic Print to test data loading.
    images, labels = next(iter(train_loader))
    print(f"Batch of images shape: {images.shape}, labels shape: {labels.shape}")
    print(f"Model: {model}")

    # 9. Model Training and Evaluation will go here in the future.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Aircraft Early Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)