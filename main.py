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
from src.data_handler.dataloader import get_data_loaders
from src.models.architectures import get_resnet50

def load_config(config_path):
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config):
    """Main function to run the aircraft detection pipeline."""

    original_data_dir = config['data']['original_data_path']
    split_data_dir = config['data']['split_data_path']
    crop_size = config['transforms']['crop_size']
    batch_size = config['training']['batch_size']

    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    seed = config['data']['seed']

    # 1. Split the data
    split_data(original_data_dir,
               split_data_dir,
               [train_split, val_split, test_split],
               seed)
    
    train_loader, val_loader, test_loader, num_classes, _, _ = get_data_loaders(split_data_dir, crop_size, batch_size)

    # 6. Load the Model
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