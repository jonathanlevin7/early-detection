# train_model.py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models.architectures import ResNet50Classifier
from src.data_handler.dataloader import get_data_loaders
import yaml
import os
import argparse

def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config_path):
    """
    Trains a model using the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.
    """

    config = load_config(config_path)

    # Data Loading
    data_dir = config['data']['split_data_path']
    crop_size = config['transforms']['crop_size']
    batch_size = config['training']['batch_size']

    train_loader, val_loader, test_loader, num_classes, mean, std = get_data_loaders(
        data_dir, crop_size, batch_size
    )

    # Model Initialization
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['epochs']
    architecture = config['model']['architecture']

    if architecture == "resnet50":
        model = ResNet50Classifier(num_classes=num_classes, learning_rate=learning_rate)
    else:
        pass

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='resnet50-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices='auto',
    )

    # Training and Testing
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('-c', "--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
    else:
        train_model(args.config)