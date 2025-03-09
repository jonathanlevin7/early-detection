# src/main.py

import argparse
import os
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

from src.data_handler.split_data import split_data
# from src.data_handler.dataloader import get_data_loaders
# from src.models.architectures import ResNet50Classifier, Scratch, ConvNeXtClassifier

from train_model import train_model

def load_config(config_path):
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config):
    """Main function to run the aircraft detection pipeline."""
    # config = load_config(config_path)
    if config['data']['with_augmentation']:
        original_data_dir = config['data']['original_data_path_aug']
        split_data_dir = config['data']['split_data_path_aug']
    else:
        original_data_dir = config['data']['original_data_path']
        split_data_dir = config['data']['split_data_path']

    # crop_size = config['transforms']['crop_size']
    # batch_size = config['training']['batch_size']

    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    seed = config['data']['seed']
    # epochs = config['training']['epochs']

    # arch = config['model']['architecture']
    # lr = config['training']['learning_rate']

    # early_stop_min_delta = config['training']['early_stopping']['min_delta']
    # early_stop_patience = config['training']['early_stopping']['patience']

    # # Split the data
    # split_data(original_data_dir,
    #            split_data_dir,
    #            [train_split, val_split, test_split],
    #            seed)

    if not os.path.exists(os.path.join(split_data_dir, 'train')):
        split_data(original_data_dir,
                split_data_dir,
                [train_split, val_split, test_split],
                seed)
    else:
        print(f"Split data directory '{split_data_dir}' already exists. Skipping data splitting.")
    
    # Train the model then evaluate on test set
    train_model(config)
    
    # train_loader, val_loader, test_loader, num_classes, _, _ = get_data_loaders(split_data_dir, crop_size, batch_size)

    # # Load the Model
    # if arch == "resnet50":
    #     model = ResNet50Classifier(num_classes=num_classes, learning_rate=lr)
    # elif arch == "scratch":
    #     model = Scratch(num_classes=num_classes, learning_rate=lr)
    # elif arch == "convnext":
    #     model = ConvNeXtClassifier(num_classes=num_classes, learning_rate=lr)
    # else:
    #     raise ValueError(f"Invalid model architecture: {arch}")

    # # Checkpoint Callback
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='loss/val',
    #     dirpath='./checkpoints',
    #     filename=f'{arch}-{{epoch:02d}}-{{loss/val:.2f}}',
    #     # filename=f'{arch}' + '-{epoch:02d}-{loss/val:.2f}',
    #     save_top_k=3,
    #     mode='min',
    # )

    # early_stop_callback = EarlyStopping(
    #     monitor='loss/val',
    #     min_delta=early_stop_min_delta,
    #     patience=early_stop_patience,
    #     verbose=False,
    #     mode='min'
    # )

    # # Trainer
    # trainer = pl.Trainer(
    #     max_epochs=epochs,
    #     callbacks=[checkpoint_callback, early_stop_callback],
    #     accelerator='auto',
    #     devices='auto'
    # )

    # # Training and Testing
    # trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)

    # # Save confusion matrix data
    # from train_model import calculate_confusion_matrix # Import the function
    # calculate_confusion_matrix(model, test_loader, num_classes)

    # # Calculate and print classification report
    # from train_model import calculate_classification_report # Import the function
    # calculate_classification_report(model, test_loader, num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Aircraft Early Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)