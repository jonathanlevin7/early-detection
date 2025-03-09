import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.architectures import ResNet50Classifier, Scratch, ConvNeXtClassifier
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data_handler.dataloader import get_data_loaders
import yaml
import os
import argparse
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import json
from sklearn.metrics import classification_report

# def load_config(config_path):
#     """Loads the configuration from a YAML file."""
#     with open(config_path, 'r') as file:
#         return yaml.safe_load(file)

def save_confusion_matrix(conf_matrix, class_names, output_dir='outputs'):
    """Saves the confusion matrix as a JSON file."""
    output_dir = str(output_dir)  # Ensure output_dir is a string
    os.makedirs(output_dir, exist_ok=True)
    
    conf_matrix_data = {
        "confusion_matrix": conf_matrix.cpu().tolist(),
        "class_names": class_names
    }
    
    with open(os.path.join(output_dir, "confusion_matrix.json"), "w") as f:
        json.dump(conf_matrix_data, f, indent=4)
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.json")

def calculate_confusion_matrix(model, test_loader, num_classes):
    """Calculates and saves the confusion matrix."""
    conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(model.device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    conf_matrix.update(all_preds, all_labels)
    save_confusion_matrix(conf_matrix.compute(), [str(i) for i in range(num_classes)])

def calculate_classification_report(model, test_loader, num_classes):
    """Calculates and prints the classification report."""
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)])
    print("Classification Report:\n", report)

def train_model(config):
    """
    Trains a model using the configuration from a YAML file.

    Args:
        config: configuration.
    """
    # config = load_config(config)

    # Seed
    seed = config['data']['seed']
    pl.seed_everything(seed)

    # Data Loading
    if config['data']['with_augmentation']:
        # original_data_dir = config['data']['original_data_path_aug']
        data_dir = config['data']['split_data_path_aug']
    else:
        # original_data_dir = config['data']['original_data_path']
        data_dir = config['data']['split_data_path']
    
    # data_dir = config['data']['split_data_path']
    crop_size = config['transforms']['crop_size']
    batch_size = config['training']['batch_size']

    train_loader, val_loader, test_loader, num_classes, mean, std = get_data_loaders(
        data_dir, crop_size, batch_size
    )

    print(f"Number of classes: {num_classes}")

    # Model Initialization
    epochs = config['training']['epochs']
    arch = config['model']['architecture']
    lr = config['training']['learning_rate']

    early_stop_min_delta = config['training']['early_stopping']['min_delta']
    early_stop_patience = config['training']['early_stopping']['patience']

    # if architecture == "resnet50":
    #     model = ResNet50Classifier(num_classes=num_classes, learning_rate=learning_rate)
    # else:
    #     raise ValueError(f"Unsupported architecture: {architecture}")

    # # Checkpoint Callback
    # os.makedirs('./checkpoints', exist_ok=True) # Ensure Checkpoint directory exists.
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='./checkpoints',
    #     filename='resnet50-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=3,
    #     mode='min',
    # )

    # # Trainer
    # trainer = pl.Trainer(
    #     max_epochs=epochs,
    #     callbacks=[checkpoint_callback],
    #     accelerator='auto',
    #     devices='auto',
    # )
    # Load the Model
    if arch == "resnet50":
        model = ResNet50Classifier(num_classes=num_classes, learning_rate=lr)
    elif arch == "scratch":
        model = Scratch(num_classes=num_classes, learning_rate=lr)
    elif arch == "convnext":
        model = ConvNeXtClassifier(num_classes=num_classes, learning_rate=lr)
    else:
        raise ValueError(f"Invalid model architecture: {arch}")

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='loss/val',
        dirpath='./checkpoints',
        filename=f'{arch}-{{epoch:02d}}-{{loss/val:.2f}}',
        # filename=f'{arch}' + '-{epoch:02d}-{loss/val:.2f}',
        save_top_k=3,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='loss/val',
        min_delta=early_stop_min_delta,
        patience=early_stop_patience,
        verbose=False,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices='auto'
    )

    # Training and Testing
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Save confusion matrix data
    calculate_confusion_matrix(model, test_loader, num_classes) # Corrected function call

    # Calculate and print classification report
    calculate_classification_report(model, test_loader, num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('-c', "--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
    else:
        train_model(args.config)