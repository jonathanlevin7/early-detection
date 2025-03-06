import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pytorch_lightning as pl
import argparse
import yaml
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def load_config(config_path):
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model_from_checkpoint(config, checkpoint_path, num_classes):
    """Loads a model from a specified checkpoint path."""
    arch = config['model']['architecture']

    if arch == "resnet50":
        from src.models.architectures import ResNet50Classifier
        model = ResNet50Classifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    elif arch == "scratch":
        from src.models.architectures import Scratch
        model = Scratch.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    elif arch == "convnext":
        from src.models.architectures import ConvNeXtClassifier
        model = ConvNeXtClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    else:
        raise ValueError(f"Invalid model architecture: {arch}")

    return model

def inference(config, image_dir, checkpoint_path):
    """Runs inference on images in a directory using the specified checkpoint and returns metrics."""

    # 2. Load Data
    class InferenceDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_paths = []
            self.labels = []
            for class_name in os.listdir(image_dir):
                class_path = os.path.join(image_dir, class_name)
                if os.path.isdir(class_path):
                    for file_name in os.listdir(class_path):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                            self.image_paths.append(os.path.join(class_path, file_name))
                            self.labels.append(class_name)
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]

    crop_size = config['transforms']['crop_size']
    mean = config['transforms']['mean']
    std = config['transforms']['std']
    batch_size = config['training']['batch_size']

    inference_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)  # Use calculated mean and std
        ])
    dataset = InferenceDataset(image_dir, transform=inference_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Calculate num_classes dynamically
    dataset_for_classes = ImageFolder(image_dir)
    class_names = dataset_for_classes.classes
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(config, checkpoint_path, num_classes).to(device)
    print(f"Loaded model from: {checkpoint_path}")

    model.eval()

    # 3. Run Inference
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            predictions.extend(preds)
            true_labels.extend(labels)

    # 4. Process Predictions and Evaluate
    label_to_index = dataset_for_classes.class_to_idx

    true_indices = [label_to_index[label] for label in true_labels]

    report = classification_report(true_indices, predictions, target_names=class_names)
    accuracy = accuracy_score(true_indices, predictions)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Aircraft Early Detection Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    # parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()
    config = load_config(args.config)

    image_directory = "/projects/dsci410_510/Levin_MAED/data/test_degraded"
    checkpoint_path = "./checkpoints/convnext-epoch=40-loss/val=0.21.ckpt"
    inference(config, image_directory, checkpoint_path)