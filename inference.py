import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pytorch_lightning as pl
import argparse
import yaml
from torchvision.datasets import ImageFolder

def load_config(config_path):
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_best_model(config, checkpoint_dir):
    """Loads the best model based on the configuration."""
    arch = config['model']['architecture']
    num_classes = config['data']['num_classes']

    # Find the best checkpoint file
    best_checkpoint = None
    best_val_loss = float('inf')

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".ckpt") and arch in filename:
            try:
                val_loss = float(filename.split("loss/val=")[1].split(".ckpt")[0])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = os.path.join(checkpoint_dir, filename)
            except IndexError:
                print(f"Warning: Could not extract validation loss from filename: {filename}")
                continue

    if best_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint file found for architecture '{arch}' in directory '{checkpoint_dir}'")

    # Load the model
    if arch == "resnet50":
        from src.models.architectures import ResNet50Classifier
        model = ResNet50Classifier.load_from_checkpoint(best_checkpoint, num_classes=num_classes)
    elif arch == "scratch":
        from src.models.architectures import Scratch
        model = Scratch.load_from_checkpoint(best_checkpoint, num_classes=num_classes)
    elif arch == "convnext":
        from src.models.architectures import ConvNeXtClassifier
        model = ConvNeXtClassifier.load_from_checkpoint(best_checkpoint, num_classes=num_classes)
    else:
        raise ValueError(f"Invalid model architecture: {arch}")

    return model, best_checkpoint

def inference(config, image_dir, output_file):
    """Runs inference on images in a directory using the best trained model."""

    checkpoint_dir = "checkpoints"  # Directory where checkpoints are saved
    model, checkpoint_path = load_best_model(config, checkpoint_dir)
    print(f"Loaded best model from: {checkpoint_path}")

    model.eval()

    # 2. Load Data
    class InferenceDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path
    
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

    # 3. Run Inference
    predictions = []
    image_paths = []

    with torch.no_grad():
        for images, paths in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            predictions.extend(preds)
            image_paths.extend(paths)

    # 4. Process Predictions
    dataset_for_classes = ImageFolder(image_dir)
    class_names = dataset_for_classes.classes

    # Construct the path to the outputs directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(current_dir, "..", "..", "outputs")

    # Ensure the output directory exists
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # Construct the full path to the output file
    output_filepath = os.path.join(outputs_dir, output_file)

    with open(output_filepath, 'w') as f:
        for path, pred in zip(image_paths, predictions):
            class_name = class_names[pred]
            f.write(f"{path}: {class_name}\n")
    print(f"Predictions saved to {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Aircraft Early Detection Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)

    image_directory = "path/to/your/inference_images"  # Replace with your image directory
    output_filename = "inference_results.txt"  # Replace with your desired output filename

    inference(config, image_directory, output_filename)