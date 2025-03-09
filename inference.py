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
import json
import shutil

from src.data_handler.simulate_degrade import process_dataset, simulate_distant_view

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model_from_checkpoint(config, checkpoint_path, num_classes):
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
            transforms.Normalize(mean=mean, std=std)
        ])
    dataset = InferenceDataset(image_dir, transform=inference_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    dataset_for_classes = ImageFolder(image_dir)
    class_names = dataset_for_classes.classes
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(config, checkpoint_path, num_classes).to(device)
    print(f"Loaded model from: {checkpoint_path}")

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            predictions.extend(preds)
            true_labels.extend(labels)

    label_to_index = dataset_for_classes.class_to_idx
    true_indices = [label_to_index[label] for label in true_labels]

    report = classification_report(true_indices, predictions, target_names=class_names, output_dict=True)
    accuracy = accuracy_score(true_indices, predictions)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    return accuracy, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with image degradation effects and inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--effect", type=str, required=True, help="Effect to apply (see config.yaml)")
    parser.add_argument("--start", type=float, required=True, help="Start value of the effect parameter")
    parser.add_argument("--stop", type=float, required=True, help="Stop value of the effect parameter")
    parser.add_argument("--step", type=float, required=True, help="Step size for the effect parameter")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save degraded images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--parameter_name", type=str, required=True, help="Name of the parameter being varied")
    parser.add_argument("--gaussian_noise_mean", type=float, help="Mean for Gaussian noise (optional)")
    parser.add_argument("--gaussian_noise_std", type=float, help="Std for Gaussian noise (optional)")
    args = parser.parse_args()

    config = load_config(args.config)

    results = []

    for param in np.arange(args.start, args.stop + args.step, args.step):
        print(f"Applying {args.effect} with parameter {param}")
        param_config = config["image_degradation"].copy()
        param_config[args.effect] = True

        # Handle gaussian_noise specifically
        if args.effect == "gaussian_noise":
            if args.gaussian_noise_mean is not None:
                param_config["gaussian_noise_mean"] = args.gaussian_noise_mean
            else:
                param_config["gaussian_noise_mean"] = config["image_degradation"].get("gaussian_noise_mean", 0)

            if args.gaussian_noise_std is not None:
                param_config["gaussian_noise_std"] = args.gaussian_noise_std
            else:
                param_config["gaussian_noise_std"] = config["image_degradation"].get("gaussian_noise_std", 1)

            param_config["gaussian_noise_std"] = param

        else:
            # Handle other effects as before
            param_config[f"{args.effect}_factor"] = param
            param_config[f"{args.effect}_radius"] = param
            param_config[f"{args.effect}_quality"] = param
            param_config[f"{args.effect}_alpha"] = param

        param_output_dir = os.path.join(args.output_dir, str(param))
        image_name = process_dataset(args.input_dir, param_output_dir, param_config)
        accuracy, report = inference(config, param_output_dir, args.checkpoint)
        result_img_name = os.path.join('outputs', os.path.basename(image_name)+f"_degraded_{args.effect}_{param}.jpg")
        shutil.copyfile(image_name, result_img_name)
        results.append({"parameter": param, "accuracy": accuracy, "report": report, "image_name": result_img_name})
        try:
            shutil.rmtree(param_output_dir)
        except:
            print(f"\n\nWarning: Unable to delete {param_output_dir}")

    results_to_save = {"parameter_name": args.parameter_name, "results": results}

    with open("outputs/inference_results.json", "w") as f:
        json.dump(results_to_save, f, indent=4)