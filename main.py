import argparse
import os
import yaml
from src.data_handler.split_data import split_data
from train_model import train_model

def load_config(config_path):
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config):
    """Main function to run the aircraft detection pipeline."""
    if config['data']['with_augmentation']:
        original_data_dir = config['data']['original_data_path_aug']
        split_data_dir = config['data']['split_data_path_aug']
    else:
        original_data_dir = config['data']['original_data_path']
        split_data_dir = config['data']['split_data_path']

    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    seed = config['data']['seed']

    if not os.path.exists(os.path.join(split_data_dir, 'train')):
        split_data(original_data_dir,
                split_data_dir,
                [train_split, val_split, test_split],
                seed)
    else:
        print(f"Split data directory '{split_data_dir}' already exists. Skipping data splitting.")
    
    # Train the model then evaluate on test set
    # train_model(config)
    test_accuracy = train_model(config)

    # Save test accuracy to a file
    with open('outputs/test_accuracy.txt', 'w') as f:
        f.write(str(test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Aircraft Early Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)