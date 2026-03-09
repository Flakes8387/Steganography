"""
Configuration Loader for Training Scripts

Loads parameters from config.yaml and allows command-line overrides.
"""

import yaml
import argparse
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def flatten_config(config: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten nested config dictionary.

    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge config file with command-line arguments.
    Command-line arguments take precedence over config file.

    Args:
        config: Configuration dictionary from YAML
        args: Command-line arguments

    Returns:
        Updated argparse.Namespace with merged values
    """
    # Flatten config for easier access
    flat_config = flatten_config(config)

    # Map config keys to argument names
    config_to_arg_map = {
        'model.image_size': 'image_size',
        'model.message_length': 'message_length',
        'training.batch_size': 'batch_size',
        'training.learning_rate': 'learning_rate',
        'training.max_epochs': 'num_epochs',
        'training.weight_decay': 'weight_decay',
        'data.num_workers': 'num_workers',
        'data.train_dir': 'train_dir',
        'data.val_dir': 'val_dir',
        'data.dataset_type': 'dataset_type',
        'data.max_train_images': 'max_train_images',
        'data.max_val_images': 'max_val_images',
        'data.use_patches': 'use_patches',
        'data.patches_per_image': 'patches_per_image',
        'data.random_crop': 'random_crop',
        'distortions.enable': 'enable_distortions',
        'distortions.jpeg_compression': 'use_jpeg',
        'distortions.gaussian_noise': 'use_noise',
        'distortions.resize_attack': 'use_resize',
        'distortions.color_jitter': 'use_color_jitter',
        'checkpoint.save_freq': 'save_freq',
        'checkpoint.checkpoint_dir': 'checkpoint_dir',
        'logging.log_dir': 'log_dir',
        'device.seed': 'seed',
    }

    # Update args with config values (only if not explicitly set via command line)
    for config_key, arg_name in config_to_arg_map.items():
        if config_key in flat_config:
            config_value = flat_config[config_key]

            # Only use config value if arg wasn't explicitly set
            # (i.e., it's still at its default value)
            if hasattr(args, arg_name):
                # For paths, use config if arg is default
                if arg_name in ['train_dir', 'val_dir', 'checkpoint_dir', 'log_dir']:
                    if config_value is not None:
                        setattr(args, arg_name, config_value)
                # For numeric/boolean values, use config if reasonable
                elif config_value is not None:
                    setattr(args, arg_name, config_value)

    return args


def print_config(config: Dict[str, Any]):
    """
    Pretty print configuration.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    sections = [
        ('Model', 'model'),
        ('Training', 'training'),
        ('Data', 'data'),
        ('Distortions', 'distortions'),
        ('Optimizer', 'optimizer'),
        ('Scheduler', 'scheduler'),
        ('Checkpointing', 'checkpoint'),
        ('Logging', 'logging'),
        ('Device', 'device'),
        ('Loss Weights', 'loss'),
    ]

    for section_name, section_key in sections:
        if section_key in config:
            print(f"\n{section_name}:")
            for key, value in config[section_key].items():
                print(f"  {key:25s}: {value}")

    print("=" * 60 + "\n")


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Configuration saved to {output_path}")


if __name__ == "__main__":
    # Test config loading
    config = load_config("config.yaml")
    print_config(config)
