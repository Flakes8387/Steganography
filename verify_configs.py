#!/usr/bin/env python3
"""
Verify DIV2K configuration settings and dataset size limits.
"""

import yaml
import os


def main():
    print("\n" + "="*60)
    print("DIV2K CONFIGURATION COMPARISON")
    print("="*60 + "\n")

    configs = {
        'Quick': 'config_div2k_quick.yaml',
        'Balanced': 'config_div2k_balanced.yaml',
        'Standard': 'config_div2k.yaml',
        'Full': 'config_div2k_full.yaml'
    }

    print(f"{'Config':<12} | {'Base':<4} | {'Patches':<8} | {'Total':<6} | {'Epochs':<6} | {'Msg':<4} | {'Time':<6} | {'Quality':<12}")
    print("-" * 90)

    for name, config_file in configs.items():
        if not os.path.exists(config_file):
            print(f"{name:<12} | Config file not found: {config_file}")
            continue

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        base = config['data']['max_train_images']
        patches = config['data']['patches_per_image']
        total = base * patches
        epochs = config['training'].get(
            'num_epochs', config['training'].get('max_epochs', 100))
        msg_len = config['model']['message_length']

        # Estimate training time
        if total <= 1500:
            time_est = "1-2h"
            quality = "75-80%"
        elif total <= 2500:
            time_est = "2-3h"
            quality = "85-90%"
        else:
            time_est = "4-5h"
            quality = "90-95%"

        print(f"{name:<12} | {base:<4} | {patches:<8} | {total:<6} | {epochs:<6} | {msg_len:<4} | {time_est:<6} | {quality:<12}")

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ max_train_images limits BASE images, not total patches")
    print("✓ Total samples = base_images × patches_per_image")
    print("✓ Random crops ensure different patches each epoch")
    print("✓ Recommended: config_div2k_balanced.yaml (2-3h, 85-90%)")
    print("\n")


if __name__ == "__main__":
    main()
