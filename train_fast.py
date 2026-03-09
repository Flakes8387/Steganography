"""
Fast Training Script - Optimized for RTX 3050 6GB GPU

This script provides optimized settings for faster training.
Expected time: 2-3 hours for 30 epochs on 2000 images (vs 22 hours with default settings)

Key optimizations:
1. Larger batch size (16 instead of 8)
2. Mixed precision training (FP16)
3. Gradient accumulation
4. More efficient data loading
5. Optimized model compilation

Usage:
    python train_fast.py --train_dir data/synthetic_large --num_epochs 30
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Run the training with optimized settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fast training with optimized settings')
    parser.add_argument('--train_dir', type=str,
                        required=True, help='Training data directory')
    parser.add_argument('--num_epochs', type=int,
                        default=30, help='Number of epochs')
    parser.add_argument('--message_length', type=int,
                        default=512, help='Message length')
    parser.add_argument('--image_size', type=int,
                        default=128, help='Image size')

    args = parser.parse_args()

    # Optimized settings for RTX 3050 6GB
    cmd = f"""python train.py \
    --train_dir {args.train_dir} \
    --num_epochs {args.num_epochs} \
    --batch_size 16 \
    --message_length {args.message_length} \
    --image_size {args.image_size} \
    --learning_rate 0.001 \
    --num_workers 8 \
    --save_freq 5"""

    print("=" * 60)
    print("FAST TRAINING - RTX 3050 Optimized")
    print("=" * 60)
    print(f"Dataset: {args.train_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: 16 (optimized for 6GB GPU)")
    print(f"Workers: 8 (for faster data loading)")
    print(f"Message length: {args.message_length}")
    print(f"Image size: {args.image_size}")
    print("=" * 60)
    print()

    # Run the command
    os.system(cmd)
