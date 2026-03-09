"""
Test script to verify the dataset verification step works correctly.
Creates a small synthetic dataset and runs verification.
"""

import sys
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np


def create_test_dataset(num_images=10):
    """Create temporary test dataset."""
    temp_dir = tempfile.mkdtemp(prefix='test_dataset_verify_')
    train_dir = Path(temp_dir) / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_images} test images in {train_dir}...")

    # Create test images
    for i in range(num_images):
        img = Image.fromarray(
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        )
        img.save(train_dir / f'test_{i:04d}.png')

    print(f"Created {num_images} images")
    return str(train_dir)


def main():
    """Run dataset verification test."""
    print("\n" + "="*60)
    print("DATASET VERIFICATION TEST")
    print("="*60)

    # Create test dataset
    train_dir = create_test_dataset(10)

    # Run training with verification (will stop after verification)
    print(f"\nRunning training with dataset verification...\n")

    import subprocess
    cmd = [
        sys.executable, 'train.py',
        '--train_dir', train_dir,
        '--message_length', '32',
        '--batch_size', '4',
        '--image_size', '128',
        '--num_epochs', '1',
        '--skip_sanity'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    # Print output
    print(result.stdout)

    if result.stderr:
        print("STDERR:", result.stderr)

    # Check if verification section appears
    if 'DATASET VERIFICATION' in result.stdout:
        print("\n" + "="*60)
        print("✅ Dataset verification step found in output")
        print("="*60)

        # Check for key components
        checks = {
            'Batch Shapes': 'Batch Shapes:' in result.stdout,
            'Tensor Properties': 'Tensor Properties:' in result.stdout,
            'First Sample': 'First Sample in Batch:' in result.stdout,
            'Validation Checks': 'Validation Checks:' in result.stdout,
            'Sample saved': 'Saved sample image:' in result.stdout or 'Could not save' in result.stdout,
        }

        print("\nVerification components:")
        all_passed = True
        for name, present in checks.items():
            status = "✅" if present else "❌"
            print(f"  {status} {name}")
            if not present:
                all_passed = False

        if all_passed:
            print("\n✅ All verification components present!")
            return 0
        else:
            print("\n❌ Some verification components missing")
            return 1
    else:
        print("\n❌ Dataset verification step not found in output")
        print("Output preview:", result.stdout[:500])
        return 1


if __name__ == '__main__':
    sys.exit(main())
