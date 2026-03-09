"""
Test script to verify DIV2K training requirements are enforced.

Tests:
1. Message length must be 16 or 32 bits (reject 8, 64, 128)
2. Batch size must be 4 (reject 8, 16, 32)
3. Distortions disabled initially
4. Loss weight is 5.0
5. CUDA required (abort if unavailable)
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path
import torch


def run_train_command(args_list):
    """Run training command and capture output."""
    cmd = [sys.executable, 'train.py', '--skip_sanity',
           '--num_epochs', '0'] + args_list
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.returncode, result.stdout, result.stderr


def create_temp_dataset(num_images=5):
    """Create temporary test dataset."""
    from PIL import Image
    import numpy as np

    temp_dir = tempfile.mkdtemp(prefix='test_div2k_')

    # Create DIV2K-like directory structure
    train_dir = Path(temp_dir) / 'DIV2K' / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    # Create test images
    for i in range(num_images):
        img = Image.fromarray(np.random.randint(
            0, 256, (256, 256, 3), dtype=np.uint8))
        img.save(train_dir / f'{i:04d}.png')

    return str(train_dir)


def test_message_length_validation():
    """Test message length must be 16 or 32 bits."""
    print("\n" + "="*60)
    print("TEST 1: Message Length Validation")
    print("="*60)

    train_dir = create_temp_dataset()

    # Test valid message lengths (16, 32)
    print("\n1a. Testing message_length=16 (should PASS)...")
    returncode, stdout, stderr = run_train_command([
        '--train_dir', train_dir,
        '--dataset_type', 'DIV2K',
        '--message_length', '16',
        '--batch_size', '4',
        '--image_size', '128',
        '--num_epochs', '1'
    ])

    if returncode == 0:
        print("   ✓ message_length=16 accepted")
    else:
        print(f"   ✗ FAILED: Should accept message_length=16")
        print(f"   Output: {stdout[-500:]}")
        return False

    print("\n1b. Testing message_length=32 (should PASS)...")
    returncode, stdout, stderr = run_train_command([
        '--train_dir', train_dir,
        '--dataset_type', 'DIV2K',
        '--message_length', '32',
        '--batch_size', '4',
        '--image_size', '128',
        '--num_epochs', '1'
    ])

    if returncode == 0:
        print("   ✓ message_length=32 accepted")
    else:
        print(f"   ✗ FAILED: Should accept message_length=32")
        return False

    # Test invalid message lengths (8, 64, 128)
    for msg_len in [8, 64, 128]:
        print(f"\n1c. Testing message_length={msg_len} (should REJECT)...")
        returncode, stdout, stderr = run_train_command([
            '--train_dir', train_dir,
            '--dataset_type', 'DIV2K',
            '--message_length', str(msg_len),
            '--batch_size', '4',
            '--image_size', '128',
            '--num_epochs', '1'
        ])

        if returncode != 0 and 'Message length must be 16 or 32 bits' in stdout:
            print(f"   ✓ message_length={msg_len} correctly rejected")
        else:
            print(f"   ✗ FAILED: Should reject message_length={msg_len}")
            print(f"   Output: {stdout[-500:]}")
            return False

    print("\n✅ Message length validation: PASSED")
    return True


def test_batch_size_validation():
    """Test batch size must be 4."""
    print("\n" + "="*60)
    print("TEST 2: Batch Size Validation")
    print("="*60)

    train_dir = create_temp_dataset()

    # Test valid batch size (4)
    print("\n2a. Testing batch_size=4 (should PASS)...")
    returncode, stdout, stderr = run_train_command([
        '--train_dir', train_dir,
        '--dataset_type', 'DIV2K',
        '--message_length', '32',
        '--batch_size', '4',
        '--image_size', '128',
        '--num_epochs', '1'
    ])

    if returncode == 0:
        print("   ✓ batch_size=4 accepted")
    else:
        print(f"   ✗ FAILED: Should accept batch_size=4")
        print(f"   Output: {stdout[-500:]}")
        return False

    # Test invalid batch sizes (8, 16, 32)
    for batch_size in [8, 16, 32]:
        print(f"\n2b. Testing batch_size={batch_size} (should REJECT)...")
        returncode, stdout, stderr = run_train_command([
            '--train_dir', train_dir,
            '--dataset_type', 'DIV2K',
            '--message_length', '32',
            '--batch_size', str(batch_size),
            '--image_size', '128',
            '--num_epochs', '1'
        ])

        if returncode != 0 and 'Batch size must be 4' in stdout:
            print(f"   ✓ batch_size={batch_size} correctly rejected")
        else:
            print(f"   ✗ FAILED: Should reject batch_size={batch_size}")
            print(f"   Output: {stdout[-500:]}")
            return False

    print("\n✅ Batch size validation: PASSED")
    return True


def test_cuda_requirement():
    """Test CUDA is required for DIV2K training."""
    print("\n" + "="*60)
    print("TEST 3: CUDA Requirement")
    print("="*60)

    train_dir = create_temp_dataset()

    if not torch.cuda.is_available():
        print("\n3a. CUDA not available - testing rejection (should REJECT)...")
        returncode, stdout, stderr = run_train_command([
            '--train_dir', train_dir,
            '--dataset_type', 'DIV2K',
            '--message_length', '32',
            '--batch_size', '4',
            '--image_size', '128',
            '--num_epochs', '1'
        ])

        if returncode != 0 and 'CUDA is REQUIRED' in stdout:
            print("   ✓ Training correctly rejected without CUDA")
        else:
            print(f"   ✗ FAILED: Should reject training without CUDA")
            print(f"   Output: {stdout[-500:]}")
            return False
    else:
        print(f"\n3a. CUDA available ({torch.cuda.get_device_name(0)})...")
        returncode, stdout, stderr = run_train_command([
            '--train_dir', train_dir,
            '--dataset_type', 'DIV2K',
            '--message_length', '32',
            '--batch_size', '4',
            '--image_size', '128',
            '--num_epochs', '1'
        ])

        if returncode == 0 and 'GPU: CUDA available' in stdout:
            print("   ✓ Training accepted with CUDA")
        else:
            print(f"   ✗ FAILED: Should accept training with CUDA")
            print(f"   Output: {stdout[-500:]}")
            return False

    print("\n✅ CUDA requirement: PASSED")
    return True


def test_config_files():
    """Test DIV2K config files have correct settings."""
    print("\n" + "="*60)
    print("TEST 4: Config File Validation")
    print("="*60)

    import yaml

    configs = {
        'config_div2k_quick.yaml': {'message_length': 16, 'batch_size': 4},
        'config_div2k_balanced.yaml': {'message_length': 32, 'batch_size': 4},
        'config_div2k_full.yaml': {'message_length': 32, 'batch_size': 4},
    }

    all_valid = True

    for config_file, expected in configs.items():
        print(f"\n4. Validating {config_file}...")

        if not os.path.exists(config_file):
            print(f"   ✗ Config file not found: {config_file}")
            all_valid = False
            continue

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Check message_length
        msg_len = config.get('model', {}).get('message_length')
        if msg_len == expected['message_length']:
            print(f"   ✓ message_length: {msg_len}")
        else:
            print(
                f"   ✗ message_length: {msg_len} (expected: {expected['message_length']})")
            all_valid = False

        # Check batch_size
        batch_size = config.get('training', {}).get('batch_size')
        if batch_size == expected['batch_size']:
            print(f"   ✓ batch_size: {batch_size}")
        else:
            print(
                f"   ✗ batch_size: {batch_size} (expected: {expected['batch_size']})")
            all_valid = False

        # Check distortions disabled
        distortions = config.get('distortions', {})
        jpeg_prob = distortions.get('jpeg', {}).get('probability', 0.0)
        if jpeg_prob == 0.0:
            print(
                f"   ✓ distortions: DISABLED initially (JPEG prob: {jpeg_prob})")
        else:
            print(
                f"   ✗ distortions: Should be disabled initially (JPEG prob: {jpeg_prob})")
            all_valid = False

        # Check loss weight
        loss_weight = config.get('training', {}).get('message_loss_weight')
        if loss_weight == 5.0:
            print(f"   ✓ message_loss_weight: {loss_weight}")
        else:
            print(f"   ✗ message_loss_weight: {loss_weight} (expected: 5.0)")
            all_valid = False

    if all_valid:
        print("\n✅ Config file validation: PASSED")
    else:
        print("\n❌ Config file validation: FAILED")

    return all_valid


def test_loss_formula():
    """Test loss formula in training script."""
    print("\n" + "="*60)
    print("TEST 5: Loss Formula Validation")
    print("="*60)

    # Read train.py and check loss formula
    with open('train.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for the loss formula
    if 'total_loss_batch = image_loss + 5.0 * message_loss' in content:
        print("\n✓ Loss formula found: total_loss = image_loss + 5.0 * message_loss")
        print("✅ Loss formula validation: PASSED")
        return True
    else:
        print("\n✗ Loss formula not found or incorrect")
        print("   Expected: total_loss_batch = image_loss + 5.0 * message_loss")
        print("❌ Loss formula validation: FAILED")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DIV2K TRAINING REQUIREMENTS VALIDATION TEST")
    print("="*60)
    print("\nThis test verifies that DIV2K training enforces:")
    print("  1. Message length: 16 or 32 bits")
    print("  2. Batch size: 4")
    print("  3. Distortions: DISABLED initially")
    print("  4. Loss: image_loss + 5.0 * message_loss")
    print("  5. GPU: CUDA required")

    results = []

    # Run tests
    try:
        results.append(("Message Length", test_message_length_validation()))
    except Exception as e:
        print(f"\n❌ Message length test failed with error: {e}")
        results.append(("Message Length", False))

    try:
        results.append(("Batch Size", test_batch_size_validation()))
    except Exception as e:
        print(f"\n❌ Batch size test failed with error: {e}")
        results.append(("Batch Size", False))

    try:
        results.append(("CUDA Requirement", test_cuda_requirement()))
    except Exception as e:
        print(f"\n❌ CUDA requirement test failed with error: {e}")
        results.append(("CUDA Requirement", False))

    try:
        results.append(("Config Files", test_config_files()))
    except Exception as e:
        print(f"\n❌ Config files test failed with error: {e}")
        results.append(("Config Files", False))

    try:
        results.append(("Loss Formula", test_loss_formula()))
    except Exception as e:
        print(f"\n❌ Loss formula test failed with error: {e}")
        results.append(("Loss Formula", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n📋 DIV2K Training Requirements:")
        print("  ✓ Message length: 16 or 32 bits")
        print("  ✓ Batch size: 4")
        print("  ✓ Distortions: DISABLED initially")
        print("  ✓ Loss: image_loss + 5.0 * message_loss")
        print("  ✓ GPU: CUDA required")
        print("\n🚀 Ready to train with:")
        print("  python train.py --config config_div2k_balanced.yaml")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
