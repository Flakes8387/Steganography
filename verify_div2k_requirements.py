"""
Simple verification script for DIV2K training requirements.
Tests the config files directly without running training.
"""

import yaml
import torch
import sys


def test_config_file(config_path):
    """Test a single config file."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_path}")
    print('='*60)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    errors = []

    # 1. Check message_length
    msg_len = config.get('model', {}).get('message_length')
    if msg_len in [16, 32]:
        print(f"OK Message length: {msg_len} bits")
    else:
        errors.append(
            f"ERROR: Message length must be 16 or 32 (got: {msg_len})")

    # 2. Check batch_size
    batch_size = config.get('training', {}).get('batch_size')
    if batch_size == 4:
        print(f"OK Batch size: {batch_size}")
    else:
        errors.append(f"ERROR: Batch size must be 4 (got: {batch_size})")

    # 3. Check distortions disabled
    distortions = config.get('distortions', {})
    jpeg_prob = distortions.get('jpeg', {}).get('probability', 0.0)
    if jpeg_prob == 0.0:
        print(f"OK Distortions: DISABLED initially (JPEG prob: {jpeg_prob})")
    else:
        errors.append(
            f"ERROR: Distortions should be disabled (JPEG prob: {jpeg_prob})")

    # 4. Check loss weight
    loss_weight = config.get('training', {}).get('message_loss_weight')
    if loss_weight == 5.0:
        print(f"OK Loss weight: {loss_weight}")
    else:
        errors.append(f"ERROR: Loss weight should be 5.0 (got: {loss_weight})")

    # 5. Check GPU setting
    gpu_device = config.get('gpu', {}).get('device')
    if gpu_device == 'cuda':
        print(f"OK GPU: CUDA required")
    else:
        errors.append(f"ERROR: GPU must be 'cuda' (got: {gpu_device})")

    if errors:
        print("\nERRORS FOUND:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("\nPASSED: All requirements met")
        return True


def test_cuda_available():
    """Test CUDA availability."""
    print(f"\n{'='*60}")
    print("Testing: CUDA Availability")
    print('='*60)

    if torch.cuda.is_available():
        print(f"OK CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"OK CUDA version: {torch.version.cuda}")
        return True
    else:
        print("ERROR: CUDA not available")
        print("  - Install NVIDIA GPU drivers")
        print("  - Install PyTorch with CUDA support")
        return False


def test_loss_formula():
    """Test loss formula in training script."""
    print(f"\n{'='*60}")
    print("Testing: Loss Formula in train.py")
    print('='*60)

    try:
        with open('train.py', 'r', encoding='utf-8') as f:
            content = f.read()

        if 'total_loss_batch = image_loss + 5.0 * message_loss' in content:
            print("OK Loss formula: total_loss = image_loss + 5.0 * message_loss")
            return True
        else:
            print("ERROR: Correct loss formula not found")
            return False
    except Exception as e:
        print(f"ERROR: Could not read train.py: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DIV2K TRAINING REQUIREMENTS VERIFICATION")
    print("="*60)
    print("\nChecking requirements:")
    print("  1. Message length: 16 or 32 bits")
    print("  2. Batch size: 4")
    print("  3. Distortions: DISABLED initially")
    print("  4. Loss: image_loss + 5.0 * message_loss")
    print("  5. GPU: CUDA required")

    results = {}

    # Test config files
    config_files = [
        'config_div2k_quick.yaml',
        'config_div2k_balanced.yaml',
        'config_div2k_full.yaml'
    ]

    for config_file in config_files:
        try:
            results[config_file] = test_config_file(config_file)
        except Exception as e:
            print(f"\nERROR testing {config_file}: {e}")
            results[config_file] = False

    # Test CUDA
    results['CUDA'] = test_cuda_available()

    # Test loss formula
    results['Loss Formula'] = test_loss_formula()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nReady to train with DIV2K:")
        print("  python train.py --config config_div2k_balanced.yaml")
        print("\nRequirements enforced:")
        print("  - Message length: 16 or 32 bits")
        print("  - Batch size: 4")
        print("  - Distortions: DISABLED initially")
        print("  - Loss: image_loss + 5.0 * message_loss")
        print("  - GPU: CUDA required")
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
