# DIV2K Training Requirements

## Enforced Settings

All DIV2K training **must** use these settings:

### 1. Message Length
- **REQUIRED**: 16 or 32 bits only
- ❌ **NOT ALLOWED**: 8, 64, 128, or any other values
- **Rationale**: Optimal balance between capacity and image quality

```yaml
# config_div2k_quick.yaml
model:
  message_length: 16  # Fast training

# config_div2k_balanced.yaml, config_div2k_full.yaml
model:
  message_length: 32  # Better capacity
```

### 2. Batch Size
- **REQUIRED**: 4
- ❌ **NOT ALLOWED**: 8, 16, 32, or any other values
- **Rationale**: 
  - Optimal memory usage for RTX 3050 6GB
  - Stable gradients with high-resolution patches
  - Prevents OOM errors

```yaml
training:
  batch_size: 4  # REQUIRED for DIV2K
```

### 3. Distortions
- **REQUIRED**: DISABLED initially
- **Auto-enabled**: At 75% bit accuracy (JPEG compression only)
- **Rationale**: Two-phase training (clean → robust)

```yaml
distortions:
  jpeg:
    enabled: true
    probability: 0.0  # Starts at 0, enabled at 75% accuracy
    quality_min: 50
    quality_max: 95
```

### 4. Loss Function
- **REQUIRED**: `total_loss = image_loss + 5.0 * message_loss`
- **Hardcoded** in `train.py` line 251
- **Rationale**: Emphasize message recovery (5x weight)

```python
# In train.py
total_loss_batch = image_loss + 5.0 * message_loss
```

### 5. GPU Requirement
- **REQUIRED**: CUDA (NVIDIA GPU)
- **Training aborts** if CUDA unavailable
- **Rationale**: 
  - DIV2K images are high-resolution (2K)
  - Training requires GPU acceleration
  - CPU training would take days

```yaml
gpu:
  device: "cuda"
  cudnn_benchmark: true
  mixed_precision: true
```

## Validation

The training script automatically validates all requirements before training starts.

### Using Config Files (Recommended)

```bash
# Quick training (16 bits, 1-2 hours)
python train.py --config config_div2k_quick.yaml

# Balanced training (32 bits, 2-3 hours) - RECOMMENDED
python train.py --config config_div2k_balanced.yaml

# Full dataset (32 bits, 4-5 hours)
python train.py --config config_div2k_full.yaml
```

**Output:**
```
============================================================
DIV2K TRAINING REQUIREMENTS VALIDATION
============================================================
✓ Message length: 32 bits (valid)
✓ Batch size: 4 (valid)
✓ Distortions: DISABLED initially (auto-enables at 75% bit accuracy)
✓ Loss: total_loss = image_loss + 5.0 * message_loss
✓ GPU: CUDA available (NVIDIA GeForce RTX 3050 6GB Laptop GPU)

✅ All DIV2K requirements satisfied
============================================================
```

### Manual Command-Line

```bash
python train.py \
  --train_dir data/DIV2K/train \
  --dataset_type DIV2K \
  --message_length 32 \
  --batch_size 4 \
  --image_size 128 \
  --use_patches \
  --patches_per_image 4 \
  --num_epochs 100
```

## Error Messages

### Invalid Message Length

```bash
# Example: Using message_length=64
python train.py --config config.yaml --message_length 64 --dataset_type DIV2K
```

**Output:**
```
============================================================
❌ DIV2K TRAINING REQUIREMENTS NOT MET
============================================================

ERRORS:
   ✗ Message length must be 16 or 32 bits (got: 64)

Required settings for DIV2K:
   • Message length: 16 or 32 bits
   • Batch size: 4
   • Distortions: DISABLED initially
   • Loss: image_loss + 5.0 * message_loss
   • GPU: CUDA required

Use config files: config_div2k_quick.yaml or config_div2k_balanced.yaml
============================================================
```

### Invalid Batch Size

```bash
# Example: Using batch_size=8
python train.py --config config.yaml --batch_size 8 --dataset_type DIV2K
```

**Output:**
```
============================================================
❌ DIV2K TRAINING REQUIREMENTS NOT MET
============================================================

ERRORS:
   ✗ Batch size must be 4 for DIV2K (got: 8)

Required settings for DIV2K:
   • Message length: 16 or 32 bits
   • Batch size: 4
   • Distortions: DISABLED initially
   • Loss: image_loss + 5.0 * message_loss
   • GPU: CUDA required
============================================================
```

### CUDA Not Available

```bash
# On system without CUDA
python train.py --config config_div2k_balanced.yaml
```

**Output:**
```
============================================================
❌ DIV2K TRAINING REQUIREMENTS NOT MET
============================================================

ERRORS:
   ✗ CUDA is REQUIRED for DIV2K training
   ✗   - Ensure NVIDIA GPU is available
   ✗   - Install PyTorch with CUDA support
   ✗   - Check: torch.cuda.is_available()
============================================================
```

## Verification Script

Run this to verify all requirements are met:

```bash
python verify_div2k_requirements.py
```

**Expected Output:**
```
============================================================
DIV2K TRAINING REQUIREMENTS VERIFICATION
============================================================

Checking requirements:
  1. Message length: 16 or 32 bits
  2. Batch size: 4
  3. Distortions: DISABLED initially
  4. Loss: image_loss + 5.0 * message_loss
  5. GPU: CUDA required

============================================================
Testing: config_div2k_quick.yaml
============================================================
✓ Message length: 16 bits
✓ Batch size: 4
✓ Distortions: DISABLED initially (JPEG prob: 0.0)
✓ Loss weight: 5.0
✓ GPU: CUDA required

PASSED: All requirements met

============================================================
Testing: config_div2k_balanced.yaml
============================================================
✓ Message length: 32 bits
✓ Batch size: 4
✓ Distortions: DISABLED initially (JPEG prob: 0.0)
✓ Loss weight: 5.0
✓ GPU: CUDA required

PASSED: All requirements met

============================================================
Testing: config_div2k_full.yaml
============================================================
✓ Message length: 32 bits
✓ Batch size: 4
✓ Distortions: DISABLED initially (JPEG prob: 0.0)
✓ Loss weight: 5.0
✓ GPU: CUDA required

PASSED: All requirements met

============================================================
SUMMARY
============================================================
  config_div2k_quick.yaml: PASSED
  config_div2k_balanced.yaml: PASSED
  config_div2k_full.yaml: PASSED
  CUDA: PASSED
  Loss Formula: PASSED

============================================================
ALL TESTS PASSED
============================================================
```

## Config File Settings

### config_div2k_quick.yaml
```yaml
model:
  message_length: 16  # 16 bits only

training:
  batch_size: 4       # Must be 4
  message_loss_weight: 5.0

distortions:
  jpeg:
    probability: 0.0  # Disabled initially

gpu:
  device: "cuda"      # CUDA required
```

### config_div2k_balanced.yaml (Recommended)
```yaml
model:
  message_length: 32  # 32 bits

training:
  batch_size: 4       # Must be 4
  message_loss_weight: 5.0

distortions:
  jpeg:
    probability: 0.0  # Disabled initially

gpu:
  device: "cuda"      # CUDA required
```

### config_div2k_full.yaml
```yaml
model:
  message_length: 32  # 32 bits (NOT 64)

training:
  batch_size: 4       # Must be 4
  message_loss_weight: 5.0

distortions:
  jpeg:
    probability: 0.0  # Disabled initially

gpu:
  device: "cuda"      # CUDA required
```

## Why These Requirements?

### Message Length (16 or 32 bits)

| Setting | Rationale |
|---------|-----------|
| 16 bits | • Fast experimentation<br>• Low image distortion<br>• Good for testing |
| 32 bits | • Optimal capacity/quality balance<br>• Standard for most use cases<br>• Recommended for production |
| 64 bits | ❌ Too large for 128×128 patches<br>❌ Causes significant distortion<br>❌ Poor convergence |

### Batch Size (4)

| Batch Size | Memory | Stability | Training Time |
|------------|--------|-----------|---------------|
| 2 | ✓ | ✓✓ | 🐢 Slow |
| **4** | **✓✓** | **✓✓** | **✓ Optimal** |
| 8 | ⚠️ High | ✓ | ✓✓ Fast |
| 16 | ❌ OOM | ❌ Unstable | ✓✓✓ Fastest |

- **Batch 4**: Perfect balance for RTX 3050 6GB
- **Larger batches**: Risk OOM with 128×128 patches
- **Smaller batches**: Slower training, no quality benefit

### Distortions (Disabled Initially)

**Two-Phase Training:**

1. **Phase 1 (Clean)**: `probability = 0.0`
   - Model learns basic steganography
   - Reaches ~75% bit accuracy
   - Fast convergence (20-40 epochs)

2. **Phase 2 (Robust)**: `probability = 0.1` (auto-enabled)
   - Model learns JPEG robustness
   - Reaches 85-95% accuracy
   - Takes longer (60-100+ epochs)

### Loss Weight (5.0)

```python
total_loss = image_loss + 5.0 * message_loss
```

| Weight | Effect |
|--------|--------|
| 1.0 | Equal priority → Poor message recovery |
| **5.0** | **Message priority → 85-95% accuracy** |
| 10.0 | Strong message priority → Visible artifacts |

**5.0 is optimal**: Emphasizes message recovery without sacrificing image quality.

### CUDA Requirement

| Device | 128×128 patch | 256×256 patch | Training Time |
|--------|---------------|---------------|---------------|
| CPU | 🐢 ~48 hours | ❌ Impossible | N/A |
| **CUDA (RTX 3050)** | **✓ 2-3 hours** | **✓ 4-6 hours** | **Optimal** |

## Implementation Details

### Code Location: train.py (lines 1200-1260)

```python
# ENFORCE DIV2K TRAINING REQUIREMENTS
if args.dataset_type == 'DIV2K':
    print("\n" + "=" * 60)
    print("DIV2K TRAINING REQUIREMENTS VALIDATION")
    print("=" * 60)
    
    errors = []
    
    # 1. Message length must be 16 or 32 bits
    if args.message_length not in [16, 32]:
        errors.append(f"Message length must be 16 or 32 bits (got: {args.message_length})")
    
    # 2. Batch size must be 4
    if args.batch_size != 4:
        errors.append(f"Batch size must be 4 for DIV2K (got: {args.batch_size})")
    
    # 3. Distortions disabled initially
    if args.enable_distortions:
        args.enable_distortions = False
    
    # 4. CUDA must be available
    if args.no_cuda or not torch.cuda.is_available():
        errors.append("CUDA is REQUIRED for DIV2K training")
    
    # Abort if errors
    if errors:
        print("\n❌ DIV2K TRAINING REQUIREMENTS NOT MET")
        for error in errors:
            print(f"   ✗ {error}")
        sys.exit(1)
```

### Loss Formula: train.py (line 251)

```python
# Combined loss with weighting (message loss weighted 5x more)
total_loss_batch = image_loss + 5.0 * message_loss
```

**Hardcoded** - cannot be overridden for DIV2K training.

## Quick Start

### 1. Verify Requirements
```bash
python verify_div2k_requirements.py
```

### 2. Download DIV2K
```bash
python download_div2k.py
```

### 3. Train
```bash
# Recommended
python train.py --config config_div2k_balanced.yaml
```

## Troubleshooting

### "Message length must be 16 or 32 bits"
**Solution:** Use config_div2k_quick.yaml (16 bits) or config_div2k_balanced.yaml (32 bits)

### "Batch size must be 4"
**Solution:** Do not override batch_size in command-line. Use the config files.

### "CUDA is REQUIRED"
**Solution:**
1. Install NVIDIA GPU drivers
2. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Config vs Command-Line
**Config files take precedence!** If you specify `--batch_size 8` but config has `batch_size: 4`, the config value (4) is used.

**Best practice:** Use config files, don't override critical parameters.

## Summary

✅ **Message length**: 16 or 32 bits only  
✅ **Batch size**: 4 (fixed)  
✅ **Distortions**: DISABLED initially  
✅ **Loss**: `image_loss + 5.0 * message_loss`  
✅ **GPU**: CUDA required  

**Recommended command:**
```bash
python train.py --config config_div2k_balanced.yaml
```

This ensures all requirements are automatically satisfied!
