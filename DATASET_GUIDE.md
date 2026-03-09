# Dataset Guide for Steganography Training

## Overview

Training a robust steganography model requires large, diverse image datasets. This guide covers recommended datasets and how to use them.

## Recommended Datasets

### 1. **DIV2K** ⭐ (Recommended for beginners)
- **Size**: 800 high-quality images (train), 100 (validation)
- **Resolution**: 2K (2040×1080 or higher)
- **Format**: PNG
- **Download**: ~3GB
- **Why**: High quality, manageable size, excellent for steganography

```bash
# Download DIV2K
python download_datasets.py --dataset div2k --output data/div2k

# Train with DIV2K
python train.py \
    --train_dir data/div2k/train/DIV2K_train_HR \
    --val_dir data/div2k/valid/DIV2K_valid_HR \
    --num_epochs 100 \
    --batch_size 16 \
    --image_size 256 \
    --message_length 1024
```

**Expected Results**:
- Training time: ~2-3 hours (GPU)
- PSNR: >35 dB
- BER: <5%

---

### 2. **COCO** (Large-scale training)
- **Size**: 118,000 train images, 5,000 validation
- **Resolution**: Variable (mostly 640×480 or higher)
- **Format**: JPEG
- **Download**: ~18GB (train), ~1GB (val)
- **Why**: Diverse scenes, standard benchmark

```bash
# Download COCO train set
python download_datasets.py --dataset coco --output data/coco --split train

# Train with COCO
python train.py \
    --train_dir data/coco/train2017 \
    --num_epochs 50 \
    --batch_size 32 \
    --image_size 256 \
    --use_jpeg \
    --use_noise
```

**Expected Results**:
- Training time: ~8-10 hours (RTX 3090)
- PSNR: >38 dB
- BER: <3%
- More robust to attacks

---

### 3. **BOSSBase** (Steganalysis research)
- **Size**: 10,000 grayscale images
- **Resolution**: 512×512
- **Format**: PGM (convert to PNG/JPEG)
- **Download**: Requires registration
- **Why**: Standard for steganalysis research, uncompressed

```bash
# See download instructions
python download_datasets.py --dataset bossbase --output data/bossbase

# Train with BOSSBase
python train.py \
    --train_dir data/bossbase \
    --num_epochs 100 \
    --batch_size 32 \
    --image_size 256
```

**Note**: BOSSBase images are grayscale. For color steganography, use DIV2K or COCO.

---

### 4. **Quick Sample Dataset** (Testing only)
For quick tests without large downloads:

```bash
# Download 500 random images
python download_datasets.py --dataset sample --output data/sample --num-images 500

# Quick training test
python train.py \
    --train_dir data/sample \
    --num_epochs 20 \
    --batch_size 16
```

---

## Dataset Preparation

### Automatic Detection
The training script automatically detects dataset structure:

```
data/
├── images/              # Flat structure ✓
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
│
├── train/              # Train/val split ✓
│   └── images/
└── val/
    └── images/
│
├── train2017/          # COCO format ✓
├── val2017/
└── annotations/
```

### Manual Preparation

If you have your own images:

```bash
# 1. Create directory
mkdir -p data/custom/train

# 2. Copy images
cp /path/to/images/*.jpg data/custom/train/

# 3. Train
python train.py --train_dir data/custom/train --num_epochs 50
```

---

## Training Recommendations

### Hardware Requirements

| Dataset | Min RAM | Recommended GPU | Training Time |
|---------|---------|-----------------|---------------|
| Sample (500 imgs) | 8GB | GTX 1060 (6GB) | 30 min |
| DIV2K (800 imgs) | 16GB | RTX 2060 (6GB) | 2-3 hours |
| COCO (118K imgs) | 32GB | RTX 3090 (24GB) | 8-10 hours |

### Batch Size Selection

```python
# Based on GPU memory
GTX 1060 (6GB):   --batch_size 4
RTX 2060 (6GB):   --batch_size 8
RTX 3060 (12GB):  --batch_size 16
RTX 3090 (24GB):  --batch_size 32
A100 (40GB):      --batch_size 64
```

### Training Duration

**Minimum epochs for convergence**:
- Small dataset (< 1K images): 50-100 epochs
- Medium dataset (1K-10K): 30-50 epochs  
- Large dataset (> 10K): 20-30 epochs

**Signs of good training**:
- PSNR > 35 dB
- BER < 5%
- Message accuracy > 95%
- Training loss < 0.1

---

## Full Training Examples

### Example 1: Quick Training (DIV2K)
```bash
# Download dataset
python download_datasets.py --dataset div2k --output data/div2k

# Train with monitoring
python train.py \
    --train_dir data/div2k/train/DIV2K_train_HR \
    --val_dir data/div2k/valid/DIV2K_valid_HR \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --image_size 256 \
    --message_length 1024 \
    --save_freq 10 \
    --checkpoint_dir checkpoints/div2k \
    --log_dir runs/div2k

# Monitor
tensorboard --logdir runs/div2k
```

### Example 2: Production Training (COCO)
```bash
# Download COCO
python download_datasets.py --dataset coco --output data/coco --split train

# Train with augmentations
python train.py \
    --train_dir data/coco/train2017 \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.0005 \
    --image_size 256 \
    --message_length 1024 \
    --use_jpeg \
    --use_noise \
    --use_resize \
    --use_blur \
    --save_freq 5 \
    --checkpoint_dir checkpoints/coco \
    --log_dir runs/coco

# Monitor
tensorboard --logdir runs/coco
```

### Example 3: Resume Training
```bash
# Resume from checkpoint
python train.py \
    --train_dir data/div2k/train/DIV2K_train_HR \
    --num_epochs 150 \
    --resume checkpoints/checkpoint_epoch_100.pth \
    --batch_size 16
```

---

## Verification

After downloading, verify your dataset:

```bash
# Count images
python -c "from pathlib import Path; print(len(list(Path('data/div2k/train').rglob('*.png'))))"

# Check image sizes
python -c "from PIL import Image; from pathlib import Path; imgs = list(Path('data/div2k/train').rglob('*.png'))[:5]; [print(f'{img.name}: {Image.open(img).size}') for img in imgs]"

# Test dataset loading
python test_training_setup.py
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 4

# Reduce image size
--image_size 128

# Use CPU (slower)
--device cpu
```

### Slow Training
```bash
# Use GPU
--device cuda

# Increase batch size (if memory allows)
--batch_size 32

# Reduce dataset size for testing
# Use only first 1000 images
```

### Poor Results
- Train for more epochs (50-100+)
- Use larger dataset (1000+ images)
- Enable augmentations: --use_jpeg --use_noise
- Reduce message length: --message_length 512
- Check PSNR/BER metrics in TensorBoard

---

## Next Steps

After training:

1. **Test encoding**: `python encode.py --image test.jpg --message "Secret" --output stego.png`
2. **Test decoding**: `python decode.py --image stego.png`
3. **Test robustness**: Open `Robustness_Test.ipynb`
4. **Evaluate**: Check PSNR, BER, SSIM metrics

For questions or issues, check the README or training documentation.
