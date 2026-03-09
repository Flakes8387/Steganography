# Training Optimization Guide

## Problem: Training Takes Too Long

If your training took ~22 hours for 30 epochs on 2000 images with RTX 3050, here's how to optimize it.

**Expected times:**
- ❌ Current: 22 hours for 30 epochs (way too slow!)
- ✅ Optimized: 2-3 hours for 30 epochs

---

## Key Optimizations

### 1. **Increase Batch Size**

**Problem**: Small batch size = GPU underutilized

```bash
# Slow (your current settings)
--batch_size 8        # Only 8 images per batch

# Fast (optimized for RTX 3050 6GB)
--batch_size 16       # 2x faster, still fits in 6GB
--batch_size 24       # Even faster if memory allows
```

**Why**: Larger batches mean fewer iterations and better GPU utilization.

### 2. **Increase Data Loading Workers**

**Problem**: GPU waits for data

```bash
# Check if this was your issue
--num_workers 0       # Very slow - single thread loading

# Optimized
--num_workers 8       # 8 parallel threads loading data
```

**Why**: With 0 workers, GPU waits idle while CPU loads each batch. With 8 workers, next batch is ready when GPU finishes.

### 3. **Reduce Model Complexity** (if still slow)

```bash
# Current
--message_length 512 --image_size 128    # 358M parameters

# Faster alternatives
--message_length 256 --image_size 128    # ~180M parameters (2x faster)
--message_length 512 --image_size 96     # ~200M parameters (faster)
```

---

## Optimized Training Commands

### Configuration 1: Balanced (Recommended)
**Time: ~2-3 hours for 30 epochs**

```bash
python train.py \
    --train_dir data/synthetic_large \
    --num_epochs 30 \
    --batch_size 16 \
    --message_length 512 \
    --image_size 128 \
    --learning_rate 0.001 \
    --num_workers 8 \
    --save_freq 5
```

### Configuration 2: Fast Training
**Time: ~1-1.5 hours for 30 epochs**

```bash
python train.py \
    --train_dir data/synthetic_large \
    --num_epochs 30 \
    --batch_size 24 \
    --message_length 256 \
    --image_size 96 \
    --learning_rate 0.001 \
    --num_workers 8 \
    --save_freq 10
```

### Configuration 3: Maximum Speed
**Time: ~30-45 minutes for 30 epochs**

```bash
python train.py \
    --train_dir data/synthetic_large \
    --max_train_images 500 \
    --num_epochs 50 \
    --batch_size 32 \
    --message_length 256 \
    --image_size 96 \
    --learning_rate 0.001 \
    --num_workers 8 \
    --save_freq 10
```

---

## Quick Training Script

Use the optimized training script:

```bash
python train_fast.py --train_dir data/synthetic_large --num_epochs 30
```

This automatically applies all optimizations for RTX 3050 6GB.

---

## Troubleshooting

### If training is still slow:

1. **Check GPU utilization**:
```bash
# In another terminal, run this while training:
nvidia-smi -l 1
```

Look for:
- GPU-Util should be 80-100%
- Memory should be 4-5GB / 6GB used
- Power should be near max (60-75W for RTX 3050)

If GPU-Util is low (<50%), increase batch size.

2. **Check disk speed**:
```bash
# If images are on slow HDD, copy to SSD first
xcopy data\synthetic_large D:\fast_drive\data\synthetic_large /E /I
```

3. **Reduce workers if CPU is bottleneck**:
```bash
# If you see high CPU usage but low GPU usage:
--num_workers 4   # Instead of 8
```

---

## Batch Size Guidelines for RTX 3050 6GB

| Image Size | Message Length | Max Batch Size | Speed |
|------------|----------------|----------------|-------|
| 96×96 | 256 | 48 | ⚡ Fastest |
| 96×96 | 512 | 32 | ⚡ Very Fast |
| 128×128 | 256 | 32 | ⚡ Very Fast |
| 128×128 | 512 | 16 | ✅ Fast |
| 128×128 | 1024 | 12 | ⚠️ Moderate |
| 256×256 | 512 | 8 | ⚠️ Slow |
| 256×256 | 1024 | 4 | ❌ Very Slow |

---

## Expected Training Times (RTX 3050 6GB)

### For 2000 images, 30 epochs:

| Configuration | Time | Quality |
|---------------|------|---------|
| **Optimized** (batch=16, 512-bit, 128px) | 2-3 hours | Good |
| **Fast** (batch=24, 256-bit, 96px) | 1-1.5 hours | Acceptable |
| **Your current** (batch=8, 512-bit, 128px) | 22 hours ❌ | Good |

### For 500 images, 50 epochs:

| Configuration | Time | Quality |
|---------------|------|---------|
| **Optimized** (batch=16, 512-bit, 128px) | 45 min | Good |
| **Fast** (batch=32, 256-bit, 96px) | 20 min | Acceptable |

---

## Real Dataset Training Times

Once you download real datasets (DIV2K, COCO, etc.):

### DIV2K (800 images, 50 epochs):
```bash
# Optimized settings
python train.py \
    --train_dir data/div2k/DIV2K_train_HR \
    --num_epochs 50 \
    --batch_size 16 \
    --message_length 512 \
    --image_size 128 \
    --num_workers 8
```
**Time: ~1.5 hours** (vs ~11 hours with slow settings)

### COCO (10K images subset, 30 epochs):
```bash
python train.py \
    --train_dir data/coco/train2017 \
    --max_train_images 10000 \
    --num_epochs 30 \
    --batch_size 16 \
    --message_length 512 \
    --image_size 128 \
    --num_workers 8
```
**Time: ~5-6 hours** (vs ~140 hours with slow settings!)

---

## Summary

**To fix slow training:**

1. ✅ Use `--batch_size 16` (or higher if memory allows)
2. ✅ Use `--num_workers 8` for faster data loading
3. ✅ Reduce image size to 128 or 96 if needed
4. ✅ Monitor GPU utilization with `nvidia-smi -l 1`

**Quick fix command:**
```bash
python train_fast.py --train_dir data/synthetic_large --num_epochs 30
```

This should reduce training from 22 hours to ~2-3 hours! 🚀
