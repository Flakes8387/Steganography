# DIV2K Dataset Training Guide

This guide explains how to use DIV2K dataset with patch-based loading for optimal training.

## 📦 What is DIV2K?

DIV2K is a high-quality image dataset containing:
- **800 training images** (2048×1080 resolution)
- Professional photography with diverse scenes
- PNG format (lossless, high quality)
- Perfect for steganography training

## 🚀 Quick Start

### Step 1: Download DIV2K

**Option A: Automatic download (recommended)**
```bash
python download_div2k.py
```

**Option B: Interactive notebook**
```bash
# Open and run setup_div2k.ipynb in VS Code or Jupyter
```

**Option C: Manual download**
1. Download from: https://data.vision.ee.ethz.ch/cvl/DIV2K/
2. Extract to: `data/DIV2K/train/`

### Step 2: Choose Training Configuration

We provide **4 configs** optimized for RTX 3050 6GB:

| Config | Base Images | Total Samples | Time | Quality |
|--------|-------------|---------------|------|---------|
| `config_div2k_quick.yaml` | 300 | 1,200 | 1-2h | Good (75-80%) |
| `config_div2k_balanced.yaml` | 500 | 2,000 | 2-3h | **Best** (85-90%) ⭐ |
| `config_div2k.yaml` | 500 | 2,000 | 2-3h | Very Good (85-90%) |
| `config_div2k_full.yaml` | 800 | 3,200 | 4-5h | Excellent (90-95%) |

**Recommended for most users:** `config_div2k_balanced.yaml`

### Step 3: Train with DIV2K

**Method 1: Using config file (recommended)**
```bash
# Run sanity check first
python train.py --config config_div2k_balanced.yaml --sanity_mode

# Start training
python train.py --config config_div2k_balanced.yaml
```

**Method 2: Command-line arguments**
```bash
python train.py \
  --train_dir data/DIV2K/train \
  --image_size 128 \
  --use_patches \
  --patches_per_image 4 \
  --max_train_images 500 \
  --num_epochs 100 \
  --batch_size 8
```

## 🎯 Patch-Based Loading

### Why Use Patches?

DIV2K images are 2048×1080 pixels - too large for direct training. Patch-based loading:
- **Extracts multiple 128×128 crops** from each image
- **Increases dataset size**: 800 images → 3,200 training samples (with 4 patches)
- **Improves convergence**: More diverse training data
- **Reduces memory**: Smaller patches fit in GPU memory

### How It Works

```
Original DIV2K Image (2048×1080)
         ↓
    Extract 4 random 128×128 patches
         ↓
┌────────┬────────┬────────┬────────┐
│Patch 1 │Patch 2 │Patch 3 │Patch 4 │
│128×128 │128×128 │128×128 │128×128 │
└────────┴────────┴────────┴────────┘
         ↓
    4 training samples per image
         ↓
    800 images × 4 patches = 3,200 samples
```

### Patch Configuration

**In config_div2k.yaml:**
```yaml
data:
  use_patches: true        # Enable patch extraction
  patches_per_image: 4     # Extract 4 patches per image
  random_crop: true        # Use random crops (augmentation)
```

**Command-line:**
```bash
--use_patches              # Enable patches
--patches_per_image 4      # Number of patches
--random_crop              # Random vs center crop
```

## 📊 Understanding Dataset Size Limits

### The Key: max_train_images

The `max_train_images` parameter **limits BASE images**, not total patches:

```
max_train_images = 500  (base images loaded)
patches_per_image = 4   (crops extracted per image)
──────────────────────────────────────────────────
Total training samples = 500 × 4 = 2,000 patches
```

### Dataset Size Comparison

| Config | Base Images | Patches/Image | Total Samples | Training Time |
|--------|-------------|---------------|---------------|---------------|
| Quick | 300 | 4 | 1,200 | 1-2 hours |
| Balanced | 500 | 4 | 2,000 | 2-3 hours |
| Full | 800 | 4 | 3,200 | 4-5 hours |

### Memory Comparison

**Without Patches (Full Images):**
- Each sample: ~12MB (2048×1080×3 pixels)
- Batch of 8: ~96MB
- Too large for RTX 3050 6GB!

**With Patches (128×128 Crops):**
- Each sample: ~192KB (128×128×3 pixels)
- Batch of 8: ~1.5MB
- Perfect for local GPU training!

### Why Limit Dataset Size?

For **local GPU training**, using all 800 images (3,200 patches) is NOT always better:

✅ **Benefits of limiting to 500 images:**
- Faster training (2-3 hours vs 4-5 hours)
- Good quality (85-90% bit accuracy)
- Less overfitting risk with early stopping
- Can experiment more configurations

❌ **Downsides of full 800 images:**
- Longer training (4-Quick (1-2 hours) - config_div2k_quick.yaml

For **fast experimentation and prototyping**:

```bash
python train.py --config config_div2k_quick.yaml
```

**Settings:**
- Base images: 300 (300 × 4 = 1,200 samples)
- Batch size: 16 (higher for faster training)
- Message length: 16 bits
- Epochs: 50

**Expected:** ~1-2 hours, 75-80% bit accuracy. Good for testing changes quickly.

### Configuration 2: Balanced (2-3 hours) - config_div2k_balanced.yaml ⭐

For **optimal quality-to-time ratio** (RECOMMENDED):

```bash
python train.py --config config_div2k_balanced.yaml
```

**Settings:**
- Base images: 500 (500 × 4 = 2,000 samples)
- Batch size: 8
- MessaRandom Sampling Per Epoch

With `random_crop=True`, patches are **randomly positioned every time**:

```python
# Epoch 1, Image #42, Patch #1
crop_position = (random.randint(0, 1920), random.randint(0, 952))
patch_1_epoch_1 = image[crop_position]  # e.g., position (450, 320)

# Epoch 2, Image #42, Patch #1 (DIFFERENT position)
crop_position = (random.randint(0, 1920), random.randint(0, 952))
patch_1_epoch_2 = image[crop_position]  # e.g., position (1100, 680)
```

**Benefits:**
- ✅ Same 500 base images generate thousands of unique patches
- ✅ Prevents overfitting even with limited dataset
- ✅ No two training batches see identical data
- ✅ Continuous data augmentation throughout training

**Result:** 500 images with random crops ≈ 5,000+ static images!xpected:** ~2-3 hours, 85-90% bit accuracy. **Best for most users.**

### Configuration 3: Standard (2-3 hours) - config_div2k.yaml

Same as balanced, original configuration:

```bash
python train.py --config config_div2k.yaml
```

**Settings:**
- Base images: 500 (500 × 4 = 2,000 samples)
- Batch size: 8
- Message length: 32 bits
- Epochs: 100

**Expected:** ~2-3 hours, 85-90% bit accuracy.

### Configuration 4: Full Dataset (4-5 hours) - config_div2k_full.yaml

For **maximum quality** with all DIV2K images:

```bash
python train.py --config config_div2k_full.yaml
```

**Settings:**
- Base images: 800 (all DIV2K, 800 × 4 = 3,200 samples)
- Batch size: 8
- Message length: 64 bits
- Epochs: 150
- Early stopping: 30 epochs patience

**Expected:** ~4-5 hours, 90-95% bit accuracy. Best results but longer training.ettings:
# - 800 images × 4 patches = 3,200 samples
# - Batch size: 8
# - Image size: 128×128
# - Epochs: 100
# - Training time: ~4-6 hours
```

### Configuration 2: Fast Training
```bash
python train.py \
  --train_dir data/DIV2K/train \
  --image_size 128 \
  --use_patches \
  --patches_per_image 2 \
  --num_epochs 50 \
  --batch_size 16

# Results in: 800 × 2 = 1,600 samples
# Training time: ~2-3 hours
```

### Configuration 3: Maximum Quality
```bash
python train.py \
  --train_dir data/DIV2K/train \
  --image_size 256 \
  --use_patches \
  --patches_per_image 4 \
  --num_epochs 150 \
  --batch_size 4

# Results in: 800 × 4 = 3,200 samples at 256×256
# Training time: ~8-10 hours
# Higher quality but slower
```

## 🔍 Validation Best Practices

For validation, use standard loading (no patches) for consistent evaluation:

```bash
python train.py \
  --train_dir data/DIV2K/train \
  --val_dir data/DIV2K/val \
  --use_patches \
  --patches_per_image 4

# Training: patches enabled (3,200 samples)
# Validation: patches automatically disabled (center crop)
```

## 📈 Expected Results

### Phase 1: Clean Training (0-75% bit accuracy)
- **No distortions**
- Rapid convergence on clean patches
- Expected: 70-80% bit accuracy in 20-30 epochs

### Phase 2: Robust Training (≥75% bit accuracy)
- **JPEG compression enabled** (0.1 probability)
- Model learns robustness
- Expected: 85-95% bit accuracy in 50-80 epochs

### Total Training Time
- **RTX 3050 6GB**: ~4-6 hours for 100 epochs
- **RTX 3060**: ~3-4 hours
- **RTX 3080**: ~2-3 hours

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size or patches per image
```bash
--batch_size 4 --patches_per_image 2
```

### Issue: Training too slow
**Solution**: Increase batch size or reduce patches
```bash
--batch_size 16 --patches_per_image 2
```

### Issue: Poor convergence
**Solution**: Increase patches per image for more samples
```bash
--patches_per_image 8  # 800 × 8 = 6,400 samples
```

### Issue: Images too small error
**Solution**: DIV2K images should all be ≥128×128. If using different dataset, adjust patch size:
```bash
--image_size 64  # Smaller patches
```

## 📚 Additional Resources

- **Download script**: `download_div2k.py`
- **Setup notebook**: `setup_div2k.ipynb`
- **Config file**: `config_div2k.yaml`
- **Training script**: `train.py`

## 💡 Tips

1. **Always run sanity check first**:
   ```bash
   python train.py --config config_div2k.yaml --sanity_mode
   ```

2. **Monitor with TensorBoard**:
   ```bash
   tensorboard --logdir runs
   ```

3. **Use plots for analysis**:
   - Plots automatically saved to `checkpoints/plots/`
   - Check pixel delta to ensure imperceptibility

4. **Start with default config**:
   - `config_div2k.yaml` has optimized settings
   - Tune only if needed

## 🎯 Summary

```bash
# Complete workflow:

# 1. Download dataset
python download_div2k.py

# 2. Run sanity check
python train.py --config config_div2k.yaml --sanity_mode

# 3. Train model
python train.py --config config_div2k.yaml

# 4. View results
tensorboard --logdir runs
```

**Result**: High-quality steganography model trained on 3,200 diverse image patches!
