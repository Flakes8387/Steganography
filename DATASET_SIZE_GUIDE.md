# Dataset Size Management for Local GPU Training

## Overview

Training on a local RTX 3050 6GB requires careful dataset size management. This guide explains how to limit dataset size while maximizing training quality.

## Key Concept: Base Images vs Total Patches

The `max_train_images` parameter **limits BASE images**, NOT total patches:

```
max_train_images = 500    (base images loaded from disk)
patches_per_image = 4     (random crops extracted per image)
────────────────────────────────────────────────────────────
Total training samples = 500 × 4 = 2,000 patches
```

## Configuration Comparison

| Configuration | Base Images | Total Samples | Training Time | Expected Accuracy | Use Case |
|--------------|-------------|---------------|---------------|-------------------|----------|
| **Quick** | 300 | 1,200 | 1-2 hours | 75-80% | Fast experimentation |
| **Balanced** ⭐ | 500 | 2,000 | 2-3 hours | 85-90% | **Recommended** |
| **Standard** | 500 | 2,000 | 2-3 hours | 85-90% | Production ready |
| **Full** | 800 | 3,200 | 4-5 hours | 90-95% | Maximum quality |

## Why Limit Dataset Size?

### Benefits of Limiting (300-500 images)

✅ **Faster training:** 2-3 hours instead of 4-5 hours  
✅ **Good quality:** 85-90% bit accuracy is excellent for steganography  
✅ **Less overfitting:** Early stopping prevents overfitting on limited data  
✅ **More experiments:** Can try different hyperparameters faster  
✅ **GPU friendly:** Lower memory pressure on 6GB GPU  

### When to Use Full Dataset (800 images)

Use the full dataset if you need:
- Maximum possible quality (90-95% bit accuracy)
- Production deployment with critical quality requirements
- Have time for 4-5 hour training sessions
- Want to hide longer messages (64+ bits)

## Random Sampling for Maximum Diversity

With `random_crop=True`, patches are **randomly positioned every time they're accessed**:

```python
# Training loop, Epoch 1
for batch in dataloader:
    # Image #42, Sample #1: Random crop at position (450, 320)
    # Image #42, Sample #2: Random crop at position (1100, 680)
    # Image #42, Sample #3: Random crop at position (780, 120)
    # ...

# Training loop, Epoch 2
for batch in dataloader:
    # Image #42, Sample #1: DIFFERENT crop at position (920, 550)
    # Image #42, Sample #2: DIFFERENT crop at position (320, 890)
    # ...
```

**Result:** Training with 500 images + random crops ≈ Training with 5,000+ static images!

## How to Set Dataset Size

### Method 1: Use Pre-configured Files

```bash
# Quick training (1-2 hours)
python train.py --config config_div2k_quick.yaml

# Balanced training (2-3 hours) - RECOMMENDED
python train.py --config config_div2k_balanced.yaml

# Full dataset (4-5 hours)
python train.py --config config_div2k_full.yaml
```

### Method 2: Modify config.yaml

```yaml
data:
  max_train_images: 500    # Set to 300, 500, or 800
  patches_per_image: 4     # Keep at 4 for good diversity
  random_crop: true        # Always use random crops
```

### Method 3: Command-Line Override

```bash
python train.py \
  --config config_div2k.yaml \
  --max_train_images 500 \
  --patches_per_image 4
```

## Memory vs Dataset Size

### Full Images (2048×1080, NO patches)

```
Single image: 2048 × 1080 × 3 × 4 bytes = ~25MB
Batch of 8: ~200MB
800 images: Would require ~20GB GPU memory!
```

❌ **Cannot fit in RTX 3050 6GB**

### Patches (128×128, WITH patches)

```
Single patch: 128 × 128 × 3 × 4 bytes = ~192KB
Batch of 8: ~1.5MB
3,200 patches (800 images × 4): ~600MB GPU memory
```

✅ **Perfect for RTX 3050 6GB**

## Training Time Breakdown

Based on RTX 3050 6GB with mixed precision (AMP):

| Dataset Size | Samples | Time/Epoch | Total Time (100 epochs) |
|--------------|---------|------------|-------------------------|
| 300 images | 1,200 | ~1 min | ~1.5 hours |
| 500 images | 2,000 | ~1.5 min | ~2.5 hours |
| 800 images | 3,200 | ~2.5 min | ~4 hours |

*Note: Includes early stopping, which typically stops at 60-80 epochs*

## Recommendations by Use Case

### 🔬 Experimentation & Prototyping
- **Config:** `config_div2k_quick.yaml`
- **Images:** 300 (1,200 samples)
- **Time:** 1-2 hours
- **Quality:** 75-80% (good enough for testing)

### 🏢 Production & Real Applications
- **Config:** `config_div2k_balanced.yaml` ⭐
- **Images:** 500 (2,000 samples)
- **Time:** 2-3 hours
- **Quality:** 85-90% (excellent for deployment)

### 🎯 Research & Maximum Quality
- **Config:** `config_div2k_full.yaml`
- **Images:** 800 (3,200 samples)
- **Time:** 4-5 hours
- **Quality:** 90-95% (best possible)

### 🚀 GPU Stress Test
- **Config:** Custom with high batch size
- **Images:** 300 (1,200 samples)
- **Batch size:** 16-32
- **Time:** <1 hour
- **Purpose:** Test GPU capabilities

## Verification

Run this to see all configurations:

```bash
python verify_configs.py
```

Output:
```
Config       | Base | Patches  | Total  | Epochs | Time   | Quality
---------------------------------------------------------------------
Quick        | 300  | 4        | 1200   | 50     | 1-2h   | 75-80%
Balanced     | 500  | 4        | 2000   | 100    | 2-3h   | 85-90% ⭐
Standard     | 500  | 4        | 2000   | 100    | 2-3h   | 85-90%
Full         | 800  | 4        | 3200   | 150    | 4-5h   | 90-95%
```

## Troubleshooting

### "Training is too slow"
→ Use `config_div2k_quick.yaml` (300 images)  
→ Increase batch size to 16  
→ Reduce epochs to 50  

### "Out of memory error"
→ You're probably loading full images, not patches  
→ Verify `use_patches: true` in config  
→ Verify `image_size: 128` (not 256 or higher)  
→ Reduce batch size to 4  

### "Model not converging well"
→ Use more images: `config_div2k_balanced.yaml` (500 images)  
→ Train longer: Increase epochs to 150  
→ Ensure random_crop is enabled for diversity  

### "Quality not good enough"
→ Use full dataset: `config_div2k_full.yaml` (800 images)  
→ Increase message length (but slower training)  
→ Train until early stopping triggers (be patient)  

## Summary

**For 99% of users:**
```bash
python train.py --config config_div2k_balanced.yaml
```

This gives you:
- ✅ 500 base images → 2,000 training samples
- ✅ 2-3 hour training time
- ✅ 85-90% bit accuracy (excellent)
- ✅ Random crops for maximum diversity
- ✅ Perfect for RTX 3050 6GB

**Remember:** Dataset size limit applies to BASE images. Total patches = base_images × patches_per_image!
