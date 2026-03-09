# Distortion Training Guide

## Overview

The training pipeline uses a **two-phase approach** for optimal model convergence:

1. **Phase 1: Clean Training** (Default)
   - Train on clean images without distortions
   - Focus: Learn basic encoding/decoding
   - Target: Achieve ≥75% bit accuracy

2. **Phase 2: Robust Training** (Auto-enabled)
   - Enable distortions (JPEG, noise, resize, crop, etc.)
   - Focus: Learn robustness to real-world attacks
   - Target: Maintain high accuracy under distortions

---

## Why Two-Phase Training?

### Problem: Training with Distortions from Start
❌ Model struggles to learn basic encoding  
❌ Loss landscape is noisy and unstable  
❌ Slow convergence or training failure  
❌ Low accuracy even on clean images  

### Solution: Progressive Training
✅ **Clean training first**: Model learns basic steganography  
✅ **Stable convergence**: Clean signal helps model optimize quickly  
✅ **High accuracy baseline**: Reach 75%+ accuracy before adding complexity  
✅ **Robust training second**: Add distortions once model is competent  

**Result:** Better final accuracy and faster training!

---

## Default Behavior

### Automatic Distortion Enabling

```python
# Default: Distortions OFF
enable_distortions = False

# During training:
# - Monitor bit accuracy each epoch
# - When accuracy >= 75%, auto-enable distortions
# - Print notification when switching modes
```

### Training Output Example

```
============================================================
Starting training...
============================================================

⚠️  DISTORTIONS DISABLED: Clean-image training mode
   Distortions will auto-enable when bit accuracy >= 75%
============================================================

Epoch 1/300
------------------------------------------------------------
Train Metrics:
  Loss: 0.523145
  Image Loss: 0.012345
  Message Loss: 0.510800
  Accuracy: 52.34%
  Time: 45.23s

...

Epoch 15/300
------------------------------------------------------------
Train Metrics:
  Loss: 0.142567
  Image Loss: 0.003456
  Message Loss: 0.139111
  Accuracy: 76.23%
  Time: 44.89s

============================================================
🎯 DISTORTIONS AUTO-ENABLED!
   Bit accuracy reached 76.23% (>= 75%)
   Switching to robust training mode with distortions
============================================================

Epoch 16/300
------------------------------------------------------------
Train Metrics:
  Loss: 0.245678
  Image Loss: 0.004567
  Message Loss: 0.241111
  Accuracy: 71.45%  ← Temporary dip is normal
  Time: 52.34s
```

---

## Configuration Options

### Option 1: Default (Recommended)

**Let distortions auto-enable at 75% accuracy**

```bash
python train.py --train_dir data/images
```

Config:
```yaml
distortions:
  enable: false  # OFF initially
```

---

### Option 2: Force Distortions ON

**Enable distortions from the start** (advanced users only)

```bash
python train.py --train_dir data/images --enable_distortions
```

Config:
```yaml
distortions:
  enable: true  # ON from start
```

⚠️ **Warning:** Training may be slower and less stable

---

### Option 3: Never Enable Distortions

**Train on clean images only** (for testing/debugging)

```bash
python train.py --train_dir data/images
# Distortions won't enable since default is False
```

Config:
```yaml
distortions:
  enable: false  # OFF always
```

Note: Model will NOT be robust to real-world distortions

---

## Distortion Types

When enabled, the model's built-in `Distortions` layer applies:

### 1. Gaussian Noise
- **Purpose:** Simulate sensor noise, compression artifacts
- **Range:** std = 0.02 (2% noise)
- **Probability:** 100% during training

### 2. Spatial Dropout
- **Purpose:** Simulate packet loss, data corruption
- **Drop rate:** 10% of pixels
- **Probability:** 10% of batches

### 3. JPEG Compression (Simulated)
- **Purpose:** Simulate lossy compression
- **Quality range:** 50-95
- **Probability:** 30% of batches
- **Note:** Simplified DCT-like noise approximation

### 4. Brightness Adjustment
- **Purpose:** Simulate lighting variations
- **Range:** ±10% brightness
- **Probability:** 30% of batches

### 5. Contrast Adjustment
- **Purpose:** Simulate display/camera variations
- **Range:** 0.8x to 1.2x contrast
- **Probability:** 30% of batches

---

## Additional Attacks

Beyond built-in distortions, you can enable **additional attacks**:

```bash
python train.py --train_dir data/images --apply_attacks
```

These include:
- **JPEG Compression** (real, quality 50-95)
- **Gaussian Noise** (std 0.01-0.03)
- **Resize Attack** (scale 0.7x-0.9x)
- **Color Jitter** (brightness/contrast/saturation variations)

Applied randomly with 30% probability during training.

---

## Training Strategies

### Strategy 1: Fast Prototyping
**Goal:** Quick results to verify code works

```bash
python train.py \
  --train_dir data/images \
  --image_size 64 \
  --message_length 8 \
  --num_epochs 50
```

- Clean training only
- Small model
- ~30 minutes on RTX 3050

---

### Strategy 2: Standard Training (Recommended)
**Goal:** Balanced accuracy and robustness

```bash
python train.py \
  --train_dir data/images \
  --image_size 128 \
  --message_length 16 \
  --num_epochs 300
```

- Auto-enable distortions at 75% accuracy
- Medium model
- ~5-8 hours on RTX 3050

---

### Strategy 3: Maximum Robustness
**Goal:** Best real-world performance

```bash
python train.py \
  --train_dir data/images \
  --image_size 128 \
  --message_length 16 \
  --num_epochs 500 \
  --apply_attacks
```

- Auto-enable distortions at 75% accuracy
- Additional attacks enabled
- ~10-15 hours on RTX 3050

---

### Strategy 4: High Capacity
**Goal:** Hide more data per image

```bash
python train.py \
  --train_dir data/images \
  --image_size 256 \
  --message_length 64 \
  --batch_size 2 \
  --num_epochs 500
```

- Auto-enable distortions at 75% accuracy
- Large model (reduce batch_size to 2)
- ~20-30 hours on RTX 3050

---

## Expected Results

### Phase 1: Clean Training (Epochs 1-15)

| Epoch | Accuracy | Image Loss | Message Loss | Status |
|-------|----------|------------|--------------|--------|
| 1     | ~50%     | 0.012      | 0.500        | Random baseline |
| 5     | ~65%     | 0.008      | 0.350        | Learning |
| 10    | ~72%     | 0.005      | 0.200        | Near threshold |
| 15    | ~76%     | 0.003      | 0.140        | ✅ **Distortions enabled** |

### Phase 2: Robust Training (Epochs 16-300)

| Epoch | Accuracy | Image Loss | Message Loss | Status |
|-------|----------|------------|--------------|--------|
| 16    | ~71%     | 0.005      | 0.240        | Temporary dip (normal) |
| 30    | ~78%     | 0.004      | 0.180        | Adapting |
| 100   | ~85%     | 0.003      | 0.120        | Good |
| 300   | ~92%+    | 0.002      | 0.060        | Excellent |

**Note:** Accuracy drop when distortions enable is **expected and normal**. The model quickly recovers and exceeds previous accuracy.

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/
```

**Key metrics to watch:**
- **Train/Accuracy**: Should reach 75%+ before distortions enable
- **Train/MessageLoss**: Should decrease steadily
- **Train/ImageLoss**: Should stay low (< 0.01)

### Console Output

Look for:
```
🎯 DISTORTIONS AUTO-ENABLED!
   Bit accuracy reached 76.23% (>= 75%)
```

This indicates successful transition to robust training.

---

## Troubleshooting

### Issue 1: Accuracy stuck below 75%

**Symptom:** Training runs for 50+ epochs but accuracy plateaus at 60-70%

**Solutions:**
```bash
# 1. Lower learning rate
python train.py --train_dir data/images --learning_rate 0.00005

# 2. Smaller model
python train.py --train_dir data/images --image_size 64 --message_length 8

# 3. More data
python train.py --train_dir larger_dataset/images
```

---

### Issue 2: Accuracy drops dramatically when distortions enable

**Symptom:** Accuracy drops from 76% to 40% and doesn't recover

**Solutions:**
```bash
# 1. Force distortions ON from start (slower but more stable)
python train.py --train_dir data/images --enable_distortions

# 2. Train longer in clean mode first (manually enable later)
# Just continue training, model will recover over 10-20 epochs
```

---

### Issue 3: Want to manually control when distortions enable

**Solution:** Modify `train.py`:

```python
# Change line ~514:
distortion_threshold = 0.85  # Enable at 85% instead of 75%

# Or disable auto-enabling entirely:
distortions_enabled = args.enable_distortions  # No auto-enable logic
```

---

### Issue 4: Training too slow after distortions enable

**Symptom:** Training speed drops by 50% after distortions enable

**Solution:**
```bash
# Reduce distortion complexity by editing models/model.py:
# Comment out expensive distortions in Distortions.forward()
```

---

## Best Practices

### ✅ DO:
- **Start with clean training** (default behavior)
- **Wait for 75% accuracy** before enabling distortions
- **Monitor accuracy trends** in TensorBoard
- **Train for 200-300 epochs** for good results
- **Use validation set** to check generalization

### ❌ DON'T:
- Enable distortions from start (unless you know what you're doing)
- Stop training immediately after distortions enable
- Expect perfect accuracy with distortions (80-90% is excellent)
- Use batch_size > 8 with distortions (slower but safer)

---

## Advanced: Custom Distortion Schedule

For advanced users who want fine control:

### Gradual Distortion Increase

Edit `train.py` to gradually increase distortion strength:

```python
# After line ~514, add distortion strength control:
distortion_strength = min(1.0, (epoch - distortion_enable_epoch) / 50)
# Pass to model.forward(..., distortion_strength=distortion_strength)
```

### Multi-threshold Schedule

```python
# Enable different distortions at different thresholds:
if accuracy >= 0.75:
    enable_noise = True
if accuracy >= 0.80:
    enable_jpeg = True
if accuracy >= 0.85:
    enable_all = True
```

---

## Summary

✅ **Default: Clean training until 75% accuracy**  
✅ **Auto-enable distortions when threshold reached**  
✅ **Two-phase training for best results**  
✅ **Temporary accuracy drop is normal**  
✅ **Final accuracy: 85-95% with distortions**  

**Recommended command:**
```bash
python train.py --train_dir data/images
```

That's it! The system handles distortion scheduling automatically. 🚀

---

## Related Documentation

- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration options
- [TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md) - Performance tips
- [README.md](README.md) - General usage guide
