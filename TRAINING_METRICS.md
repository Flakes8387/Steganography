# Training Metrics and Logging for DIV2K

## Overview

The training script now provides comprehensive metrics logging optimized for DIV2K training on local GPUs. All key metrics are tracked, logged, and saved automatically.

## Metrics Tracked

### Per-Batch Metrics (Every 10 Batches)

```
[Epoch   1] [ 12.5%] BitAcc: 67.45% | BER: 0.3255 | Loss: 0.5234 | PixelΔ: 0.0087 [OK]
[Epoch   1] [ 25.0%] BitAcc: 71.23% | BER: 0.2877 | Loss: 0.4891 | PixelΔ: 0.0092 [OK]
[Epoch   1] [ 37.5%] BitAcc: 74.56% | BER: 0.2544 | Loss: 0.4523 | PixelΔ: 0.0098 [OK]
```

**Displayed:**
- Epoch number
- Progress percentage
- Bit Accuracy (%)
- BER (Bit Error Rate)
- Total Loss
- Pixel Delta (image quality metric)

### Per-Epoch Summary

```
============================================================
EPOCH 25/100 SUMMARY
============================================================

Message Recovery Performance:
  Bit Accuracy : 87.45% [GOOD]
  BER          : 0.125500

Loss Breakdown:
  Total Loss   : 0.234567
  Image Loss   : 0.012345
  Message Loss : 0.044444

Image Quality:
  Pixel Delta  : 0.009876 [OK]

Training Speed:
  Epoch Time   : 145.2s (2.4 min)
  Throughput   : 13.8 samples/sec
```

**Status Indicators:**
- `[EXCELLENT]`: ≥95% bit accuracy - Production ready
- `[GOOD]`: ≥85% bit accuracy - Suitable for most uses
- `[OK]`: ≥75% bit accuracy - Acceptable, may need more training
- `[TRAINING]`: <75% bit accuracy - Still learning

### Best Model Tracking

When a new best model is achieved:

```
[SUCCESS] NEW BEST MODEL!
  Previous Best : 86.23%
  Current       : 87.45%
  Improvement   : +1.22%

  Saved: checkpoints/best_model_local.pth
```

**Automatic Saving:**
- Best model saved automatically when bit accuracy improves
- Both training and validation metrics considered
- No manual intervention required

## Saved Files

### 1. Training Metrics CSV

**Location:** `checkpoints/training_metrics.csv`

**Format:**
```csv
Epoch,Train_BitAccuracy,Train_BER,Train_Loss,Val_BitAccuracy,Val_BER,Val_Loss
1,65.23,0.3477,0.6543,,,
2,71.45,0.2855,0.5234,,,
3,76.89,0.2311,0.4567,,,
...
```

**Use Cases:**
- Import into Excel/Python for analysis
- Plot custom graphs
- Compare different training runs
- Calculate statistics

### 2. Model Checkpoints

**Best Model:** `checkpoints/best_model_local.pth`
- Automatically saved when bit accuracy improves
- Contains model weights and optimizer state
- Ready for inference or resuming training

**Latest Checkpoint:** `checkpoints/checkpoint_latest.pth`
- Saved every `save_freq` epochs (default: 10)
- Can resume training from last epoch

**Periodic Checkpoints:** `checkpoints/checkpoint_epoch_N.pth`
- Saved every 10 epochs
- Provides training history snapshots

### 3. Training Plots

**Location:** `checkpoints/plots/`

Generated automatically:
- `bit_accuracy.png`: Bit accuracy over epochs
- `ber.png`: Bit error rate over epochs
- `loss.png`: Training loss over epochs
- `combined_metrics.png`: All metrics in one view

### 4. TensorBoard Logs

**Location:** `runs/YYYYMMDD_HHMMSS/`

**View with:**
```bash
tensorboard --logdir runs
```

**Available Metrics:**
- Train/EpochBitAccuracy
- Train/EpochBER
- Train/EpochLoss
- Train/ImageLoss
- Train/MessageLoss
- Train/PixelDelta
- Train/LearningRate
- Val/* (if validation set provided)

## Training Progress Display

### During Training

```
============================================================
EPOCH 50/100 SUMMARY
============================================================

Message Recovery Performance:
  Bit Accuracy : 89.76% [GOOD]
  BER          : 0.102400

Loss Breakdown:
  Total Loss   : 0.189234
  Image Loss   : 0.008765
  Message Loss : 0.036093

Image Quality:
  Pixel Delta  : 0.011234 [OK]

Training Speed:
  Epoch Time   : 142.8s (2.4 min)
  Throughput   : 14.0 samples/sec

Validation Results:
  Bit Accuracy : 88.34%
  BER          : 0.116600
  Total Loss   : 0.201234

[SUCCESS] NEW BEST MODEL!
  Previous Best : 88.12%
  Current       : 88.34%
  Improvement   : +0.22%

  Saved: checkpoints/best_model_local.pth
  Saved: checkpoints/checkpoint_latest.pth

============================================================
Learning Rate: 0.000100
Best Model   : 88.34% bit accuracy
============================================================
```

### Training Completion

```
============================================================
TRAINING COMPLETED SUCCESSFULLY
============================================================

Final Results:
  Epochs Trained    : 87
  Best Bit Accuracy : 91.23%
  Final Train Acc   : 90.87%
  Final Train BER   : 0.091300
  Final Val Acc     : 89.45%
  Final Val BER     : 0.105500

Model Performance:
  Status: GOOD - Suitable for most uses

Saved Files:
  Best Model     : checkpoints/best_model_local.pth
  Latest Model   : checkpoints/checkpoint_latest.pth
  Metrics CSV    : checkpoints/training_metrics.csv
  Training Plots : checkpoints/plots
  TensorBoard    : runs/20251214_143052

To visualize training:
  tensorboard --logdir runs/20251214_143052
============================================================
```

## Metrics Explanation

### Bit Accuracy
**Definition:** Percentage of message bits correctly recovered

**Formula:** `1.0 - BER`

**Interpretation:**
- 95-100%: Excellent - Almost perfect recovery
- 85-95%: Good - Reliable steganography
- 75-85%: OK - Usable but may have errors
- <75%: Poor - Needs more training

**Target:** ≥85% for production use

### BER (Bit Error Rate)
**Definition:** Proportion of incorrectly recovered bits

**Formula:** `(wrong_bits / total_bits)`

**Interpretation:**
- 0.0-0.05: Excellent (5% or fewer errors)
- 0.05-0.15: Good (5-15% errors)
- 0.15-0.25: Acceptable (15-25% errors)
- >0.25: Poor (>25% errors)

**Target:** ≤0.15 for reliable use

### Pixel Delta
**Definition:** Average absolute difference between cover and stego images

**Formula:** `mean(|stego_image - cover_image|)`

**Interpretation:**
- <0.005: Too imperceptible (may indicate underfitting)
- 0.005-0.02: Good (imperceptible to human eye)
- >0.02: Too visible (artifacts may be noticeable)

**Target:** 0.005-0.02 for optimal imperceptibility

### Training Speed
**Throughput:** Samples processed per second

**Typical Values (RTX 3050 6GB):**
- 128×128 images, batch_size=4: ~12-15 samples/sec
- 256×256 images, batch_size=2: ~5-7 samples/sec

**Epoch Time Estimates:**
- 2000 samples @ 14 samples/sec: ~2.4 minutes
- 3200 samples @ 14 samples/sec: ~3.8 minutes

## Example Training Session

### Quick Training (config_div2k_quick.yaml)
```bash
python train.py --config config_div2k_quick.yaml
```

**Expected Output:**
- ~50 epochs
- ~1-2 hours
- 75-80% final accuracy
- Good for testing and experimentation

### Balanced Training (config_div2k_balanced.yaml)
```bash
python train.py --config config_div2k_balanced.yaml
```

**Expected Output:**
- ~80-100 epochs
- ~2-3 hours
- 85-90% final accuracy
- Recommended for production

### Full Training (config_div2k_full.yaml)
```bash
python train.py --config config_div2k_full.yaml
```

**Expected Output:**
- ~100-150 epochs
- ~4-5 hours
- 90-95% final accuracy
- Maximum quality

## Analyzing Results

### 1. Check CSV File
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('checkpoints/training_metrics.csv')

# Plot bit accuracy
plt.plot(df['Epoch'], df['Train_BitAccuracy'], label='Train')
plt.plot(df['Epoch'], df['Val_BitAccuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Bit Accuracy (%)')
plt.legend()
plt.show()

# Check best epoch
best_epoch = df.loc[df['Train_BitAccuracy'].idxmax()]
print(f"Best epoch: {best_epoch['Epoch']}")
print(f"Best accuracy: {best_epoch['Train_BitAccuracy']}%")
```

### 2. Use TensorBoard
```bash
tensorboard --logdir runs
```

**Navigate to:** http://localhost:6006

**Features:**
- Interactive plots
- Compare multiple runs
- Zoom and filter
- Export data

### 3. Check Saved Plots
```bash
# View plots
ls checkpoints/plots/

# Open in image viewer
open checkpoints/plots/combined_metrics.png  # macOS
xdg-open checkpoints/plots/combined_metrics.png  # Linux
start checkpoints/plots/combined_metrics.png  # Windows
```

## Troubleshooting

### Bit Accuracy Not Improving
**Symptoms:**
- Accuracy stuck at 50-60%
- BER not decreasing

**Solutions:**
1. Reduce message_length (32 → 16 bits)
2. Reduce learning_rate (0.001 → 0.0005)
3. Check dataset is loading correctly
4. Verify distortions are disabled initially

### Training Too Slow
**Symptoms:**
- <5 samples/sec
- >5 minutes per epoch

**Solutions:**
1. Reduce batch_size (but keep at 4 for DIV2K)
2. Reduce num_workers in config
3. Close other GPU applications
4. Verify GPU is being used (check "Using device: cuda")

### Pixel Delta Too High
**Symptoms:**
- Pixel Delta >0.02
- Visible artifacts in stego images

**Solutions:**
1. Increase image_loss weight in code
2. Reduce message_length
3. Increase image_size

### Metrics Not Saved
**Symptoms:**
- No CSV file created
- No plots generated

**Solutions:**
1. Check checkpoint_dir permissions
2. Verify training completed (not interrupted)
3. Check disk space

## Related Documentation

- [DIV2K_REQUIREMENTS.md](DIV2K_REQUIREMENTS.md): Training requirements
- [DIV2K_TRAINING_QUICKSTART.md](DIV2K_TRAINING_QUICKSTART.md): Quick start guide
- [DATASET_VERIFICATION.md](DATASET_VERIFICATION.md): Dataset validation

## Summary

**Automatic Logging:**
- ✓ BER and bit accuracy logged every batch
- ✓ Comprehensive epoch summaries
- ✓ Best model saved automatically
- ✓ Metrics exported to CSV
- ✓ Training plots generated
- ✓ TensorBoard integration
- ✓ Clear progress display for local GPU runs

**No manual intervention required** - all metrics are tracked and saved automatically!
