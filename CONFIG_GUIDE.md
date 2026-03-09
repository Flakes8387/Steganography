# Configuration System Guide

## Overview

The training system uses `config.yaml` for centralized parameter management. This makes experiments reproducible and simplifies command-line usage.

## Quick Start

### 1. Train with defaults
```bash
python train.py --train_dir data/images
```

### 2. Override specific parameters
```bash
python train.py --train_dir data/images --batch_size 8 --learning_rate 0.001
```

### 3. Use custom config file
```bash
python train.py --config experiments/my_config.yaml --train_dir data/images
```

---

## Configuration File Structure

### config.yaml

```yaml
# Model Architecture
model:
  image_size: 128              # Image dimensions (128x128)
  message_length: 16           # Secret message length in bits
  
# Training Parameters
training:
  batch_size: 4                # Batch size (adjust for GPU memory)
  learning_rate: 0.0001        # Learning rate (1e-4)
  max_epochs: 300              # Number of training epochs
  num_workers: 8               # Data loader threads
  weight_decay: 1e-5           # L2 regularization
  
# Data Loading
data:
  max_train_images: 1000       # Limit training images (default: 1000 for local GPU)
  max_val_images: 200          # Limit validation images
  train_dir: null              # Training directory (override in CLI)
  val_dir: null                # Validation directory
  
# Distortions (Robustness Training)
distortions:
  enable: false                # Master switch for distortions
  jpeg:
    enabled: false             # JPEG compression attack
    quality_min: 50            # Minimum JPEG quality
    quality_max: 95            # Maximum JPEG quality
  gaussian_noise:
    enabled: false             # Gaussian noise attack
    std_min: 0.01              # Minimum noise std
    std_max: 0.1               # Maximum noise std
  resize:
    enabled: false             # Resize attack
    scale_min: 0.5             # Minimum scale factor
    scale_max: 0.9             # Maximum scale factor
  color_jitter:
    enabled: false             # Color jitter attack
    brightness: 0.2            # Brightness variation
    contrast: 0.2              # Contrast variation
    saturation: 0.2            # Saturation variation
    
# Optimizer Settings
optimizer:
  type: adam                   # Optimizer type (adam/sgd/adamw)
  betas: [0.9, 0.999]         # Adam beta parameters
  eps: 1e-8                    # Adam epsilon
  
# Learning Rate Scheduler
scheduler:
  type: plateau                # Scheduler type (plateau/step/cosine)
  mode: min                    # Mode for ReduceLROnPlateau
  factor: 0.5                  # LR reduction factor
  patience: 10                 # Epochs to wait before reducing
  min_lr: 1e-6                 # Minimum learning rate
  
# Checkpointing
checkpoint:
  save_freq: 10                # Save checkpoint every N epochs
  checkpoint_dir: ./checkpoints
  resume: null                 # Resume from checkpoint path
  
# Logging
logging:
  log_dir: ./runs              # TensorBoard log directory
  print_freq: 100              # Print every N batches
  
# Device Settings
device:
  no_cuda: false               # Disable CUDA (force CPU)
  seed: 42                     # Random seed for reproducibility
  
# Loss Weights
loss_weights:
  image_loss: 1.0              # Weight for image reconstruction
  message_loss: 1.0            # Weight for message recovery
```

---

## Common Configurations

### Configuration 1: Quick Testing (Small Model)
```yaml
model:
  image_size: 64
  message_length: 8
training:
  batch_size: 8
  max_epochs: 50
distortions:
  enable: false
```

**Use case:** Fast prototyping, verifying code works  
**Training time:** ~30 minutes on RTX 3050

---

### Configuration 2: Standard Training (RTX 3050 6GB)
```yaml
model:
  image_size: 128
  message_length: 16
training:
  batch_size: 4
  max_epochs: 300
  learning_rate: 0.0001
distortions:
  enable: false
```

**Use case:** Default training, balanced quality/speed  
**Training time:** ~5-8 hours on RTX 3050

---

### Configuration 3: High Capacity Model
```yaml
model:
  image_size: 256
  message_length: 64
training:
  batch_size: 2
  max_epochs: 500
  learning_rate: 0.00005
distortions:
  enable: true
  jpeg:
    enabled: true
  gaussian_noise:
    enabled: true
```

**Use case:** Maximum message capacity with robustness  
**Training time:** ~20-30 hours on RTX 3050  
**Requires:** More VRAM, reduce batch_size to 2

---

### Configuration 4: Robustness Training
```yaml
model:
  image_size: 128
  message_length: 16
training:
  batch_size: 4
  max_epochs: 500
distortions:
  enable: true
  jpeg:
    enabled: true
    quality_min: 50
  gaussian_noise:
    enabled: true
    std_max: 0.15
  resize:
    enabled: true
  color_jitter:
    enabled: true
```

**Use case:** Train robust model for real-world scenarios  
**Training time:** ~10-15 hours on RTX 3050

---

## Command-Line Override Rules

**CLI arguments always take precedence over config file.**

### Example: Config says batch_size=4, CLI overrides to 8
```bash
python train.py --train_dir data/images --batch_size 8
```
Result: Uses batch_size=8

### Example: Config has distortions OFF, CLI enables them
```bash
python train.py --train_dir data/images --enable_distortions
```
Result: Distortions enabled

---

## Parameter-to-Config Mapping

| CLI Argument | Config Path | Type |
|--------------|-------------|------|
| `--image_size` | `model.image_size` | int |
| `--message_length` | `model.message_length` | int |
| `--batch_size` | `training.batch_size` | int |
| `--learning_rate` | `training.learning_rate` | float |
| `--num_epochs` | `training.max_epochs` | int |
| `--num_workers` | `training.num_workers` | int |
| `--weight_decay` | `training.weight_decay` | float |
| `--enable_distortions` | `distortions.enable` | bool |
| `--checkpoint_dir` | `checkpoint.checkpoint_dir` | str |
| `--log_dir` | `logging.log_dir` | str |
| `--save_freq` | `checkpoint.save_freq` | int |
| `--seed` | `device.seed` | int |
| `--resume` | `checkpoint.resume` | str |
| `--no_cuda` | `device.no_cuda` | bool |

---

## Creating Custom Configs

### Method 1: Copy and Modify
```bash
# Copy default config
cp config.yaml experiments/my_experiment.yaml

# Edit with your parameters
nano experiments/my_experiment.yaml

# Train with it
python train.py --config experiments/my_experiment.yaml --train_dir data/images
```

### Method 2: Create from Scratch
```yaml
# minimal_config.yaml
model:
  image_size: 64
  message_length: 8
training:
  batch_size: 8
  max_epochs: 50
```

```bash
python train.py --config minimal_config.yaml --train_dir data/images
```

---

## Troubleshooting

### Issue 1: Config file not found
```
WARNING: Config file 'config.yaml' not found. Using defaults from arguments only.
```

**Solution:** Create `config.yaml` or specify path:
```bash
python train.py --config path/to/config.yaml --train_dir data/images
```

---

### Issue 2: Missing required argument
```
ERROR: --train_dir is required
```

**Solution:** Even with config, `--train_dir` must be specified:
```bash
python train.py --train_dir data/images
```

Or add it to config:
```yaml
data:
  train_dir: data/images
```

---

### Issue 3: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch_size in config:
```yaml
training:
  batch_size: 2  # Reduce from 4 to 2
```

Or via CLI:
```bash
python train.py --train_dir data/images --batch_size 2
```

---

## Best Practices

### 1. **Start with defaults**
Use `config.yaml` defaults for RTX 3050 6GB, they're optimized for your GPU.

### 2. **Enable distortions gradually**
Start with `distortions.enable: false`, train for 50-100 epochs, then enable distortions.

### 3. **Save configs for experiments**
```bash
cp config.yaml experiments/exp1_baseline.yaml
cp config.yaml experiments/exp2_robust.yaml
```

### 4. **Use version control**
Commit config files with your code:
```bash
git add config.yaml experiments/*.yaml
git commit -m "Add training configurations"
```

### 5. **Document changes**
Add comments in config files:
```yaml
training:
  batch_size: 2  # Reduced from 4 due to OOM on 6GB GPU
```

---

## GPU-Specific Recommendations

### RTX 3050 6GB (Your GPU)
```yaml
model:
  image_size: 128
  message_length: 16
training:
  batch_size: 4
  num_workers: 8
```

### RTX 3060 12GB
```yaml
model:
  image_size: 256
  message_length: 32
training:
  batch_size: 8
  num_workers: 12
```

### Google Colab (T4 15GB)
```yaml
model:
  image_size: 256
  message_length: 64
training:
  batch_size: 16
  num_workers: 2
```

### CPU Only
```yaml
model:
  image_size: 64
  message_length: 8
training:
  batch_size: 2
  num_workers: 4
device:
  no_cuda: true
```

---

## Advanced: Programmatic Config Loading

```python
from utils.config_loader import load_config, save_config

# Load config
config = load_config('config.yaml')

# Modify programmatically
config['training']['batch_size'] = 8
config['training']['learning_rate'] = 0.001

# Save modified config
save_config(config, 'experiments/modified_config.yaml')
```

---

## Summary

✅ **Config file = centralized parameter management**  
✅ **CLI overrides = flexibility for experiments**  
✅ **Reproducible training = commit configs to git**  
✅ **GPU-optimized defaults = start training immediately**

**Recommended workflow:**
1. Use default `config.yaml` for first training
2. Monitor results
3. Create custom config for experiments
4. Override specific parameters via CLI as needed
