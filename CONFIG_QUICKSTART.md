# Config System - Quick Reference

## ✅ Configuration System Successfully Integrated!

The training system now uses `config.yaml` for centralized parameter management with command-line override capability.

---

## What's New

### Files Created/Modified:

1. **config.yaml** - Central configuration file with GPU-optimized defaults for RTX 3050 6GB
2. **utils/config_loader.py** - Configuration loading and merging utility
3. **train.py** - Updated to load config.yaml and merge with CLI arguments
4. **requirements.txt** - Added pyyaml>=6.0 dependency
5. **CONFIG_GUIDE.md** - Comprehensive configuration guide
6. **README.md** - Updated with config-based training instructions

---

## How to Use

### Method 1: Config File Only (Easiest)
```bash
# Uses all defaults from config.yaml
python train.py --train_dir data/images
```

### Method 2: Config + CLI Overrides (Flexible)
```bash
# Override specific parameters
python train.py --train_dir data/images --batch_size 8 --num_epochs 200
```

### Method 3: Custom Config File
```bash
# Use completely custom config
python train.py --config experiments/my_config.yaml --train_dir data/images
```

---

## Default Configuration (RTX 3050 6GB Optimized)

```yaml
model:
  image_size: 128          # Smaller images for faster training
  message_length: 16       # 16-bit messages

training:
  batch_size: 4            # Safe for 6GB VRAM
  learning_rate: 0.0001    # Conservative LR
  max_epochs: 300          # Extended training
  
data:
  num_workers: 8           # Multi-threaded loading

distortions:
  enable: false            # OFF initially (enable after 100 epochs)
```

---

## Key Features

✅ **GPU-Optimized Defaults** - Pre-configured for RTX 3050 6GB  
✅ **CLI Override** - Command-line arguments take precedence  
✅ **Reproducible** - Save configs for each experiment  
✅ **Flexible** - Use default, custom, or hybrid approach  
✅ **No Long Commands** - Forget complex CLI arguments  

---

## Configuration Priority

**Precedence order (highest to lowest):**
1. ⭐ Command-line arguments (e.g., `--batch_size 8`)
2. ⭐ Config file values (e.g., `config.yaml`)
3. ⭐ Hardcoded defaults in train.py

**Example:**
- Config says: `batch_size: 4`
- CLI says: `--batch_size 8`
- **Result:** Uses `batch_size: 8` (CLI wins)

---

## Common Commands

### Train with defaults
```bash
python train.py --train_dir data/synthetic_large
```

### Train with custom batch size
```bash
python train.py --train_dir data/images --batch_size 8
```

### Train with distortions enabled
```bash
python train.py --train_dir data/images --enable_distortions
```

### Resume training
```bash
python train.py --train_dir data/images --resume checkpoints/best_model.pth
```

### Quick test (small model)
```bash
python train.py --train_dir data/images --image_size 64 --message_length 8 --num_epochs 50
```

---

## Benefits Over Old System

### Before (Without Config):
```bash
python train.py --train_dir data/images --num_epochs 300 --batch_size 4 --learning_rate 0.0001 --image_size 128 --message_length 16 --num_workers 8 --checkpoint_dir checkpoints --log_dir runs --save_freq 10 --seed 42 --weight_decay 0.0001
```
❌ Long, error-prone  
❌ Hard to reproduce  
❌ Can't version control easily  

### After (With Config):
```bash
python train.py --train_dir data/images
```
✅ Short and clean  
✅ Reproducible (commit config.yaml)  
✅ Easy to manage experiments  

---

## Experiment Management

### Save configs for each experiment:
```bash
# Experiment 1: Baseline
cp config.yaml experiments/exp1_baseline.yaml
python train.py --config experiments/exp1_baseline.yaml --train_dir data/images

# Experiment 2: Higher capacity
cp config.yaml experiments/exp2_highcap.yaml
# Edit exp2_highcap.yaml: message_length: 32
python train.py --config experiments/exp2_highcap.yaml --train_dir data/images

# Experiment 3: With distortions
cp config.yaml experiments/exp3_robust.yaml
# Edit exp3_robust.yaml: distortions.enable: true
python train.py --config experiments/exp3_robust.yaml --train_dir data/images
```

### Version control:
```bash
git add config.yaml experiments/*.yaml
git commit -m "Add training configurations for experiments 1-3"
```

---

## Troubleshooting

### Issue: Config file not found
```
WARNING: Config file 'config.yaml' not found
```
**Solution:** Create `config.yaml` or specify path:
```bash
python train.py --config path/to/config.yaml --train_dir data/images
```

---

### Issue: train_dir required
```
ERROR: --train_dir is required
```
**Solution:** Always specify training directory:
```bash
python train.py --train_dir data/images
```

---

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch_size:
```bash
python train.py --train_dir data/images --batch_size 2
```

Or edit config.yaml:
```yaml
training:
  batch_size: 2
```

---

## Next Steps

1. **Test the config system:**
   ```bash
   python train.py --train_dir data/synthetic_small --num_epochs 5
   ```

2. **Read the detailed guide:**
   - [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Full configuration reference
   - [TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md) - Performance tips

3. **Create your first experiment:**
   ```bash
   cp config.yaml experiments/my_first_experiment.yaml
   nano experiments/my_first_experiment.yaml
   python train.py --config experiments/my_first_experiment.yaml --train_dir data/images
   ```

4. **Monitor training:**
   ```bash
   tensorboard --logdir runs/
   ```

---

## Summary

✅ Config system fully integrated  
✅ GPU-optimized defaults for RTX 3050 6GB  
✅ Command-line override capability  
✅ PyYAML dependency added  
✅ Documentation complete  
✅ Ready to train!

**Start training now:**
```bash
python train.py --train_dir data/synthetic_large
```

Good luck with your training! 🚀
