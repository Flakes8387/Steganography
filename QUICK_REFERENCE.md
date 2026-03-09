# Quick Reference: Training Commands

## Download Dataset

```bash
# Automatic download (recommended)
python download_div2k.py

# Interactive notebook
# Open and run: setup_div2k.ipynb
```

## Training Commands

### Quick Training (1-2 hours)
```bash
# 300 images → 1,200 samples
python train.py --config config_div2k_quick.yaml
```

### Balanced Training (2-3 hours) ⭐ RECOMMENDED
```bash
# 500 images → 2,000 samples
python train.py --config config_div2k_balanced.yaml
```

### Standard Training (2-3 hours)
```bash
# 500 images → 2,000 samples
python train.py --config config_div2k.yaml
```

### Full Dataset (4-5 hours)
```bash
# 800 images → 3,200 samples
python train.py --config config_div2k_full.yaml
```

## Sanity Check

Always run sanity check first:
```bash
python train.py --config config_div2k_balanced.yaml --sanity_mode
```

## Custom Configuration

```bash
# Override specific parameters
python train.py \
  --config config_div2k.yaml \
  --max_train_images 400 \
  --batch_size 16 \
  --num_epochs 80
```

## Monitor Training

```bash
# Watch training progress (real-time)
tensorboard --logdir runs/

# Check latest plots
ls -la plots/
```

## Verify Configurations

```bash
# Show all config comparisons
python verify_configs.py
```

## Common Options

```bash
--config <file>          # Config file to use
--max_train_images <n>   # Limit base images (300/500/800)
--batch_size <n>         # Batch size (4/8/16)
--num_epochs <n>         # Maximum epochs
--sanity_mode            # Quick test (5 steps only)
--use_patches            # Enable patch-based loading
--patches_per_image <n>  # Patches per image (default: 4)
--random_crop            # Random vs center crop
```

## Training on Custom Dataset

```bash
# Your own images
python train.py \
  --train_dir path/to/your/images \
  --image_size 128 \
  --use_patches \
  --patches_per_image 4 \
  --max_train_images 500 \
  --num_epochs 100 \
  --batch_size 8
```

## Resume Training

```bash
# Resume from checkpoint
python train.py \
  --config config_div2k_balanced.yaml \
  --resume checkpoints/best_model_local.pth
```

## Documentation

- **DIV2K Guide:** [DIV2K_TRAINING_GUIDE.md](DIV2K_TRAINING_GUIDE.md)
- **Dataset Size:** [DATASET_SIZE_GUIDE.md](DATASET_SIZE_GUIDE.md)
- **Model Docs:** [STEGOMODEL_DOCUMENTATION.md](STEGOMODEL_DOCUMENTATION.md)
- **Project Summary:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## Tips

💡 **Start with balanced config:** Best quality-to-time ratio  
💡 **Use sanity mode first:** Test configuration before full training  
💡 **Monitor early stopping:** Training auto-stops when not improving  
💡 **Check plots:** Saved automatically in `plots/` directory  
💡 **Random crops:** Increase diversity without downloading more data  
💡 **Limit dataset size:** Keep training time reasonable on local GPU  
