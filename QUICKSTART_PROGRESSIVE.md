# 🚀 Quick Start - Progressive Training for 90%+ Accuracy

## One-Line Start (Windows)

```batch
train_progressive.bat
```

## Manual Start (All Platforms)

```bash
python train_progressive_90plus.py --train_dir data/DIV2K/train --max_images 400
```

---

## What Happens?

The script will:
1. ✅ Load 400 DIV2K images (1,600 training samples with patches)
2. ✅ Train through 6 progressive phases automatically
3. ✅ Save checkpoints after each phase
4. ✅ Monitor pixel delta and auto-adjust if needed
5. ✅ Use cyclical learning rate (1e-4 to 5e-3)
6. ✅ Evaluate final model on all attacks

---

## Expected Results

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Clean | 84.75% | 94-96% | +10% |
| JPEG | 84.00% | 91-93% | +7% |
| Blur | 84.12% | 90-92% | +6% |
| Resize | 83.50% | 90-92% | +7% |
| ColorJitter | 86.00% | 91-93% | +5% |
| **Combined** | **77.00%** | **88-91%** | **+11%** |

---

## Training Time

- **Total:** ~10-15 hours on GPU
- **Phase 1 (Clean):** 2-3 hours → 94%+
- **Phase 2 (JPEG):** 1-2 hours → 90%+
- **Phase 3 (Blur):** 1-2 hours → 90%+
- **Phase 4 (Resize):** 1-2 hours → 90%+
- **Phase 5 (All):** 1-2 hours → 90%+
- **Phase 6 (Combined):** 2-3 hours → 88%+

---

## Monitor Progress

```bash
tensorboard --logdir ./runs
```

Open: http://localhost:6006

---

## Checkpoints

All saved in `checkpoints/`:
- `model_phase1_clean_94.pth` - Clean training
- `model_phase2_jpeg_90.pth` - + JPEG
- `model_phase3_blur_90.pth` - + Blur
- `model_phase4_resize_90.pth` - + Resize
- `model_phase5_all_90.pth` - All individual
- `model_phase6_combined_88.pth` - **FINAL MODEL** ⭐

---

## Evaluate Results

```bash
python evaluate_progressive_models.py
```

---

## Full Documentation

- **[PROGRESSIVE_TRAINING_GUIDE.md](PROGRESSIVE_TRAINING_GUIDE.md)** - Complete guide
- **[PROGRESSIVE_TRAINING_SUMMARY.md](PROGRESSIVE_TRAINING_SUMMARY.md)** - Implementation details
- **[train_progressive_90plus.py](train_progressive_90plus.py)** - Training script (1000+ lines)

---

## Key Features

✅ **Cyclical Learning Rate** - Escapes local minima  
✅ **Progressive Training** - 6 phases, clean → combined attacks  
✅ **400 Images** - Expanded from 200 for better generalization  
✅ **Auto-Checkpointing** - Each phase loads previous best model  
✅ **Pixel Delta Monitoring** - Auto-adjusts to stay < 0.02  
✅ **Early Stopping** - Stops if no improvement for 10 epochs  
✅ **TensorBoard Logging** - Track all metrics in real-time  

---

## Troubleshooting

### Out of Memory?
```bash
python train_progressive_90plus.py --batch_size 4
```

### Training Too Slow?
```bash
python train_progressive_90plus.py --max_images 200 --max_epochs_per_phase 50
```

### Need Different Dataset?
```bash
python train_progressive_90plus.py --train_dir data/images
```

---

## Console Output Preview

```
================================================================================
🚀 PROGRESSIVE MULTI-PHASE TRAINING FOR 90%+ ACCURACY
================================================================================

✓ Device: cuda
✓ Loaded 400 images from data/DIV2K/train
✓ Total training samples (with patches): 1600

================================================================================
🎯 PHASE 1/6: CLEAN TRAINING (Target: >94%)
================================================================================
Configuration:
  • Alpha (image loss weight): 1.0
  • Beta (message loss weight): 1.0
  • Attacks: {}
  • Min Epochs: 20
  • Max Epochs: 100

✓ Initialized new model
✓ Optimizer: Adam with CyclicLR
  • Base LR: 1.00e-04
  • Max LR: 5.00e-03
  • Step size: 200

Epoch 1/100 [PHASE1_CLEAN]: 100%|████| Loss: 0.0234, Acc: 89.2%, PixΔ: 0.0178

Epoch 1/100 Summary:
  • Loss: 0.0234
  • Bit Accuracy: 89.23%
  • Pixel Delta: 0.0178

...

  ✅ Target accuracy 94% reached!
  💾 Saved checkpoint: model_phase1_clean_94.pth

✅ Phase 1 Complete!
   • Best Accuracy: 94.53%
   • Target: 94%
   • Status: ✓ PASSED
```

---

## Requirements

```bash
pip install torch torchvision pillow pyyaml tensorboard tqdm numpy matplotlib
```

---

## 🎉 Ready to Train?

```bash
# Windows
train_progressive.bat

# Linux/Mac
python train_progressive_90plus.py --train_dir data/DIV2K/train
```

**Target: 90%+ accuracy on all attacks in 10-15 hours!** 🚀
