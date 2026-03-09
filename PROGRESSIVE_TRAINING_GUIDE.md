# Progressive Training Guide - 90%+ Accuracy

## 🚀 Quick Start

```bash
# Step 1: Ensure you have DIV2K dataset (400 images)
python download_div2k.py

# Step 2: Run progressive training
python train_progressive_90plus.py --train_dir data/DIV2K/train --max_images 400

# Step 3: Monitor with TensorBoard
tensorboard --logdir ./runs
```

---

## 📋 Training Pipeline Overview

### **6-Phase Progressive Training Strategy**

| Phase | Description | Target | Loss Weights | Attacks |
|-------|-------------|--------|--------------|---------|
| **1** | Clean Training | >94% | α=1.0, β=1.0 | None |
| **2** | JPEG Compression | >90% | α=0.9, β=2.0 | JPEG (50%) |
| **3** | Add Gaussian Blur | >90% | α=0.9, β=2.2 | JPEG (30%) + Blur (50%) |
| **4** | Add Resize Attack | >90% | α=0.85, β=2.5 | JPEG (25%) + Blur (35%) + Resize (50%) |
| **5** | All Individual | >90% | α=0.8, β=2.5 | All 4 attacks individually |
| **6** | Combined Attacks | >88% | α=0.8, β=3.0 | All 4 attacks simultaneously |

---

## 🎯 Key Features

### **1. Cyclical Learning Rate (CyclicLR)**
- **Base LR:** 1e-4
- **Max LR:** 5e-3
- **Mode:** triangular2 (progressively smaller cycles)
- **Benefit:** Escapes local minima, explores better solutions

### **2. Expanded Dataset**
- **400 DIV2K images** (vs 200 previously)
- **4 patches per image** = 1,600 training samples
- **Better generalization** from diverse data

### **3. Pixel Delta Monitoring**
- **Constraint:** Pixel delta < 0.02
- **Auto-adjustment:** If exceeded, alpha increases by 0.1
- **Ensures:** Model remains imperceptible

### **4. Auto-Checkpoint Loading**
- Each phase loads the previous phase's best model
- Smooth transition between phases
- No manual intervention needed

### **5. Early Stopping**
- Stops if no improvement for 10 epochs
- Saves training time
- Prevents overfitting

---

## 🔧 Command Line Options

### **Basic Usage**
```bash
python train_progressive_90plus.py --train_dir data/DIV2K/train
```

### **All Options**
```bash
python train_progressive_90plus.py \
  --train_dir data/DIV2K/train \
  --val_dir data/DIV2K/val \
  --max_images 400 \
  --message_length 16 \
  --image_size 128 \
  --batch_size 8 \
  --min_epochs_per_phase 20 \
  --max_epochs_per_phase 100 \
  --base_lr 1e-4 \
  --max_lr 5e-3 \
  --cyclic_step_size 200 \
  --max_pixel_delta 0.02 \
  --checkpoint_dir checkpoints \
  --log_dir runs
```

### **For Faster Testing (Smaller Dataset)**
```bash
python train_progressive_90plus.py \
  --train_dir data/images \
  --max_images 100 \
  --max_epochs_per_phase 30
```

---

## 📊 Expected Results

### **Phase 1: Clean Training**
- **Epochs:** 30-50
- **Expected Accuracy:** 94-96%
- **Pixel Delta:** 0.015-0.020
- **Time:** ~2-3 hours

### **Phase 2: JPEG**
- **Epochs:** 20-40
- **Expected Accuracy:** 90-93%
- **Time:** ~1-2 hours

### **Phase 3: Blur**
- **Epochs:** 20-40
- **Expected Accuracy:** 90-92%
- **Time:** ~1-2 hours

### **Phase 4: Resize**
- **Epochs:** 20-40
- **Expected Accuracy:** 90-92%
- **Time:** ~1-2 hours

### **Phase 5: All Individual**
- **Epochs:** 20-40
- **Expected Accuracy:** 90-92%
- **Time:** ~1-2 hours

### **Phase 6: Combined**
- **Epochs:** 30-50
- **Expected Accuracy:** 88-91%
- **Time:** ~2-3 hours

### **Total Training Time:** ~10-15 hours on GPU

---

## 📈 Monitoring Progress

### **Console Output**
```
🎯 PHASE 1/6: CLEAN TRAINING (Target: >94%)
================================================================
Configuration:
  • Alpha (image loss weight): 1.0
  • Beta (message loss weight): 1.0
  • Attacks: {}
  • Min Epochs: 20
  • Max Epochs: 100

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

### **TensorBoard Metrics**
- Loss curves per phase
- Bit accuracy per phase
- Pixel delta over time
- Learning rate schedule
- Attack-specific metrics

```bash
tensorboard --logdir ./runs
# Open http://localhost:6006 in browser
```

---

## 🎯 Checkpoints

All checkpoints are saved in `checkpoints/`:

```
checkpoints/
  ├── model_phase1_clean_94.pth      # Clean training (94%+)
  ├── model_phase2_jpeg_90.pth       # + JPEG (90%+)
  ├── model_phase3_blur_90.pth       # + Blur (90%+)
  ├── model_phase4_resize_90.pth     # + Resize (90%+)
  ├── model_phase5_all_90.pth        # All individual (90%+)
  └── model_phase6_combined_88.pth   # Combined attacks (88%+) ⭐ FINAL
```

### **Load Final Model**
```python
import torch
from models.model import StegoModel

model = StegoModel(message_length=16)
checkpoint = torch.load('checkpoints/model_phase6_combined_88.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🧪 Evaluation

After training completes, the script automatically evaluates on all attack types:

```
📊 FINAL EVALUATION
================================================================

Final Model Performance:
------------------------------------------------------------
Clean (No Attack)        → Accuracy: 94.53%, Pixel Δ: 0.0178
JPEG Compression         → Accuracy: 91.23%, Pixel Δ: 0.0180
Gaussian Blur            → Accuracy: 90.67%, Pixel Δ: 0.0179
Resize Attack            → Accuracy: 90.45%, Pixel Δ: 0.0181
Color Jitter             → Accuracy: 92.11%, Pixel Δ: 0.0178
ALL COMBINED             → Accuracy: 88.92%, Pixel Δ: 0.0182
```

---

## 🐛 Troubleshooting

### **Issue: Out of Memory**
```bash
# Reduce batch size
python train_progressive_90plus.py --batch_size 4
```

### **Issue: Training Too Slow**
```bash
# Reduce max epochs per phase
python train_progressive_90plus.py --max_epochs_per_phase 50

# Or use fewer images for testing
python train_progressive_90plus.py --max_images 200
```

### **Issue: Accuracy Not Reaching Target**
- Let training run longer (increase `--max_epochs_per_phase`)
- Increase max learning rate (`--max_lr 7e-3`)
- Increase message loss weight (edit beta values in script)

### **Issue: Pixel Delta Too High**
- The script auto-adjusts alpha if pixel delta > 0.02
- Can manually set lower beta values for less aggressive message recovery

---

## 🎓 How It Works

### **Progressive Learning Strategy**
1. **Clean Training First:** Model learns basic encoding/decoding without distractions
2. **Gradual Attack Introduction:** Attacks added one at a time
3. **Increasing Difficulty:** Each phase builds on previous knowledge
4. **Combined Training Last:** Final phase applies all attacks simultaneously

### **Cyclical Learning Rate**
- Learning rate cycles between base (1e-4) and max (5e-3)
- Helps escape local minima
- Progressively smaller cycles prevent instability

### **Loss Weight Adjustment**
- **Alpha:** Controls image similarity (imperceptibility)
- **Beta:** Controls message recovery (accuracy)
- Later phases increase beta to prioritize message recovery

---

## 📚 Related Files

- [train_progressive_90plus.py](train_progressive_90plus.py) - Main training script
- [evaluate_transformations.py](evaluate_transformations.py) - Evaluation script
- [TWO_PHASE_TRAINING.md](TWO_PHASE_TRAINING.md) - Original two-phase guide
- [DISTORTION_TRAINING_GUIDE.md](DISTORTION_TRAINING_GUIDE.md) - Distortion details

---

## 🎉 Ready to Train?

```bash
python train_progressive_90plus.py --train_dir data/DIV2K/train --max_images 400
```

Expected final accuracy: **90-95%** on all attacks! 🚀
