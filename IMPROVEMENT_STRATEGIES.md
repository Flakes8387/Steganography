# Strategies to Improve Overall Model Accuracy

## Current Performance
- Individual attacks: 83-86% ✓
- **ALL Combined: 77%** ← needs improvement
- Pixel delta: ~0.019 ✓

---

## Top 5 Strategies (Ranked by Impact)

### 1. ⭐ Train on Combined Attacks (Highest Impact - 77% → 85%+)

**Problem:** Model rarely sees all 4 attacks together during training
**Solution:** Created `train_combined_attacks.py`

**Run this:**
```bash
python train_combined_attacks.py
```

**What it does:**
- Trains with 50-80% probability of ALL attacks combined
- Uses higher message loss weight (beta=2.5)
- Fine-tunes from improved blur model
- Target: 85%+ on combined attacks

**Expected improvement:** +8-10% on ALL Combined

---

### 2. 📊 Use More Training Data (High Impact)

**Current:** 200 images
**Recommended:** 800+ images

**Changes needed:**
```python
# In train_combined_attacks.py, line 23
max_images=200  # Change to 800 or None
```

**Also add DIV2K validation set:**
- Download DIV2K validation (100 images)
- Add path: `'data/DIV2K/valid'`

**Expected improvement:** +2-3% across all attacks

---

### 3. 🎯 Adjust Loss Function Weights (Medium Impact)

**Current:** `alpha=1.0, beta=2.0`
**Recommended:** `alpha=0.8, beta=2.5`

**Why:** Prioritizes message recovery over perfect imperceptibility

**Where to change:**
```python
# In train_combined_attacks.py, line 88
loss_dict = model.compute_loss(cover_images, binary_messages, 
                               alpha=0.8, beta=2.5)  # ← Already done!
```

**Expected improvement:** +1-2% on difficult attacks

---

### 4. 🔢 Add Error Correction (High Impact)

**Current:** 16-bit raw message
**Recommended:** 32-bit with Reed-Solomon or repetition coding

**Implementation options:**

**A. Simple Repetition (Easy):**
```python
# Encode: Repeat each bit 3 times
encoded_msg = message.repeat_interleave(3)  # 16 → 48 bits

# Decode: Majority voting
decoded = decoded_bits.reshape(-1, 3)
final_msg = (decoded.sum(dim=1) > 1.5).float()  # Majority
```

**B. Reed-Solomon (Better, requires library):**
```bash
pip install reedsolo
```

**Expected improvement:** +5-8% on ALL Combined

---

### 5. 🏗️ Architectural Improvements (Medium Impact, More Work)

**A. Add Skip Connections in Decoder:**
```python
# In models/decoder.py
# Connect encoder features directly to decoder
# Helps preserve spatial information through distortions
```

**B. Attention Mechanism:**
```python
# Add self-attention layers to focus on important regions
# Use channel attention to emphasize message-carrying features
```

**C. Multi-Scale Processing:**
```python
# Process at different resolutions (64x64, 128x128, 256x256)
# Combine features from multiple scales
```

**Expected improvement:** +3-5% (requires retraining from scratch)

---

## Quick Action Plan

### Phase 1: Immediate (1-2 hours)
```bash
# Step 1: Train on combined attacks
python train_combined_attacks.py

# Step 2: Evaluate
python evaluate_transformations.py
```
**Expected: 77% → 83-85% on ALL Combined**

### Phase 2: Data Enhancement (2-3 hours)
```bash
# Download more DIV2K images
# Modify max_images to 800
# Retrain with more data
```
**Expected: Additional +2-3%**

### Phase 3: Error Correction (3-4 hours)
```python
# Implement repetition coding or Reed-Solomon
# Retrain model with 32-bit messages
```
**Expected: Additional +5-8%**

---

## Other Optimization Tips

### 6. Fine-tune Distortion Parameters
Currently using reduced blur. You could:
- Test different sigma ranges
- Adjust JPEG quality range (currently 70-95)
- Modify resize scale range (currently 0.5-0.9)

### 7. Increase Model Capacity
- Add more conv layers
- Increase channel dimensions
- Use larger PrepNetwork

**Caution:** Increases training time and memory usage

### 8. Data Augmentation
During training, add:
- Random flips (horizontal/vertical)
- Random rotations (90°, 180°, 270°)
- Random crops at different positions

### 9. Ensemble Methods
Train 3-5 models with different seeds
Average their predictions
**Expected:** +2-3% accuracy

### 10. Adversarial Training
Add adversarial examples during training
Makes model more robust to unexpected attacks

---

## Recommended Sequence

1. ✅ **Start with combined attacks training** (quickest, highest impact)
2. Then add more data if needed
3. Then implement error correction if still not at target
4. Consider architectural changes only if above don't work

---

## Expected Final Performance

After implementing strategies 1-4:
- Clean: 84-86%
- JPEG: 84-86%
- Gaussian Blur: 84-86%
- Resize: 84-86%
- Color Jitter: 86-88%
- **ALL Combined: 85-88%** ⭐ (up from 77%)

Pixel delta: Maintained at ~0.02
