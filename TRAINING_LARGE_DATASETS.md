# Training on Large Datasets - Quick Start Guide

## ✅ Currently Running

**Training Status**: Active
- **Dataset**: 2000 synthetic images in `data/synthetic_large`
- **Model Size**: 1.1B parameters (1024-bit messages, 256×256 images)
- **Training**: 30 epochs, batch size 16
- **Location**: `runs/20251213_012432`

Monitor training:
```bash
tensorboard --logdir runs/
```

## 📊 How to Get Real Large Datasets

### Option 1: DIV2K (Recommended) ⭐
**Best for: Quality training with manageable size**

```bash
# Manual download (recommended)
# 1. Visit: https://data.vision.ee.ethz.ch/cvl/DIV2K/
# 2. Download:
#    - DIV2K_train_HR.zip (~2.5GB)
#    - DIV2K_valid_HR.zip (~300MB)
# 3. Extract to: data/div2k/

# Then train:
python train.py \
    --train_dir data/div2k/DIV2K_train_HR \
    --val_dir data/div2k/DIV2K_valid_HR \
    --num_epochs 50 \
    --batch_size 16 \
    --message_length 1024 \
    --image_size 256
```

**Expected Results**:
- Training time: 2-3 hours (GPU) or 10-12 hours (CPU)
- Final PSNR: >38 dB
- Message accuracy: >98%

---

### Option 2: COCO (Production Scale)
**Best for: State-of-the-art results**

```bash
# Download COCO train2017
# Visit: https://cocodataset.org/#download
# Download: train2017.zip (~18GB)
# Extract to: data/coco/

# Train:
python train.py \
    --train_dir data/coco/train2017 \
    --num_epochs 30 \
    --batch_size 32 \
    --message_length 1024 \
    --image_size 256 \
    --enable_distortions \
    --apply_attacks
```

**Expected Results**:
- Training time: 8-10 hours (RTX 3090) or 3-4 days (CPU)
- Final PSNR: >40 dB
- Message accuracy: >99%
- Robust to JPEG, noise, resize attacks

---

### Option 3: ImageNet (Large Scale)
**Best for: Maximum diversity**

```bash
# Download ImageNet ILSVRC2012
# Visit: https://www.image-net.org/download.php
# Requires registration
# Download: ILSVRC2012_img_train.tar (~150GB)

# Train:
python train.py \
    --train_dir data/imagenet/train \
    --num_epochs 20 \
    --batch_size 64 \
    --message_length 1024 \
    --image_size 256
```

---

### Option 4: Flickr30k (Free Alternative)
**Best for: Free large dataset**

```bash
# Download from Kaggle:
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
# Download: flickr30k_images.zip (~4GB)
# Extract to: data/flickr30k/

python train.py \
    --train_dir data/flickr30k \
    --num_epochs 40 \
    --batch_size 16 \
    --message_length 1024 \
    --image_size 256
```

---

## 🚀 Training Configurations

### Configuration 1: Fast Training (CPU)
```bash
python train.py \
    --train_dir data/YOUR_DATASET \
    --num_epochs 30 \
    --batch_size 8 \
    --message_length 512 \
    --image_size 128 \
    --save_freq 5
```

### Configuration 2: Balanced (GPU)
```bash
python train.py \
    --train_dir data/YOUR_DATASET \
    --num_epochs 50 \
    --batch_size 32 \
    --message_length 1024 \
    --image_size 256 \
    --enable_distortions \
    --save_freq 5
```

### Configuration 3: Production (High-End GPU)
```bash
python train.py \
    --train_dir data/YOUR_DATASET \
    --num_epochs 100 \
    --batch_size 64 \
    --message_length 2048 \
    --image_size 512 \
    --enable_distortions \
    --apply_attacks \
    --learning_rate 0.0005 \
    --save_freq 10
```

---

## 📁 Using Your Own Images

If you have your own image collection:

```bash
# 1. Organize images in a folder
mkdir -p data/my_images
cp /path/to/your/images/*.jpg data/my_images/

# 2. Train
python train.py \
    --train_dir data/my_images \
    --num_epochs 50 \
    --batch_size 16 \
    --message_length 1024 \
    --image_size 256
```

**Minimum Requirements**:
- At least 500 images (1000+ recommended)
- JPEG or PNG format
- Resolution > 256×256
- Diverse scenes (not all similar images)

---

## 📊 Monitoring Training

### TensorBoard (Recommended)
```bash
tensorboard --logdir runs/
```

Open: http://localhost:6006

**Key Metrics**:
- **Loss**: Should decrease to <0.1
- **PSNR**: Should increase to >35 dB
- **Message Accuracy**: Should reach >95%
- **Learning Rate**: Will decrease when loss plateaus

### Command Line
Watch the terminal output for:
- Epoch progress
- Loss values
- Training time per epoch

---

## ⚙️ Optimization Tips

### 1. Speed Up Training
```bash
# Use smaller image size
--image_size 128

# Reduce message length
--message_length 512

# Increase batch size (if memory allows)
--batch_size 32

# Use GPU
--device cuda
```

### 2. Improve Quality
```bash
# Enable distortions for robustness
--enable_distortions

# Train longer
--num_epochs 100

# Use larger images
--image_size 512

# Reduce learning rate
--learning_rate 0.0001
```

### 3. Memory Issues
```bash
# Reduce batch size
--batch_size 4

# Reduce image size
--image_size 128

# Reduce message length
--message_length 512
```

---

## 🎯 Training Targets

### Good Training Results:
- PSNR > 35 dB (image quality)
- BER < 5% (bit error rate)
- Message accuracy > 95%
- Training loss < 0.1

### Excellent Results:
- PSNR > 40 dB
- BER < 1%
- Message accuracy > 99%
- Training loss < 0.05

---

## 🔄 Resume Training

If training is interrupted:

```bash
python train.py \
    --train_dir data/YOUR_DATASET \
    --num_epochs 100 \
    --resume checkpoints/checkpoint_epoch_50.pth \
    --batch_size 16
```

---

## 📝 After Training

Once training completes:

1. **Test Encoding**:
```bash
python encode.py \
    --image test.jpg \
    --message "Secret message" \
    --output stego.png \
    --model checkpoints/best_model.pth \
    --show-stats
```

2. **Test Decoding**:
```bash
python decode.py \
    --image stego.png \
    --model checkpoints/best_model.pth \
    --show-confidence
```

3. **Test Robustness**:
```bash
jupyter notebook Robustness_Test.ipynb
```

---

## 🔥 Current Training

Your training is currently running:
- **Dataset**: 2000 synthetic images
- **Progress**: Check terminal output
- **Logs**: `runs/20251213_012432`

To view logs:
```bash
tensorboard --logdir runs/20251213_012432
```

**Note**: Synthetic images are good for testing the pipeline, but for real results, you need natural images (DIV2K, COCO, Flickr30k, etc.)

---

## 📚 More Information

- Dataset guide: [DATASET_GUIDE.md](DATASET_GUIDE.md)
- Training guide: [README.md](README.md)
- Model documentation: [STEGOMODEL_DOCUMENTATION.md](STEGOMODEL_DOCUMENTATION.md)
