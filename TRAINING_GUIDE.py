"""
Training Script Usage Guide

This guide shows how to use train.py to train the deep learning steganography model.
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

# Minimal command (replace with your image directory)
python train.py - -train_dir ./images

# Full example with all common options
python train.py \
    - -train_dir ./images/train \
    - -val_dir ./images/val \
    - -num_epochs 100 \
    - -batch_size 16 \
    - -learning_rate 0.001 \
    - -message_length 1024 \
    - -image_size 256 \
    - -checkpoint_dir ./checkpoints \
    - -log_dir ./runs \
    - -save_freq 5

# =============================================================================
# TRAINING ON DIFFERENT DATASETS
# =============================================================================

# COCO Dataset
python train.py \
    - -train_dir ./coco/train2017 \
    - -val_dir ./coco/val2017 \
    - -batch_size 32 \
    - -num_workers 8

# ImageNet Dataset
python train.py \
    - -train_dir ./imagenet/train \
    - -val_dir ./imagenet/val \
    - -batch_size 64 \
    - -learning_rate 0.0005

# BOSSBase Dataset
python train.py \
    - -train_dir ./bossbase \
    - -batch_size 16 \
    - -image_size 512

# Custom Dataset (limited images for quick testing)
python train.py \
    - -train_dir ./my_images \
    - -max_train_images 1000 \
    - -num_epochs 20 \
    - -batch_size 8

# =============================================================================
# ADVANCED TRAINING OPTIONS
# =============================================================================

# With additional attacks during training (more robust but slower)
python train.py \
    - -train_dir ./images \
    - -apply_attacks \
    - -num_epochs 150

# Resume from checkpoint
python train.py \
    - -train_dir ./images \
    - -resume ./checkpoints/checkpoint_best.pth

# CPU-only training (no GPU)
python train.py \
    - -train_dir ./images \
    - -no_cuda \
    - -batch_size 4

# Large message capacity
python train.py \
    - -train_dir ./images \
    - -message_length 4096 \
    - -batch_size 8

# High resolution images
python train.py \
    - -train_dir ./images \
    - -image_size 512 \
    - -batch_size 4

# =============================================================================
# MONITORING TRAINING
# =============================================================================

# Start TensorBoard to monitor training
tensorboard - -logdir ./runs

# Then open browser to http://localhost:6006
# You'll see:
# - Training/validation loss curves
# - Image loss (imperceptibility)
# - Message loss (recoverability)
# - Bit accuracy
# - Learning rate schedule

# =============================================================================
# CHECKPOINTS
# =============================================================================

# Checkpoints are saved to ./checkpoints/ by default:
# - checkpoint_latest.pth  : Latest model
# - checkpoint_best.pth    : Best validation loss
# - checkpoint_epoch_N.pth : Periodic checkpoints (every 10 epochs)

# Load a checkpoint for inference:
"""
from models.model import StegoModel
import torch

model = StegoModel(message_length=1024, image_size=256)
checkpoint = torch.load('./checkpoints/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
"""

# =============================================================================
# TYPICAL TRAINING WORKFLOW
# =============================================================================

"""
Step 1: Prepare your dataset
-----------------------------
Organize images in a directory:
  ./images/train/
  ./images/val/

Step 2: Start training
----------------------
python train.py \
    --train_dir ./images/train \
    --val_dir ./images/val \
    --num_epochs 100 \
    --batch_size 16

Step 3: Monitor with TensorBoard
---------------------------------
tensorboard --logdir ./runs
(Open http://localhost:6006)

Step 4: Training will output:
------------------------------
Epoch [1/100]
  Train Loss: 0.234567
  Train Image Loss: 0.012345
  Train Message Loss: 0.123456
  Train Accuracy: 98.76%
  
  Val Loss: 0.245678
  Val Image Loss: 0.013456
  Val Message Loss: 0.134567
  Val Accuracy: 97.89%
  ✓ New best validation loss!

Step 5: Use trained model
--------------------------
Load checkpoint_best.pth and use for inference
"""

# =============================================================================
# PERFORMANCE TIPS
# =============================================================================

"""
1. GPU Training:
   - Use CUDA-enabled GPU for 10-50x speedup
   - Increase batch_size to utilize GPU memory
   - Use num_workers=4-8 for data loading

2. Memory Optimization:
   - Reduce batch_size if out of memory
   - Reduce image_size (256 is good default)
   - Reduce num_workers if CPU memory limited

3. Training Speed:
   - Larger batch_size = faster but needs more memory
   - More num_workers = faster data loading
   - Don't use --apply_attacks for first training

4. Model Quality:
   - Train longer (100+ epochs) for better results
   - Use validation set to prevent overfitting
   - Monitor TensorBoard for convergence

5. Robustness:
   - Use --enable_distortions (default)
   - Add --apply_attacks for extra robustness
   - Train with diverse image dataset
"""

# =============================================================================
# EXPECTED TRAINING TIME
# =============================================================================

"""
On NVIDIA RTX 3090 (24GB):
- 10,000 images, batch_size=32: ~5 min/epoch
- 50,000 images, batch_size=32: ~25 min/epoch
- 100 epochs: ~8-40 hours

On NVIDIA GTX 1080 (8GB):
- 10,000 images, batch_size=16: ~8 min/epoch
- 50,000 images, batch_size=16: ~40 min/epoch
- 100 epochs: ~13-66 hours

On CPU (Intel i7):
- 1,000 images, batch_size=4: ~15 min/epoch
- 100 epochs: ~25 hours
(CPU training not recommended for large datasets)
"""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
Problem: CUDA out of memory
Solution: Reduce --batch_size (try 8, 4, or 2)

Problem: Training very slow
Solution: 
  - Increase --num_workers (try 4 or 8)
  - Check GPU is being used (not --no_cuda)
  - Reduce --image_size if possible

Problem: Model not converging
Solution:
  - Check dataset has enough images (>1000)
  - Try lower --learning_rate (0.0001)
  - Train longer (more epochs)
  - Check TensorBoard for loss curves

Problem: Accuracy not improving
Solution:
  - Model might need more capacity
  - Dataset might be too challenging
  - Try without --apply_attacks first
  - Check images are loading correctly

Problem: Can't find images
Solution:
  - Check --train_dir path is correct
  - Ensure images are .jpg, .png, or .bmp
  - Script searches recursively in subdirectories
"""

print(__doc__)
