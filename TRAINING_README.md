# Deep Learning Steganography Training

Complete training pipeline for deep learning-based steganography using PyTorch.

## Quick Start

### 1. Verify Setup
```bash
python test_training_setup.py
```

### 2. Start Training
```bash
# Basic training
python train.py --train_dir ./images

# Full training with validation
python train.py \
    --train_dir ./images/train \
    --val_dir ./images/val \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001
```

### 3. Monitor Training
```bash
tensorboard --logdir ./runs
```
Open browser to http://localhost:6006

## Training Features

✅ **Complete Deep Learning Pipeline**
- Encoder → Distortions → Decoder architecture
- Automatic dataset loading from any folder
- Support for COCO, ImageNet, BOSSBase structures
- Auto-resizing and normalization

✅ **Loss Computation**
- Image loss (MSE) - measures imperceptibility
- Message loss (BCE) - measures recoverability  
- Combined weighted loss function
- Bit accuracy metric

✅ **Optimization**
- Adam optimizer with weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping (prevents exploding gradients)
- Automatic checkpoint saving

✅ **TensorBoard Logging**
- Training/validation loss curves
- Image loss and message loss separately
- Bit accuracy tracking
- Learning rate schedule

✅ **Checkpoint Management**
- `checkpoint_latest.pth` - Latest model
- `checkpoint_best.pth` - Best validation loss
- `checkpoint_epoch_N.pth` - Periodic saves
- Resume training from any checkpoint

✅ **Robustness Features**
- Built-in distortions during training
- Optional additional attacks (JPEG, noise, resize, color jitter)
- Prepares model for real-world conditions

## Command Line Arguments

### Dataset
- `--train_dir` - Training images directory (required)
- `--val_dir` - Validation images directory (optional)
- `--max_train_images` - Limit training set size
- `--max_val_images` - Limit validation set size

### Model
- `--message_length` - Binary message length (default: 1024)
- `--image_size` - Image size H=W (default: 256)
- `--enable_distortions` - Enable training distortions (default: True)
- `--apply_attacks` - Apply extra attacks during training

### Training
- `--num_epochs` - Number of epochs (default: 100)
- `--batch_size` - Batch size (default: 16)
- `--learning_rate` - Learning rate (default: 0.001)
- `--weight_decay` - L2 regularization (default: 1e-5)
- `--num_workers` - Data loader workers (default: 4)

### Checkpoints
- `--checkpoint_dir` - Checkpoint directory (default: ./checkpoints)
- `--log_dir` - TensorBoard logs (default: ./runs)
- `--save_freq` - Save every N epochs (default: 5)
- `--resume` - Resume from checkpoint path

### Other
- `--seed` - Random seed (default: 42)
- `--no_cuda` - Disable GPU training

## Training Examples

### Standard Training
```bash
python train.py \
    --train_dir ./images \
    --num_epochs 100 \
    --batch_size 16
```

### High Capacity Training
```bash
python train.py \
    --train_dir ./images \
    --message_length 4096 \
    --batch_size 8
```

### Robust Training (with attacks)
```bash
python train.py \
    --train_dir ./images \
    --apply_attacks \
    --num_epochs 150
```

### Resume Training
```bash
python train.py \
    --train_dir ./images \
    --resume ./checkpoints/checkpoint_best.pth
```

## Expected Output

```
Using device: cuda

Loading datasets...
Loaded 10000 images from ./images/train
Loaded 2000 images from ./images/val

Initializing model...
Encoder parameters: 1,234,567
Decoder parameters: 1,345,678
Total parameters: 2,580,245

============================================================
Starting training...
============================================================

Epoch [1/100]
------------------------------------------------------------
Batch [0/625] Loss: 1.234567 ImgLoss: 0.123456 MsgLoss: 1.111111 Acc: 51.23%
Batch [10/625] Loss: 0.987654 ImgLoss: 0.098765 MsgLoss: 0.888889 Acc: 65.43%
...

Train Metrics:
  Loss: 0.543210
  Image Loss: 0.054321
  Message Loss: 0.488889
  Accuracy: 87.65%
  Time: 123.45s

Validation Metrics:
  Loss: 0.567890
  Image Loss: 0.056789
  Message Loss: 0.511111
  Accuracy: 85.43%
  ✓ New best validation loss: 0.567890

Saved checkpoint to ./checkpoints/checkpoint_latest.pth
Saved best checkpoint to ./checkpoints/checkpoint_best.pth
```

## Performance

**On NVIDIA RTX 3090:**
- 10K images, batch=32: ~5 min/epoch
- 100 epochs: ~8 hours

**On NVIDIA GTX 1080:**
- 10K images, batch=16: ~8 min/epoch
- 100 epochs: ~13 hours

**On CPU:**
- Not recommended for large datasets
- 1K images, batch=4: ~15 min/epoch

## Using Trained Model

```python
import torch
from models.model import StegoModel
from PIL import Image
from torchvision import transforms

# Load model
model = StegoModel(message_length=1024, image_size=256)
checkpoint = torch.load('./checkpoints/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and process image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
image = Image.open('cover.jpg').convert('RGB')
image = transform(image).unsqueeze(0)

# Create message
message = torch.randint(0, 2, (1, 1024)).float()

# Encode
with torch.no_grad():
    stego = model.encode(image, message)

# Decode
with torch.no_grad():
    decoded = model.decode(stego)

# Calculate accuracy
accuracy = (decoded == message).float().mean()
print(f"Accuracy: {accuracy.item()*100:.2f}%")
```

## Troubleshooting

**Out of memory:**
- Reduce `--batch_size` (try 8, 4, or 2)
- Reduce `--image_size` (try 128)

**Training slow:**
- Increase `--num_workers` (try 4-8)
- Check GPU is being used
- Reduce `--image_size`

**Model not converging:**
- Check dataset has enough images (>1000)
- Try lower learning rate (0.0001)
- Train longer
- Check TensorBoard

## Files

- `train.py` - Main training script
- `test_training_setup.py` - Verify setup works
- `TRAINING_GUIDE.py` - Detailed usage examples
- `models/model.py` - StegoModel (Encoder + Decoder)
- `attacks/` - Distortion modules
- `defense/` - Defense mechanisms

## Next Steps

After training:
1. Evaluate model on test set
2. Test robustness against attacks
3. Export for deployment
4. Fine-tune on specific use case

## Citation

This replaces classical LSB steganography with deep learning approach for improved imperceptibility and robustness.
