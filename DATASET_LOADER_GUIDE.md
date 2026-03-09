# Dataset Loader Integration Guide

## Overview

The `utils/dataset.py` module provides a unified dataset loader that works for both generic image folders and DIV2K high-resolution images. It supports patch-based loading for efficient training on local GPUs.

## Features

✅ **Unified Interface** - Same API for all dataset types  
✅ **DIV2K Support** - Optimized for high-resolution images with patch extraction  
✅ **Auto-Detection** - Automatically detects dataset structure  
✅ **Patch-Based Loading** - Extract multiple crops from large images  
✅ **Dataset Size Limits** - Control base images to manage training time  
✅ **Standard Return Format** - Always returns `(image_tensor, random_binary_message)`  

## Usage

### Basic Usage (Generic Dataset)

```python
from utils.dataset import SteganographyDataset

# Create dataset
dataset = SteganographyDataset(
    root_dir='path/to/images',
    image_size=256,
    message_length=1024,
    dataset_type='auto'  # Auto-detect structure
)

# Get sample
image_tensor, random_binary_message = dataset[0]

print(f"Image: {image_tensor.shape}")  # (3, 256, 256)
print(f"Message: {random_binary_message.shape}")  # (1024,)
```

### DIV2K with Patch-Based Loading

```python
from utils.dataset import SteganographyDataset

# Create DIV2K dataset with patches
dataset = SteganographyDataset(
    root_dir='data/DIV2K/train',
    image_size=128,              # Patch size
    message_length=32,
    dataset_type='DIV2K',        # Explicit DIV2K flag
    use_patches=True,            # Enable patch extraction
    patches_per_image=4,         # 4 random crops per image
    random_crop=True,            # Random positioning
    max_images=500               # Limit to 500 base images
)

# Dataset info
print(f"Base images: {len(dataset.image_paths)}")      # 500
print(f"Total samples: {len(dataset)}")                # 2,000 (500 × 4)

# Get sample (random patch from a random image)
image_tensor, random_binary_message = dataset[0]

print(f"Patch: {image_tensor.shape}")  # (3, 128, 128)
```

### Using DataLoader

```python
from utils.dataset import create_dataloader

# Create DataLoader for DIV2K
dataloader = create_dataloader(
    root_dir='data/DIV2K/train',
    batch_size=8,
    image_size=128,
    message_length=32,
    dataset_type='DIV2K',
    use_patches=True,
    patches_per_image=4,
    max_images=500,
    num_workers=8,
    shuffle=True
)

# Training loop
for images, messages in dataloader:
    # images: (8, 3, 128, 128)
    # messages: (8, 32)
    
    # Your training code here
    pass
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | str | required | Root directory containing images |
| `image_size` | int | 256 | Target size (patch size if use_patches=True) |
| `message_length` | int | 1024 | Length of binary message |

### Dataset Type Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_type` | str | 'auto' | 'auto', 'DIV2K', 'coco', 'imagenet', 'flat' |

**Supported dataset types:**
- `'auto'` - Auto-detect based on directory structure
- `'DIV2K'` - DIV2K high-resolution images
- `'coco'` - MS COCO dataset
- `'imagenet'` - ImageNet-style class subdirectories
- `'flat'` - Flat directory with images

### Patch-Based Loading (DIV2K)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_patches` | bool | False | Enable patch extraction |
| `patches_per_image` | int | 4 | Number of patches per image |
| `random_crop` | bool | True | Random (True) or center (False) crop |

### Dataset Size Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_images` | int | None | Limit BASE images (not total patches) |

**Important:** `max_images` limits BASE images, not total samples:
```
max_images = 500
patches_per_image = 4
→ Total samples = 500 × 4 = 2,000
```

## Return Format

All methods return the same format:

```python
image_tensor, random_binary_message = dataset[idx]
```

- **`image_tensor`**: `torch.Tensor` of shape `(3, image_size, image_size)`
  - Normalized to [0, 1] range
  - RGB channels in CHW format
  
- **`random_binary_message`**: `torch.Tensor` of shape `(message_length,)`
  - Binary values {0, 1} as float32
  - Randomly generated for each access

## Dataset Type Detection

The loader automatically detects dataset structure:

### DIV2K Detection
Triggers when:
- Path contains "div2k" or "DIV2K"
- Directory structure: `train/`, `valid/`, or flat with .png files

### COCO Detection
Triggers when:
- Contains `train2017/`, `val2017/`, etc.

### ImageNet Detection
Triggers when:
- Contains >10 class subdirectories with images

### Flat Detection
Triggers when:
- Images directly in root directory

## Examples

### Example 1: Quick Training (300 images)

```python
from utils.dataset import create_dataloader

dataloader = create_dataloader(
    root_dir='data/DIV2K/train',
    batch_size=16,
    image_size=128,
    message_length=16,
    dataset_type='DIV2K',
    use_patches=True,
    patches_per_image=4,
    max_images=300,      # 300 × 4 = 1,200 samples
    random_crop=True
)

# Training time: ~1-2 hours on RTX 3050 6GB
```

### Example 2: Balanced Training (500 images) - RECOMMENDED

```python
from utils.dataset import create_dataloader

dataloader = create_dataloader(
    root_dir='data/DIV2K/train',
    batch_size=8,
    image_size=128,
    message_length=32,
    dataset_type='DIV2K',
    use_patches=True,
    patches_per_image=4,
    max_images=500,      # 500 × 4 = 2,000 samples
    random_crop=True
)

# Training time: ~2-3 hours on RTX 3050 6GB
# Quality: 85-90% bit accuracy
```

### Example 3: Full Dataset (800 images)

```python
from utils.dataset import create_dataloader

dataloader = create_dataloader(
    root_dir='data/DIV2K/train',
    batch_size=8,
    image_size=128,
    message_length=64,
    dataset_type='DIV2K',
    use_patches=True,
    patches_per_image=4,
    max_images=800,      # 800 × 4 = 3,200 samples
    random_crop=True
)

# Training time: ~4-5 hours on RTX 3050 6GB
# Quality: 90-95% bit accuracy
```

### Example 4: Generic Image Folder

```python
from utils.dataset import create_dataloader

dataloader = create_dataloader(
    root_dir='path/to/your/images',
    batch_size=16,
    image_size=256,
    message_length=1024,
    dataset_type='auto',     # Auto-detect
    use_patches=False,       # No patches for generic images
    max_images=1000
)
```

## Integration with Training Script

```python
from utils.dataset import create_dataloader
import torch

# Create DataLoader
train_loader = create_dataloader(
    root_dir='data/DIV2K/train',
    batch_size=8,
    image_size=128,
    message_length=32,
    dataset_type='DIV2K',
    use_patches=True,
    patches_per_image=4,
    max_images=500
)

# Training loop
for epoch in range(num_epochs):
    for images, messages in train_loader:
        # Move to GPU
        images = images.to(device)
        messages = messages.to(device)
        
        # Forward pass
        encoded = encoder(images, messages)
        decoded_messages = decoder(encoded)
        
        # Compute loss
        loss = criterion(decoded_messages, messages)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

## Comparison: train.py vs utils/dataset.py

Both implementations work identically:

### Using train.py (old way)
```python
from train import SteganographyDataset

dataset = SteganographyDataset(
    image_dir='data/DIV2K/train',
    image_size=128,
    message_length=32,
    use_patches=True,
    patches_per_image=4,
    random_crop=True,
    max_images=500
)
```

### Using utils/dataset.py (new way)
```python
from utils.dataset import SteganographyDataset

dataset = SteganographyDataset(
    root_dir='data/DIV2K/train',
    image_size=128,
    message_length=32,
    dataset_type='DIV2K',      # Explicit flag
    use_patches=True,
    patches_per_image=4,
    random_crop=True,
    max_images=500
)
```

**Key differences:**
- `root_dir` instead of `image_dir`
- Added `dataset_type` parameter for explicit control
- Auto-detection support
- Same functionality, unified interface

## Verification

Run the integration test:

```bash
python test_div2k_integration.py
```

Expected output:
```
✅ ALL TESTS PASSED!

📋 Summary:
  ✓ Generic dataset loading works
  ✓ DIV2K can be selected with dataset_type='DIV2K'
  ✓ Patch-based loading works for DIV2K
  ✓ Same loader logic works for both generic and DIV2K
  ✓ Returns (image_tensor, random_binary_message)
```

## Troubleshooting

### "No images found in directory"
- Check that `root_dir` path is correct
- Verify images have supported extensions (.png, .jpg, etc.)
- For DIV2K, images should be in `data/DIV2K/train/`

### "Out of memory error"
- Reduce `batch_size` (try 4 or 8)
- Reduce `max_images` (try 300 instead of 500)
- Ensure `use_patches=True` (not loading full-size images)
- Check `image_size` is 128 (not 256 or higher)

### "Dataset too small"
- Increase `patches_per_image` (try 8 instead of 4)
- Increase `max_images` (try 800 for full dataset)
- Disable size limit: `max_images=None`

### "Training too slow"
- Reduce `max_images` to 300
- Increase `batch_size` to 16
- Ensure `use_patches=True` for DIV2K

## Summary

✅ **Use `dataset_type='DIV2K'`** to select DIV2K explicitly  
✅ **Same loader logic** works for both generic folders and DIV2K  
✅ **Always returns** `(image_tensor, random_binary_message)`  
✅ **Patch-based loading** multiplies dataset size without downloading more data  
✅ **Dataset size limits** control training time on local GPU  

**Recommended configuration:**
```python
dataset = SteganographyDataset(
    root_dir='data/DIV2K/train',
    image_size=128,
    message_length=32,
    dataset_type='DIV2K',
    use_patches=True,
    patches_per_image=4,
    random_crop=True,
    max_images=500  # 2,000 training samples, 2-3 hours
)
```
