# StegoModel - Unified Steganography Pipeline

## Overview

`models/model.py` provides a unified interface for the complete steganography pipeline, combining the Encoder and Decoder with distortion layers for robust training.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         StegoModel                  │
                    ├─────────────────────────────────────┤
                    │                                     │
Cover Image ───────►│  Encoder                            │
Binary Message ────►│    ├─ PrepNetwork                  │
                    │    └─ HidingNetwork                 │
                    │          ↓                          │
                    │    Stego Image                      │
                    │          ↓                          │
                    │  Distortions (optional, training)   │
                    │    ├─ Gaussian Noise                │
                    │    ├─ Dropout                       │
                    │    ├─ JPEG Compression              │
                    │    ├─ Brightness Adjustment         │
                    │    └─ Contrast Adjustment           │
                    │          ↓                          │
                    │    Distorted Stego                  │
                    │          ↓                          │
                    │  Decoder                            │
                    │    ├─ RevealNetwork                 │
                    │    └─ MessageExtractor              │
                    │          ↓                          │
                    │  Decoded Message                    │
                    └─────────────────────────────────────┘
```

## Main Components

### 1. StegoModel Class

The main unified model class that orchestrates the entire pipeline.

#### Initialization

```python
from models.model import StegoModel

model = StegoModel(
    message_length=1024,      # Number of bits in the message
    image_size=256,           # Size of square images (H=W)
    enable_distortions=True   # Enable distortion layer for training
)
```

#### Key Methods

##### `encode(cover_image, binary_message)`

Embeds a binary message into a cover image.

**Parameters:**
- `cover_image`: Tensor of shape `(batch_size, 3, H, W)`, values in [0, 1]
- `binary_message`: Tensor of shape `(batch_size, message_length)`, values in {0, 1}

**Returns:**
- `stego_image`: Tensor of shape `(batch_size, 3, H, W)`, values in [0, 1]

**Example:**
```python
import torch

cover = torch.rand(4, 3, 256, 256)
message = torch.randint(0, 2, (4, 1024)).float()

stego = model.encode(cover, message)
```

##### `decode(stego_image, return_logits=False)`

Extracts the hidden message from a stego image.

**Parameters:**
- `stego_image`: Tensor of shape `(batch_size, 3, H, W)`, values in [0, 1]
- `return_logits`: If True, returns raw logits instead of binary predictions

**Returns:**
- `binary_message`: Tensor of shape `(batch_size, message_length)`, values in {0, 1}
- Or `logits`: Raw logits for training

**Example:**
```python
# Binary predictions
decoded_message = model.decode(stego)

# Raw logits (for training with BCEWithLogitsLoss)
logits = model.decode(stego, return_logits=True)
```

##### `forward(cover_image, binary_message, apply_distortions=None)`

Complete pipeline: encode → distortions → decode.

**Parameters:**
- `cover_image`: Tensor of shape `(batch_size, 3, H, W)`
- `binary_message`: Tensor of shape `(batch_size, message_length)`
- `apply_distortions`: Override distortion behavior (None = use self.training)

**Returns:**
Dictionary containing:
- `'stego_image'`: Encoded image before distortions
- `'distorted_stego'`: Stego image after distortions
- `'decoded_logits'`: Raw logits from decoder
- `'decoded_message'`: Binary predictions

**Example:**
```python
model.train()  # Enable distortions
outputs = model(cover, message)

print(outputs['stego_image'].shape)        # (4, 3, 256, 256)
print(outputs['decoded_message'].shape)    # (4, 1024)
```

##### `compute_loss(cover_image, binary_message, alpha=1.0, beta=1.0)`

Computes training loss combining imperceptibility and recoverability.

**Loss Formula:**
```
total_loss = alpha * image_loss + beta * message_loss

where:
  image_loss = MSE(stego_image, cover_image)       # Imperceptibility
  message_loss = BCE(decoded_logits, binary_message)  # Recoverability
```

**Parameters:**
- `cover_image`: Tensor of shape `(batch_size, 3, H, W)`
- `binary_message`: Tensor of shape `(batch_size, message_length)`
- `alpha`: Weight for image loss (imperceptibility)
- `beta`: Weight for message loss (recoverability)

**Returns:**
Dictionary containing:
- `'total_loss'`: Combined loss
- `'image_loss'`: MSE between cover and stego
- `'message_loss'`: BCE for message recovery
- `'accuracy'`: Bit accuracy of decoded message

**Example:**
```python
loss_dict = model.compute_loss(cover, message, alpha=1.0, beta=1.0)

optimizer.zero_grad()
loss_dict['total_loss'].backward()
optimizer.step()

print(f"Loss: {loss_dict['total_loss'].item():.4f}")
print(f"Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")
```

##### `save_model(path)` / `load_model(path)`

Save and load model checkpoints.

**Example:**
```python
# Save
model.save_model('checkpoint.pth')

# Load
new_model = StegoModel(1024, 256)
checkpoint = new_model.load_model('checkpoint.pth')
```

### 2. Distortions Class

Applies realistic distortions to stego images during training for robustness.

#### Available Distortions

**Basic Distortions:**
1. **Gaussian Noise**: Adds random noise (σ=0.02)
2. **Spatial Dropout**: Randomly zeros out 10% of pixels
3. **JPEG Compression**: Simulates compression artifacts
4. **Brightness Adjustment**: Random brightness changes (±0.1)
5. **Contrast Adjustment**: Random contrast changes (0.8-1.2×)

**Advanced Attack-Based Distortions (NEW):**
6. **Gaussian Blur**: Blurs image with kernel size 3-7, sigma 0.5-2.0
7. **Resize Attack**: Downscales (0.5-0.9×) then upscales back to original size
8. **Color Jitter**: Combined brightness, contrast, saturation, and hue adjustments

#### Usage

```python
from models.model import Distortions

distortions = Distortions()
distortions.train()  # Enable distortions

# Apply to images
distorted_images = distortions(clean_images)
```

#### Customization

```python
distortions = Distortions(
    dropout_prob=0.1,              # Probability of applying dropout
    jpeg_quality_range=(50, 95)    # JPEG quality range
)

# The attack modules are automatically initialized with default parameters:
# - GaussianBlur: kernel_size_range=(3, 7), sigma_range=(0.5, 2.0)
# - ResizeAttack: scale_range=(0.5, 0.9)
# - ColorJitter: brightness=(-0.2, 0.2), contrast=(0.7, 1.3), 
#                saturation=(0.7, 1.3), hue=(-0.1, 0.1)
```

#### Distortion Probability

Each distortion is applied with 30% probability during training:
- Gaussian noise: Always applied
- Dropout: 10% probability
- JPEG compression: 30% probability
- Brightness: 30% probability
- Contrast: 30% probability
- **Gaussian blur: 30% probability** (NEW)
- **Resize attack: 30% probability** (NEW)
- **Color jitter: 30% probability** (NEW)

## Training Example

### Basic Training Loop

```python
import torch
import torch.optim as optim
from models.model import StegoModel

# Create model
model = StegoModel(message_length=1024, image_size=256, enable_distortions=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    for cover_images, binary_messages in dataloader:
        # Forward pass
        optimizer.zero_grad()
        loss_dict = model.compute_loss(
            cover_images, 
            binary_messages,
            alpha=1.0,  # Image loss weight
            beta=1.0    # Message loss weight
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Log metrics
        print(f"Loss: {loss_dict['total_loss'].item():.4f}, "
              f"Acc: {loss_dict['accuracy'].item()*100:.2f}%")
```

### Advanced Training with Custom Loss Weights

```python
# Start with high alpha (prioritize imperceptibility)
alpha = 2.0
beta = 1.0

for epoch in range(num_epochs):
    for batch in dataloader:
        loss_dict = model.compute_loss(batch[0], batch[1], alpha=alpha, beta=beta)
        
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
    
    # Gradually increase message recovery importance
    if epoch > 10:
        alpha = 1.0
        beta = 2.0
```

## Inference Example

### Basic Inference

```python
# Load trained model
model = StegoModel(1024, 256, enable_distortions=False)
model.load_model('trained_model.pth')
model.eval()

# Prepare data
cover_image = load_image('cover.png')  # Implement this
binary_message = text_to_binary('Secret message')  # Implement this

# Encode
with torch.no_grad():
    stego_image = model.encode(cover_image, binary_message)

# Save stego image
save_image(stego_image, 'stego.png')  # Implement this

# Later: Decode
with torch.no_grad():
    decoded_message = model.decode(stego_image)
    
decoded_text = binary_to_text(decoded_message)  # Implement this
print(f"Decoded: {decoded_text}")
```

### Testing Robustness

```python
model.eval()

# Original stego
with torch.no_grad():
    stego = model.encode(cover, message)
    decoded_clean = model.decode(stego)

accuracy_clean = (decoded_clean == message).float().mean()
print(f"Clean accuracy: {accuracy_clean*100:.2f}%")

# With distortions
distortions = Distortions()
distortions.train()

with torch.no_grad():
    distorted_stego = distortions(stego, apply_all=True)
    decoded_distorted = model.decode(distorted_stego)

accuracy_distorted = (decoded_distorted == message).float().mean()
print(f"Distorted accuracy: {accuracy_distorted*100:.2f}%")
```

## Model Parameters

```
Total parameters: ~1.17 billion
├─ Encoder: ~1.15 billion
└─ Decoder: ~20 million
```

## Performance Metrics

### Image Quality
- **MSE**: Mean Squared Error between cover and stego
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (closer to 1 is better)

### Message Recovery
- **Bit Accuracy**: Percentage of correctly decoded bits
- **Message Loss**: Binary Cross-Entropy between original and decoded
- **BER**: Bit Error Rate (1 - accuracy)

### Example Calculation

```python
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# Calculate metrics
psnr = calculate_psnr(cover_image, stego_image)
accuracy = (decoded == original).float().mean()
ber = 1 - accuracy

print(f"PSNR: {psnr:.2f} dB")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"BER: {ber*100:.2f}%")
```

## Tips and Best Practices

### 1. Loss Weight Tuning
- Start with `alpha=1.0, beta=1.0`
- If stego images are too different: increase `alpha`
- If message recovery is poor: increase `beta`
- Typical ranges: `alpha ∈ [0.5, 2.0]`, `beta ∈ [0.5, 2.0]`

### 2. Training Stability
- Use learning rate scheduling (e.g., ReduceLROnPlateau)
- Start with small learning rate (1e-4)
- Monitor both losses separately
- Use gradient clipping if training is unstable

### 3. Distortion Usage
- **Training**: Enable distortions for robustness
- **Validation**: Disable to measure clean performance
- **Testing**: Test both with and without distortions

### 4. Memory Management
- Large model (~1.17B parameters) requires significant GPU memory
- Reduce batch size if OOM errors occur
- Use mixed precision training (AMP) to save memory
- Consider gradient checkpointing for very deep models

### 5. Data Augmentation
- Apply geometric transforms to cover images
- Use different message patterns (random, structured)
- Vary image sources and content types

## File Structure

```
models/
├── __init__.py
├── encoder.py          # Encoder model
├── decoder.py          # Decoder model
└── model.py           # Unified StegoModel (this file)

example_usage.py       # Usage examples
test_models.py         # Model tests
```

## See Also

- `MODEL_DOCUMENTATION.md` - Detailed encoder/decoder documentation
- `example_usage.py` - Complete usage examples
- `test_models.py` - Model testing script

## Future Enhancements

1. **Attention Mechanisms**: Add channel/spatial attention
2. **Variable Message Length**: Support dynamic message sizes
3. **Multi-Resolution**: Process images at multiple scales
4. **Adversarial Training**: Add discriminator for steganalysis resistance
5. **Model Compression**: Reduce encoder size with knowledge distillation
