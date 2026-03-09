# Steganography Models - Documentation

## Overview
This project implements deep neural network-based steganography models for hiding and extracting binary messages in images without using LSB techniques.

## Models

### 1. Encoder (`models/encoder.py`)

**Purpose**: Embeds a binary message into a cover image to produce a stego image.

**Architecture Components**:
- **ResidualBlock**: Convolutional blocks with skip connections for better gradient flow
- **PrepNetwork**: Transforms binary message into spatial feature maps
  - Fully connected layers expand message to image dimensions
  - Convolutional layers create message features (output: 64 channels)
- **HidingNetwork**: Combines cover image and message features
  - Encoder path: 64 → 128 → 128 channels with residual blocks
  - Residual blocks for deep feature extraction
  - Decoder path: 128 → 64 → 32 → 3 channels
  - Residual connection from cover image for minimal perturbation

**Input**:
- `cover_image`: RGB image tensor of shape `(batch_size, 3, H, W)`, values in [0, 1]
- `binary_message`: Binary message tensor of shape `(batch_size, message_length)`, values in {0, 1}

**Output**:
- `stego_image`: RGB image tensor of shape `(batch_size, 3, H, W)`, values in [0, 1]

**Parameters**: ~1.1 billion (1,146,427,395)

**Usage**:
```python
from models.encoder import Encoder
import torch

# Create encoder
encoder = Encoder(message_length=1024, image_size=256)

# Prepare inputs
cover_image = torch.rand(4, 3, 256, 256)  # Batch of 4 images
binary_message = torch.randint(0, 2, (4, 1024)).float()  # 1024 bits

# Generate stego image
stego_image = encoder(cover_image, binary_message)
```

### 2. Decoder (`models/decoder.py`)

**Purpose**: Extracts the hidden binary message from a stego image.

**Architecture Components**:
- **ResidualBlock**: Same as encoder for consistency
- **RevealNetwork**: Extracts message features from stego image
  - Encoder path: 3 → 64 → 128 → 256 channels
  - 4 residual blocks for deep feature extraction
  - Decoder path: 256 → 128 → 64 → 32 → 1 channel
  - Outputs spatial message features
- **MessageExtractor**: Converts spatial features to binary message
  - Adaptive pooling to fixed size (32×32)
  - Fully connected layers: 1024 → 4096 → 2048 → message_length
  - Dropout layers for regularization
  - Outputs logits for each bit

**Input**:
- `stego_image`: RGB image tensor of shape `(batch_size, 3, H, W)`, values in [0, 1]
- `return_logits`: bool, if True returns raw logits instead of binary predictions

**Output**:
- `binary_message`: Binary message tensor of shape `(batch_size, message_length)`, values in {0, 1}
- Or `logits`: Raw logits for training with BCEWithLogitsLoss

**Parameters**: ~20 million (20,184,001)

**Usage**:
```python
from models.decoder import Decoder
import torch

# Create decoder
decoder = Decoder(message_length=1024, image_size=256)

# Decode message
stego_image = torch.rand(4, 3, 256, 256)
decoded_message = decoder(stego_image)  # Binary predictions

# Or get raw logits for training
logits = decoder(stego_image, return_logits=True)

# Or get probabilities
probabilities = decoder.get_probabilities(stego_image)
```

## Key Features

### ✅ No LSB Techniques
- Uses deep neural networks instead of simple bit manipulation
- More robust to image processing attacks
- Harder to detect with steganalysis

### ✅ Residual Blocks
- Skip connections improve gradient flow
- Enable deeper networks
- Better feature extraction

### ✅ U-Net Style Architecture (Encoder)
- Encoder-decoder structure with bottleneck
- Preserves spatial information
- Minimal visual artifacts

### ✅ Multi-Scale Feature Processing
- Progressive channel expansion/reduction
- Captures both low and high-level features
- Better message embedding

### ✅ Residual Connection from Cover Image
- Encourages minimal perturbation
- Preserves cover image quality
- Reduces detectability

## Training Considerations

### Loss Functions
```python
# Encoder loss (combined)
image_loss = F.mse_loss(stego_image, cover_image)  # Imperceptibility
message_loss = F.binary_cross_entropy_with_logits(
    decoder(stego_image, return_logits=True), 
    binary_message
)  # Recoverability
total_loss = alpha * image_loss + beta * message_loss

# Decoder loss
decoder_loss = F.binary_cross_entropy_with_logits(logits, binary_message)
```

### Training Tips
1. **Joint Training**: Train encoder and decoder together
2. **Loss Weighting**: Balance imperceptibility (α) vs recoverability (β)
   - Start with α=1.0, β=1.0
   - Adjust based on validation metrics
3. **Learning Rate**: Start with 1e-4, use scheduler
4. **Batch Size**: 4-8 images (memory intensive due to large encoder)
5. **Image Normalization**: [0, 1] range (as implemented)
6. **Data Augmentation**: Add noise, JPEG compression to improve robustness

## Testing

Run the test script to verify both models:
```bash
python test_models.py
```

Expected output:
- Models load successfully
- Forward pass completes without errors
- Output shapes are correct
- Random accuracy ~50% (untrained baseline)

## Model Files

```
models/
├── __init__.py
├── encoder.py          # Encoder model (1.1B parameters)
├── decoder.py          # Decoder model (20M parameters)
└── __pycache__/
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0
pillow>=9.0.0
```

Install with:
```bash
pip install torch torchvision torchaudio
```

## Future Improvements

1. **Reduce Encoder Size**: Current model has 1.1B parameters
   - Use depthwise separable convolutions
   - Reduce channel dimensions
   - Share weights across blocks

2. **Add Attention Mechanisms**: 
   - Channel attention
   - Spatial attention
   - Better message-image fusion

3. **Adversarial Training**:
   - Add discriminator for steganalysis
   - Make stego images harder to detect

4. **Robustness**:
   - Train with JPEG compression
   - Add noise layers
   - Test against common attacks

5. **Variable Message Length**:
   - Dynamic message embedding
   - Padding/masking for different lengths

## Citation

If you use this code, please cite appropriately.

## License

See LICENSE file for details.
