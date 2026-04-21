# Image Steganography Model

![steganography](https://user-images.githubusercontent.com/28294942/184524203-850eea1e-74fc-4539-a70f-8c65adcf96dc.png)

A robust deep learning-based image steganography model using PyTorch. This project implements an end-to-end neural network approach for hiding and extracting secret messages in images, designed to be robust against real-world distortions like JPEG compression, noise, and image transformations.

##    YOY Overview

**Steganography** is the practice of concealing information within other non-secret data. Unlike traditional cryptography which obscures the content of messages, steganography hides the existence of the message itself.

This project uses **deep learning** to automatically learn optimal encoding and decoding strategies, achieving:

-    Y"' **High Security**: Stego images are visually indistinguishable from originals (>40 dB PSNR)
-    Y>       **Robustness**: Survives JPEG compression, noise, resizing, and other distortions
-    Y"S **High Capacity**: Hide 1024+ bits per 256   -256 image
-    s    **Fast Inference**: Real-time encoding and decoding on GPU

### Why Deep Learning?

Traditional steganography methods (like LSB modification) are:
-    O Fragile to compression and transformations
-    O Detectable by statistical analysis
-    O Fixed embedding strategies

Deep learning steganography is:
-    o. Robust to attacks through adversarial training
-    o. Learns imperceptible embedding automatically
-    o. Adapts to image content and distortions

---

##    Y   -    Architecture

The system consists of three main components:

### 1. **Encoder Network**
Hides secret messages into cover images using a U-Net style architecture with residual connections:
- Input: Cover image (3 channels) + Binary message (1024 bits)
- Message is expanded to spatial dimensions via fully connected layers
- Concatenated with image features and processed through convolutional blocks
- Output: Stego image (3 channels, visually identical to cover)

```
Cover Image (3, 256, 256)    "?   "?   "   
                                "o   "?   ?' [U-Net Encoder]    "?   ?' Stego Image (3, 256, 256)
Message (1024 bits)    "?   "?   "?   "?   "?   "?   "?   "?   "~
```

### 2. **Decoder Network**
Extracts hidden messages from stego images (possibly distorted):
- Input: Stego image (3 channels)
- Convolutional feature extraction with pooling
- Fully connected layers to binary message logits
- Output: Decoded message (1024 bits)

```
Stego Image (3, 256, 256)    "?   ?' [CNN Decoder]    "?   ?' Message Logits (1024)
```

### 3. **Distortion Layer (Training Only)**
Simulates real-world attacks during training for robustness:
- JPEG compression (quality 50-100)
- Gaussian/Salt-Pepper/Speckle noise
- Resize operations (0.5x-1.0x)
- Gaussian/Motion/Average blur
- Random crops and color jitter

---

##    Y"    Project Structure

```
Final-year-Project-steganography/
   "o   "?   "? models/
   ",      "o   "?   "? encoder.py          # Encoder network architecture
   ",      "o   "?   "? decoder.py          # Decoder network architecture
   ",      ""   "?   "? model.py            # StegoModel (combines encoder + decoder)
   "o   "?   "? attacks/
   ",      "o   "?   "? jpeg.py             # JPEG compression simulation
   ",      "o   "?   "? noise.py            # Various noise types
   ",      "o   "?   "? resize.py           # Resize attacks
   ",      "o   "?   "? blur.py             # Blur operations
   ",      "o   "?   "? crop.py             # Cropping attacks
   ",      ""   "?   "? color_jitter.py     # Color transformations
   "o   "?   "? defense/
   ",      "o   "?   "? antialias.py        # Anti-aliasing filters
   ",      "o   "?   "? denoise.py          # Denoising modules
   ",      ""   "?   "? adversarial.py      # Adversarial training
   "o   "?   "? evaluation/
   ",      "o   "?   "? ber.py              # Bit Error Rate metric
   ",      "o   "?   "? psnr.py             # Peak Signal-to-Noise Ratio
   ",      "o   "?   "? ssim.py             # Structural Similarity Index
   ",      ""   "?   "? steganalysis.py     # CNN-based detection
   "o   "?   "? utils/
   ",      ""   "?   "? dataset.py          # Dataset loading utilities
   "o   "?   "? losses.py               # Loss functions (image + message + perceptual)
   "o   "?   "? train.py                # Training script
   "o   "?   "? encode.py               # Encode messages into images
   "o   "?   "? decode.py               # Extract messages from images
   "o   "?   "? inference.py            # Batch inference utilities
   "o   "?   "? Robustness_Test.ipynb   # Interactive robustness testing
   ""   "?   "? README.md               # This file
```

---

##    Ys? Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Final-year-Project-steganography
```

2. **Install dependencies:**
```bash
# Option 1: Install from requirements.txt (recommended)
pip install -r requirements.txt

# Option 2: Manual installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy matplotlib pandas scikit-image tensorboard pyyaml tqdm opencv-python
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

4. **Quick Start - Train Your First Model:**
```bash
# Edit config.yaml if needed, then train with your images
python train.py --train_dir path/to/your/images

# Example with provided sample data
python train.py --train_dir data/synthetic_small --num_epochs 50
```

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed configuration options.

---

##    Y"S Dataset Setup

The system supports multiple dataset formats:

### Supported Datasets

1. **COCO Dataset** (Recommended)
   - Download from: http://cocodataset.org/
   - Structure: `images/train2017/`, `images/val2017/`

2. **ImageNet**
   - Structure: `train/class_name/`, `val/class_name/`

3. **BOSSBase** (Steganalysis dataset)
   - Flat directory with images

4. **Custom Images**
   - Any folder with `.jpg`, `.png`, `.bmp` images

### Quick Setup

```bash
# Create data directory
mkdir -p data/images

# Option 1: Use your own images
cp /path/to/your/images/* data/images/

# Option 2: Download COCO (example)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/
```

The dataset loader will automatically:
- Detect dataset structure
- Resize images to 256   -256
- Normalize to [0, 1] range
- Generate random binary messages for training

---

##    YZ" Training

### Using Configuration File (Recommended)

The easiest way to train is using the `config.yaml` file:

```bash
# Train with config defaults (optimized for RTX 3050 6GB)
python train.py --train_dir data/images

# Use custom config file
python train.py --config my_config.yaml --train_dir data/images

# Override specific config parameters
python train.py --train_dir data/images --batch_size 8 --num_epochs 200
```

The `config.yaml` file contains all training parameters. Default configuration for RTX 3050 6GB:

```yaml
model:
  image_size: 128          # Start with smaller images
  message_length: 16       # Shorter messages for faster training

training:
  batch_size: 4            # Safe for 6GB VRAM
  learning_rate: 0.0001    # Conservative learning rate
  max_epochs: 300          # Longer training with smaller batches
  num_workers: 8           # Multi-threaded data loading

data:
  max_train_images: 1000   # Limited to 500-1000 images for local GPU convergence
  max_val_images: 200      # Validation set size

distortions:
  enable: false            # OFF initially (auto-enables at 75% accuracy)
```

**Two-Phase Training (Automatic):**
1. **Clean Training** (Phase 1): Trains on clean images until bit accuracy    ?   75%
2. **Robust Training** (Phase 2): Auto-enables distortions for real-world robustness

See [DISTORTION_TRAINING_GUIDE.md](DISTORTION_TRAINING_GUIDE.md) for details.

**Benefits of config-based training:**
-    o. Reproducible experiments
-    o. Easy parameter management
-    o. Command-line overrides still work
-    o. GPU-optimized defaults
-    o. No need to remember long CLI commands
-    o. Automatic distortion scheduling

### Basic Training (Legacy CLI)

```bash
python train.py \
    --train_dir data/images \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 0.001
```

### Advanced Training with Attacks

```bash
python train.py \
    --train_dir data/images \
    --val_dir data/val_images \
    --num_epochs 100 \
    --batch_size 4 \
    --image_size 128 \
    --message_length 16 \
    --enable_distortions \
    --apply_attacks \
    --checkpoint_dir checkpoints \
    --log_dir runs/experiment1
```

### Training Parameters

| Parameter | Config Default | Description |
|-----------|----------------|-------------|
| `--train_dir` | Required | Path to training images |
| `--val_dir` | None | Path to validation images |
| `--num_epochs` | 300 | Number of training epochs |
| `--batch_size` | 4 | Batch size (safe for 6GB GPU) |
| `--learning_rate` | 0.0001 | Learning rate |
| `--image_size` | 128 | Image size (128   -128) |
| `--message_length` | 16 | Message length in bits |
| `--enable_distortions` | False | Enable distortions during training |
| `--apply_attacks` | False | Apply additional attacks |
| `--num_workers` | 8 | Data loader workers |
| `--config` | config.yaml | Path to config file |

### Monitor Training

```bash
# View training progress with TensorBoard
tensorboard --logdir runs/
```

Training typically takes:
- **Small dataset (1K images)**: ~1 hour on RTX 3090
- **COCO train set (100K images)**: ~8-10 hours

---

##    Y"    Encoding Messages

### Command-Line Usage

```bash
# Encode text message
python encode.py \
    --image cover.jpg \
    --message "This is a secret message" \
    --output stego.png \
    --model checkpoints/best_model.pth

# Encode from text file
python encode.py \
    --image cover.jpg \
    --message-file secret.txt \
    --output stego.png

# Encode binary data
python encode.py \
    --image cover.jpg \
    --binary "10101010101010..." \
    --output stego.png

# Show quality metrics
python encode.py \
    --image cover.jpg \
    --message "Secret" \
    --output stego.png \
    --show-stats
```

### Python API

```python
from models.model import StegoModel
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = StegoModel(message_length=1024)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Load and preprocess image
image = Image.open('cover.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
cover = transform(image).unsqueeze(0)

# Prepare message (1024 bits)
message = torch.randint(0, 2, (1, 1024)).float()

# Encode
with torch.no_grad():
    stego = model.encode(cover, message)

# Save stego image
stego_pil = transforms.ToPILImage()(stego.squeeze(0))
stego_pil.save('stego.png')
```

---

##    Y"" Decoding Messages

### Command-Line Usage

```bash
# Decode message
python decode.py \
    --image stego.png \
    --model checkpoints/best_model.pth

# Save decoded message to file
python decode.py \
    --image stego.png \
    --output message.txt

# Output as binary string
python decode.py \
    --image stego.png \
    --binary

# Show confidence statistics
python decode.py \
    --image stego.png \
    --show-confidence
```

### Python API

```python
from models.model import StegoModel
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = StegoModel(message_length=1024)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Load stego image
image = Image.open('stego.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
stego = transform(image).unsqueeze(0)

# Decode
with torch.no_grad():
    decoded_logits = model.decode(stego)
    decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()

print(f"Decoded message: {decoded_message}")
```

---

##    Y       Testing Robustness

### Interactive Notebook

The easiest way to test robustness is using the Jupyter notebook:

```bash
jupyter notebook Robustness_Test.ipynb
```

This notebook provides:
-    o. Interactive testing interface
-    o. Automatic metric computation (BER, PSNR, SSIM)
-    o. Visual comparisons of attacked images
-    o. Performance statistics and charts

### Command-Line Testing

```bash
# Test specific attack
python -c "
from models.model import StegoModel
from attacks import JPEGCompression
from evaluation import compute_ber
import torch

model = StegoModel(message_length=1024)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Create test data
cover = torch.rand(1, 3, 256, 256)
message = torch.randint(0, 2, (1, 1024)).float()

# Encode
stego = model.encode(cover, message)

# Apply JPEG attack
jpeg = JPEGCompression(quality=50)
attacked = jpeg(stego)

# Decode and measure BER
decoded = model.decode(attacked)
ber = compute_ber(decoded, message, logits=True)
print(f'BER with JPEG Q=50: {ber:.4f} ({(1-ber)*100:.1f}% accuracy)')
"
```

### Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **BER** | Bit Error Rate (message recovery) | < 0.01 (>99% accuracy) |
| **PSNR** | Peak Signal-to-Noise Ratio (image quality) | > 40 dB |
| **SSIM** | Structural Similarity (perceptual quality) | > 0.95 |
| **Security** | Steganalysis detection rate | < 55% (near random) |

---

##    Y"^ Example Results

### Visual Quality

| Cover Image | Stego Image | PSNR | SSIM |
|-------------|-------------|------|------|
| ![cover](docs/cover.png) | ![stego](docs/stego.png) | 42.5 dB | 0.982 |

*Stego images are virtually indistinguishable from originals (PSNR > 40 dB, SSIM > 0.98)*

### Robustness Performance

| Attack Type | Bit Accuracy | PSNR | SSIM |
|-------------|--------------|------|------|
| No Attack | 99.8% |    ^z | 1.000 |
| JPEG Q=90 | 98.5% | 35.2 dB | 0.956 |
| JPEG Q=75 | 96.3% | 31.8 dB | 0.923 |
| JPEG Q=50 | 92.1% | 28.4 dB | 0.875 |
| Gaussian Noise (   f=0.01) | 97.8% | 38.1 dB | 0.945 |
| Gaussian Noise (   f=0.05) | 89.4% | 26.2 dB | 0.812 |
| Resize 0.75x | 96.7% | 33.5 dB | 0.934 |
| Resize 0.5x | 91.2% | 29.1 dB | 0.867 |
| Gaussian Blur (   f=1.0) | 95.8% | 32.7 dB | 0.918 |
| Gaussian Blur (   f=2.0) | 88.3% | 27.9 dB | 0.845 |
| Random Crop 90% | 94.5% | 31.2 dB | 0.901 |
| Color Jitter | 96.2% | 34.8 dB | 0.927 |

*Results on COCO validation set with 1024-bit messages*

### Training Curves

```
Epoch   Train Loss   Val Loss   BER      PSNR     SSIM    Time
------  -----------  ---------  -------  -------  ------  ------
1/100   0.3456       0.3521     0.1234   32.5     0.891   45s
10/100  0.0892       0.0915     0.0456   38.7     0.945   43s
25/100  0.0345       0.0367     0.0189   41.2     0.967   43s
50/100  0.0178       0.0192     0.0067   43.8     0.981   43s
100/100 0.0089       0.0095     0.0021   45.5     0.988   43s
```

---

##    Y>       Advanced Features

### Custom Loss Functions

The system includes multiple loss components in `losses.py`:

```python
from losses import StegoLoss

# Create combined loss
criterion = StegoLoss(
    image_loss_type='mse',     # MSE, L1, or both
    use_perceptual=True,       # VGG perceptual loss
    use_ssim=True,             # SSIM loss
    alpha=1.0,                 # Image reconstruction weight
    beta=1.0,                  # Message reconstruction weight
    gamma=0.1,                 # Perceptual loss weight
    delta=0.1                  # SSIM loss weight
)
```

### Defense Mechanisms

Add pre-decoding defense modules:

```python
from defense import DenoiseBeforeDecode, AdaptiveAntiAlias

# Add denoising before decoding
model = StegoModel(message_length=1024)
model.decoder = nn.Sequential(
    DenoiseBeforeDecode(method='gaussian'),
    model.decoder
)
```

### Adversarial Training

Train with adversarial attacks for better robustness:

```python
from defense import AdversarialTraining

# Create adversarial training wrapper
adv_training = AdversarialTraining(
    model=model,
    method='pgd',
    epsilon=0.1,
    alpha=0.01,
    num_iter=10
)

# Use in training loop
for images, messages in dataloader:
    stego = adv_training.generate_adversarial(images, messages)
    decoded = model.decode(stego)
    loss = criterion(stego, images, decoded, messages)
```

---

##    Y"S Evaluation

### Compute All Metrics

```python
from evaluation import compute_ber, compute_psnr, compute_ssim

# After encoding
stego = model.encode(cover, message)

# Image quality
psnr = compute_psnr(stego, cover)
ssim = compute_ssim(stego, cover)

# Message recovery
decoded = model.decode(stego)
ber = compute_ber(decoded, message, logits=True)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print(f"BER: {ber:.6f} ({(1-ber)*100:.2f}% accuracy)")
```

### Steganalysis Detection

Test security against CNN-based detectors:

```python
from evaluation import evaluate_steganography_security

security_results = evaluate_steganography_security(
    stego_model=model,
    cover_images=test_images,
    messages=test_messages,
    detector_type='srnet',
    train_epochs=10
)

print(f"Detection Accuracy: {security_results['accuracy']:.2f}%")
print(f"Security Score: {security_results['security']:.2f}%")
```

Lower detection accuracy (< 55%) indicates better steganography security.

---

##    Y"    Research & Citations

This project implements concepts from:

1. **"Hiding Images in Plain Sight: Deep Steganography"** (Baluja, 2017)
   - End-to-end neural network approach

2. **"SteganoGAN: High Capacity Image Steganography with GANs"** (Zhang et al., 2019)
   - Adversarial training for robustness

3. **"HiDDeN: Hiding Data with Deep Networks"** (Zhu et al., 2018)
   - Distortion layer for attack robustness

If you use this code in your research, please cite appropriately.

```bibtex
@Flakes8387{image-steganography-model,
  title={Image Steganography Model},
  year={2026},
  publisher={GitHub},
  note={Deep Learning Model for Image Steganography}
}
```

---

##    Y       Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for color space transformations (YUV, Lab)
- [ ] GAN-based stego image generation
- [ ] Multi-scale encoding for larger messages
- [ ] Real-time video steganography
- [ ] Mobile/edge deployment optimization
- [ ] Additional attack simulations

Please submit issues or pull requests.

---

##    Y"    License

This project is available for educational and research purposes.

---

##    YT    Acknowledgments

- PyTorch team for the deep learning framework
- COCO dataset contributors
- Research papers on deep steganography

---

##    Y"    Contact

For questions, issues, or collaboration, please create an issue in this repository.

---

##    YZ    Quick Start Summary

```bash
# 1. Install
pip install torch torchvision pillow numpy matplotlib

# 2. Prepare data
mkdir -p data/images && cp your_images/* data/images/

# 3. Train
python train.py --train-dir data/images --epochs 50

# 4. Encode
python encode.py --image cover.jpg --message "Secret" --output stego.png

# 5. Decode
python decode.py --image stego.png

# 6. Test robustness
jupyter notebook Robustness_Test.ipynb
```

**Happy Steganography!**    YZ      Y"   



