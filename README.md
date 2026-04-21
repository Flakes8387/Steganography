# Image Steganography Model

![steganography](https://user-images.githubusercontent.com/28294942/184524203-850eea1e-74fc-4539-a70f-8c65adcf96dc.png)

A deep learning-based image steganography project in PyTorch for hiding and extracting secret messages, designed for robustness against JPEG compression, noise, and image transformations.

## Overview

Steganography hides the existence of a message inside non-secret media. This project uses neural networks to learn encoding and decoding strategies that balance security, robustness, and message capacity.

Key goals:
- High visual quality in stego images
- Robust message recovery under common distortions
- Practical training and inference workflows

## Architecture

The system has three components:

1. Encoder network
- Inputs: cover image (3 channels) and binary message (1024 bits)
- Message is expanded to spatial dimensions
- Features are fused and transformed through convolutional blocks
- Output: stego image (3 channels)

2. Decoder network
- Input: stego image
- CNN feature extraction plus pooling
- Fully connected layers map features to message logits
- Output: recovered message bits

3. Distortion layer (training only)
- JPEG compression
- Gaussian, salt-pepper, and speckle noise
- Resize operations
- Blur, crops, and color jitter

## Project Structure

```text
Final-year-Project-steganography/
|-- models/
|   |-- encoder.py
|   |-- decoder.py
|   `-- model.py
|-- attacks/
|   |-- jpeg.py
|   |-- noise.py
|   |-- resize.py
|   |-- blur.py
|   |-- crop.py
|   `-- color_jitter.py
|-- defense/
|   |-- antialias.py
|   |-- denoise.py
|   `-- adversarial.py
|-- evaluation/
|   |-- ber.py
|   |-- psnr.py
|   |-- ssim.py
|   `-- steganalysis.py
|-- utils/
|   `-- dataset.py
|-- losses.py
|-- train.py
|-- encode.py
|-- decode.py
|-- inference.py
|-- Robustness_Test.ipynb
`-- README.md
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone <repository-url>
cd Final-year-Project-steganography
pip install -r requirements.txt
```

### Quick Start

```bash
python train.py --train_dir data/images
python encode.py --image cover.jpg --message "Secret" --output stego.png
python decode.py --image stego.png
```

## Dataset Setup

Supported datasets:
- COCO
- ImageNet
- BOSSBase
- Custom image folders (.jpg, .png, .bmp)

Example setup:

```bash
mkdir -p data/images
cp /path/to/your/images/* data/images/
```

## Training

### Config-based training (recommended)

```bash
python train.py --train_dir data/images
python train.py --config my_config.yaml --train_dir data/images
python train.py --train_dir data/images --batch_size 8 --num_epochs 200
```

Two-phase training workflow:
1. Clean training phase
2. Robust training phase with distortions enabled

### Legacy CLI examples

```bash
python train.py --train_dir data/images --num_epochs 100 --batch_size 4 --learning_rate 0.001
python train.py --train_dir data/images --val_dir data/val_images --num_epochs 100 --batch_size 4 --image_size 128 --message_length 16 --enable_distortions --apply_attacks
```

### Monitor training

```bash
tensorboard --logdir runs/
```

## Encoding Messages

```bash
python encode.py --image cover.jpg --message "This is a secret message" --output stego.png --model checkpoints/best_model.pth
python encode.py --image cover.jpg --message-file secret.txt --output stego.png
python encode.py --image cover.jpg --binary "101010..." --output stego.png
```

## Decoding Messages

```bash
python decode.py --image stego.png --model checkpoints/best_model.pth
python decode.py --image stego.png --output message.txt
python decode.py --image stego.png --binary
```

## Testing Robustness

Notebook workflow:

```bash
jupyter notebook Robustness_Test.ipynb
```

Evaluation metrics:
- BER: bit error rate for message recovery
- PSNR: image fidelity
- SSIM: structural similarity
- Security: steganalysis detection rate

## Example Results

Typical outcomes reported in this project:
- PSNR above 40 dB on clean images
- High bit accuracy on moderate JPEG and noise perturbations
- Graceful degradation under stronger attacks

## Advanced Features

- Combined image/message/perceptual losses
- Defense modules before decode
- Adversarial training integration

## Evaluation

```python
from evaluation import compute_ber, compute_psnr, compute_ssim
```

Use these metrics to track visual quality and message recovery.

## Research References

The implementation follows ideas from:
- Hiding Images in Plain Sight: Deep Steganography (Baluja, 2017)
- SteganoGAN: High Capacity Image Steganography with GANs (Zhang et al., 2019)
- HiDDeN: Hiding Data with Deep Networks (Zhu et al., 2018)

## Contributing

Contributions are welcome via issues and pull requests.

## License

Available for educational and research purposes.

## Acknowledgments

- PyTorch contributors
- COCO dataset contributors
- Deep steganography research community

## Contact

Open an issue in this repository for questions or collaboration.

## Quick Start Summary

```bash
pip install torch torchvision pillow numpy matplotlib
mkdir -p data/images
python train.py --train_dir data/images --num_epochs 50
python encode.py --image cover.jpg --message "Secret" --output stego.png
python decode.py --image stego.png
```
