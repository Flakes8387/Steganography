"""
Test the Final Model with All Transformations
Loads: model_ALL_TRANSFORMATIONS_95_DIV2K.pth
Tests: JPEG, Gaussian Blur, Resize Attack, Color Jitter
"""

import torch
from models.model import StegoModel
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def test_model_with_transformations():
    """Test the final trained model with all transformations."""

    print("\n" + "="*80)
    print("TESTING FINAL MODEL WITH ALL TRANSFORMATIONS")
    print("="*80 + "\n")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Configuration
    message_length = 16
    image_size = 128
    checkpoint_path = 'checkpoints/model_ALL_TRANSFORMATIONS_95_DIV2K.pth'

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = StegoModel(message_length, image_size, enable_distortions=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    # Set encoder/decoder to eval, but keep distortions in train mode to apply attacks
    model.encoder.eval()
    model.decoder.eval()
    model.distortions.train()  # Keep distortions active for testing attacks

    print(f"✓ Model loaded")
    print(f"  Training Accuracy: {checkpoint['accuracy']*100:.2f}%")
    print(f"  Training Pixel Delta: {checkpoint['pixel_delta']:.6f}\n")

    # Load a test image from DIV2K
    import glob
    div2k_images = glob.glob('data/DIV2K/train/**/*.png', recursive=True)
    if not div2k_images:
        print("No DIV2K images found, using random image...")
        cover_image = torch.rand(1, 3, image_size, image_size).to(device)
    else:
        # Load real image
        test_img_path = div2k_images[0]
        print(f"Test image: {test_img_path}")
        img = Image.open(test_img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])
        cover_image = transform(img).unsqueeze(0).to(device)
        print(f"✓ Loaded cover image\n")

    # Generate random binary message
    binary_message = torch.randint(
        0, 2, (1, message_length)).float().to(device)
    print(
        f"Secret message (16 bits): {binary_message[0].cpu().numpy().astype(int)}\n")

    print("="*80)
    print("RUNNING TESTS")
    print("="*80 + "\n")

    # Test 1: Clean encoding (no distortions)
    print("Test 1: Clean Encoding/Decoding")
    print("-" * 40)
    with torch.no_grad():
        outputs = model(cover_image, binary_message, apply_distortions=False)
        stego = outputs['stego_image']
        decoded = outputs['decoded_message']

        pixel_delta = torch.mean(torch.abs(stego - cover_image)).item()
        decoded_bits = (decoded > 0.5).float()
        accuracy = (decoded_bits == binary_message).float().mean().item()
        psnr = 20 * \
            torch.log10(torch.tensor(
                1.0 / (torch.mean((stego - cover_image) ** 2) ** 0.5))).item()

        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Pixel Delta: {pixel_delta:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Decoded: {decoded_bits[0].cpu().numpy().astype(int)}\n")

    # Test 2: With JPEG compression
    print("Test 2: JPEG Compression Attack")
    print("-" * 40)
    with torch.no_grad():
        # Enable only JPEG - force application
        def forward_jpeg(images, apply_all=False, jpeg_only=False):
            # Always apply regardless of training mode
            return model.distortions.apply_jpeg_compression(images, probability=1.0)

        original_forward = model.distortions.forward
        model.distortions.forward = forward_jpeg

        outputs = model(cover_image, binary_message, apply_distortions=True)
        stego = outputs['stego_image']
        decoded = outputs['decoded_message']

        decoded_bits = (decoded > 0.5).float()
        accuracy = (decoded_bits == binary_message).float().mean().item()
        bit_errors = (decoded_bits != binary_message).sum().item()

        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Bit Errors: {bit_errors}/{message_length}")
        print(f"  Decoded: {decoded_bits[0].cpu().numpy().astype(int)}\n")

        model.distortions.forward = original_forward

    # Test 3: With Gaussian Blur
    print("Test 3: Gaussian Blur Attack")
    print("-" * 40)
    with torch.no_grad():
        def forward_blur(images, apply_all=False, jpeg_only=False):
            return model.distortions.apply_gaussian_blur_attack(images, probability=1.0)

        model.distortions.forward = forward_blur

        outputs = model(cover_image, binary_message, apply_distortions=True)
        decoded = outputs['decoded_message']

        decoded_bits = (decoded > 0.5).float()
        accuracy = (decoded_bits == binary_message).float().mean().item()
        bit_errors = (decoded_bits != binary_message).sum().item()

        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Bit Errors: {bit_errors}/{message_length}")
        print(f"  Decoded: {decoded_bits[0].cpu().numpy().astype(int)}\n")

        model.distortions.forward = original_forward

    # Test 4: With Resize Attack
    print("Test 4: Resize Attack")
    print("-" * 40)
    with torch.no_grad():
        def forward_resize(images, apply_all=False, jpeg_only=False):
            return model.distortions.apply_resize_attack(images, probability=1.0)

        model.distortions.forward = forward_resize

        outputs = model(cover_image, binary_message, apply_distortions=True)
        decoded = outputs['decoded_message']

        decoded_bits = (decoded > 0.5).float()
        accuracy = (decoded_bits == binary_message).float().mean().item()
        bit_errors = (decoded_bits != binary_message).sum().item()

        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Bit Errors: {bit_errors}/{message_length}")
        print(f"  Decoded: {decoded_bits[0].cpu().numpy().astype(int)}\n")

        model.distortions.forward = original_forward

    # Test 5: With Color Jitter
    print("Test 5: Color Jitter Attack")
    print("-" * 40)
    with torch.no_grad():
        def forward_jitter(images, apply_all=False, jpeg_only=False):
            return model.distortions.apply_color_jitter_attack(images, probability=1.0)

        model.distortions.forward = forward_jitter

        outputs = model(cover_image, binary_message, apply_distortions=True)
        decoded = outputs['decoded_message']

        decoded_bits = (decoded > 0.5).float()
        accuracy = (decoded_bits == binary_message).float().mean().item()
        bit_errors = (decoded_bits != binary_message).sum().item()

        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Bit Errors: {bit_errors}/{message_length}")
        print(f"  Decoded: {decoded_bits[0].cpu().numpy().astype(int)}\n")

        model.distortions.forward = original_forward

    # Test 6: With ALL transformations combined
    print("Test 6: ALL TRANSFORMATIONS (JPEG + Blur + Resize + Color Jitter)")
    print("-" * 40)
    with torch.no_grad():
        def forward_all(images, apply_all=False, jpeg_only=False):
            images = model.distortions.apply_jpeg_compression(
                images, probability=1.0)
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=1.0)
            images = model.distortions.apply_resize_attack(
                images, probability=1.0)
            images = model.distortions.apply_color_jitter_attack(
                images, probability=1.0)
            return images

        model.distortions.forward = forward_all

        outputs = model(cover_image, binary_message, apply_distortions=True)
        decoded = outputs['decoded_message']

        decoded_bits = (decoded > 0.5).float()
        accuracy = (decoded_bits == binary_message).float().mean().item()
        bit_errors = (decoded_bits != binary_message).sum().item()

        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Bit Errors: {bit_errors}/{message_length}")
        print(f"  Original: {binary_message[0].cpu().numpy().astype(int)}")
        print(f"  Decoded:  {decoded_bits[0].cpu().numpy().astype(int)}\n")

        model.distortions.forward = original_forward

    print("="*80)
    print("TESTING COMPLETE")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  Model: {checkpoint_path}")
    print(f"  Message Length: {message_length} bits")
    print(f"  Image Size: {image_size}x{image_size}")
    print(f"  All transformation attacks tested successfully!\n")


if __name__ == "__main__":
    test_model_with_transformations()
