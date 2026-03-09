"""
Comprehensive Evaluation with Multiple Test Samples
Tests each transformation on 50+ images to get real accuracy statistics
"""

import torch
from models.model import StegoModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import numpy as np


class DIV2KTestDataset(Dataset):
    """DIV2K test dataset."""

    def __init__(self, image_dir, message_length, image_size=128, max_images=50):
        self.message_length = message_length
        self.image_size = image_size

        # Find all images
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.image_paths.extend(
                glob.glob(image_dir + f'/**/*{ext}', recursive=True))

        # Limit number
        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Random message
        message = torch.randint(0, 2, (self.message_length,)).float()

        return image, message


def evaluate_attack(model, dataloader, attack_name, attack_fn, device):
    """Evaluate model on specific attack across all test samples."""

    model.encoder.eval()
    model.decoder.eval()
    model.distortions.train()  # Keep distortions active

    total_accuracy = 0
    total_bit_errors = 0
    total_bits = 0
    num_samples = 0

    # Override forward function
    original_forward = model.distortions.forward
    model.distortions.forward = attack_fn

    with torch.no_grad():
        for images, messages in dataloader:
            images = images.to(device)
            messages = messages.to(device)

            # Forward pass
            outputs = model(images, messages, apply_distortions=True)
            decoded = outputs['decoded_message']

            # Calculate metrics
            decoded_bits = (decoded > 0.5).float()
            correct = (decoded_bits == messages).float()
            accuracy = correct.mean().item()
            bit_errors = (decoded_bits != messages).sum().item()

            total_accuracy += accuracy * len(images)
            total_bit_errors += bit_errors
            total_bits += messages.numel()
            num_samples += len(images)

    # Restore forward
    model.distortions.forward = original_forward

    avg_accuracy = total_accuracy / num_samples
    ber = total_bit_errors / total_bits

    return avg_accuracy, ber, total_bit_errors, total_bits


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE TRANSFORMATION EVALUATION")
    print("Testing on 50 DIV2K images with random messages")
    print("="*80 + "\n")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    message_length = 16
    image_size = 128
    batch_size = 8
    num_test_images = 50

    # Load dataset
    print(f"Loading {num_test_images} test images from DIV2K...")
    dataset = DIV2KTestDataset(
        'data/DIV2K/train', message_length, image_size, num_test_images)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    print(f"✓ Loaded {len(dataset)} images\n")

    # Load BEST COMBINED model
    print("Loading model: model_BEST_COMBINED.pth")
    model = StegoModel(message_length, image_size, enable_distortions=True)
    checkpoint = torch.load(
        'checkpoints/model_BEST_COMBINED.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(
        f"✓ Model loaded (Combined accuracy: {checkpoint.get('combined_accuracy', 0)*100:.2f}%)\n")

    print("="*80)
    print("RUNNING EVALUATIONS")
    print("="*80 + "\n")

    results = {}

    # Test 1: Clean (no distortions)
    print("1. Clean Encoding/Decoding")
    print("-" * 40)

    def forward_clean(images, apply_all=False, jpeg_only=False):
        return images

    acc, ber, errors, total = evaluate_attack(
        model, dataloader, "Clean", forward_clean, device)
    results['Clean'] = (acc, ber, errors, total)
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  BER: {ber*100:.2f}%")
    print(f"  Bit Errors: {errors}/{total}\n")

    # Test 2: JPEG Compression
    print("2. JPEG Compression Attack")
    print("-" * 40)

    def forward_jpeg(images, apply_all=False, jpeg_only=False):
        return model.distortions.apply_jpeg_compression(images, probability=1.0)

    acc, ber, errors, total = evaluate_attack(
        model, dataloader, "JPEG", forward_jpeg, device)
    results['JPEG'] = (acc, ber, errors, total)
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  BER: {ber*100:.2f}%")
    print(f"  Bit Errors: {errors}/{total}\n")

    # Test 3: Gaussian Blur
    print("3. Gaussian Blur Attack")
    print("-" * 40)

    def forward_blur(images, apply_all=False, jpeg_only=False):
        return model.distortions.apply_gaussian_blur_attack(images, probability=1.0)

    acc, ber, errors, total = evaluate_attack(
        model, dataloader, "Blur", forward_blur, device)
    results['Gaussian Blur'] = (acc, ber, errors, total)
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  BER: {ber*100:.2f}%")
    print(f"  Bit Errors: {errors}/{total}\n")

    # Test 4: Resize Attack
    print("4. Resize Attack")
    print("-" * 40)

    def forward_resize(images, apply_all=False, jpeg_only=False):
        return model.distortions.apply_resize_attack(images, probability=1.0)

    acc, ber, errors, total = evaluate_attack(
        model, dataloader, "Resize", forward_resize, device)
    results['Resize'] = (acc, ber, errors, total)
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  BER: {ber*100:.2f}%")
    print(f"  Bit Errors: {errors}/{total}\n")

    # Test 5: Color Jitter
    print("5. Color Jitter Attack")
    print("-" * 40)

    def forward_jitter(images, apply_all=False, jpeg_only=False):
        return model.distortions.apply_color_jitter_attack(images, probability=1.0)

    acc, ber, errors, total = evaluate_attack(
        model, dataloader, "Jitter", forward_jitter, device)
    results['Color Jitter'] = (acc, ber, errors, total)
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  BER: {ber*100:.2f}%")
    print(f"  Bit Errors: {errors}/{total}\n")

    # Test 6: ALL Combined (DISABLED FOR NOW)
    # print("6. ALL TRANSFORMATIONS (JPEG + Blur + Resize + Jitter)")
    # print("-" * 40)

    # def forward_all(images, apply_all=False, jpeg_only=False):
    #     images = model.distortions.apply_jpeg_compression(
    #         images, probability=1.0)
    #     images = model.distortions.apply_gaussian_blur_attack(
    #         images, probability=1.0)
    #     images = model.distortions.apply_resize_attack(images, probability=1.0)
    #     images = model.distortions.apply_color_jitter_attack(
    #         images, probability=1.0)
    #     return images

    # acc, ber, errors, total = evaluate_attack(
    #     model, dataloader, "All", forward_all, device)
    # results['ALL Combined'] = (acc, ber, errors, total)
    # print(f"  Accuracy: {acc*100:.2f}%")
    # print(f"  BER: {ber*100:.2f}%")
    # print(f"  Bit Errors: {errors}/{total}\n")

    # Summary Table
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print(f"{'Transformation':<25} {'Accuracy':<12} {'BER':<12} {'Bit Errors'}")
    print("-" * 80)

    for name, (acc, ber, errors, total) in results.items():
        print(
            f"{name:<25} {acc*100:>6.2f}%      {ber*100:>6.2f}%      {errors}/{total}")

    print("\n" + "="*80)
    print(f"Evaluated on {len(dataset)} test images")
    print(f"Total bits tested per attack: {total}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
