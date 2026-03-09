"""
Quick Evaluation Script for Progressive Training Models

Tests all 6 phase checkpoints and compares performance.
"""

import torch
import argparse
from pathlib import Path
from models.model import StegoModel
from attacks import JPEGCompression, ResizeAttack, ColorJitter
from attacks.blur import GaussianBlur
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os


class SimpleDataset:
    """Simple dataset for evaluation"""
    def __init__(self, image_dir, message_length=16, image_size=128, max_images=50):
        self.message_length = message_length
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, '**', f'*{ext}'), recursive=True))
        
        if max_images and max_images < len(self.image_paths):
            self.image_paths = self.image_paths[:max_images]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        message = torch.randint(0, 2, (self.message_length,)).float()
        return image, message


def evaluate_model(model, dataloader, device, attack_config):
    """Evaluate model with specific attack configuration"""
    model.eval()
    
    # Initialize attacks
    jpeg = JPEGCompression(quality_range=(70, 95)).to(device) if attack_config.get('jpeg') else None
    blur = GaussianBlur(kernel_size_range=(3, 5), sigma_range=(0.3, 1.2)).to(device) if attack_config.get('blur') else None
    resize = ResizeAttack(scale_range=(0.6, 0.9)).to(device) if attack_config.get('resize') else None
    color_jitter = ColorJitter().to(device) if attack_config.get('color_jitter') else None
    
    total_accuracy = 0.0
    total_pixel_delta = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for cover_images, binary_messages in dataloader:
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)
            
            # Encode
            stego_images = model.encode(cover_images, binary_messages)
            distorted_stego = stego_images.clone()
            
            # Apply attacks
            if jpeg and attack_config.get('combined'):
                distorted_stego = jpeg(distorted_stego)
            if blur and attack_config.get('combined'):
                distorted_stego = blur(distorted_stego)
            if resize and attack_config.get('combined'):
                distorted_stego = resize(distorted_stego)
            if color_jitter and attack_config.get('combined'):
                distorted_stego = color_jitter(distorted_stego)
            
            # Single attack mode
            if not attack_config.get('combined'):
                if jpeg:
                    distorted_stego = jpeg(distorted_stego)
                elif blur:
                    distorted_stego = blur(distorted_stego)
                elif resize:
                    distorted_stego = resize(distorted_stego)
                elif color_jitter:
                    distorted_stego = color_jitter(distorted_stego)
            
            # Decode
            decoded_logits = model.decode(distorted_stego, return_logits=True)
            decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()
            
            # Metrics
            accuracy = (decoded_message == binary_messages).float().mean()
            pixel_delta = (stego_images - cover_images).abs().mean()
            
            total_accuracy += accuracy.item()
            total_pixel_delta += pixel_delta.item()
            num_batches += 1
    
    return {
        'accuracy': total_accuracy / num_batches * 100,
        'pixel_delta': total_pixel_delta / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Progressive Training Models')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing phase checkpoints')
    parser.add_argument('--test_dir', type=str, default='data/DIV2K/val',
                        help='Test images directory')
    parser.add_argument('--message_length', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load test dataset
    dataset = SimpleDataset(args.test_dir, args.message_length, args.image_size, max_images=50)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} test images\n")
    
    # Define phase checkpoints
    phase_checkpoints = [
        'model_phase1_clean_94.pth',
        'model_phase2_jpeg_90.pth',
        'model_phase3_blur_90.pth',
        'model_phase4_resize_90.pth',
        'model_phase5_all_90.pth',
        'model_phase6_combined_88.pth',
    ]
    
    # Attack configurations for evaluation
    attack_configs = [
        {'name': 'Clean (No Attack)', 'config': {}},
        {'name': 'JPEG Compression', 'config': {'jpeg': True}},
        {'name': 'Gaussian Blur', 'config': {'blur': True}},
        {'name': 'Resize Attack', 'config': {'resize': True}},
        {'name': 'Color Jitter', 'config': {'color_jitter': True}},
        {'name': 'ALL COMBINED', 'config': {'jpeg': True, 'blur': True, 'resize': True, 'color_jitter': True, 'combined': True}},
    ]
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    print("="*80)
    print("PROGRESSIVE TRAINING MODEL EVALUATION")
    print("="*80)
    print()
    
    # Evaluate each phase checkpoint
    for i, checkpoint_name in enumerate(phase_checkpoints, 1):
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"⚠ Phase {i} checkpoint not found: {checkpoint_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"PHASE {i}: {checkpoint_name}")
        print(f"{'='*80}")
        
        # Load model
        model = StegoModel(message_length=args.message_length).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded checkpoint (Epoch {checkpoint.get('epoch', 'N/A')})")
        print(f"  Training Accuracy: {checkpoint.get('accuracy', 0)*100:.2f}%")
        print(f"  Training Pixel Δ: {checkpoint.get('pixel_delta', 0):.4f}")
        print()
        
        # Evaluate on all attack types
        print("Evaluation Results:")
        print("-" * 60)
        
        for attack_config in attack_configs:
            results = evaluate_model(model, dataloader, device, attack_config['config'])
            status = "✓" if results['accuracy'] >= 90 else ("⚠" if results['accuracy'] >= 85 else "✗")
            print(f"{status} {attack_config['name']:20} → Acc: {results['accuracy']:5.2f}%, "
                  f"Pixel Δ: {results['pixel_delta']:.4f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nLegend:")
    print("  ✓ = Accuracy >= 90%")
    print("  ⚠ = Accuracy 85-90%")
    print("  ✗ = Accuracy < 85%")
    print()


if __name__ == '__main__':
    main()
