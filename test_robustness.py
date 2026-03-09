"""
Comprehensive Robustness Test for Steganography Model

Tests model's robustness against various attacks:
- JPEG Compression (multiple quality levels)
- Gaussian Noise
- Gaussian Blur
- Resize attacks
- Color Jitter
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.model import StegoModel
from attacks.jpeg import JPEGCompression
from attacks.noise import GaussianNoise
from attacks.blur import GaussianBlur
from attacks.resize import ResizeAttack
from attacks.color_jitter import ColorJitter
from evaluation.ber import BERMetric
from evaluation.psnr import PSNRMetric
from evaluation.ssim import SSIMMetric


def load_model(checkpoint_path='checkpoints/best_model_local.pth', device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer model parameters from checkpoint structure
    state_dict = checkpoint['model_state_dict']
    message_length = state_dict['encoder.prep_network.fc1.weight'].shape[1]
    fc3_output = state_dict['encoder.prep_network.fc3.weight'].shape[0]
    image_size = int(np.sqrt(fc3_output))
    
    print(f"Inferred model parameters:")
    print(f"  Message Length: {message_length}")
    print(f"  Image Size: {image_size}x{image_size}")
    
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, message_length, image_size, checkpoint


def test_jpeg_compression(model, message_length, image_size, device, num_tests=10):
    """Test robustness against JPEG compression."""
    print("\n" + "="*70)
    print("TEST 1: JPEG COMPRESSION ROBUSTNESS")
    print("="*70)
    
    # Use PIL-based JPEG compression instead of the differentiable approximation
    from PIL import Image
    from io import BytesIO
    
    ber_metric = BERMetric().to(device)
    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIMMetric().to(device)
    
    quality_levels = [95, 90, 85, 80, 75, 70, 60, 50]
    results = {'quality': quality_levels, 'ber': [], 'accuracy': [], 'psnr': [], 'ssim': []}
    
    def apply_jpeg(tensor, quality):
        """Apply real JPEG compression using PIL."""
        # Convert to PIL Image
        img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
        
        # Compress with JPEG
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        compressed_tensor = transforms.ToTensor()(compressed_img).unsqueeze(0).to(device)
        return compressed_tensor
    
    for quality in quality_levels:
        bers, accs, psnrs, ssims = [], [], [], []
        
        for _ in range(num_tests):
            cover = torch.rand(1, 3, image_size, image_size).to(device)
            message = torch.randint(0, 2, (1, message_length)).float().to(device)
            
            with torch.no_grad():
                stego = model.encode(cover, message)
                compressed = apply_jpeg(stego, quality)
                decoded = model.decode(compressed)
                
                ber = ber_metric(decoded, message).item()
                psnr = psnr_metric(compressed, stego).item()
                ssim = ssim_metric(compressed, stego).item()
                
                bers.append(ber)
                accs.append(1 - ber)
                psnrs.append(psnr)
                ssims.append(ssim)
        
        avg_ber = np.mean(bers)
        avg_acc = np.mean(accs)
        results['ber'].append(avg_ber)
        results['accuracy'].append(avg_acc)
        results['psnr'].append(np.mean(psnrs))
        results['ssim'].append(np.mean(ssims))
        
        status = "✓ EXCELLENT" if avg_acc >= 0.90 else "✓ GOOD" if avg_acc >= 0.80 else "⚠ MODERATE" if avg_acc >= 0.70 else "✗ POOR"
        print(f"Quality {quality:3d}: Accuracy={avg_acc*100:5.2f}%, BER={avg_ber:.4f} {status}")
    
    return results


def test_gaussian_noise(model, message_length, image_size, device, num_tests=10):
    """Test robustness against Gaussian noise."""
    print("\n" + "="*70)
    print("TEST 2: GAUSSIAN NOISE ROBUSTNESS")
    print("="*70)
    
    noise_attack = GaussianNoise().to(device)
    ber_metric = BERMetric().to(device)
    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIMMetric().to(device)
    
    noise_levels = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    results = {'noise_std': noise_levels, 'ber': [], 'accuracy': [], 'psnr': [], 'ssim': []}
    
    for noise_std in noise_levels:
        bers, accs, psnrs, ssims = [], [], [], []
        
        for _ in range(num_tests):
            cover = torch.rand(1, 3, image_size, image_size).to(device)
            message = torch.randint(0, 2, (1, message_length)).float().to(device)
            
            with torch.no_grad():
                stego = model.encode(cover, message)
                noisy = noise_attack(stego, std=noise_std)
                decoded = model.decode(noisy)
                
                ber = ber_metric(decoded, message).item()
                psnr = psnr_metric(noisy, stego).item()
                ssim = ssim_metric(noisy, stego).item()
                
                bers.append(ber)
                accs.append(1 - ber)
                psnrs.append(psnr)
                ssims.append(ssim)
        
        avg_ber = np.mean(bers)
        avg_acc = np.mean(accs)
        results['ber'].append(avg_ber)
        results['accuracy'].append(avg_acc)
        results['psnr'].append(np.mean(psnrs))
        results['ssim'].append(np.mean(ssims))
        
        status = "✓ EXCELLENT" if avg_acc >= 0.90 else "✓ GOOD" if avg_acc >= 0.80 else "⚠ MODERATE" if avg_acc >= 0.70 else "✗ POOR"
        print(f"Noise σ={noise_std:.3f}: Accuracy={avg_acc*100:5.2f}%, BER={avg_ber:.4f} {status}")
    
    return results


def test_gaussian_blur(model, message_length, image_size, device, num_tests=10):
    """Test robustness against Gaussian blur."""
    print("\n" + "="*70)
    print("TEST 3: GAUSSIAN BLUR ROBUSTNESS")
    print("="*70)
    
    blur_attack = GaussianBlur().to(device)
    ber_metric = BERMetric().to(device)
    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIMMetric().to(device)
    
    kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
    results = {'kernel_size': kernel_sizes, 'ber': [], 'accuracy': [], 'psnr': [], 'ssim': []}
    
    for kernel_size in kernel_sizes:
        bers, accs, psnrs, ssims = [], [], [], []
        
        for _ in range(num_tests):
            cover = torch.rand(1, 3, image_size, image_size).to(device)
            message = torch.randint(0, 2, (1, message_length)).float().to(device)
            
            with torch.no_grad():
                stego = model.encode(cover, message)
                blurred = blur_attack(stego, kernel_size=kernel_size)
                decoded = model.decode(blurred)
                
                ber = ber_metric(decoded, message).item()
                psnr = psnr_metric(blurred, stego).item()
                ssim = ssim_metric(blurred, stego).item()
                
                bers.append(ber)
                accs.append(1 - ber)
                psnrs.append(psnr)
                ssims.append(ssim)
        
        avg_ber = np.mean(bers)
        avg_acc = np.mean(accs)
        results['ber'].append(avg_ber)
        results['accuracy'].append(avg_acc)
        results['psnr'].append(np.mean(psnrs))
        results['ssim'].append(np.mean(ssims))
        
        status = "✓ EXCELLENT" if avg_acc >= 0.90 else "✓ GOOD" if avg_acc >= 0.80 else "⚠ MODERATE" if avg_acc >= 0.70 else "✗ POOR"
        print(f"Kernel={kernel_size:2d}×{kernel_size:2d}: Accuracy={avg_acc*100:5.2f}%, BER={avg_ber:.4f} {status}")
    
    return results


def test_resize_attack(model, message_length, image_size, device, num_tests=10):
    """Test robustness against resize attacks."""
    print("\n" + "="*70)
    print("TEST 4: RESIZE ATTACK ROBUSTNESS")
    print("="*70)
    
    resize_attack = ResizeAttack().to(device)
    ber_metric = BERMetric().to(device)
    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIMMetric().to(device)
    
    scale_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5]
    results = {'scale': scale_factors, 'ber': [], 'accuracy': [], 'psnr': [], 'ssim': []}
    
    for scale in scale_factors:
        bers, accs, psnrs, ssims = [], [], [], []
        
        for _ in range(num_tests):
            cover = torch.rand(1, 3, image_size, image_size).to(device)
            message = torch.randint(0, 2, (1, message_length)).float().to(device)
            
            with torch.no_grad():
                stego = model.encode(cover, message)
                resized = resize_attack(stego, scale_factor=scale)
                decoded = model.decode(resized)
                
                ber = ber_metric(decoded, message).item()
                psnr = psnr_metric(resized, stego).item()
                ssim = ssim_metric(resized, stego).item()
                
                bers.append(ber)
                accs.append(1 - ber)
                psnrs.append(psnr)
                ssims.append(ssim)
        
        avg_ber = np.mean(bers)
        avg_acc = np.mean(accs)
        results['ber'].append(avg_ber)
        results['accuracy'].append(avg_acc)
        results['psnr'].append(np.mean(psnrs))
        results['ssim'].append(np.mean(ssims))
        
        status = "✓ EXCELLENT" if avg_acc >= 0.90 else "✓ GOOD" if avg_acc >= 0.80 else "⚠ MODERATE" if avg_acc >= 0.70 else "✗ POOR"
        print(f"Scale={scale:.2f}×: Accuracy={avg_acc*100:5.2f}%, BER={avg_ber:.4f} {status}")
    
    return results


def test_color_jitter(model, message_length, image_size, device, num_tests=10):
    """Test robustness against color jitter."""
    print("\n" + "="*70)
    print("TEST 5: COLOR JITTER ROBUSTNESS")
    print("="*70)
    
    color_attack = ColorJitter().to(device)
    ber_metric = BERMetric().to(device)
    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIMMetric().to(device)
    
    intensity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {'intensity': intensity_levels, 'ber': [], 'accuracy': [], 'psnr': [], 'ssim': []}
    
    for intensity in intensity_levels:
        bers, accs, psnrs, ssims = [], [], [], []
        
        for _ in range(num_tests):
            cover = torch.rand(1, 3, image_size, image_size).to(device)
            message = torch.randint(0, 2, (1, message_length)).float().to(device)
            
            with torch.no_grad():
                stego = model.encode(cover, message)
                jittered = color_attack(stego, brightness=intensity, contrast=intensity, 
                                       saturation=intensity, hue=intensity/2)
                decoded = model.decode(jittered)
                
                ber = ber_metric(decoded, message).item()
                psnr = psnr_metric(jittered, stego).item()
                ssim = ssim_metric(jittered, stego).item()
                
                bers.append(ber)
                accs.append(1 - ber)
                psnrs.append(psnr)
                ssims.append(ssim)
        
        avg_ber = np.mean(bers)
        avg_acc = np.mean(accs)
        results['ber'].append(avg_ber)
        results['accuracy'].append(avg_acc)
        results['psnr'].append(np.mean(psnrs))
        results['ssim'].append(np.mean(ssims))
        
        status = "✓ EXCELLENT" if avg_acc >= 0.90 else "✓ GOOD" if avg_acc >= 0.80 else "⚠ MODERATE" if avg_acc >= 0.70 else "✗ POOR"
        print(f"Intensity={intensity:.2f}: Accuracy={avg_acc*100:5.2f}%, BER={avg_ber:.4f} {status}")
    
    return results


def plot_robustness_results(jpeg_results, noise_results, blur_results, resize_results, color_results):
    """Create comprehensive visualization of all robustness tests."""
    fig = plt.figure(figsize=(18, 12))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: JPEG Compression
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(jpeg_results['quality'], [acc*100 for acc in jpeg_results['accuracy']], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Accuracy')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('JPEG Quality', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('JPEG Compression', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Gaussian Noise
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(noise_results['noise_std'], [acc*100 for acc in noise_results['accuracy']], 
             marker='s', linewidth=2, markersize=8, color='#A23B72', label='Accuracy')
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Gaussian Noise', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gaussian Blur
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(blur_results['kernel_size'], [acc*100 for acc in blur_results['accuracy']], 
             marker='^', linewidth=2, markersize=8, color='#F18F01', label='Accuracy')
    ax3.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Kernel Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Gaussian Blur', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Resize Attack
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(resize_results['scale'], [acc*100 for acc in resize_results['accuracy']], 
             marker='D', linewidth=2, markersize=8, color='#C73E1D', label='Accuracy')
    ax4.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Scale Factor', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Resize Attack', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Color Jitter
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(color_results['intensity'], [acc*100 for acc in color_results['accuracy']], 
             marker='*', linewidth=2, markersize=12, color='#6A994E', label='Accuracy')
    ax5.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax5.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax5.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.set_xlabel('Jitter Intensity', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Color Jitter', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary Comparison
    ax6 = fig.add_subplot(gs[2, 1])
    attacks = ['JPEG\n(Q=80)', 'Noise\n(σ=0.05)', 'Blur\n(k=7)', 'Resize\n(0.7×)', 'Color\n(0.3)']
    accuracies = [
        jpeg_results['accuracy'][jpeg_results['quality'].index(80)] * 100,
        noise_results['accuracy'][noise_results['noise_std'].index(0.05)] * 100,
        blur_results['accuracy'][blur_results['kernel_size'].index(7)] * 100,
        resize_results['accuracy'][resize_results['scale'].index(0.7)] * 100,
        color_results['accuracy'][color_results['intensity'].index(0.3)] * 100,
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax6.bar(attacks, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent')
    ax6.axhline(y=80, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Good')
    ax6.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Moderate')
    ax6.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Attack Comparison (Moderate Settings)', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 100])
    ax6.legend(loc='lower right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('Comprehensive Robustness Test Results', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('robustness_test_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive plot saved: robustness_test_results.png")


def main():
    """Run comprehensive robustness tests."""
    print("="*70)
    print("COMPREHENSIVE ROBUSTNESS TEST")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/best_model_local.pth'
    
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    model, message_length, image_size, checkpoint = load_model(checkpoint_path, device)
    
    print(f"\nTraining Info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Training BER: {checkpoint['metrics']['ber']:.4f}")
    print(f"  Training Accuracy: {(1 - checkpoint['metrics']['ber'])*100:.2f}%")
    
    # Run all tests
    num_tests = 10
    print(f"\nRunning {num_tests} tests per configuration...")
    
    jpeg_results = test_jpeg_compression(model, message_length, image_size, device, num_tests)
    noise_results = test_gaussian_noise(model, message_length, image_size, device, num_tests)
    blur_results = test_gaussian_blur(model, message_length, image_size, device, num_tests)
    resize_results = test_resize_attack(model, message_length, image_size, device, num_tests)
    color_results = test_color_jitter(model, message_length, image_size, device, num_tests)
    
    # Generate plots
    plot_robustness_results(jpeg_results, noise_results, blur_results, resize_results, color_results)
    
    # Final Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    print("\n📊 Performance Rating Guide:")
    print("  ✓ EXCELLENT: ≥90% accuracy")
    print("  ✓ GOOD:      ≥80% accuracy")
    print("  ⚠ MODERATE:  ≥70% accuracy")
    print("  ✗ POOR:      <70% accuracy")
    
    print("\n🎯 Key Findings:")
    print(f"  Best JPEG Quality (95):     {jpeg_results['accuracy'][0]*100:.2f}%")
    print(f"  Moderate JPEG (80):         {jpeg_results['accuracy'][jpeg_results['quality'].index(80)]*100:.2f}%")
    print(f"  Low Noise (σ=0.02):         {noise_results['accuracy'][1]*100:.2f}%")
    print(f"  Small Blur (k=5):           {blur_results['accuracy'][1]*100:.2f}%")
    print(f"  Downscale (0.7×):           {resize_results['accuracy'][resize_results['scale'].index(0.7)]*100:.2f}%")
    print(f"  Moderate Color Jitter (0.3): {color_results['accuracy'][2]*100:.2f}%")
    
    print("\n💡 Recommendations:")
    if jpeg_results['accuracy'][jpeg_results['quality'].index(80)] >= 0.80:
        print("  ✓ Model handles typical JPEG compression well")
    else:
        print("  ⚠ Consider training with more JPEG augmentation")
    
    if noise_results['accuracy'][1] >= 0.70:
        print("  ✓ Model has good noise resilience")
    else:
        print("  ⚠ Model struggles with noise - add noise augmentation during training")
    
    if blur_results['accuracy'][1] >= 0.70:
        print("  ✓ Model handles blur reasonably well")
    else:
        print("  ⚠ Model is sensitive to blur - consider blur augmentation")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\n📁 Results saved to: robustness_test_results.png")


if __name__ == "__main__":
    main()
