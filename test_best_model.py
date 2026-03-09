"""
Comprehensive test script for the best saved model
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from models.model import StegoModel
from evaluation.ber import BERMetric
from evaluation.psnr import PSNRMetric
from evaluation.ssim import SSIMMetric


def test_model_comprehensive():
    """Comprehensive test of the best saved model"""
    
    print("=" * 70)
    print("COMPREHENSIVE MODEL TEST")
    print("=" * 70)
    
    # Model parameters (inferred from checkpoint)
    checkpoint_path = 'checkpoints/best_model_local.pth'
    message_length = 16
    image_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Message Length: {message_length} bits")
    print(f"Image Size: {image_size}x{image_size}")
    
    # Load model
    print("\n" + "-" * 70)
    print("Loading Model...")
    print("-" * 70)
    
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Initialize evaluation metrics
    ber_metric = BERMetric().to(device)
    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIMMetric().to(device)
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"✓ Trained for: {checkpoint['epoch']} epochs")
    print(f"✓ Training metrics:")
    for key, value in checkpoint['metrics'].items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    
    # Test with sample image
    print("\n" + "-" * 70)
    print("Test 1: Single Image Encoding/Decoding")
    print("-" * 70)
    
    # Load test image
    test_image_path = 'checkpoints/dataset_sample.png'
    if os.path.exists(test_image_path):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        cover_image = Image.open(test_image_path).convert('RGB')
        cover_tensor = transform(cover_image).unsqueeze(0).to(device)
        
        # Generate random binary message
        binary_message = torch.randint(0, 2, (1, message_length)).float().to(device)
        
        print(f"Cover image shape: {cover_tensor.shape}")
        print(f"Binary message: {binary_message[0].cpu().int().tolist()}")
        
        # Encode
        with torch.no_grad():
            stego_tensor = model.encode(cover_tensor, binary_message)
        
        print(f"Stego image shape: {stego_tensor.shape}")
        
        # Decode
        with torch.no_grad():
            decoded_message = model.decode(stego_tensor)
        
        print(f"Decoded message: {decoded_message[0].round().cpu().int().tolist()}")
        
        # Calculate metrics
        ber = ber_metric(decoded_message, binary_message).item()
        bit_accuracy = (1 - ber) * 100
        psnr = psnr_metric(stego_tensor, cover_tensor).item()
        ssim = ssim_metric(stego_tensor, cover_tensor).item()
        
        print(f"\nMetrics:")
        print(f"  Bit Error Rate (BER): {ber:.6f}")
        print(f"  Bit Accuracy: {bit_accuracy:.2f}%")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        
        # Save stego image
        stego_image = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu())
        stego_image.save('test_stego_comprehensive.png')
        print(f"\n✓ Stego image saved to: test_stego_comprehensive.png")
    
    else:
        print(f"Warning: Test image not found at {test_image_path}")
    
    # Test with multiple random messages
    print("\n" + "-" * 70)
    print("Test 2: Multiple Random Messages (Statistical Analysis)")
    print("-" * 70)
    
    num_tests = 20
    bers = []
    psnrs = []
    ssims = []
    
    print(f"Running {num_tests} tests with random messages...")
    
    if os.path.exists(test_image_path):
        for i in range(num_tests):
            # Generate random message
            message = torch.randint(0, 2, (1, message_length)).float().to(device)
            
            with torch.no_grad():
                stego = model.encode(cover_tensor, message)
                decoded = model.decode(stego)
            
            # Calculate metrics
            ber = ber_metric(decoded, message).item()
            psnr = psnr_metric(stego, cover_tensor).item()
            ssim = ssim_metric(stego, cover_tensor).item()
            
            bers.append(ber)
            psnrs.append(psnr)
            ssims.append(ssim)
        
        print(f"\nStatistical Results ({num_tests} tests):")
        print(f"  BER:")
        print(f"    Mean: {np.mean(bers):.6f}")
        print(f"    Std: {np.std(bers):.6f}")
        print(f"    Min: {np.min(bers):.6f}")
        print(f"    Max: {np.max(bers):.6f}")
        print(f"  Bit Accuracy:")
        print(f"    Mean: {(1 - np.mean(bers)) * 100:.2f}%")
        print(f"  PSNR (dB):")
        print(f"    Mean: {np.mean(psnrs):.2f}")
        print(f"    Std: {np.std(psnrs):.2f}")
        print(f"    Min: {np.min(psnrs):.2f}")
        print(f"    Max: {np.max(psnrs):.2f}")
        print(f"  SSIM:")
        print(f"    Mean: {np.mean(ssims):.4f}")
        print(f"    Std: {np.std(ssims):.4f}")
        print(f"    Min: {np.min(ssims):.4f}")
        print(f"    Max: {np.max(ssims):.4f}")
    
    # Model architecture summary
    print("\n" + "-" * 70)
    print("Model Architecture Summary")
    print("-" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    test_model_comprehensive()
