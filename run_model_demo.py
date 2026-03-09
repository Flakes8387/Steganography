"""
Comprehensive demo of the trained steganography model
"""

import torch
from models.model import StegoModel
import numpy as np

def main():
    print("\n" + "=" * 70)
    print("STEGANOGRAPHY MODEL - TRAINED MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Configuration (matching the trained model)
    checkpoint_path = "checkpoints/best_model_local.pth"
    message_length = 16
    image_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  Epochs trained: {checkpoint['epoch']}")
    print(f"  Training accuracy: {checkpoint['metrics']['bit_accuracy']*100:.2f}%")
    print(f"  Training BER: {checkpoint['metrics']['ber']:.4f}")
    
    # Run multiple tests
    print(f"\n" + "=" * 70)
    print("RUNNING 10 TEST CASES")
    print("=" * 70)
    
    accuracies = []
    bers = []
    
    for test_num in range(1, 11):
        # Generate random cover image
        cover_image = torch.rand(1, 3, image_size, image_size).to(device)
        
        # Generate random binary message
        binary_message = torch.randint(0, 2, (1, message_length)).float().to(device)
        
        # Encode
        with torch.no_grad():
            stego_image = model.encode(cover_image, binary_message)
            
        # Decode
        with torch.no_grad():
            decoded_message = model.decode(stego_image)
        
        # Calculate metrics
        correct_bits = (binary_message == decoded_message).float().sum().item()
        accuracy = correct_bits / message_length
        ber = 1.0 - accuracy
        
        # Image quality
        mse = torch.mean((cover_image - stego_image) ** 2).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        accuracies.append(accuracy)
        bers.append(ber)
        
        # Show first 5 tests in detail
        if test_num <= 5:
            print(f"\nTest {test_num}:")
            print(f"  Original:  {binary_message.cpu().numpy().astype(int)[0]}")
            print(f"  Decoded:   {decoded_message.cpu().numpy().astype(int)[0]}")
            print(f"  Accuracy: {accuracy*100:.2f}%  |  BER: {ber:.4f}  |  PSNR: {psnr:.2f} dB")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print("SUMMARY STATISTICS (10 tests)")
    print("=" * 70)
    print(f"  Average Accuracy: {np.mean(accuracies)*100:.2f}%")
    print(f"  Min Accuracy: {np.min(accuracies)*100:.2f}%")
    print(f"  Max Accuracy: {np.max(accuracies)*100:.2f}%")
    print(f"  Average BER: {np.mean(bers):.4f}")
    print(f"  Std Dev: {np.std(accuracies)*100:.2f}%")
    
    # Performance assessment
    avg_acc = np.mean(accuracies)
    print(f"\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)
    
    if avg_acc >= 0.95:
        print("  🌟 EXCELLENT - Model performs very well!")
        print("  The model can reliably hide and recover messages.")
    elif avg_acc >= 0.85:
        print("  ✓ GOOD - Model performs well!")
        print("  The model successfully hides and recovers most messages.")
    elif avg_acc >= 0.75:
        print("  ⚠ FAIR - Model shows promise but could improve.")
        print("  Consider training for more epochs or adjusting hyperparameters.")
    else:
        print("  ⚠ NEEDS IMPROVEMENT - Model needs more training.")
        print("  Consider longer training or checking the loss functions.")
    
    print(f"\n" + "=" * 70)
    print("MODEL CAPABILITIES")
    print("=" * 70)
    print(f"  Message capacity: {message_length} bits ({message_length//8} bytes)")
    print(f"  Image size: {image_size}x{image_size} pixels")
    print(f"  Stego imperceptibility: High (minimal visual changes)")
    print(f"  Recovery rate: {avg_acc*100:.1f}% of bits correctly recovered")
    
    print(f"\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
