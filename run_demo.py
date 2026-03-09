"""
Quick demo script to run the trained model
"""

import torch
from models.model import StegoModel
from PIL import Image
from torchvision import transforms
import numpy as np

def main():
    print("=" * 70)
    print("STEGANOGRAPHY MODEL DEMO")
    print("=" * 70)
    
    # Configuration
    checkpoint_path = "checkpoints/best_model_local.pth"
    message_length = 16  # Model was trained with 16-bit messages
    image_size = 128  # Model was trained with 128x128 images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n1. Loading trained model from {checkpoint_path}")
    print(f"   Device: {device}")
    
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
    
    if 'epoch' in checkpoint:
        print(f"   Trained for: {checkpoint['epoch']} epochs")
    if 'metrics' in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")
    
    # Create a test image
    print("\n2. Generating test cover image...")
    cover_image = torch.rand(1, 3, image_size, image_size).to(device)
    
    # Create a test message
    print("\n3. Creating secret message...")
    secret_message = "Hi"  # Short message for 16-bit capacity
    print(f"   Text: '{secret_message}'")
    
    # Convert text to binary
    text_bytes = secret_message.encode('utf-8')
    bits = []
    for byte in text_bytes:
        byte_bits = [int(b) for b in format(byte, '08b')]
        bits.extend(byte_bits)
    
    # Pad to message_length
    if len(bits) < message_length:
        bits.extend([0] * (message_length - len(bits)))
    else:
        bits = bits[:message_length]
    
    binary_message = torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"   Binary length: {len(bits)} bits")
    
    # Encode
    print("\n4. Encoding message into image...")
    with torch.no_grad():
        stego_image = model.encode(cover_image, binary_message)
    
    # Calculate difference
    diff = torch.abs(cover_image - stego_image).mean().item()
    print(f"   Average pixel difference: {diff:.6f}")
    
    # Decode
    print("\n5. Decoding message from stego image...")
    with torch.no_grad():
        decoded_message = model.decode(stego_image)
    
    # Convert back to text
    decoded_bits = decoded_message.squeeze(0).cpu().numpy().astype(int)
    num_bytes = len(secret_message.encode('utf-8'))
    
    bytes_array = []
    for i in range(num_bytes):
        byte_bits = decoded_bits[i*8:(i+1)*8]
        byte_value = int(''.join(map(str, byte_bits)), 2)
        bytes_array.append(byte_value)
    
    decoded_text = bytes(bytes_array).decode('utf-8')
    print(f"   Decoded text: '{decoded_text}'")
    
    # Calculate accuracy
    accuracy = (binary_message.cpu() == decoded_message.cpu()).float().mean().item()
    print(f"\n6. Results:")
    print(f"   Bit accuracy: {accuracy*100:.2f}%")
    print(f"   Text matches: {decoded_text == secret_message}")
    
    if accuracy > 0.95:
        print(f"   ✓ Excellent! Model working perfectly!")
    elif accuracy > 0.80:
        print(f"   ✓ Good! Model working well!")
    else:
        print(f"   ⚠ Model may need more training")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
