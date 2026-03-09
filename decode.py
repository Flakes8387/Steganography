"""
Decode Script - Extract hidden messages from stego images.

Usage:
    python decode.py --image stego.png
    python decode.py --image stego.png --output message.txt
    python decode.py --image stego.png --binary

Loads a trained model and extracts the hidden message from a stego image.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_model(model_path, device='cuda', message_length=None, image_size=None):
    """
    Load trained steganography model.

    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
        message_length: Message length in bits (if None, infers from weights)
        image_size: Image size (if None, infers from model weights)

    Returns:
        Loaded model in eval mode, message_length, image_size, device
    """
    from models.model import StegoModel
    import math

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint['model_state_dict'] if isinstance(
        checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    # Infer image_size from model weights if not provided
    if image_size is None:
        fc3_weight_shape = state_dict['encoder.prep_network.fc3.weight'].shape
        image_size = int(math.sqrt(fc3_weight_shape[0]))
        print(f"✓ Inferred image_size from checkpoint: {image_size}")

    # Infer message_length from model weights if not provided
    if message_length is None:
        fc1_weight_shape = state_dict['encoder.prep_network.fc1.weight'].shape
        message_length = fc1_weight_shape[1]
        print(f"✓ Inferred message_length from checkpoint: {message_length}")

    # Create model
    model = StegoModel(message_length=message_length, image_size=image_size)

    # Load weights
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from {model_path}")
    print(f"  Message length: {message_length} bits")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Device: {device}")

    return model, message_length, image_size, device

    return model, message_length, device


def load_image(image_path, target_size=256):
    """
    Load and preprocess stego image.

    Args:
        image_path: Path to stego image
        target_size: Target size for resizing

    Returns:
        Image tensor (1, 3, H, W)
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # Transform to tensor
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    print(f"✓ Image loaded: {image_path}")
    print(f"  Size: {original_size}")
    print(f"  Resized to: {target_size}x{target_size}")

    return image_tensor


def decode_message(model, stego_image, device):
    """
    Decode message from stego image.

    Args:
        model: Trained steganography model
        stego_image: Stego image tensor
        device: Device to run on

    Returns:
        Decoded binary message tensor
    """
    with torch.no_grad():
        stego_image = stego_image.to(device)

        # Decode
        decoded_logits = model.decode(stego_image)

        # Convert to binary
        decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()

    return decoded_message


def binary_to_text(binary_tensor):
    """
    Convert binary message tensor to text.

    Expects first 32 bits to contain the message length.

    Args:
        binary_tensor: Binary message tensor (1, message_length)

    Returns:
        Decoded text string
    """
    # Convert to binary string
    binary_list = binary_tensor.squeeze(0).cpu().numpy().astype(int)
    binary_str = ''.join(str(bit) for bit in binary_list)

    # Extract length (first 32 bits)
    try:
        length_bits = binary_str[:32]
        message_length = int(length_bits, 2)

        # Validate length
        if message_length <= 0 or message_length > len(binary_str) - 32:
            # No valid length prefix, try to decode entire message
            message_length = len(binary_str) - 32

        # Extract message bits
        message_bits = binary_str[32:32 + message_length]

    except (ValueError, IndexError):
        # If length extraction fails, use entire message
        message_bits = binary_str

    # Convert bits to bytes
    try:
        # Pad to multiple of 8
        if len(message_bits) % 8 != 0:
            padding = 8 - (len(message_bits) % 8)
            message_bits = message_bits + '0' * padding

        # Convert to bytes
        message_bytes = bytearray()
        for i in range(0, len(message_bits), 8):
            byte = message_bits[i:i+8]
            message_bytes.append(int(byte, 2))

        # Decode to text
        text = message_bytes.decode('utf-8', errors='ignore')

        # Remove null characters and control characters
        text = ''.join(char for char in text if char.isprintable()
                       or char in '\n\r\t')
        text = text.rstrip('\x00')

        return text

    except Exception as e:
        print(f"⚠ Warning: Text decoding failed ({e})")
        return None


def binary_to_string(binary_tensor):
    """
    Convert binary tensor to binary string representation.

    Args:
        binary_tensor: Binary message tensor

    Returns:
        Binary string
    """
    binary_list = binary_tensor.squeeze(0).cpu().numpy().astype(int)
    return ''.join(str(bit) for bit in binary_list)


def main():
    parser = argparse.ArgumentParser(
        description='Decode secret message from stego image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode and print message
  python decode.py --image stego.png
  
  # Save decoded message to file
  python decode.py --image stego.png --output message.txt
  
  # Output as binary string
  python decode.py --image stego.png --binary
  
  # Use custom model
  python decode.py --image stego.png --model my_model.pth
        """
    )

    # Required arguments
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to stego image')

    # Optional arguments
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save decoded message (optional)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--message-length', type=int, default=None,
                        help='Message length in bits (if not specified, tries to load from checkpoint)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Image size for processing (default: auto-detect from model)')
    parser.add_argument('--binary', '-b', action='store_true',
                        help='Output as binary string instead of text')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--show-confidence', action='store_true',
                        help='Show bit confidence statistics')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"✗ Error: Image file not found: {args.image}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"✗ Error: Model file not found: {args.model}")
        print(f"  Please train a model first or specify correct path with --model")
        sys.exit(1)

    print("=" * 60)
    print("Steganography Decoder")
    print("=" * 60)

    # Load model
    try:
        model, message_length, model_image_size, device = load_model(
            args.model, args.device, args.message_length, args.image_size)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Use model's image size if not explicitly provided
    target_image_size = args.image_size if args.image_size is not None else model_image_size

    # Load image
    try:
        stego_image = load_image(args.image, target_image_size)
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        sys.exit(1)

    # Decode
    print(f"\n{'='*60}")
    print("Decoding hidden message...")
    print("=" * 60)

    try:
        decoded_message = decode_message(model, stego_image, device)
        print("✓ Message decoded successfully")
        print(f"  Message length: {message_length} bits")
    except Exception as e:
        print(f"✗ Error during decoding: {e}")
        sys.exit(1)

    # Show confidence if requested
    if args.show_confidence:
        with torch.no_grad():
            stego_image_gpu = stego_image.to(device)
            logits = model.decode(stego_image_gpu)
            probs = torch.sigmoid(logits).squeeze(0).cpu()

            # Calculate confidence (how far from 0.5)
            confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]

            print(f"\n{'='*60}")
            print("Bit Confidence Statistics:")
            print("=" * 60)
            print(f"  Mean confidence: {confidence.mean():.4f}")
            print(f"  Min confidence: {confidence.min():.4f}")
            print(f"  Max confidence: {confidence.max():.4f}")
            print(
                f"  Bits > 90% confident: {(confidence > 0.9).sum().item()}/{len(confidence)}")
            print(
                f"  Bits < 60% confident: {(confidence < 0.6).sum().item()}/{len(confidence)}")

    # Process output
    print(f"\n{'='*60}")
    print("Decoded Message:")
    print("=" * 60)

    if args.binary:
        # Output as binary string
        binary_string = binary_to_string(decoded_message)

        # Show preview
        if len(binary_string) > 100:
            print(f"\n{binary_string[:100]}...")
            print(f"\n(showing first 100 of {len(binary_string)} bits)")
        else:
            print(f"\n{binary_string}")

        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(binary_string)
            print(f"\n✓ Binary message saved to: {args.output}")

    else:
        # Output as text
        text_message = binary_to_text(decoded_message)

        if text_message is None or len(text_message.strip()) == 0:
            print("\n⚠ Warning: Could not decode valid text message")
            print("  The message may be binary data or corrupted.")
            print("  Try using --binary flag to see raw binary.")
        else:
            # Show message
            if len(text_message) > 500:
                print(f"\n{text_message[:500]}...")
                print(
                    f"\n(showing first 500 of {len(text_message)} characters)")
            else:
                print(f"\n{text_message}")

            print(f"\nMessage length: {len(text_message)} characters")

            # Save to file if specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(text_message)
                print(f"✓ Message saved to: {args.output}")

    print(f"\n{'='*60}")
    print("✅ Decoding completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
