"""
Encode Script - Hide messages in images using trained steganography model.

Usage:
    python encode.py --image cover.jpg --message "Hello World" --output stego.png
    python encode.py --image cover.jpg --message-file secret.txt --output stego.png
    python encode.py --image cover.jpg --binary 101010... --output stego.png

Loads a trained model and embeds a secret message into a cover image.
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
        message_length: Message length in bits (if None, tries to load from checkpoint or infers from weights)
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


def load_image(image_path, target_size=256):
    """
    Load and preprocess cover image.

    Args:
        image_path: Path to cover image
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
    print(f"  Original size: {original_size}")
    print(f"  Resized to: {target_size}x{target_size}")

    return image_tensor, original_size


def text_to_binary(text, message_length):
    """
    Convert text to binary message tensor.

    Args:
        text: Text message to encode
        message_length: Target binary message length

    Returns:
        Binary tensor (1, message_length)
    """
    # Encode text to bytes
    text_bytes = text.encode('utf-8')

    # Convert to binary string
    binary_str = ''.join(format(byte, '08b') for byte in text_bytes)

    # Add length prefix (32 bits for length)
    length_bits = format(len(binary_str), '032b')
    binary_str = length_bits + binary_str

    # Pad or truncate
    if len(binary_str) < message_length:
        binary_str = binary_str + '0' * (message_length - len(binary_str))
    else:
        binary_str = binary_str[:message_length]
        print(f"⚠ Warning: Message truncated to {message_length} bits")

    # Convert to tensor
    binary_list = [float(bit) for bit in binary_str]
    binary_tensor = torch.tensor([binary_list], dtype=torch.float32)

    return binary_tensor


def binary_string_to_tensor(binary_str, message_length):
    """
    Convert binary string to tensor.

    Args:
        binary_str: Binary string (e.g., "10101010...")
        message_length: Target message length

    Returns:
        Binary tensor (1, message_length)
    """
    # Remove spaces and non-binary characters
    binary_str = ''.join(c for c in binary_str if c in '01')

    # Pad or truncate
    if len(binary_str) < message_length:
        binary_str = binary_str + '0' * (message_length - len(binary_str))
    else:
        binary_str = binary_str[:message_length]

    # Convert to tensor
    binary_list = [float(bit) for bit in binary_str]
    binary_tensor = torch.tensor([binary_list], dtype=torch.float32)

    return binary_tensor


def encode_message(model, cover_image, message, device):
    """
    Encode message into cover image.

    Args:
        model: Trained steganography model
        cover_image: Cover image tensor
        message: Binary message tensor
        device: Device to run on

    Returns:
        Stego image tensor
    """
    with torch.no_grad():
        cover_image = cover_image.to(device)
        message = message.to(device)

        # Encode
        stego_image = model.encode(cover_image, message)

    return stego_image


def save_stego_image(stego_tensor, output_path, original_size=None):
    """
    Save stego image to file.

    Args:
        stego_tensor: Stego image tensor (1, 3, H, W)
        output_path: Output file path
        original_size: Optional original size for resizing back
    """
    # Convert to PIL image
    stego_np = stego_tensor.squeeze(0).cpu().numpy()
    stego_np = np.transpose(stego_np, (1, 2, 0))
    stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

    stego_image = Image.fromarray(stego_np)

    # Resize back to original size if specified
    if original_size is not None:
        stego_image = stego_image.resize(original_size, Image.LANCZOS)

    # Save
    stego_image.save(output_path)

    print(f"✓ Stego image saved: {output_path}")
    if original_size:
        print(f"  Resized back to: {original_size}")


def compute_psnr(cover, stego):
    """Compute PSNR between cover and stego images."""
    # Ensure both tensors are on CPU for comparison
    cover = cover.cpu() if cover.is_cuda else cover
    stego = stego.cpu() if stego.is_cuda else stego
    mse = torch.mean((cover - stego) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def main():
    parser = argparse.ArgumentParser(
        description='Encode secret message into image using steganography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode text message
  python encode.py --image cover.jpg --message "Secret message" --output stego.png
  
  # Encode from text file
  python encode.py --image cover.jpg --message-file secret.txt --output stego.png
  
  # Encode binary message
  python encode.py --image cover.jpg --binary "101010101010" --output stego.png
  
  # Use custom model
  python encode.py --image cover.jpg --message "Hello" --model my_model.pth --output stego.png
        """
    )

    # Required arguments
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to cover image')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output stego image')

    # Message input (one of these is required)
    message_group = parser.add_mutually_exclusive_group(required=True)
    message_group.add_argument('--message', '-m', type=str,
                               help='Text message to encode')
    message_group.add_argument('--message-file', '-f', type=str,
                               help='Path to text file containing message')
    message_group.add_argument('--binary', '-b', type=str,
                               help='Binary message string (e.g., "101010...")')

    # Optional arguments
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--message-length', type=int, default=None,
                        help='Message length in bits (if not specified, tries to load from checkpoint)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Image size for processing (default: auto-detect from model)')
    parser.add_argument('--keep-size', action='store_true',
                        help='Resize output back to original image size')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show quality statistics (PSNR)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"✗ Error: Image file not found: {args.image}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"✗ Error: Model file not found: {args.model}")
        print(f"  Please train a model first or specify correct path with --model")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output).parent
    if output_dir != Path('.') and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Steganography Encoder")
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
        cover_image, original_size = load_image(args.image, target_image_size)
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        sys.exit(1)

    # Prepare message
    print(f"\n{'='*60}")
    print("Preparing message...")
    print("=" * 60)

    try:
        if args.message:
            message_text = args.message
            print(f"✓ Text message: \"{message_text}\"")
            print(f"  Length: {len(message_text)} characters")
            message_tensor = text_to_binary(message_text, message_length)

        elif args.message_file:
            with open(args.message_file, 'r', encoding='utf-8') as f:
                message_text = f.read().strip()
            print(f"✓ Message loaded from: {args.message_file}")
            print(f"  Length: {len(message_text)} characters")
            if len(message_text) > 100:
                print(f"  Preview: \"{message_text[:100]}...\"")
            else:
                print(f"  Content: \"{message_text}\"")
            message_tensor = text_to_binary(message_text, message_length)

        elif args.binary:
            print(f"✓ Binary message")
            print(f"  Length: {len(args.binary)} bits")
            message_tensor = binary_string_to_tensor(
                args.binary, message_length)

    except Exception as e:
        print(f"✗ Error preparing message: {e}")
        sys.exit(1)

    # Encode
    print(f"\n{'='*60}")
    print("Encoding message into image...")
    print("=" * 60)

    try:
        stego_image = encode_message(
            model, cover_image, message_tensor, device)
        print("✓ Message encoded successfully")
    except Exception as e:
        print(f"✗ Error during encoding: {e}")
        sys.exit(1)

    # Show statistics if requested
    if args.show_stats:
        print(f"\n{'='*60}")
        print("Quality Statistics:")
        print("=" * 60)
        psnr = compute_psnr(cover_image, stego_image)
        print(f"  PSNR: {psnr:.2f} dB")
        if psnr > 40:
            print(f"  Quality: Excellent (imperceptible)")
        elif psnr > 35:
            print(f"  Quality: Very Good")
        elif psnr > 30:
            print(f"  Quality: Good")
        else:
            print(f"  Quality: Fair")

    # Save output
    print(f"\n{'='*60}")
    print("Saving stego image...")
    print("=" * 60)

    try:
        save_stego_image(
            stego_image,
            args.output,
            original_size if args.keep_size else None
        )
    except Exception as e:
        print(f"✗ Error saving image: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("✅ Encoding completed successfully!")
    print("=" * 60)
    print(f"\nTo decode the message:")
    print(f"  python decode.py --image {args.output} --model {args.model}")


if __name__ == "__main__":
    main()
