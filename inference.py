"""
Inference Script - Use Trained Model for Steganography

Demonstrates how to load a trained model and use it for:
1. Encoding messages into images
2. Decoding messages from stego images
3. Batch processing
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np

from models.model import StegoModel


def load_trained_model(checkpoint_path, message_length=1024, image_size=256, device='cpu'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        message_length: Length of binary messages
        image_size: Image size
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=False  # Disable for inference
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'metrics' in checkpoint:
        print(f"Training metrics: {checkpoint['metrics']}")

    return model


def encode_message(model, cover_image_path, binary_message, output_path, device='cpu'):
    """
    Encode a binary message into a cover image.

    Args:
        model: Trained StegoModel
        cover_image_path: Path to cover image
        binary_message: Binary message as torch.Tensor or list
        output_path: Path to save stego image
        device: Device to use

    Returns:
        Stego image as torch.Tensor
    """
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((model.image_size, model.image_size)),
        transforms.ToTensor(),
    ])

    cover_image = Image.open(cover_image_path).convert('RGB')
    original_size = cover_image.size
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)

    # Prepare message
    if isinstance(binary_message, list):
        binary_message = torch.tensor(binary_message, dtype=torch.float32)
    binary_message = binary_message.unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        stego_tensor = model.encode(cover_tensor, binary_message)

    # Convert to image and save
    stego_tensor = stego_tensor.squeeze(0).cpu()
    stego_image = transforms.ToPILImage()(stego_tensor)

    # Resize back to original size if needed
    if original_size != (model.image_size, model.image_size):
        stego_image = stego_image.resize(original_size, Image.LANCZOS)

    stego_image.save(output_path)
    print(f"Saved stego image to {output_path}")

    return stego_tensor


def decode_message(model, stego_image_path, device='cpu'):
    """
    Decode binary message from stego image.

    Args:
        model: Trained StegoModel
        stego_image_path: Path to stego image
        device: Device to use

    Returns:
        Decoded binary message as torch.Tensor
    """
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((model.image_size, model.image_size)),
        transforms.ToTensor(),
    ])

    stego_image = Image.open(stego_image_path).convert('RGB')
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)

    # Decode
    with torch.no_grad():
        decoded_message = model.decode(stego_tensor)

    decoded_message = decoded_message.squeeze(0).cpu()

    return decoded_message


def message_to_text(binary_message, encoding='utf-8'):
    """
    Convert binary message to text string.

    Args:
        binary_message: Binary tensor
        encoding: Text encoding

    Returns:
        Decoded text string
    """
    # Convert to bytes
    bits = binary_message.numpy().astype(int)

    # Group into bytes (8 bits)
    num_bytes = len(bits) // 8
    bytes_array = []

    for i in range(num_bytes):
        byte_bits = bits[i*8:(i+1)*8]
        byte_value = int(''.join(map(str, byte_bits)), 2)
        bytes_array.append(byte_value)

    # Convert to string, stop at null terminator
    try:
        text = bytes(bytes_array).decode(encoding)
        # Remove null characters and non-printable characters
        text = text.rstrip('\x00')
        return text
    except:
        return None


def text_to_message(text, message_length, encoding='utf-8'):
    """
    Convert text string to binary message.

    Args:
        text: Text string to encode
        message_length: Length of binary message
        encoding: Text encoding

    Returns:
        Binary tensor
    """
    # Convert text to bytes
    text_bytes = text.encode(encoding)

    # Convert to binary
    bits = []
    for byte in text_bytes:
        byte_bits = [int(b) for b in format(byte, '08b')]
        bits.extend(byte_bits)

    # Pad or truncate to message_length
    if len(bits) < message_length:
        bits.extend([0] * (message_length - len(bits)))
    else:
        bits = bits[:message_length]

    return torch.tensor(bits, dtype=torch.float32)


# =============================================================================
# Main Functions
# =============================================================================

def main_encode(args):
    """Encode message into image."""
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')

    # Load model
    model = load_trained_model(
        args.checkpoint, args.message_length, args.image_size, device)

    # Prepare message
    if args.text:
        # Encode text message
        binary_message = text_to_message(args.text, args.message_length)
        print(f"Encoding text: '{args.text}'")
    elif args.message_file:
        # Load message from file
        binary_message = torch.load(args.message_file)
        print(f"Loaded message from {args.message_file}")
    else:
        # Random message
        binary_message = torch.randint(0, 2, (args.message_length,)).float()
        print(f"Generated random message")

    # Encode
    encode_message(model, args.cover_image,
                   binary_message, args.output, device)

    # Save message for later verification
    if args.save_message:
        torch.save(binary_message, args.save_message)
        print(f"Saved message to {args.save_message}")


def main_decode(args):
    """Decode message from image."""
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')

    # Load model
    model = load_trained_model(
        args.checkpoint, args.message_length, args.image_size, device)

    # Decode
    decoded_message = decode_message(model, args.stego_image, device)

    print(f"Decoded message length: {len(decoded_message)}")
    print(f"Unique values: {decoded_message.unique()}")

    # Try to decode as text
    text = message_to_text(decoded_message)
    if text:
        print(f"Decoded text: '{text}'")

    # Compare with original if provided
    if args.original_message:
        original = torch.load(args.original_message)
        accuracy = (decoded_message == original).float().mean()
        print(f"Accuracy vs original: {accuracy.item()*100:.2f}%")

        bit_errors = (decoded_message != original).sum().item()
        print(f"Bit errors: {bit_errors}/{len(decoded_message)}")

    # Save decoded message
    if args.output:
        torch.save(decoded_message, args.output)
        print(f"Saved decoded message to {args.output}")


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Learning Steganography Inference')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Encode command
    encode_parser = subparsers.add_parser(
        'encode', help='Encode message into image')
    encode_parser.add_argument(
        '--checkpoint', type=str, required=True, help='Path to model checkpoint')
    encode_parser.add_argument(
        '--cover_image', type=str, required=True, help='Path to cover image')
    encode_parser.add_argument(
        '--output', type=str, required=True, help='Path to save stego image')
    encode_parser.add_argument(
        '--text', type=str, help='Text message to encode')
    encode_parser.add_argument(
        '--message_file', type=str, help='Path to binary message file')
    encode_parser.add_argument(
        '--save_message', type=str, help='Path to save message for verification')
    encode_parser.add_argument('--message_length', type=int, default=1024)
    encode_parser.add_argument('--image_size', type=int, default=256)
    encode_parser.add_argument('--no_cuda', action='store_true')

    # Decode command
    decode_parser = subparsers.add_parser(
        'decode', help='Decode message from image')
    decode_parser.add_argument(
        '--checkpoint', type=str, required=True, help='Path to model checkpoint')
    decode_parser.add_argument(
        '--stego_image', type=str, required=True, help='Path to stego image')
    decode_parser.add_argument(
        '--output', type=str, help='Path to save decoded message')
    decode_parser.add_argument(
        '--original_message', type=str, help='Path to original message for comparison')
    decode_parser.add_argument('--message_length', type=int, default=1024)
    decode_parser.add_argument('--image_size', type=int, default=256)
    decode_parser.add_argument('--no_cuda', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'encode':
        main_encode(args)
    elif args.command == 'decode':
        main_decode(args)
    else:
        print("Please specify 'encode' or 'decode' command")
        print("Examples:")
        print("  python inference.py encode --checkpoint model.pth --cover_image img.jpg --output stego.png --text 'Hello'")
        print("  python inference.py decode --checkpoint model.pth --stego_image stego.png")
