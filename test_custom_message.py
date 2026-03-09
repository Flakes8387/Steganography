"""
Interactive Message Testing
Allow user to input custom 16-bit messages and test the model
"""

import torch
from models.model import StegoModel
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def string_to_binary(text, max_bits=16):
    """Convert text to binary representation."""
    binary_str = ''.join(format(ord(char), '08b') for char in text)

    # Truncate or pad to max_bits
    if len(binary_str) > max_bits:
        binary_str = binary_str[:max_bits]
    else:
        binary_str = binary_str.ljust(max_bits, '0')

    return [int(bit) for bit in binary_str]


def binary_to_string(binary_list):
    """Convert binary list back to string."""
    binary_str = ''.join(str(int(bit)) for bit in binary_list)

    # Split into 8-bit chunks
    chars = []
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            try:
                char = chr(int(byte, 2))
                if char.isprintable():
                    chars.append(char)
                else:
                    chars.append('?')
            except:
                chars.append('?')

    return ''.join(chars)


def load_test_image(image_size=128):
    """Load a random test image from DIV2K dataset."""
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_paths.extend(
            glob.glob('data/DIV2K/train/**/*' + ext, recursive=True))

    if not image_paths:
        print("❌ No images found in data/DIV2K/train/")
        return None

    # Random image
    img_path = np.random.choice(image_paths)

    # Load and transform
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
    ])

    return transform(image).unsqueeze(0), img_path


def display_results(cover, stego, original_msg, decoded_msg, attack_name="None"):
    """Display encoding/decoding results."""

    # Convert tensors to numpy for display
    cover_np = cover.squeeze().permute(1, 2, 0).cpu().numpy()
    stego_np = stego.squeeze().permute(1, 2, 0).cpu().numpy()
    diff = np.abs(stego_np - cover_np) * 10  # Amplify difference

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cover_np)
    axes[0].set_title('Cover Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(stego_np)
    axes[1].set_title('Stego Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(diff)
    axes[2].set_title('Difference (10×)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('test_result.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visual comparison saved: test_result.png")
    plt.show()

    # Print message comparison
    print("\n" + "="*80)
    print("MESSAGE COMPARISON")
    print("="*80)
    print(f"Attack Applied: {attack_name}")
    print(f"\nOriginal Message (16 bits):")
    print(f"  Binary: {''.join(str(int(b)) for b in original_msg)}")
    print(f"  Decimal: {[int(b) for b in original_msg]}")

    print(f"\nDecoded Message (16 bits):")
    print(f"  Binary: {''.join(str(int(b)) for b in decoded_msg)}")
    print(f"  Decimal: {[int(b) for b in decoded_msg]}")

    # Calculate accuracy
    correct = (original_msg == decoded_msg).sum().item()
    accuracy = (correct / 16) * 100

    print(f"\n{'✓' if accuracy == 100 else '⚠'} Bit Accuracy: {accuracy:.2f}% ({correct}/16 bits correct)")

    if accuracy < 100:
        errors = torch.where(original_msg != decoded_msg)[0]
        print(f"  Errors at positions: {errors.tolist()}")

    # Pixel delta
    pixel_delta = torch.mean(torch.abs(stego - cover)).item()
    print(
        f"\n📊 Pixel Delta: {pixel_delta:.6f} ({'✓ Imperceptible' if pixel_delta < 0.02 else '⚠ Slightly visible'})")
    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("CUSTOM MESSAGE TESTING")
    print("Test the steganography model with your own 16-bit message")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading model: model_BEST_COMBINED.pth")
    model = StegoModel(message_length=16, image_size=128,
                       enable_distortions=True)

    try:
        checkpoint = torch.load(
            'checkpoints/model_BEST_COMBINED.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
    except FileNotFoundError:
        print("⚠ model_BEST_COMBINED.pth not found, trying model_IMPROVED_BLUR.pth")
        checkpoint = torch.load(
            'checkpoints/model_IMPROVED_BLUR.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")

    model = model.to(device)
    model.eval()

    # Input method selection
    print("\n" + "-"*80)
    print("MESSAGE INPUT OPTIONS")
    print("-"*80)
    print("1. Enter 16-bit binary string (e.g., 1010110011001100)")
    print("2. Enter text (will be converted to binary, max 2 chars)")
    print("3. Use random 16-bit message")
    print("-"*80)

    choice = input("Choose option (1/2/3): ").strip()

    if choice == '1':
        # Binary input
        binary_input = input("\nEnter 16-bit binary message: ").strip()

        # Validate
        if len(binary_input) != 16 or not all(c in '01' for c in binary_input):
            print("❌ Invalid input! Must be exactly 16 bits (0s and 1s)")
            return

        message_bits = [int(b) for b in binary_input]
        print(f"✓ Binary message: {binary_input}")

    elif choice == '2':
        # Text input
        text_input = input("\nEnter text (max 2 characters): ").strip()

        if len(text_input) > 2:
            text_input = text_input[:2]
            print(f"⚠ Truncated to 2 characters: '{text_input}'")

        message_bits = string_to_binary(text_input, max_bits=16)
        print(f"✓ Text: '{text_input}'")
        print(f"  Binary: {''.join(str(b) for b in message_bits)}")

        # Try to decode back
        decoded_text = binary_to_string(message_bits)
        print(f"  Decoded back: '{decoded_text}'")

    else:  # choice == '3' or invalid
        # Random
        message_bits = torch.randint(0, 2, (16,)).tolist()
        print(f"✓ Random message: {''.join(str(b) for b in message_bits)}")

    # Convert to tensor
    message = torch.tensor(
        message_bits, dtype=torch.float32).unsqueeze(0).to(device)

    # Load test image
    print("\nLoading random test image from DIV2K...")
    cover_image, img_path = load_test_image(image_size=128)

    if cover_image is None:
        return

    cover_image = cover_image.to(device)
    print(f"✓ Loaded: {os.path.basename(img_path)}")

    # Attack selection
    print("\n" + "-"*80)
    print("ATTACK OPTIONS")
    print("-"*80)
    print("1. No attack (clean)")
    print("2. JPEG compression (quality=75)")
    print("3. Gaussian blur (kernel=5, sigma=1.0)")
    print("4. Resize attack (scale=0.7)")
    print("5. Color jitter (brightness=0.2)")
    print("6. ALL attacks combined")
    print("-"*80)

    attack_choice = input("Choose attack (1-6): ").strip()

    with torch.no_grad():
        # Encode
        print("\n🔒 Encoding message into image...")
        stego_image = model.encode(cover_image, message)
        print("✓ Message hidden in image")

        # Apply attack
        attacked_image = stego_image
        attack_name = "None"

        if attack_choice == '2':
            attacked_image = model.distortions.apply_jpeg_compression(
                stego_image, probability=1.0)
            attack_name = "JPEG Compression"
        elif attack_choice == '3':
            attacked_image = model.distortions.apply_gaussian_blur_attack(
                stego_image, probability=1.0)
            attack_name = "Gaussian Blur"
        elif attack_choice == '4':
            attacked_image = model.distortions.apply_resize_attack(
                stego_image, probability=1.0)
            attack_name = "Resize Attack"
        elif attack_choice == '5':
            attacked_image = model.distortions.apply_color_jitter_attack(
                stego_image, probability=1.0)
            attack_name = "Color Jitter"
        elif attack_choice == '6':
            attacked_image = model.distortions.apply_jpeg_compression(
                stego_image, probability=1.0)
            attacked_image = model.distortions.apply_gaussian_blur_attack(
                attacked_image, probability=1.0)
            attacked_image = model.distortions.apply_resize_attack(
                attacked_image, probability=1.0)
            attacked_image = model.distortions.apply_color_jitter_attack(
                attacked_image, probability=1.0)
            attack_name = "ALL Combined"

        if attack_name != "None":
            print(f"⚔️  Applied attack: {attack_name}")

        # Decode
        print("🔓 Extracting message from image...")
        decoded_logits = model.decode(attacked_image)
        decoded_message = (decoded_logits > 0.5).float()
        print("✓ Message extracted")

        # Display results
        display_results(cover_image, stego_image, message.squeeze(),
                        decoded_message.squeeze(), attack_name)

    # Save stego image
    save_choice = input("Save stego image? (y/n): ").strip().lower()
    if save_choice == 'y':
        stego_np = stego_image.squeeze().permute(1, 2, 0).cpu().numpy()
        stego_pil = Image.fromarray((stego_np * 255).astype(np.uint8))
        stego_pil.save('stego_output.png')
        print("✓ Stego image saved: stego_output.png")

    # Test again?
    again = input("\nTest another message? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("\n✓ Testing complete!\n")


if __name__ == "__main__":
    main()
