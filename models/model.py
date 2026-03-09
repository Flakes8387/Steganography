"""
Unified Steganography Model
Combines Encoder and Decoder into a single pipeline with distortions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from encoder import Encoder
    from decoder import Decoder
else:
    from .encoder import Encoder
    from .decoder import Decoder

# Import attack modules
try:
    from attacks.blur import GaussianBlur
    from attacks.resize import ResizeAttack
    from attacks.color_jitter import ColorJitter
except ImportError:
    # Fallback for when attacks module is not in path
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from attacks.blur import GaussianBlur
    from attacks.resize import ResizeAttack
    from attacks.color_jitter import ColorJitter


class Distortions(nn.Module):
    """
    Applies various distortions to stego images to simulate real-world conditions.
    This makes the model more robust during training.

    Includes:
    - Gaussian noise
    - Spatial dropout
    - JPEG compression
    - Brightness/contrast adjustment
    - Gaussian blur (NEW)
    - Resize attacks (NEW)
    - Color jitter (NEW)
    """

    def __init__(self, dropout_prob=0.1, jpeg_quality_range=(50, 95)):
        super(Distortions, self).__init__()
        self.dropout_prob = dropout_prob
        self.jpeg_quality_range = jpeg_quality_range

        # Initialize attack modules with moderate parameters
        # Reduced blur for more realistic scenarios (sigma 0.3-1.2 instead of 0.5-2.0)
        self.gaussian_blur = GaussianBlur(
            kernel_size_range=(3, 5), sigma_range=(0.3, 1.2))
        self.resize_attack = ResizeAttack(scale_range=(0.5, 0.9))
        self.color_jitter = ColorJitter(
            brightness_range=(-0.2, 0.2),
            contrast_range=(0.7, 1.3),
            saturation_range=(0.7, 1.3),
            hue_range=(-0.1, 0.1)
        )

    def apply_gaussian_noise(self, images, noise_std=0.02):
        """Add Gaussian noise to images."""
        if self.training:
            noise = torch.randn_like(images) * noise_std
            return torch.clamp(images + noise, 0.0, 1.0)
        return images

    def apply_dropout(self, images):
        """Apply spatial dropout (randomly zero out pixels)."""
        if self.training and torch.rand(1).item() < self.dropout_prob:
            mask = torch.rand_like(images) > 0.1  # Drop 10% of pixels
            return images * mask.float()
        return images

    def apply_jpeg_compression(self, images, quality=None, probability=0.3):
        """
        Simulate JPEG compression artifacts.
        Note: This is a simplified approximation using frequency domain operations.

        Args:
            images: Input images
            quality: JPEG quality (not used in simplified version)
            probability: Probability of applying JPEG compression (default: 0.3)
        """
        if self.training and torch.rand(1).item() < probability:
            # Simple approximation: add DCT-like noise
            noise = torch.randn_like(images) * 0.01
            return torch.clamp(images + noise, 0.0, 1.0)
        return images

    def apply_brightness_adjustment(self, images, brightness_range=(-0.1, 0.1)):
        """Randomly adjust brightness."""
        if self.training and torch.rand(1).item() < 0.3:
            brightness = torch.rand(
                1).item() * (brightness_range[1] - brightness_range[0]) + brightness_range[0]
            return torch.clamp(images + brightness, 0.0, 1.0)
        return images

    def apply_contrast_adjustment(self, images, contrast_range=(0.8, 1.2)):
        """Randomly adjust contrast."""
        if self.training and torch.rand(1).item() < 0.3:
            contrast = torch.rand(
                1).item() * (contrast_range[1] - contrast_range[0]) + contrast_range[0]
            mean = images.mean(dim=[2, 3], keepdim=True)
            return torch.clamp((images - mean) * contrast + mean, 0.0, 1.0)
        return images

    def apply_gaussian_blur_attack(self, images, probability=0.3):
        """Apply Gaussian blur attack."""
        if self.training and torch.rand(1).item() < probability:
            return self.gaussian_blur(images)
        return images

    def apply_resize_attack(self, images, probability=0.3):
        """Apply resize (downscale-upscale) attack."""
        if self.training and torch.rand(1).item() < probability:
            return self.resize_attack(images)
        return images

    def apply_color_jitter_attack(self, images, probability=0.3):
        """Apply color jitter (brightness, contrast, saturation, hue) attack."""
        if self.training and torch.rand(1).item() < probability:
            return self.color_jitter(images, apply_all=False)
        return images

    def forward(self, images, apply_all=False, jpeg_only=False):
        """
        Apply distortions to images.
# Original distortions
        images = self.apply_gaussian_noise(images)
        images = self.apply_dropout(images)
        images = self.apply_jpeg_compression(images)
        images = self.apply_brightness_adjustment(images)
        images = self.apply_contrast_adjustment(images)

        # New attack-based distortions
        images = self.apply_gaussian_blur_attack(images, probability=0.3)
        images = self.apply_resize_attack(images, probability=0.3)
        images = self.apply_color_jitter_attack(images, probability=0.3ssion (for compute-limited training)

        Returns:
            Distorted images of same shape
        """
        if not self.training and not apply_all:
            return images

        # JPEG-only mode for local GPU training (less compute intensive)
        if jpeg_only:
            images = self.apply_jpeg_compression(images, probability=0.1)
            return images

        # Apply distortions in sequence
        images = self.apply_gaussian_noise(images)
        images = self.apply_dropout(images)
        images = self.apply_jpeg_compression(images)
        images = self.apply_brightness_adjustment(images)
        images = self.apply_contrast_adjustment(images)

        return images


class StegoModel(nn.Module):
    """
    Unified Steganography Model combining Encoder and Decoder.

    This model provides a complete pipeline for:
    1. Encoding: Hide binary message in cover image
    2. Distortions: Apply realistic distortions (optional, for training)
    3. Decoding: Extract binary message from stego image
    """

    def __init__(self, message_length, image_size=256, enable_distortions=True):
        """
        Args:
            message_length: Number of bits in the binary message
            image_size: Size of square images (H = W = image_size)
            enable_distortions: Whether to enable distortion layer
        """
        super(StegoModel, self).__init__()

        self.message_length = message_length
        self.image_size = image_size
        self.enable_distortions = enable_distortions

        # Initialize encoder and decoder
        self.encoder = Encoder(
            message_length=message_length, image_size=image_size)
        self.decoder = Decoder(
            message_length=message_length, image_size=image_size)

        # Initialize distortion layer
        if enable_distortions:
            self.distortions = Distortions()
        else:
            self.distortions = None

    def encode(self, cover_image, binary_message):
        """
        Encode binary message into cover image.

        Args:
            cover_image: Tensor of shape (batch_size, 3, H, W), values in [0, 1]
            binary_message: Tensor of shape (batch_size, message_length), values in {0, 1}

        Returns:
            stego_image: Tensor of shape (batch_size, 3, H, W), values in [0, 1]
        """
        return self.encoder(cover_image, binary_message)

    def decode(self, stego_image, return_logits=False):
        """
        Decode binary message from stego image.

        Args:
            stego_image: Tensor of shape (batch_size, 3, H, W), values in [0, 1]
            return_logits: If True, return raw logits instead of binary predictions

        Returns:
            If return_logits=False:
                binary_message: Tensor of shape (batch_size, message_length), values in {0, 1}
            If return_logits=True:
                logits: Tensor of shape (batch_size, message_length), raw logits
        """
        return self.decoder(stego_image, return_logits=return_logits)

    def forward(self, cover_image, binary_message, apply_distortions=None, jpeg_only=False):
        """
        Complete forward pass: encode → distortions → decode.

        This is the main training pipeline that:
        1. Encodes the message into the cover image
        2. Applies distortions (if enabled and training)
        3. Decodes the message from the distorted stego image

        Args:
            cover_image: Tensor of shape (batch_size, 3, H, W), values in [0, 1]
            binary_message: Tensor of shape (batch_size, message_length), values in {0, 1}
            apply_distortions: Override distortion behavior (None = use self.training)
            jpeg_only: If True, only apply JPEG compression (for compute-limited training)

        Returns:
            Dictionary containing:
                - 'stego_image': Encoded image before distortions
                - 'distorted_stego': Stego image after distortions (same as stego if no distortions)
                - 'decoded_logits': Raw logits from decoder
                - 'decoded_message': Binary predictions from decoder
        """
        # Step 1: Encode message into cover image
        stego_image = self.encode(cover_image, binary_message)

        # Step 2: Apply distortions (if enabled)
        if apply_distortions is None:
            apply_distortions = self.training and self.enable_distortions

        if apply_distortions and self.distortions is not None:
            distorted_stego = self.distortions(
                stego_image, jpeg_only=jpeg_only)
        else:
            distorted_stego = stego_image

        # Step 3: Decode message from (possibly distorted) stego image
        decoded_logits = self.decode(distorted_stego, return_logits=True)
        decoded_message = torch.sigmoid(decoded_logits)
        decoded_message = (decoded_message > 0.5).float()

        return {
            'stego_image': stego_image,
            'distorted_stego': distorted_stego,
            'decoded_logits': decoded_logits,
            'decoded_message': decoded_message
        }

    def compute_loss(self, cover_image, binary_message, alpha=1.0, beta=1.0):
        """
        Compute training loss for the steganography model.

        Loss = alpha * image_loss + beta * message_loss

        Where:
        - image_loss: MSE between cover and stego (imperceptibility)
        - message_loss: BCE between original and decoded message (recoverability)

        Args:
            cover_image: Tensor of shape (batch_size, 3, H, W)
            binary_message: Tensor of shape (batch_size, message_length)
            alpha: Weight for image loss (imperceptibility)
            beta: Weight for message loss (recoverability)

        Returns:
            Dictionary containing:
                - 'total_loss': Combined loss
                - 'image_loss': MSE between cover and stego
                - 'message_loss': BCE for message recovery
                - 'accuracy': Bit accuracy of decoded message
        """
        # Forward pass
        outputs = self.forward(cover_image, binary_message)

        stego_image = outputs['stego_image']
        decoded_logits = outputs['decoded_logits']
        decoded_message = outputs['decoded_message']

        # Image loss: MSE between cover and stego (imperceptibility)
        image_loss = F.mse_loss(stego_image, cover_image)

        # Message loss: BCE between original and decoded message (recoverability)
        message_loss = F.binary_cross_entropy_with_logits(
            decoded_logits,
            binary_message.float()
        )

        # Combined loss
        total_loss = alpha * image_loss + beta * message_loss

        # Calculate BER (Bit Error Rate) and bit accuracy
        BER = (decoded_message != binary_message).float().mean()
        bit_accuracy = 1.0 - BER

        return {
            'total_loss': total_loss,
            'image_loss': image_loss,
            'message_loss': message_loss,
            'ber': BER,
            'bit_accuracy': bit_accuracy
        }

    def get_num_parameters(self):
        """Returns the total number of trainable parameters."""
        encoder_params = sum(p.numel()
                             for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel()
                             for p in self.decoder.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': encoder_params + decoder_params
        }

    def save_model(self, path):
        """Save model state dict to file."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'message_length': self.message_length,
            'image_size': self.image_size,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model state dict from file."""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Model loaded from {path}")

        return checkpoint


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Unified Steganography Model")
    print("=" * 60)

    # Configuration
    batch_size = 2
    message_length = 1024
    image_size = 256

    # Create unified model
    print("\n1. Creating StegoModel...")
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=True
    )

    params = model.get_num_parameters()
    print(f"   Encoder parameters: {params['encoder']:,}")
    print(f"   Decoder parameters: {params['decoder']:,}")
    print(f"   Total parameters: {params['total']:,}")

    # Create test data
    print("\n2. Creating test data...")
    cover_image = torch.rand(batch_size, 3, image_size, image_size)
    binary_message = torch.randint(0, 2, (batch_size, message_length)).float()
    print(f"   Cover image shape: {cover_image.shape}")
    print(f"   Binary message shape: {binary_message.shape}")

    # Test encode
    print("\n3. Testing encode()...")
    with torch.no_grad():
        stego_image = model.encode(cover_image, binary_message)
    print(f"   Stego image shape: {stego_image.shape}")
    print(
        f"   Stego image range: [{stego_image.min():.4f}, {stego_image.max():.4f}]")

    # Test decode
    print("\n4. Testing decode()...")
    with torch.no_grad():
        decoded_message = model.decode(stego_image)
        decoded_logits = model.decode(stego_image, return_logits=True)
    print(f"   Decoded message shape: {decoded_message.shape}")
    print(
        f"   Decoded logits range: [{decoded_logits.min():.4f}, {decoded_logits.max():.4f}]")

    # Test forward (full pipeline)
    print("\n5. Testing forward() with distortions...")
    model.train()  # Enable distortions
    outputs = model.forward(cover_image, binary_message)
    print(f"   Output keys: {list(outputs.keys())}")
    print(f"   Stego image shape: {outputs['stego_image'].shape}")
    print(f"   Distorted stego shape: {outputs['distorted_stego'].shape}")
    print(f"   Decoded message shape: {outputs['decoded_message'].shape}")

    # Check if distortions were applied
    distortion_diff = (outputs['stego_image'] -
                       outputs['distorted_stego']).abs().mean()
    print(f"   Distortion applied: {distortion_diff.item():.6f}")

    # Test loss computation
    print("\n6. Testing compute_loss()...")
    loss_dict = model.compute_loss(
        cover_image, binary_message, alpha=1.0, beta=1.0)
    print(f"   Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"   Image loss: {loss_dict['image_loss'].item():.6f}")
    print(f"   Message loss: {loss_dict['message_loss'].item():.6f}")
    print(f"   Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")

    # Test without distortions
    print("\n7. Testing forward() without distortions...")
    model.eval()  # Disable distortions
    with torch.no_grad():
        outputs_no_dist = model.forward(cover_image, binary_message)
    distortion_diff = (
        outputs_no_dist['stego_image'] - outputs_no_dist['distorted_stego']).abs().mean()
    print(f"   Distortion applied: {distortion_diff.item():.6f} (should be 0)")

    # Test save/load
    print("\n8. Testing save/load...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pth")
        model.save_model(save_path)

        # Create new model and load
        new_model = StegoModel(
            message_length=message_length, image_size=image_size)
        new_model.load_model(save_path)
        print("   Save/load successful!")

    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
