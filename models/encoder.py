"""
Encoder Network for Deep Steganography
Embeds a binary message into a cover image using deep neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connections."""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class PrepNetwork(nn.Module):
    """Preparation network to process the binary message."""

    def __init__(self, message_length, image_size=256):
        super(PrepNetwork, self).__init__()
        self.image_size = image_size
        self.message_length = message_length

        # Expand message to image dimensions
        self.fc1 = nn.Linear(message_length, 4096)
        self.fc2 = nn.Linear(4096, 16384)
        self.fc3 = nn.Linear(16384, image_size * image_size)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, message):
        # message shape: (batch_size, message_length)
        batch_size = message.size(0)

        # Expand message through fully connected layers
        x = F.relu(self.fc1(message))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Reshape to image format
        x = x.view(batch_size, 1, self.image_size, self.image_size)

        # Convolutional layers to create message features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x


class HidingNetwork(nn.Module):
    """Hiding network that combines cover image and processed message."""

    def __init__(self):
        super(HidingNetwork, self).__init__()

        # Initial convolution (3 channels from image + 64 from prep network)
        self.conv1 = nn.Conv2d(67, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Residual blocks
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.res_block3 = ResidualBlock(128)

        # Decoder path with skip connections
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Final layer to produce stego image
        self.conv_final = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, cover_image, prepared_message):
        # Concatenate cover image and prepared message
        x = torch.cat([cover_image, prepared_message], dim=1)

        # Encoder path
        x = F.relu(self.bn1(self.conv1(x)))
        skip1 = x  # Save for skip connection (64 channels)

        x = F.relu(self.bn2(self.conv2(x)))
        skip2 = x  # Save for skip connection (128 channels)

        x = F.relu(self.bn3(self.conv3(x)))

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Decoder path with skip connections
        x = F.relu(self.bn4(self.conv4(x)))
        # Skip connection - concatenate instead of add since channels don't match

        x = F.relu(self.bn5(self.conv5(x)))
        # Skip connection - concatenate instead of add since channels don't match

        # Final convolution to produce 3-channel image
        stego_image = self.conv_final(x)

        # Add residual connection from original cover image
        stego_image = stego_image + cover_image

        # Clamp to valid image range
        stego_image = torch.clamp(stego_image, 0.0, 1.0)

        return stego_image


class Encoder(nn.Module):
    """
    Complete Encoder architecture for steganography.
    Embeds binary message into cover image.
    """

    def __init__(self, message_length, image_size=256):
        super(Encoder, self).__init__()
        self.message_length = message_length
        self.image_size = image_size

        self.prep_network = PrepNetwork(message_length, image_size)
        self.hiding_network = HidingNetwork()

    def forward(self, cover_image, binary_message):
        """
        Args:
            cover_image: torch.Tensor of shape (batch_size, 3, H, W), values in [0, 1]
            binary_message: torch.Tensor of shape (batch_size, message_length), values in {0, 1}

        Returns:
            stego_image: torch.Tensor of shape (batch_size, 3, H, W), values in [0, 1]
        """
        # Convert binary message to float if needed
        if binary_message.dtype != torch.float32:
            binary_message = binary_message.float()

        # Prepare message features
        prepared_message = self.prep_network(binary_message)

        # Hide message in cover image
        stego_image = self.hiding_network(cover_image, prepared_message)

        return stego_image

    def get_num_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test the encoder
    batch_size = 4
    message_length = 1024  # 1024 bits
    image_size = 256

    # Create encoder
    encoder = Encoder(message_length=message_length, image_size=image_size)

    # Create dummy inputs
    cover_image = torch.rand(batch_size, 3, image_size, image_size)
    binary_message = torch.randint(0, 2, (batch_size, message_length)).float()

    # Forward pass
    stego_image = encoder(cover_image, binary_message)

    print(f"Encoder Architecture:")
    print(
        f"  Input: Cover Image {cover_image.shape}, Binary Message {binary_message.shape}")
    print(f"  Output: Stego Image {stego_image.shape}")
    print(f"  Total Parameters: {encoder.get_num_parameters():,}")
    print(
        f"  Output Range: [{stego_image.min().item():.4f}, {stego_image.max().item():.4f}]")

    # Check that stego image is different from cover image
    diff = torch.abs(stego_image - cover_image).mean()
    print(f"  Average Pixel Difference: {diff.item():.6f}")
