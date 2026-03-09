"""
Decoder Network for Deep Steganography
Extracts hidden binary message from stego image.
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


class RevealNetwork(nn.Module):
    """Reveal network to extract message features from stego image."""

    def __init__(self):
        super(RevealNetwork, self).__init__()

        # Encoder path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Residual blocks for feature extraction
        self.res_block1 = ResidualBlock(256)
        self.res_block2 = ResidualBlock(256)
        self.res_block3 = ResidualBlock(256)
        self.res_block4 = ResidualBlock(256)

        # Decoder path with skip connections
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        # Output layer for message features
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, stego_image):
        # Encoder path
        x = F.relu(self.bn1(self.conv1(stego_image)))
        skip1 = x  # Save for skip connection (64 channels)

        x = F.relu(self.bn2(self.conv2(x)))
        skip2 = x  # Save for skip connection (128 channels)

        x = F.relu(self.bn3(self.conv3(x)))
        skip3 = x  # Save for skip connection (256 channels)

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Decoder path without skip connections (channels don't match in original architecture)
        x = F.relu(self.bn4(self.conv4(x)))
        # Removed skip connection - channel mismatch (128 vs 256)

        x = F.relu(self.bn5(self.conv5(x)))
        # Removed skip connection - channel mismatch (64 vs 128)

        x = F.relu(self.bn6(self.conv6(x)))
        # Removed skip connection - channel mismatch (32 vs 64)

        # Output message features
        message_features = self.conv_out(x)

        return message_features


class MessageExtractor(nn.Module):
    """Extracts binary message from message features."""

    def __init__(self, message_length, image_size=256):
        super(MessageExtractor, self).__init__()
        self.message_length = message_length
        self.image_size = image_size

        # Global average pooling and fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        self.fc1 = nn.Linear(32 * 32, 4096)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(2048, message_length)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2048)

    def forward(self, message_features):
        # message_features shape: (batch_size, 1, H, W)
        batch_size = message_features.size(0)

        # Pool to fixed size
        x = self.adaptive_pool(message_features)

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        # Output layer
        logits = self.fc3(x)

        return logits


class Decoder(nn.Module):
    """
    Complete Decoder architecture for steganography.
    Extracts hidden binary message from stego image.
    """

    def __init__(self, message_length, image_size=256):
        super(Decoder, self).__init__()
        self.message_length = message_length
        self.image_size = image_size

        self.reveal_network = RevealNetwork()
        self.message_extractor = MessageExtractor(message_length, image_size)

    def forward(self, stego_image, return_logits=False):
        """
        Args:
            stego_image: torch.Tensor of shape (batch_size, 3, H, W), values in [0, 1]
            return_logits: bool, if True returns raw logits instead of binary predictions

        Returns:
            If return_logits=False:
                binary_message: torch.Tensor of shape (batch_size, message_length), values in {0, 1}
            If return_logits=True:
                logits: torch.Tensor of shape (batch_size, message_length), raw logits
        """
        # Extract message features from stego image
        message_features = self.reveal_network(stego_image)

        # Extract binary message
        logits = self.message_extractor(message_features)

        if return_logits:
            return logits
        else:
            # Convert logits to binary predictions
            binary_message = torch.sigmoid(logits)
            binary_message = (binary_message > 0.5).float()
            return binary_message

    def get_probabilities(self, stego_image):
        """
        Returns probability distribution for each bit.

        Args:
            stego_image: torch.Tensor of shape (batch_size, 3, H, W)

        Returns:
            probabilities: torch.Tensor of shape (batch_size, message_length), values in [0, 1]
        """
        message_features = self.reveal_network(stego_image)
        logits = self.message_extractor(message_features)
        probabilities = torch.sigmoid(logits)
        return probabilities

    def get_num_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test the decoder
    batch_size = 4
    message_length = 1024  # 1024 bits
    image_size = 256

    # Create decoder
    decoder = Decoder(message_length=message_length, image_size=image_size)

    # Create dummy stego image
    stego_image = torch.rand(batch_size, 3, image_size, image_size)

    # Forward pass
    decoded_message = decoder(stego_image)
    logits = decoder(stego_image, return_logits=True)
    probabilities = decoder.get_probabilities(stego_image)

    print(f"Decoder Architecture:")
    print(f"  Input: Stego Image {stego_image.shape}")
    print(f"  Output: Binary Message {decoded_message.shape}")
    print(f"  Total Parameters: {decoder.get_num_parameters():,}")
    print(
        f"  Decoded Message Range: [{decoded_message.min().item()}, {decoded_message.max().item()}]")
    print(
        f"  Logits Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(
        f"  Probabilities Range: [{probabilities.min().item():.4f}, {probabilities.max().item():.4f}]")

    # Show bit distribution
    ones_ratio = decoded_message.mean().item()
    print(
        f"  Decoded Message Distribution: {ones_ratio*100:.1f}% ones, {(1-ones_ratio)*100:.1f}% zeros")
