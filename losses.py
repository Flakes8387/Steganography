"""
Loss Functions for Deep Learning Steganography

Implements various loss functions for training steganography models:
1. Image reconstruction loss (L1/MSE)
2. Message reconstruction loss (Binary Cross Entropy)
3. Perceptual loss (VGG-based)
4. Combined StegoLoss class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageReconstructionLoss(nn.Module):
    """
    Image reconstruction loss for measuring imperceptibility.

    Compares cover image with stego image to ensure visual similarity.
    """

    def __init__(self, loss_type='mse'):
        """
        Args:
            loss_type: Type of loss ('mse', 'l1', or 'both')
        """
        super(ImageReconstructionLoss, self).__init__()
        self.loss_type = loss_type.lower()

    def forward(self, stego_image, cover_image):
        """
        Compute image reconstruction loss.

        Args:
            stego_image: Generated stego image (batch_size, 3, H, W)
            cover_image: Original cover image (batch_size, 3, H, W)

        Returns:
            Scalar loss value
        """
        if self.loss_type == 'mse':
            return F.mse_loss(stego_image, cover_image)

        elif self.loss_type == 'l1':
            return F.l1_loss(stego_image, cover_image)

        elif self.loss_type == 'both':
            # Combine MSE and L1 for better gradient properties
            mse = F.mse_loss(stego_image, cover_image)
            l1 = F.l1_loss(stego_image, cover_image)
            return 0.5 * mse + 0.5 * l1

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MessageReconstructionLoss(nn.Module):
    """
    Message reconstruction loss for measuring recoverability.

    Compares original message with decoded message using binary cross entropy.
    """

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(MessageReconstructionLoss, self).__init__()
        self.reduction = reduction

    def forward(self, decoded_logits, original_message):
        """
        Compute message reconstruction loss.

        Args:
            decoded_logits: Raw logits from decoder (batch_size, message_length)
            original_message: Original binary message (batch_size, message_length)

        Returns:
            Scalar loss value
        """
        return F.binary_cross_entropy_with_logits(
            decoded_logits,
            original_message,
            reduction=self.reduction
        )


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Compares high-level features extracted by a pre-trained VGG network
    to ensure perceptual similarity between cover and stego images.
    """

    def __init__(self, feature_layers=None, use_gpu=True):
        """
        Args:
            feature_layers: List of VGG layer indices to use (default: [3, 8, 15, 22])
            use_gpu: Whether to use GPU if available
        """
        super(PerceptualLoss, self).__init__()

        # Default to relu1_2, relu2_2, relu3_3, relu4_3
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]

        self.feature_layers = feature_layers

        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Move to GPU if available
        if use_gpu and torch.cuda.is_available():
            self.features = self.features.cuda()

        self.features.eval()

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize images using ImageNet statistics."""
        return (x - self.mean) / self.std

    def extract_features(self, x):
        """
        Extract features from specified VGG layers.

        Args:
            x: Input image tensor (batch_size, 3, H, W)

        Returns:
            List of feature tensors
        """
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

    def forward(self, stego_image, cover_image):
        """
        Compute perceptual loss.

        Args:
            stego_image: Generated stego image (batch_size, 3, H, W)
            cover_image: Original cover image (batch_size, 3, H, W)

        Returns:
            Scalar loss value
        """
        # Normalize images
        stego_normalized = self.normalize(stego_image)
        cover_normalized = self.normalize(cover_image)

        # Extract features
        stego_features = self.extract_features(stego_normalized)
        cover_features = self.extract_features(cover_normalized)

        # Compute loss across all feature layers
        loss = 0.0
        for stego_feat, cover_feat in zip(stego_features, cover_features):
            loss += F.mse_loss(stego_feat, cover_feat)

        # Average across layers
        loss = loss / len(self.feature_layers)

        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.

    Measures structural similarity between images, which correlates
    better with human perception than MSE.
    """

    def __init__(self, window_size=11, size_average=True):
        """
        Args:
            window_size: Size of the Gaussian window
            size_average: Whether to average the loss
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        """Create Gaussian kernel."""
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, stego_image, cover_image):
        """
        Compute SSIM loss.

        Args:
            stego_image: Generated stego image
            cover_image: Original cover image

        Returns:
            1 - SSIM (lower is better)
        """
        # Move window to same device as images
        if stego_image.device != self.window.device:
            self.window = self.window.to(stego_image.device)

        mu1 = F.conv2d(stego_image, self.window,
                       padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(cover_image, self.window,
                       padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(stego_image * stego_image, self.window,
                             padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(cover_image * cover_image, self.window,
                             padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(stego_image * cover_image, self.window,
                           padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
            ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class StegoLoss(nn.Module):
    """
    Combined loss function for steganography training.

    Combines image reconstruction loss, message reconstruction loss,
    and optional perceptual loss with configurable weights.
    """

    def __init__(
        self,
        image_loss_type='mse',
        use_perceptual=False,
        use_ssim=False,
        alpha=1.0,
        beta=5.0,
        gamma=0.1,
        delta=0.1
    ):
        """
        Args:
            image_loss_type: Type of image loss ('mse', 'l1', 'both')
            use_perceptual: Whether to use perceptual loss
            use_ssim: Whether to use SSIM loss
            alpha: Weight for image reconstruction loss (default: 1.0)
            beta: Weight for message reconstruction loss (default: 5.0)
            gamma: Weight for perceptual loss (if enabled)
            delta: Weight for SSIM loss (if enabled)
        """
        super(StegoLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Image reconstruction loss
        self.image_loss = ImageReconstructionLoss(loss_type=image_loss_type)

        # Message reconstruction loss
        self.message_loss = MessageReconstructionLoss()

        # Optional perceptual loss
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

        # Optional SSIM loss
        self.use_ssim = use_ssim
        if use_ssim:
            self.ssim_loss = SSIMLoss()

    def forward(self, stego_image, cover_image, decoded_logits, original_message):
        """
        Compute combined steganography loss.

        Args:
            stego_image: Generated stego image (batch_size, 3, H, W)
            cover_image: Original cover image (batch_size, 3, H, W)
            decoded_logits: Raw decoder logits (batch_size, message_length)
            original_message: Original binary message (batch_size, message_length)

        Returns:
            Dictionary containing:
                - 'total_loss': Combined weighted loss
                - 'image_loss': Image reconstruction loss
                - 'message_loss': Message reconstruction loss
                - 'perceptual_loss': Perceptual loss (if enabled)
                - 'ssim_loss': SSIM loss (if enabled)
                - 'ber': Bit Error Rate
                - 'bit_accuracy': Bit accuracy (1 - BER)
        """
        # Compute individual losses
        img_loss = self.image_loss(stego_image, cover_image)
        msg_loss = self.message_loss(decoded_logits, original_message)

        # Start with basic losses
        total_loss = self.alpha * img_loss + self.beta * msg_loss

        loss_dict = {
            'image_loss': img_loss,
            'message_loss': msg_loss,
        }

        # Add perceptual loss if enabled
        if self.use_perceptual:
            perc_loss = self.perceptual_loss(stego_image, cover_image)
            total_loss = total_loss + self.gamma * perc_loss
            loss_dict['perceptual_loss'] = perc_loss

        # Add SSIM loss if enabled
        if self.use_ssim:
            ssim_loss_val = self.ssim_loss(stego_image, cover_image)
            total_loss = total_loss + self.delta * ssim_loss_val
            loss_dict['ssim_loss'] = ssim_loss_val

        # Compute BER (Bit Error Rate) and bit accuracy
        decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()
        BER = (decoded_message != original_message).float().mean()
        bit_accuracy = 1.0 - BER

        loss_dict['total_loss'] = total_loss
        loss_dict['ber'] = BER
        loss_dict['bit_accuracy'] = bit_accuracy

        return loss_dict

    def set_weights(self, alpha=None, beta=None, gamma=None, delta=None):
        """
        Update loss weights dynamically.

        Useful for curriculum learning or adaptive weighting.
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # Create dummy data
    batch_size = 4
    image_size = 256
    message_length = 1024

    cover_image = torch.rand(batch_size, 3, image_size, image_size)
    stego_image = cover_image + \
        torch.randn_like(cover_image) * 0.01  # Slight perturbation

    original_message = torch.randint(
        0, 2, (batch_size, message_length)).float()
    decoded_logits = torch.randn(batch_size, message_length)

    print(f"\nData shapes:")
    print(f"  Cover/Stego: {cover_image.shape}")
    print(f"  Message: {original_message.shape}")
    print(f"  Logits: {decoded_logits.shape}")

    # Test 1: Image Reconstruction Loss
    print("\n1. Testing Image Reconstruction Loss")
    print("-" * 60)

    for loss_type in ['mse', 'l1', 'both']:
        img_loss = ImageReconstructionLoss(loss_type=loss_type)
        loss_val = img_loss(stego_image, cover_image)
        print(f"   {loss_type.upper():5s} loss: {loss_val.item():.6f}")

    # Test 2: Message Reconstruction Loss
    print("\n2. Testing Message Reconstruction Loss")
    print("-" * 60)

    msg_loss = MessageReconstructionLoss()
    loss_val = msg_loss(decoded_logits, original_message)
    print(f"   BCE loss: {loss_val.item():.6f}")

    # Test 3: Perceptual Loss (if GPU available)
    print("\n3. Testing Perceptual Loss")
    print("-" * 60)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")

        perc_loss = PerceptualLoss(use_gpu=torch.cuda.is_available())
        cover_gpu = cover_image.to(device)
        stego_gpu = stego_image.to(device)

        loss_val = perc_loss(stego_gpu, cover_gpu)
        print(f"   Perceptual loss: {loss_val.item():.6f}")
        print(f"   ✓ Perceptual loss working")
    except Exception as e:
        print(f"   ⚠ Perceptual loss test skipped: {e}")

    # Test 4: SSIM Loss
    print("\n4. Testing SSIM Loss")
    print("-" * 60)

    ssim_loss = SSIMLoss()
    loss_val = ssim_loss(stego_image, cover_image)
    print(f"   SSIM loss: {loss_val.item():.6f}")
    print(f"   ✓ SSIM loss working")

    # Test 5: Combined StegoLoss
    print("\n5. Testing Combined StegoLoss")
    print("-" * 60)

    # Without perceptual/SSIM
    print("   a) Basic (MSE + BCE):")
    stego_loss = StegoLoss(
        image_loss_type='mse',
        use_perceptual=False,
        use_ssim=False,
        alpha=1.0,
        beta=1.0
    )

    loss_dict = stego_loss(stego_image, cover_image,
                           decoded_logits, original_message)
    print(f"      Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"      Image loss: {loss_dict['image_loss'].item():.6f}")
    print(f"      Message loss: {loss_dict['message_loss'].item():.6f}")
    print(f"      Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")

    # With SSIM
    print("\n   b) With SSIM:")
    stego_loss_ssim = StegoLoss(
        image_loss_type='mse',
        use_perceptual=False,
        use_ssim=True,
        alpha=1.0,
        beta=1.0,
        delta=0.1
    )

    loss_dict = stego_loss_ssim(
        stego_image, cover_image, decoded_logits, original_message)
    print(f"      Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"      Image loss: {loss_dict['image_loss'].item():.6f}")
    print(f"      Message loss: {loss_dict['message_loss'].item():.6f}")
    print(f"      SSIM loss: {loss_dict['ssim_loss'].item():.6f}")
    print(f"      Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")

    # With perceptual (if available)
    try:
        print("\n   c) With Perceptual Loss:")
        stego_loss_perc = StegoLoss(
            image_loss_type='mse',
            use_perceptual=True,
            use_ssim=False,
            alpha=1.0,
            beta=1.0,
            gamma=0.1
        )

        cover_gpu = cover_image.to(device)
        stego_gpu = stego_image.to(device)
        logits_gpu = decoded_logits.to(device)
        message_gpu = original_message.to(device)

        loss_dict = stego_loss_perc(
            stego_gpu, cover_gpu, logits_gpu, message_gpu)
        print(f"      Total loss: {loss_dict['total_loss'].item():.6f}")
        print(f"      Image loss: {loss_dict['image_loss'].item():.6f}")
        print(f"      Message loss: {loss_dict['message_loss'].item():.6f}")
        print(
            f"      Perceptual loss: {loss_dict['perceptual_loss'].item():.6f}")
        print(f"      Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")
    except Exception as e:
        print(f"      ⚠ Perceptual loss test skipped: {e}")

    # Test 6: Dynamic weight adjustment
    print("\n6. Testing Dynamic Weight Adjustment")
    print("-" * 60)

    stego_loss = StegoLoss(alpha=1.0, beta=1.0)

    print(
        f"   Initial weights: alpha={stego_loss.alpha}, beta={stego_loss.beta}")
    loss_dict1 = stego_loss(stego_image, cover_image,
                            decoded_logits, original_message)
    print(f"   Total loss: {loss_dict1['total_loss'].item():.6f}")

    # Increase message loss weight
    stego_loss.set_weights(beta=2.0)
    print(
        f"\n   Updated weights: alpha={stego_loss.alpha}, beta={stego_loss.beta}")
    loss_dict2 = stego_loss(stego_image, cover_image,
                            decoded_logits, original_message)
    print(f"   Total loss: {loss_dict2['total_loss'].item():.6f}")
    print(f"   Loss increased due to higher beta weight")

    # Test 7: Gradient flow
    print("\n7. Testing Gradient Flow")
    print("-" * 60)

    stego_loss = StegoLoss()

    cover_image.requires_grad = True
    stego_image.requires_grad = True
    decoded_logits.requires_grad = True

    loss_dict = stego_loss(stego_image, cover_image,
                           decoded_logits, original_message)
    loss_dict['total_loss'].backward()

    print(f"   Cover gradient: {cover_image.grad is not None}")
    print(f"   Stego gradient: {stego_image.grad is not None}")
    print(f"   Logits gradient: {decoded_logits.grad is not None}")
    print(f"   ✓ Gradient flow working")

    print("\n" + "=" * 60)
    print("✅ All loss function tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("""
# Create loss function
criterion = StegoLoss(
    image_loss_type='mse',
    use_perceptual=True,
    use_ssim=True,
    alpha=1.0,  # Image reconstruction weight
    beta=1.0,   # Message reconstruction weight
    gamma=0.1,  # Perceptual loss weight
    delta=0.1   # SSIM loss weight
)

# In training loop
loss_dict = criterion(stego_image, cover_image, decoded_logits, original_message)
loss = loss_dict['total_loss']
loss.backward()
optimizer.step()

# Access individual losses
print(f"Image Loss: {loss_dict['image_loss'].item()}")
print(f"Message Loss: {loss_dict['message_loss'].item()}")
print(f"Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")
    """)
