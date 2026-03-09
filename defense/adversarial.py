"""
Adversarial Training Defense Module

Implements adversarial training techniques (FGSM, PGD) to improve
robustness of steganography models against attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM(nn.Module):
    """
    Fast Gradient Sign Method (FGSM) for adversarial training.

    Generates adversarial examples by taking a step in the direction
    of the gradient of the loss with respect to the input.
    """

    def __init__(self, epsilon=0.03, enabled=True):
        """
        Args:
            epsilon: Magnitude of adversarial perturbation
            enabled: Whether adversarial training is enabled
        """
        super(FGSM, self).__init__()
        self.epsilon = epsilon
        self.enabled = enabled

    def generate_adversarial(self, images, loss_fn, *args, **kwargs):
        """
        Generate adversarial examples using FGSM.

        Args:
            images: Input images (requires_grad=True)
            loss_fn: Loss function to compute gradient
            *args, **kwargs: Additional arguments for loss_fn

        Returns:
            Adversarial images
        """
        if not self.enabled:
            return images

        # Ensure requires_grad
        images.requires_grad = True

        # Forward pass
        loss = loss_fn(images, *args, **kwargs)

        # Compute gradient
        grad = torch.autograd.grad(loss, images, create_graph=False)[0]

        # Generate adversarial perturbation
        perturbation = self.epsilon * grad.sign()

        # Apply perturbation
        adversarial_images = images + perturbation
        adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

        return adversarial_images.detach()

    def forward(self, images, model, binary_message, criterion):
        """
        Apply FGSM adversarial training.

        Args:
            images: Input images
            model: Steganography model
            binary_message: Binary message to hide
            criterion: Loss criterion

        Returns:
            Adversarial images
        """
        if not self.enabled:
            return images

        # Define loss function for FGSM
        def loss_fn(img):
            outputs = model(img, binary_message)
            stego = outputs['stego_image']
            decoded_logits = outputs['decoded_logits']

            # Loss combines image and message objectives
            image_loss = F.mse_loss(stego, img)
            message_loss = F.binary_cross_entropy_with_logits(
                decoded_logits, binary_message)

            return image_loss + message_loss

        # Generate adversarial examples
        adversarial = self.generate_adversarial(images, loss_fn)

        return adversarial

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled


class PGD(nn.Module):
    """
    Projected Gradient Descent (PGD) for adversarial training.

    More powerful than FGSM - performs multiple gradient steps
    with projection back to epsilon ball.
    """

    def __init__(self, epsilon=0.03, alpha=0.01, num_iter=7, enabled=True):
        """
        Args:
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of PGD iterations
            enabled: Whether adversarial training is enabled
        """
        super(PGD, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.enabled = enabled

    def generate_adversarial(self, images, loss_fn, *args, **kwargs):
        """
        Generate adversarial examples using PGD.

        Args:
            images: Input images
            loss_fn: Loss function to compute gradient
            *args, **kwargs: Additional arguments for loss_fn

        Returns:
            Adversarial images
        """
        if not self.enabled:
            return images

        # Start with random perturbation
        adversarial_images = images.clone().detach()
        adversarial_images = adversarial_images + \
            torch.empty_like(
                adversarial_images).uniform_(-self.epsilon, self.epsilon)
        adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

        # PGD iterations
        for _ in range(self.num_iter):
            adversarial_images.requires_grad = True

            # Compute loss and gradient
            loss = loss_fn(adversarial_images, *args, **kwargs)
            grad = torch.autograd.grad(
                loss, adversarial_images, create_graph=False)[0]

            # Take step in gradient direction
            adversarial_images = adversarial_images.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball
            perturbation = torch.clamp(
                adversarial_images - images, -self.epsilon, self.epsilon)
            adversarial_images = torch.clamp(images + perturbation, 0.0, 1.0)

        return adversarial_images.detach()

    def forward(self, images, model, binary_message, criterion):
        """
        Apply PGD adversarial training.

        Args:
            images: Input images
            model: Steganography model
            binary_message: Binary message to hide
            criterion: Loss criterion

        Returns:
            Adversarial images
        """
        if not self.enabled:
            return images

        # Define loss function for PGD
        def loss_fn(img):
            outputs = model(img, binary_message)
            stego = outputs['stego_image']
            decoded_logits = outputs['decoded_logits']

            # Loss combines image and message objectives
            image_loss = F.mse_loss(stego, img)
            message_loss = F.binary_cross_entropy_with_logits(
                decoded_logits, binary_message)

            return image_loss + message_loss

        # Generate adversarial examples
        adversarial = self.generate_adversarial(images, loss_fn)

        return adversarial

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled


class AdversarialTraining(nn.Module):
    """
    Unified adversarial training module.

    Provides flexible interface for adversarial training with FGSM or PGD.
    """

    def __init__(self, method='fgsm', epsilon=0.03, enabled=True):
        """
        Args:
            method: Adversarial method ('fgsm' or 'pgd')
            epsilon: Perturbation magnitude
            enabled: Whether adversarial training is enabled
        """
        super(AdversarialTraining, self).__init__()
        self.method = method
        self.enabled = enabled

        # Initialize adversarial methods
        self.fgsm = FGSM(epsilon=epsilon, enabled=True)
        self.pgd = PGD(epsilon=epsilon, alpha=epsilon /
                       4, num_iter=7, enabled=True)

    def forward(self, images, model, binary_message, criterion=None, method=None):
        """
        Apply adversarial training.

        Args:
            images: Input images
            model: Steganography model
            binary_message: Binary message to hide
            criterion: Loss criterion (optional)
            method: Override default method ('fgsm' or 'pgd')

        Returns:
            Adversarial images or original images if disabled
        """
        if not self.enabled:
            return images

        method = method or self.method

        if method == 'fgsm':
            return self.fgsm(images, model, binary_message, criterion)
        elif method == 'pgd':
            return self.pgd(images, model, binary_message, criterion)
        else:
            return images

    def set_method(self, method):
        """Change adversarial method."""
        self.method = method

    def enable(self):
        """Enable adversarial training."""
        self.enabled = True
        self.fgsm.enable()
        self.pgd.enable()

    def disable(self):
        """Disable adversarial training."""
        self.enabled = False
        self.fgsm.disable()
        self.pgd.disable()

    def toggle(self):
        """Toggle adversarial training on/off."""
        self.enabled = not self.enabled


class AdversarialAugmentation(nn.Module):
    """
    Adversarial augmentation for data augmentation during training.

    Generates adversarial examples as additional training samples
    to improve model robustness.
    """

    def __init__(self, method='fgsm', epsilon=0.03, mix_ratio=0.5, enabled=True):
        """
        Args:
            method: Adversarial method
            epsilon: Perturbation magnitude
            mix_ratio: Ratio of adversarial examples in batch
            enabled: Whether augmentation is enabled
        """
        super(AdversarialAugmentation, self).__init__()
        self.adversarial_training = AdversarialTraining(
            method=method, epsilon=epsilon, enabled=True)
        self.mix_ratio = mix_ratio
        self.enabled = enabled

    def forward(self, images, model, binary_message, criterion=None):
        """
        Generate mixed batch of clean and adversarial examples.

        Args:
            images: Input images
            model: Steganography model
            binary_message: Binary message
            criterion: Loss criterion

        Returns:
            Tuple of (mixed_images, mixed_messages, is_adversarial_mask)
        """
        if not self.enabled:
            return images, binary_message, torch.zeros(images.size(0), dtype=torch.bool, device=images.device)

        batch_size = images.size(0)
        num_adversarial = int(batch_size * self.mix_ratio)

        if num_adversarial == 0:
            return images, binary_message, torch.zeros(batch_size, dtype=torch.bool, device=images.device)

        # Generate adversarial examples
        adversarial_images = self.adversarial_training(
            images[:num_adversarial],
            model,
            binary_message[:num_adversarial],
            criterion
        )

        # Mix clean and adversarial
        mixed_images = torch.cat(
            [adversarial_images, images[num_adversarial:]], dim=0)

        # Create mask indicating which samples are adversarial
        is_adversarial = torch.cat([
            torch.ones(num_adversarial, dtype=torch.bool,
                       device=images.device),
            torch.zeros(batch_size - num_adversarial,
                        dtype=torch.bool, device=images.device)
        ])

        return mixed_images, binary_message, is_adversarial

    def enable(self):
        self.enabled = True
        self.adversarial_training.enable()

    def disable(self):
        self.enabled = False
        self.adversarial_training.disable()

    def toggle(self):
        self.enabled = not self.enabled


# Test function
if __name__ == "__main__":
    print("Testing Adversarial Training Defense Modules...")

    batch_size = 2
    image_size = 256
    message_length = 1024

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.conv = nn.Conv2d(3, 3, 1)
            self.fc = nn.Linear(image_size * image_size * 3, message_length)

        def forward(self, images, binary_message):
            stego = self.conv(images)
            decoded = self.fc(stego.view(stego.size(0), -1))
            return {
                'stego_image': stego,
                'decoded_logits': decoded,
                'decoded_message': (decoded > 0).float()
            }

    model = DummyModel()

    # Create test data
    images = torch.rand(batch_size, 3, image_size, image_size)
    binary_message = torch.randint(0, 2, (batch_size, message_length)).float()

    print("\n1. Testing FGSM...")
    fgsm = FGSM(epsilon=0.03, enabled=True)
    adversarial = fgsm(images, model, binary_message, None)
    perturbation = (images - adversarial).abs().mean()
    print(f"   Perturbation magnitude: {perturbation:.6f}")
    print(f"   Max perturbation: {(images - adversarial).abs().max():.6f}")

    print("\n2. Testing PGD...")
    pgd = PGD(epsilon=0.03, alpha=0.01, num_iter=7, enabled=True)
    adversarial = pgd(images, model, binary_message, None)
    perturbation = (images - adversarial).abs().mean()
    print(f"   Perturbation magnitude: {perturbation:.6f}")
    print(f"   Max perturbation: {(images - adversarial).abs().max():.6f}")

    print("\n3. Testing Adversarial Training...")
    adv_training = AdversarialTraining(
        method='fgsm', epsilon=0.03, enabled=True)

    for method in ['fgsm', 'pgd']:
        adversarial = adv_training(
            images, model, binary_message, None, method=method)
        perturbation = (images - adversarial).abs().mean()
        print(f"   Method '{method}': perturbation = {perturbation:.6f}")

    print("\n4. Testing Adversarial Augmentation...")
    augmentation = AdversarialAugmentation(
        method='fgsm', epsilon=0.03, mix_ratio=0.5, enabled=True)
    mixed_images, mixed_messages, is_adversarial = augmentation(
        images, model, binary_message, None)

    print(f"   Mixed batch size: {mixed_images.size(0)}")
    print(f"   Adversarial samples: {is_adversarial.sum().item()}")
    print(f"   Clean samples: {(~is_adversarial).sum().item()}")

    print("\n5. Testing enable/disable...")
    adv_training.disable()
    disabled_output = adv_training(images, model, binary_message, None)
    print(
        f"   Disabled: {(images - disabled_output).abs().mean():.6f} (should be 0)")

    adv_training.enable()
    enabled_output = adv_training(images, model, binary_message, None)
    print(
        f"   Enabled: {(images - enabled_output).abs().mean():.6f} (should be > 0)")

    print("\n✅ All adversarial training defense tests passed!")
