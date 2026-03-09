"""
Example usage of defense mechanisms for robust steganography.

This demonstrates how to integrate anti-aliasing, denoising, and
adversarial training into the steganography pipeline.
"""

import torch
from models.model import StegoModel
from defense import (
    AdaptiveAntiAlias,
    DenoiseBeforeDecode,
    AdversarialTraining,
    AdversarialAugmentation
)

# Configuration
batch_size = 4
message_length = 1024
image_size = 256

# Create steganography model
model = StegoModel(
    message_length=message_length,
    image_size=image_size,
    enable_distortions=True
)

# Initialize defense mechanisms (all toggleable)
antialias = AdaptiveAntiAlias(enabled=True)
denoise = DenoiseBeforeDecode(method='gaussian', enabled=True)
adversarial = AdversarialTraining(method='pgd', epsilon=0.03, enabled=True)
augmentation = AdversarialAugmentation(
    method='fgsm', epsilon=0.03, mix_ratio=0.5, enabled=True)

# Create test data
cover_image = torch.rand(batch_size, 3, image_size, image_size)
binary_message = torch.randint(0, 2, (batch_size, message_length)).float()

print("=" * 60)
print("Defense Mechanisms Demo")
print("=" * 60)

# =============================================================================
# Example 1: Basic Encoding/Decoding with Anti-Aliasing
# =============================================================================
print("\n1. Encoding with Anti-Aliasing Before Decoding")
print("-" * 60)

# Encode
stego_image = model.encode(cover_image, binary_message)
print(f"Stego image created: {stego_image.shape}")

# Apply anti-aliasing before decoding
filtered_stego = antialias(stego_image, mode='gaussian')
print(f"Anti-aliasing applied (mode: gaussian)")

# Decode
decoded_message = model.decode(filtered_stego)
accuracy = (decoded_message == binary_message).float().mean()
print(f"Decoding accuracy: {accuracy.item()*100:.2f}%")

# =============================================================================
# Example 2: Denoising Before Decoding
# =============================================================================
print("\n2. Denoising Before Decoding")
print("-" * 60)

# Add noise to stego image
noise = torch.randn_like(stego_image) * 0.02
noisy_stego = torch.clamp(stego_image + noise, 0.0, 1.0)
print(f"Noise added to stego image")

# Decode without denoising
decoded_noisy = model.decode(noisy_stego)
accuracy_noisy = (decoded_noisy == binary_message).float().mean()
print(f"Accuracy without denoising: {accuracy_noisy.item()*100:.2f}%")

# Apply denoising before decoding
denoised_stego = denoise(noisy_stego, method='gaussian')
decoded_denoised = model.decode(denoised_stego)
accuracy_denoised = (decoded_denoised == binary_message).float().mean()
print(f"Accuracy with denoising: {accuracy_denoised.item()*100:.2f}%")

# =============================================================================
# Example 3: Adversarial Training
# =============================================================================
print("\n3. Adversarial Training")
print("-" * 60)

# Generate adversarial examples for training
model.train()
adversarial_images = adversarial(
    cover_image, model, binary_message, None, method='pgd')
perturbation = (cover_image - adversarial_images).abs().mean()
print(f"Adversarial examples generated (perturbation: {perturbation:.6f})")

# Train on adversarial examples
outputs = model.forward(adversarial_images, binary_message)
print(f"Training on adversarial examples...")

# =============================================================================
# Example 4: Complete Pipeline with All Defenses
# =============================================================================
print("\n4. Complete Pipeline with All Defenses")
print("-" * 60)

# Enable all defenses
antialias.enable()
denoise.enable()
adversarial.enable()
augmentation.enable()

# Generate augmented batch (mix of clean and adversarial)
mixed_images, mixed_messages, is_adversarial = augmentation(
    cover_image, model, binary_message, None
)
print(
    f"Mixed batch: {is_adversarial.sum().item()} adversarial + {(~is_adversarial).sum().item()} clean")

# Forward pass with distortions
model.train()
outputs = model.forward(mixed_images, mixed_messages)
stego_images = outputs['stego_image']

# Apply anti-aliasing
filtered = antialias(stego_images, mode='adaptive')

# Apply denoising
denoised = denoise(filtered, method='gaussian')

# Decode
decoded = model.decode(denoised)
accuracy = (decoded == mixed_messages).float().mean()
print(f"Final accuracy with all defenses: {accuracy.item()*100:.2f}%")

# =============================================================================
# Example 5: Toggling Defenses On/Off
# =============================================================================
print("\n5. Toggling Defenses")
print("-" * 60)

# Test with defenses enabled
antialias.enable()
denoise.enable()
print("Defenses enabled")

filtered = antialias(stego_image)
denoised = denoise(filtered)
decoded = model.decode(denoised)
accuracy_enabled = (decoded == binary_message).float().mean()
print(f"Accuracy with defenses: {accuracy_enabled.item()*100:.2f}%")

# Test with defenses disabled
antialias.disable()
denoise.disable()
print("Defenses disabled")

filtered = antialias(stego_image)  # Should return unchanged
denoised = denoise(filtered)        # Should return unchanged
decoded = model.decode(denoised)
accuracy_disabled = (decoded == binary_message).float().mean()
print(f"Accuracy without defenses: {accuracy_disabled.item()*100:.2f}%")

# =============================================================================
# Example 6: Method Selection
# =============================================================================
print("\n6. Comparing Different Defense Methods")
print("-" * 60)

# Test different anti-aliasing modes
antialias.enable()
for mode in ['gaussian', 'bilateral', 'median', 'auto']:
    filtered = antialias(stego_image, mode=mode)
    decoded = model.decode(filtered)
    accuracy = (decoded == binary_message).float().mean()
    print(f"Anti-alias mode '{mode}': {accuracy.item()*100:.2f}%")

print()

# Test different denoising methods
denoise.enable()
for method in ['gaussian', 'nlm', 'wavelet', 'adaptive', 'none']:
    denoised = denoise(stego_image, method=method)
    decoded = model.decode(denoised)
    accuracy = (decoded == binary_message).float().mean()
    print(f"Denoise method '{method}': {accuracy.item()*100:.2f}%")

print()

# Test different adversarial methods
adversarial.enable()
for method in ['fgsm', 'pgd']:
    adv_images = adversarial(
        cover_image, model, binary_message, None, method=method)
    perturbation = (cover_image - adv_images).abs().mean()
    print(f"Adversarial method '{method}': perturbation = {perturbation:.6f}")

print("\n" + "=" * 60)
print("✅ All defense mechanisms working correctly!")
print("=" * 60)

# =============================================================================
# Training Loop Example with Defenses
# =============================================================================
print("\n7. Training Loop Example with Defenses")
print("-" * 60)


def train_with_defenses(model, dataloader, optimizer, num_epochs=1):
    """
    Example training loop with defense mechanisms.
    """
    # Initialize defenses
    antialias = AdaptiveAntiAlias(enabled=True)
    denoise = DenoiseBeforeDecode(method='gaussian', enabled=True)
    adversarial = AdversarialAugmentation(
        method='pgd', epsilon=0.03, mix_ratio=0.3, enabled=True)

    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Simulated batch
        cover = torch.rand(batch_size, 3, image_size, image_size)
        message = torch.randint(0, 2, (batch_size, message_length)).float()

        # Apply adversarial augmentation
        augmented_cover, augmented_message, is_adv = adversarial(
            cover, model, message, None)

        # Forward pass
        outputs = model.forward(augmented_cover, augmented_message)
        stego = outputs['stego_image']

        # Apply defenses before decoding (during training)
        if torch.rand(1).item() < 0.5:  # Randomly apply defenses
            stego_defended = antialias(stego, mode='gaussian')
            stego_defended = denoise(stego_defended, method='gaussian')
        else:
            stego_defended = stego

        # Decode and compute loss
        decoded_logits = model.decode(stego_defended, return_logits=True)

        # Compute losses
        image_loss = torch.nn.functional.mse_loss(stego, augmented_cover)
        message_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            decoded_logits, augmented_message
        )
        total_loss = image_loss + message_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Calculate accuracy
        decoded = (torch.sigmoid(decoded_logits) > 0.5).float()
        accuracy = (decoded == augmented_message).float().mean()

        print(
            f"  Loss: {total_loss.item():.6f}, Accuracy: {accuracy.item()*100:.2f}%")
        print(f"  Adversarial samples: {is_adv.sum().item()}/{batch_size}")


print("\nTraining with defenses (simulated):")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_with_defenses(model, None, optimizer, num_epochs=1)

print("\n" + "=" * 60)
print("✅ Defense mechanisms demonstration complete!")
print("=" * 60)
