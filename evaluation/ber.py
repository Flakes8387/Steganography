"""
Bit Error Rate (BER) evaluation for steganography.

Computes the bit error rate between original and decoded messages.
BER measures the recoverability of hidden messages.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple


class BERMetric(nn.Module):
    """
    Bit Error Rate (BER) metric.

    BER = (Number of incorrect bits) / (Total number of bits)

    Lower BER indicates better message recovery.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for converting probabilities to binary (default: 0.5)
        """
        super(BERMetric, self).__init__()
        self.threshold = threshold

    def forward(
        self,
        decoded_message: torch.Tensor,
        original_message: torch.Tensor,
        logits: bool = False
    ) -> torch.Tensor:
        """
        Compute bit error rate.

        Args:
            decoded_message: Decoded binary message or logits (batch_size, message_length)
            original_message: Original binary message (batch_size, message_length)
            logits: Whether decoded_message contains raw logits (needs sigmoid)

        Returns:
            BER value (scalar tensor)
        """
        # Convert logits to probabilities if needed
        if logits:
            decoded_message = torch.sigmoid(decoded_message)

        # Threshold to binary
        decoded_binary = (decoded_message > self.threshold).float()

        # Compute bit errors
        bit_errors = (decoded_binary != original_message).float()

        # BER = total errors / total bits
        ber = bit_errors.sum() / bit_errors.numel()

        return ber

    def compute_per_sample(
        self,
        decoded_message: torch.Tensor,
        original_message: torch.Tensor,
        logits: bool = False
    ) -> torch.Tensor:
        """
        Compute BER for each sample in batch.

        Args:
            decoded_message: Decoded messages (batch_size, message_length)
            original_message: Original messages (batch_size, message_length)
            logits: Whether decoded_message contains raw logits

        Returns:
            BER for each sample (batch_size,)
        """
        if logits:
            decoded_message = torch.sigmoid(decoded_message)

        decoded_binary = (decoded_message > self.threshold).float()
        bit_errors = (decoded_binary != original_message).float()

        # BER per sample
        ber_per_sample = bit_errors.mean(dim=1)

        return ber_per_sample

    def compute_accuracy(
        self,
        decoded_message: torch.Tensor,
        original_message: torch.Tensor,
        logits: bool = False
    ) -> torch.Tensor:
        """
        Compute bit accuracy (1 - BER).

        Args:
            decoded_message: Decoded messages
            original_message: Original messages
            logits: Whether decoded_message contains raw logits

        Returns:
            Bit accuracy (scalar tensor)
        """
        ber = self.forward(decoded_message, original_message, logits)
        return 1.0 - ber


def compute_ber(
    decoded_message: Union[torch.Tensor, np.ndarray],
    original_message: Union[torch.Tensor, np.ndarray],
    logits: bool = False,
    threshold: float = 0.5
) -> float:
    """
    Standalone function to compute BER.

    Args:
        decoded_message: Decoded binary message or logits
        original_message: Original binary message
        logits: Whether decoded_message contains raw logits
        threshold: Threshold for binarization

    Returns:
        BER value as float
    """
    # Convert to torch tensors if needed
    if isinstance(decoded_message, np.ndarray):
        decoded_message = torch.from_numpy(decoded_message)
    if isinstance(original_message, np.ndarray):
        original_message = torch.from_numpy(original_message)

    metric = BERMetric(threshold=threshold)
    ber = metric(decoded_message, original_message, logits=logits)

    return ber.item()


def compute_ber_statistics(
    decoded_messages: torch.Tensor,
    original_messages: torch.Tensor,
    logits: bool = False,
    threshold: float = 0.5
) -> dict:
    """
    Compute BER statistics across multiple samples.

    Args:
        decoded_messages: Batch of decoded messages (batch_size, message_length)
        original_messages: Batch of original messages (batch_size, message_length)
        logits: Whether decoded_messages contains raw logits
        threshold: Threshold for binarization

    Returns:
        Dictionary with BER statistics:
            - mean: Mean BER across all samples
            - std: Standard deviation of BER
            - min: Minimum BER
            - max: Maximum BER
            - median: Median BER
            - per_sample: BER for each sample
    """
    metric = BERMetric(threshold=threshold)

    # Compute per-sample BER
    ber_per_sample = metric.compute_per_sample(
        decoded_messages,
        original_messages,
        logits=logits
    )

    # Compute statistics
    stats = {
        'mean': ber_per_sample.mean().item(),
        'std': ber_per_sample.std().item(),
        'min': ber_per_sample.min().item(),
        'max': ber_per_sample.max().item(),
        'median': ber_per_sample.median().item(),
        'per_sample': ber_per_sample.cpu().numpy()
    }

    return stats


def evaluate_ber_with_attacks(
    model: nn.Module,
    images: torch.Tensor,
    messages: torch.Tensor,
    attacks: list = None,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate BER under various attacks.

    Args:
        model: Steganography model with encode() and decode() methods
        images: Cover images (batch_size, 3, H, W)
        messages: Secret messages (batch_size, message_length)
        attacks: List of attack functions to apply
        device: Device to run on

    Returns:
        Dictionary with BER for each attack condition
    """
    model = model.to(device)
    model.eval()

    images = images.to(device)
    messages = messages.to(device)

    results = {}

    with torch.no_grad():
        # Encode messages
        stego_images = model.encode(images, messages)

        # No attack baseline
        decoded_logits = model.decode(stego_images)
        ber_no_attack = compute_ber(decoded_logits, messages, logits=True)
        results['no_attack'] = ber_no_attack

        # Test with attacks if provided
        if attacks is not None:
            for attack_name, attack_fn in attacks:
                # Apply attack
                attacked_stego = attack_fn(stego_images)

                # Decode
                decoded_logits = model.decode(attacked_stego)
                ber = compute_ber(decoded_logits, messages, logits=True)
                results[attack_name] = ber

    return results


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing BER Metric")
    print("=" * 60)

    # Test 1: Perfect recovery
    print("\n1. Perfect Recovery (BER = 0)")
    print("-" * 60)

    batch_size = 8
    message_length = 1024

    original = torch.randint(0, 2, (batch_size, message_length)).float()
    decoded = original.clone()  # Perfect recovery

    ber = compute_ber(decoded, original)
    print(f"   BER: {ber:.6f}")
    print(f"   Expected: 0.0")
    print(f"   ✓ Test passed: {abs(ber) < 1e-6}")

    # Test 2: Random guessing (BER ≈ 0.5)
    print("\n2. Random Guessing (BER ≈ 0.5)")
    print("-" * 60)

    original = torch.randint(0, 2, (batch_size, message_length)).float()
    decoded = torch.randint(0, 2, (batch_size, message_length)).float()

    ber = compute_ber(decoded, original)
    print(f"   BER: {ber:.6f}")
    print(f"   Expected: ~0.5")
    print(f"   ✓ Test passed: {0.4 < ber < 0.6}")

    # Test 3: With logits
    print("\n3. With Logits (sigmoid conversion)")
    print("-" * 60)

    original = torch.randint(0, 2, (batch_size, message_length)).float()

    # Create logits that will produce correct bits after sigmoid
    logits = torch.where(original == 1,
                         torch.tensor(5.0),   # sigmoid(5) ≈ 0.993 > 0.5
                         torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007 < 0.5

    ber = compute_ber(logits, original, logits=True)
    print(f"   BER: {ber:.6f}")
    print(f"   Expected: ~0.0")
    print(f"   ✓ Test passed: {ber < 0.01}")

    # Test 4: Per-sample BER
    print("\n4. Per-Sample BER")
    print("-" * 60)

    metric = BERMetric()
    original = torch.randint(0, 2, (batch_size, message_length)).float()

    # Create varying quality decoding
    decoded = original.clone()
    decoded[0] = torch.randint(0, 2, (message_length,)).float()  # Random (bad)
    decoded[1] = original[1]  # Perfect

    ber_per_sample = metric.compute_per_sample(decoded, original)

    print(f"   Sample 0 (random): {ber_per_sample[0]:.4f}")
    print(f"   Sample 1 (perfect): {ber_per_sample[1]:.4f}")
    print(f"   ✓ Sample 0 BER high: {ber_per_sample[0] > 0.3}")
    print(f"   ✓ Sample 1 BER zero: {ber_per_sample[1] < 0.01}")

    # Test 5: BER statistics
    print("\n5. BER Statistics")
    print("-" * 60)

    original = torch.randint(0, 2, (100, message_length)).float()
    decoded = original.clone()

    # Introduce varying amounts of errors
    for i in range(100):
        num_errors = int(message_length * (i / 1000))  # 0% to 10% errors
        error_indices = torch.randperm(message_length)[:num_errors]
        decoded[i, error_indices] = 1 - decoded[i, error_indices]

    stats = compute_ber_statistics(decoded, original)

    print(f"   Mean BER: {stats['mean']:.6f}")
    print(f"   Std BER: {stats['std']:.6f}")
    print(f"   Min BER: {stats['min']:.6f}")
    print(f"   Max BER: {stats['max']:.6f}")
    print(f"   Median BER: {stats['median']:.6f}")
    print(f"   ✓ Statistics computed")

    # Test 6: Bit accuracy
    print("\n6. Bit Accuracy (1 - BER)")
    print("-" * 60)

    metric = BERMetric()
    original = torch.randint(0, 2, (batch_size, message_length)).float()

    # 90% correct bits
    decoded = original.clone()
    num_errors = int(0.1 * batch_size * message_length)
    error_mask = torch.zeros_like(decoded.view(-1))
    error_mask[:num_errors] = 1
    error_mask = error_mask[torch.randperm(
        error_mask.size(0))].view_as(decoded)
    decoded = torch.where(error_mask.bool(), 1 - decoded, decoded)

    ber = metric(decoded, original)
    accuracy = metric.compute_accuracy(decoded, original)

    print(f"   BER: {ber:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Sum: {(ber + accuracy):.4f}")
    print(f"   ✓ BER + Accuracy = 1.0: {abs(ber + accuracy - 1.0) < 0.01}")

    # Test 7: Different thresholds
    print("\n7. Different Thresholds")
    print("-" * 60)

    original = torch.zeros(batch_size, message_length)
    decoded_probs = torch.ones(
        batch_size, message_length) * 0.6  # Slightly above 0.5

    for threshold in [0.3, 0.5, 0.7]:
        ber = compute_ber(decoded_probs, original,
                          logits=False, threshold=threshold)
        print(f"   Threshold {threshold}: BER = {ber:.4f}")

    # Test 8: Numpy compatibility
    print("\n8. NumPy Compatibility")
    print("-" * 60)

    original_np = np.random.randint(
        0, 2, (batch_size, message_length)).astype(np.float32)
    decoded_np = original_np.copy()

    ber = compute_ber(decoded_np, original_np)
    print(f"   BER with NumPy input: {ber:.6f}")
    print(f"   ✓ NumPy arrays work")

    # Test 9: Edge cases
    print("\n9. Edge Cases")
    print("-" * 60)

    # Single bit
    original = torch.tensor([[1.0]])
    decoded = torch.tensor([[1.0]])
    ber = compute_ber(decoded, original)
    print(f"   Single bit (correct): BER = {ber:.4f}")

    decoded = torch.tensor([[0.0]])
    ber = compute_ber(decoded, original)
    print(f"   Single bit (wrong): BER = {ber:.4f}")

    # All zeros vs all ones
    original = torch.zeros(10, 100)
    decoded = torch.ones(10, 100)
    ber = compute_ber(decoded, original)
    print(f"   All wrong: BER = {ber:.4f}")
    print(f"   ✓ BER = 1.0: {abs(ber - 1.0) < 0.01}")

    print("\n" + "=" * 60)
    print("✅ All BER tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("""
# Basic usage
from evaluation.ber import compute_ber

decoded_message = model.decode(stego_image)
original_message = get_original_message()

ber = compute_ber(decoded_message, original_message, logits=True)
print(f"Bit Error Rate: {ber:.4f}")
print(f"Accuracy: {(1-ber)*100:.2f}%")

# Statistics across dataset
from evaluation.ber import compute_ber_statistics

stats = compute_ber_statistics(decoded_messages, original_messages, logits=True)
print(f"Mean BER: {stats['mean']:.4f} ± {stats['std']:.4f}")
print(f"Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")

# Evaluate with attacks
from evaluation.ber import evaluate_ber_with_attacks

attacks = [
    ('jpeg_50', lambda x: jpeg_compression(x, quality=50)),
    ('gaussian_noise', lambda x: add_gaussian_noise(x, std=0.1))
]

results = evaluate_ber_with_attacks(model, images, messages, attacks)
for attack, ber in results.items():
    print(f"{attack}: BER = {ber:.4f}")
    """)
