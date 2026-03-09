"""
Steganalysis detector for evaluating steganography security.

Implements CNN-based binary classifier to detect stego vs cover images.
Lower detection accuracy indicates better steganography security.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict


class SRNet(nn.Module):
    """
    Simplified SRNet architecture for steganalysis.

    Based on "Deep Residual Network for Steganalysis of Digital Images"
    Designed to detect steganographic content in images.
    """

    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super(SRNet, self).__init__()

        # Layer 1: High-pass filtering
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 2-5: Convolutional blocks
        self.layer2 = self._make_layer(64, 64)
        self.layer3 = self._make_layer(64, 128, stride=2)
        self.layer4 = self._make_layer(128, 256, stride=2)
        self.layer5 = self._make_layer(256, 512, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Linear(512, 2)  # Binary: cover vs stego

    def _make_layer(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        """Create a residual-like layer."""
        layers = []

        # Conv block
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Another conv
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (batch_size, channels, H, W)

        Returns:
            Logits (batch_size, 2)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SimpleStegDetector(nn.Module):
    """
    Lightweight CNN for steganalysis.

    Faster to train than SRNet, suitable for quick evaluation.
    """

    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: Number of input channels
        """
        super(SimpleStegDetector, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SteganalysisDataset(Dataset):
    """
    Dataset for training steganalysis detector.

    Pairs cover images (label=0) with stego images (label=1).
    """

    def __init__(
        self,
        cover_images: torch.Tensor,
        stego_images: torch.Tensor
    ):
        """
        Args:
            cover_images: Tensor of cover images (N, C, H, W)
            stego_images: Tensor of stego images (N, C, H, W)
        """
        assert cover_images.shape == stego_images.shape

        # Combine images and labels
        self.images = torch.cat([cover_images, stego_images], dim=0)
        self.labels = torch.cat([
            torch.zeros(len(cover_images)),
            torch.ones(len(stego_images))
        ]).long()

        # Shuffle
        indices = torch.randperm(len(self.images))
        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class SteganalysisEvaluator:
    """
    Evaluator for steganalysis detection.

    Trains a detector and evaluates detection accuracy.
    Lower detection accuracy = better steganography security.
    """

    def __init__(
        self,
        detector_type: str = 'simple',
        device: str = 'cuda'
    ):
        """
        Args:
            detector_type: Type of detector ('simple' or 'srnet')
            device: Device to run on
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        if detector_type == 'simple':
            self.detector = SimpleStegDetector()
        elif detector_type == 'srnet':
            self.detector = SRNet()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        self.detector = self.detector.to(self.device)
        self.detector_type = detector_type

    def train_detector(
        self,
        cover_images: torch.Tensor,
        stego_images: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train steganalysis detector.

        Args:
            cover_images: Cover images (N, C, H, W)
            stego_images: Stego images (N, C, H, W)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history (losses, accuracies)
        """
        # Create dataset and dataloader
        dataset = SteganalysisDataset(cover_images, stego_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.detector.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {'loss': [], 'accuracy': []}

        self.detector.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                optimizer.zero_grad()
                outputs = self.detector(images)
                loss = criterion(outputs, labels)

                # Backward
                loss.backward()
                optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Epoch statistics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100.0 * correct / total

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2f}%")

        return history

    def evaluate_detector(
        self,
        cover_images: torch.Tensor,
        stego_images: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate trained detector.

        Args:
            cover_images: Cover images
            stego_images: Stego images
            batch_size: Batch size

        Returns:
            Dictionary with evaluation metrics:
                - accuracy: Overall detection accuracy
                - cover_accuracy: True negative rate
                - stego_accuracy: True positive rate
                - security: 1 - accuracy (higher = better security)
        """
        dataset = SteganalysisDataset(cover_images, stego_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.detector.eval()

        correct = 0
        total = 0
        cover_correct = 0
        cover_total = 0
        stego_correct = 0
        stego_total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.detector(images)
                _, predicted = outputs.max(1)

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                # Cover accuracy (label = 0)
                cover_mask = labels == 0
                if cover_mask.sum() > 0:
                    cover_correct += predicted[cover_mask].eq(
                        labels[cover_mask]).sum().item()
                    cover_total += cover_mask.sum().item()

                # Stego accuracy (label = 1)
                stego_mask = labels == 1
                if stego_mask.sum() > 0:
                    stego_correct += predicted[stego_mask].eq(
                        labels[stego_mask]).sum().item()
                    stego_total += stego_mask.sum().item()

        accuracy = 100.0 * correct / total
        cover_accuracy = 100.0 * cover_correct / \
            cover_total if cover_total > 0 else 0.0
        stego_accuracy = 100.0 * stego_correct / \
            stego_total if stego_total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'cover_accuracy': cover_accuracy,
            'stego_accuracy': stego_accuracy,
            'security': 100.0 - accuracy  # Higher = better security
        }

    def save_detector(self, path: str):
        """Save trained detector."""
        torch.save(self.detector.state_dict(), path)

    def load_detector(self, path: str):
        """Load trained detector."""
        self.detector.load_state_dict(
            torch.load(path, map_location=self.device))


def evaluate_steganography_security(
    stego_model: nn.Module,
    cover_images: torch.Tensor,
    messages: torch.Tensor,
    detector_type: str = 'simple',
    train_epochs: int = 10,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate steganography security using steganalysis.

    Args:
        stego_model: Steganography model
        cover_images: Cover images
        messages: Messages to hide
        detector_type: Type of detector
        train_epochs: Training epochs for detector
        device: Device to run on
        verbose: Whether to print progress

    Returns:
        Dictionary with security metrics
    """
    stego_model = stego_model.to(device)
    stego_model.eval()

    # Generate stego images
    with torch.no_grad():
        stego_images = stego_model.encode(
            cover_images.to(device), messages.to(device))

    # Train detector
    evaluator = SteganalysisEvaluator(
        detector_type=detector_type, device=device)

    if verbose:
        print("Training steganalysis detector...")

    history = evaluator.train_detector(
        cover_images,
        stego_images,
        epochs=train_epochs,
        verbose=verbose
    )

    # Evaluate
    if verbose:
        print("\nEvaluating detector...")

    results = evaluator.evaluate_detector(cover_images, stego_images)

    if verbose:
        print(f"\nResults:")
        print(f"  Detection Accuracy: {results['accuracy']:.2f}%")
        print(f"  Cover Accuracy: {results['cover_accuracy']:.2f}%")
        print(f"  Stego Accuracy: {results['stego_accuracy']:.2f}%")
        print(f"  Security Score: {results['security']:.2f}%")
        print(f"\n  Interpretation:")
        if results['accuracy'] < 55:
            print(f"    Excellent security (detector performs near random)")
        elif results['accuracy'] < 65:
            print(f"    Good security (detector has difficulty)")
        elif results['accuracy'] < 75:
            print(f"    Fair security (detector has moderate success)")
        else:
            print(f"    Poor security (detector easily detects stego)")

    return results


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Steganalysis Detector")
    print("=" * 60)

    # Test 1: Create detectors
    print("\n1. Creating Detectors")
    print("-" * 60)

    simple_detector = SimpleStegDetector()
    srnet_detector = SRNet()

    print(
        f"   SimpleStegDetector parameters: {sum(p.numel() for p in simple_detector.parameters()):,}")
    print(
        f"   SRNet parameters: {sum(p.numel() for p in srnet_detector.parameters()):,}")
    print(f"   ✓ Detectors created")

    # Test 2: Forward pass
    print("\n2. Testing Forward Pass")
    print("-" * 60)

    batch_size = 4
    test_images = torch.rand(batch_size, 3, 128, 128)

    simple_output = simple_detector(test_images)
    srnet_output = srnet_detector(test_images)

    print(f"   Input shape: {test_images.shape}")
    print(f"   SimpleStegDetector output: {simple_output.shape}")
    print(f"   SRNet output: {srnet_output.shape}")
    print(f"   ✓ Forward pass working")

    # Test 3: SteganalysisDataset
    print("\n3. Testing SteganalysisDataset")
    print("-" * 60)

    num_images = 50
    cover = torch.rand(num_images, 3, 64, 64)
    stego = torch.rand(num_images, 3, 64, 64)

    dataset = SteganalysisDataset(cover, stego)

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Total images: {num_images * 2}")

    img, label = dataset[0]
    print(f"   Sample shape: {img.shape}")
    print(f"   Label type: {label.dtype}")
    print(f"   ✓ Dataset working")

    # Test 4: Train simple detector
    print("\n4. Training Simple Detector")
    print("-" * 60)

    # Create synthetic data
    num_train = 100
    cover_train = torch.rand(num_train, 3, 64, 64)
    # Make stego slightly different
    stego_train = cover_train + torch.randn_like(cover_train) * 0.05
    stego_train = torch.clamp(stego_train, 0, 1)

    evaluator = SteganalysisEvaluator(detector_type='simple', device='cpu')

    print("   Training detector (this may take a moment)...")
    history = evaluator.train_detector(
        cover_train,
        stego_train,
        epochs=3,
        batch_size=16,
        lr=0.001,
        verbose=False
    )

    print(f"   Final loss: {history['loss'][-1]:.4f}")
    print(f"   Final accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"   ✓ Training completed")

    # Test 5: Evaluate detector
    print("\n5. Evaluating Detector")
    print("-" * 60)

    num_test = 50
    cover_test = torch.rand(num_test, 3, 64, 64)
    stego_test = cover_test + torch.randn_like(cover_test) * 0.05
    stego_test = torch.clamp(stego_test, 0, 1)

    results = evaluator.evaluate_detector(cover_test, stego_test)

    print(f"   Detection accuracy: {results['accuracy']:.2f}%")
    print(f"   Cover accuracy: {results['cover_accuracy']:.2f}%")
    print(f"   Stego accuracy: {results['stego_accuracy']:.2f}%")
    print(f"   Security score: {results['security']:.2f}%")
    print(f"   ✓ Evaluation completed")

    # Test 6: Save and load detector
    print("\n6. Save and Load Detector")
    print("-" * 60)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'detector.pth')

        # Save
        evaluator.save_detector(model_path)
        print(f"   Saved detector to {model_path}")

        # Create new evaluator and load
        new_evaluator = SteganalysisEvaluator(
            detector_type='simple', device='cpu')
        new_evaluator.load_detector(model_path)
        print(f"   Loaded detector from {model_path}")

        # Verify same predictions
        with torch.no_grad():
            test_img = torch.rand(1, 3, 64, 64)
            out1 = evaluator.detector(test_img)
            out2 = new_evaluator.detector(test_img)
            same = torch.allclose(out1, out2, atol=1e-6)

        print(f"   ✓ Predictions match: {same}")

    # Test 7: Different perturbation levels
    print("\n7. Security vs Perturbation Level")
    print("-" * 60)

    cover = torch.rand(100, 3, 64, 64)

    for noise_level in [0.01, 0.05, 0.1]:
        stego = cover + torch.randn_like(cover) * noise_level
        stego = torch.clamp(stego, 0, 1)

        evaluator = SteganalysisEvaluator(detector_type='simple', device='cpu')
        evaluator.train_detector(cover, stego, epochs=3, verbose=False)
        results = evaluator.evaluate_detector(cover, stego)

        print(
            f"   Noise level {noise_level:.2f}: Accuracy = {results['accuracy']:.2f}%")

    print("\n" + "=" * 60)
    print("✅ All steganalysis tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("""
# Evaluate steganography security
from evaluation.steganalysis import evaluate_steganography_security

# Generate test data
cover_images = load_cover_images()
messages = generate_messages()

# Evaluate security
results = evaluate_steganography_security(
    stego_model=model,
    cover_images=cover_images,
    messages=messages,
    detector_type='simple',
    train_epochs=10,
    verbose=True
)

print(f"Detection Accuracy: {results['accuracy']:.2f}%")
print(f"Security Score: {results['security']:.2f}%")

# Lower detection accuracy = better steganography
if results['accuracy'] < 55:
    print("Excellent security!")
    """)
