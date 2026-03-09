"""
Quick test to verify training setup works.

Creates a small synthetic dataset and runs a few training iterations
to ensure everything is configured correctly.
"""

import shutil
import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

print("=" * 60)
print("Training Setup Verification Test")
print("=" * 60)

# Create temporary directory for test images
temp_dir = tempfile.mkdtemp()
train_dir = Path(temp_dir) / "train"
train_dir.mkdir(parents=True, exist_ok=True)

print(f"\n1. Creating synthetic test images in {train_dir}...")
# Create 20 random test images
for i in range(20):
    # Create random RGB image
    img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(train_dir / f"test_image_{i:03d}.jpg")

print(f"   Created 20 test images")

# Test imports
print("\n2. Testing imports...")
try:
    from models.model import StegoModel
    print("   ✓ StegoModel imported")
except Exception as e:
    print(f"   ✗ Failed to import StegoModel: {e}")
    exit(1)

try:
    from attacks import JPEGCompression, GaussianNoise
    print("   ✓ Attack modules imported")
except Exception as e:
    print(f"   ✗ Failed to import attacks: {e}")
    exit(1)

# Test dataset
print("\n3. Testing dataset loading...")


class TestDataset(Dataset):
    def __init__(self, image_dir, message_length=1024, image_size=256):
        self.image_paths = glob.glob(str(Path(image_dir) / "*.jpg"))
        self.message_length = message_length
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        message = torch.randint(0, 2, (self.message_length,)).float()
        return image, message


dataset = TestDataset(train_dir, message_length=512, image_size=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"   ✓ Dataset loaded: {len(dataset)} images")
print(f"   ✓ DataLoader created: {len(dataloader)} batches")

# Test model
print("\n4. Testing model initialization...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

model = StegoModel(
    message_length=512,
    image_size=128,
    enable_distortions=True
)
model = model.to(device)
print(f"   ✓ Model initialized")

params = model.get_num_parameters()
print(f"   Total parameters: {params['total']:,}")

# Test forward pass
print("\n5. Testing forward pass...")
images, messages = next(iter(dataloader))
images = images.to(device)
messages = messages.to(device)

with torch.no_grad():
    outputs = model.forward(images, messages)

print(f"   ✓ Forward pass successful")
print(f"   Input shape: {images.shape}")
print(f"   Stego shape: {outputs['stego_image'].shape}")
print(f"   Decoded shape: {outputs['decoded_message'].shape}")

# Test loss computation
print("\n6. Testing loss computation...")
loss_dict = model.compute_loss(images, messages)
print(f"   ✓ Loss computed")
print(f"   Total loss: {loss_dict['total_loss'].item():.6f}")
print(f"   Image loss: {loss_dict['image_loss'].item():.6f}")
print(f"   Message loss: {loss_dict['message_loss'].item():.6f}")
print(f"   Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")

# Test backward pass
print("\n7. Testing backward pass...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

images, messages = next(iter(dataloader))
images = images.to(device)
messages = messages.to(device)

outputs = model.forward(images, messages)
loss_dict = model.compute_loss(images, messages)
loss = loss_dict['total_loss']

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"   ✓ Backward pass successful")
print(f"   Gradients computed and optimizer step completed")

# Test checkpoint saving
print("\n8. Testing checkpoint saving...")
checkpoint_dir = Path(temp_dir) / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_path)

print(f"   ✓ Checkpoint saved to {checkpoint_path}")

# Test checkpoint loading
print("\n9. Testing checkpoint loading...")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(f"   ✓ Checkpoint loaded successfully")

# Test mini training loop
print("\n10. Testing mini training loop (3 iterations)...")
model.train()

for i in range(3):
    images, messages = next(iter(dataloader))
    images = images.to(device)
    messages = messages.to(device)

    outputs = model.forward(images, messages)
    loss_dict = model.compute_loss(images, messages)
    loss = loss_dict['total_loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"   Iteration {i+1}: Loss={loss.item():.6f}, Acc={loss_dict['accuracy'].item()*100:.2f}%")

print(f"   ✓ Training loop working")

# Cleanup
print("\n11. Cleaning up temporary files...")
shutil.rmtree(temp_dir)
print(f"   ✓ Temporary files removed")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour training setup is ready to use!")
print("\nTo start training:")
print("  python train.py --train_dir ./your_images --num_epochs 10")
print("\nTo monitor training:")
print("  tensorboard --logdir ./runs")
print("=" * 60)
