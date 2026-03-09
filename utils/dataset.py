"""
Dataset utilities for deep learning steganography.

Supports loading images from various directory structures:
- DIV2K: High-resolution images with patch-based loading
- COCO: images/train2017/, images/val2017/
- ImageNet: train/class_name/, val/class_name/
- BOSSBase: flat directory with .pgm or other image files
- Generic: any folder with images

Returns (image_tensor, random_binary_message) pairs for training.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SteganographyDataset(Dataset):
    """
    Dataset for steganography training.

    Loads images from directory and generates random binary messages.
    Supports multiple directory structures and image formats, including
    DIV2K with patch-based loading.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        message_length: int = 1024,
        normalize: bool = True,
        augment: bool = False,
        dataset_type: str = 'auto',
        extensions: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        use_patches: bool = False,
        patches_per_image: int = 4,
        random_crop: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing images
            image_size: Target size for images (patch size if use_patches=True)
            message_length: Length of binary message to generate
            normalize: Whether to normalize images to [0, 1] range
            augment: Whether to apply data augmentation
            dataset_type: Type of dataset structure ('auto', 'DIV2K', 'coco', 'imagenet', 'flat')
            extensions: List of valid image extensions (default: common formats)
            max_images: Maximum number of BASE images to load (not total patches)
            use_patches: If True, extract multiple patches per image (for DIV2K)
            patches_per_image: Number of patches to extract per image
            random_crop: If True, use random crops; if False, use center crop
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.message_length = message_length
        self.normalize = normalize
        self.augment = augment
        self.dataset_type = dataset_type.lower()
        self.use_patches = use_patches
        self.patches_per_image = patches_per_image
        self.random_crop = random_crop

        # Default image extensions (prioritize .png for DIV2K)
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg',
                          '.bmp', '.pgm', '.ppm', '.tiff', '.tif']
        self.extensions = [ext.lower() for ext in extensions]

        # Find all image paths
        self.image_paths = self._collect_image_paths()

        # Shuffle for random sampling
        random.shuffle(self.image_paths)

        # Limit number of BASE images if specified (not total patches)
        if max_images is not None and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]
            print(
                f"Loaded {len(self.image_paths)} images from {root_dir} (limited to {max_images} for local GPU)")
        else:
            print(f"Loaded {len(self.image_paths)} images from {root_dir}")

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {root_dir} with extensions {extensions}")

        # Calculate total dataset size with patches
        if use_patches:
            self.total_samples = len(self.image_paths) * patches_per_image
            print(f"✓ Patch-based loading enabled:")
            print(f"  - Patches per image: {patches_per_image}")
            print(f"  - Base images: {len(self.image_paths)}")
            print(f"  - Total training samples: {self.total_samples}")
            print(f"  - Patches are randomly sampled each epoch for diversity")
        else:
            self.total_samples = len(self.image_paths)

        # Setup transforms
        self.transform = self._build_transform()

    def _collect_image_paths(self) -> List[Path]:
        """Collect all image paths based on dataset type."""

        if self.dataset_type == 'auto':
            # Auto-detect dataset structure
            self.dataset_type = self._detect_dataset_type()
            print(f"Auto-detected dataset type: {self.dataset_type}")

        if self.dataset_type == 'div2k':
            return self._collect_div2k_images()
        elif self.dataset_type == 'coco':
            return self._collect_coco_images()
        elif self.dataset_type == 'imagenet':
            return self._collect_imagenet_images()
        elif self.dataset_type == 'bossbase' or self.dataset_type == 'flat':
            return self._collect_flat_images()
        else:
            # Default: recursively find all images
            return self._collect_recursive_images()

    def _detect_dataset_type(self) -> str:
        """Auto-detect dataset structure."""

        # Check for DIV2K structure
        root_name = self.root_dir.name.lower()
        if 'div2k' in root_name or 'DIV2K' in str(self.root_dir):
            return 'div2k'

        # Check for COCO structure
        if (self.root_dir / 'train2017').exists() or (self.root_dir / 'val2017').exists():
            return 'coco'

        # Check for ImageNet structure (multiple class directories)
        subdirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if len(subdirs) > 10:  # Likely ImageNet-style if many subdirectories
            # Check if subdirs contain images
            for subdir in subdirs[:5]:
                files = list(subdir.iterdir())
                if any(f.suffix.lower() in self.extensions for f in files):
                    return 'imagenet'

        # Check for flat structure (BOSSBase style)
        files = list(self.root_dir.glob('*'))
        image_files = [f for f in files if f.is_file(
        ) and f.suffix.lower() in self.extensions]
        if len(image_files) > 0:
            return 'flat'

        # Default to recursive search
        return 'recursive'

    def _collect_div2k_images(self) -> List[Path]:
        """Collect images from DIV2K structure."""
        image_paths = []

        # DIV2K has flat structure with high-res PNG images
        # Look for common DIV2K directories
        for split in ['train', 'valid', 'DIV2K_train_HR', 'DIV2K_valid_HR']:
            split_dir = self.root_dir / split
            if split_dir.exists():
                paths = [f for f in split_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in self.extensions]
                image_paths.extend(paths)
                print(f"  Found {len(paths)} images in {split}")

        # If no subdirectories, check root directly
        if len(image_paths) == 0:
            paths = [f for f in self.root_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in self.extensions]
            image_paths.extend(paths)

        return sorted(image_paths)

    def _collect_coco_images(self) -> List[Path]:
        """Collect images from COCO structure."""
        image_paths = []

        # Check common COCO directories
        for split in ['train2017', 'val2017', 'test2017', 'train2014', 'val2014']:
            split_dir = self.root_dir / split
            if split_dir.exists():
                paths = [f for f in split_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in self.extensions]
                image_paths.extend(paths)
                print(f"  Found {len(paths)} images in {split}")

        return sorted(image_paths)

    def _collect_imagenet_images(self) -> List[Path]:
        """Collect images from ImageNet structure (class subdirectories)."""
        image_paths = []

        # Look for train/val directories
        for split in ['train', 'val', 'test']:
            split_dir = self.root_dir / split
            if split_dir.exists():
                # Iterate through class directories
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        paths = [f for f in class_dir.iterdir()
                                 if f.is_file() and f.suffix.lower() in self.extensions]
                        image_paths.extend(paths)
                print(f"  Found {len(image_paths)} images in {split}")

        # If no train/val structure, look directly in root subdirectories
        if len(image_paths) == 0:
            for class_dir in self.root_dir.iterdir():
                if class_dir.is_dir():
                    paths = [f for f in class_dir.iterdir()
                             if f.is_file() and f.suffix.lower() in self.extensions]
                    image_paths.extend(paths)

        return sorted(image_paths)

    def _collect_flat_images(self) -> List[Path]:
        """Collect images from flat directory structure."""
        image_paths = [f for f in self.root_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in self.extensions]
        return sorted(image_paths)

    def _collect_recursive_images(self) -> List[Path]:
        """Recursively collect all images."""
        image_paths = []
        for ext in self.extensions:
            image_paths.extend(self.root_dir.rglob(f'*{ext}'))
        return sorted(image_paths)

    def _build_transform(self) -> Callable:
        """Build image transformation pipeline."""
        transform_list = []

        # For patch-based loading (DIV2K), use crop instead of resize
        if self.use_patches:
            # Extract patches from high-resolution images
            if self.random_crop:
                transform_list.append(transforms.RandomCrop(self.image_size))
            else:
                transform_list.append(transforms.CenterCrop(self.image_size))
        else:
            # Standard resize for smaller images
            transform_list.append(transforms.Resize(
                (self.image_size, self.image_size)))

        # Data augmentation (if enabled and not using patches)
        # Note: patches already provide augmentation via random crops
        if self.augment and not self.use_patches:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
            ])

        # Convert to tensor (automatically normalizes to [0, 1])
        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)

    def _generate_random_message(self) -> torch.Tensor:
        """Generate random binary message."""
        return torch.randint(0, 2, (self.message_length,), dtype=torch.float32)

    def __len__(self) -> int:
        """Return dataset size (total patches if patch-based, else number of images)."""
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (image_tensor, random_binary_message) pair.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, random_binary_message):
                - image_tensor: Tensor of shape (3, image_size, image_size), normalized to [0, 1]
                - random_binary_message: Binary tensor of shape (message_length,)
        """
        # Map sample index to image index
        if self.use_patches:
            image_idx = idx // self.patches_per_image
            # Not used, but could be for debugging
            patch_idx = idx % self.patches_per_image
        else:
            image_idx = idx

        # Load image
        image_path = self.image_paths[image_idx]

        try:
            image = Image.open(image_path).convert('RGB')

            # For patch-based loading, ensure image is large enough
            if self.use_patches:
                width, height = image.size
                if width < self.image_size or height < self.image_size:
                    # If image is smaller than patch size, resize it first
                    scale = max(self.image_size / width,
                                self.image_size / height)
                    new_width = int(width * scale) + 1
                    new_height = int(height * scale) + 1
                    image = image.resize(
                        (new_width, new_height), Image.BICUBIC)

        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # Return a random other image
            return self.__getitem__((idx + 1) % len(self))

        # Apply transforms (crop/resize and normalize to [0,1])
        image_tensor = self.transform(image)

        # Generate random binary message
        random_binary_message = self._generate_random_message()

        return image_tensor, random_binary_message

    def get_image_path(self, idx: int) -> Path:
        """Get path of image at index."""
        if self.use_patches:
            image_idx = idx // self.patches_per_image
        else:
            image_idx = idx
        return self.image_paths[image_idx]


# Utility function
def create_dataloader(
    root_dir: str,
    batch_size: int = 16,
    image_size: int = 256,
    message_length: int = 1024,
    normalize: bool = True,
    augment: bool = False,
    dataset_type: str = 'auto',
    num_workers: int = 4,
    shuffle: bool = True,
    max_images: Optional[int] = None,
    use_patches: bool = False,
    patches_per_image: int = 4,
    random_crop: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for steganography training.

    Args:
        root_dir: Root directory containing images
        batch_size: Batch size
        image_size: Target image size (patch size if use_patches=True)
        message_length: Length of binary message
        normalize: Whether to normalize images
        augment: Whether to apply data augmentation
        dataset_type: Type of dataset structure ('auto', 'DIV2K', 'coco', 'imagenet', 'flat')
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        max_images: Maximum number of BASE images to load (not total patches)
        use_patches: Enable patch-based loading (recommended for DIV2K)
        patches_per_image: Number of patches to extract per image
        random_crop: Use random crops (True) or center crop (False)

    Returns:
        DataLoader instance
    """
    dataset = SteganographyDataset(
        root_dir=root_dir,
        image_size=image_size,
        message_length=message_length,
        normalize=normalize,
        augment=augment,
        dataset_type=dataset_type,
        max_images=max_images,
        use_patches=use_patches,
        patches_per_image=patches_per_image,
        random_crop=random_crop
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader


# Quick test
if __name__ == "__main__":
    print("Testing DIV2K-integrated dataset...")

    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        for i in range(5):
            img = Image.fromarray(np.random.randint(
                0, 255, (512, 512, 3), dtype=np.uint8))
            img.save(tmpdir / f"test_{i}.jpg")

        # Test generic loading
        dataset = SteganographyDataset(
            root_dir=str(tmpdir),
            image_size=128,
            message_length=32,
            dataset_type='flat'
        )

        image, message = dataset[0]
        print(f"✓ Generic: image {image.shape}, message {message.shape}")

        # Test patch-based loading
        dataset_patches = SteganographyDataset(
            root_dir=str(tmpdir),
            image_size=128,
            message_length=32,
            dataset_type='flat',
            use_patches=True,
            patches_per_image=4,
            max_images=3
        )

        image, message = dataset_patches[0]
        print(
            f"✓ Patches: image {image.shape}, message {message.shape}, total samples {len(dataset_patches)}")

        print("✅ All tests passed!")
