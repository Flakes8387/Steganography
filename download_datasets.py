"""
Download and prepare large datasets for steganography training.

Supports:
- DIV2K (800 high-quality images)
- COCO (100K+ images)
- ImageNet subset
- BOSSBase (10K grayscale images for steganalysis research)

Usage:
    python download_datasets.py --dataset div2k --output data/div2k
    python download_datasets.py --dataset coco --output data/coco --split train
    python download_datasets.py --dataset bossbase --output data/bossbase
"""

import argparse
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm
import shutil


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading from {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path}")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def download_div2k(output_dir):
    """
    Download DIV2K dataset.
    - 800 high-quality 2K resolution images
    - Excellent for training steganography models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        'train': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'valid': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
    }

    for split, url in urls.items():
        print(f"\n{'='*60}")
        print(f"Downloading DIV2K {split} set")
        print('='*60)

        zip_path = output_dir / f'DIV2K_{split}_HR.zip'

        if not zip_path.exists():
            try:
                download_url(url, str(zip_path))
            except Exception as e:
                print(f"✗ Error downloading: {e}")
                print(f"Please manually download from: {url}")
                continue

        # Extract
        extract_to = output_dir / split
        extract_to.mkdir(exist_ok=True)
        extract_archive(str(zip_path), str(extract_to))

        # Cleanup zip
        if zip_path.exists():
            zip_path.unlink()
            print(f"✓ Cleaned up {zip_path}")

    print(f"\n✅ DIV2K dataset ready at: {output_dir}")
    print(
        f"   Train images: {len(list((output_dir / 'train').rglob('*.png')))}")
    print(
        f"   Valid images: {len(list((output_dir / 'valid').rglob('*.png')))}")


def download_bossbase(output_dir):
    """
    Download BOSSBase dataset.
    - 10,000 grayscale images (512×512)
    - Standard dataset for steganalysis research
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("BOSSBase Dataset Information")
    print('='*60)
    print("BOSSBase is a standard dataset for steganalysis research.")
    print("It contains 10,000 grayscale images at 512×512 resolution.")
    print()
    print("Download instructions:")
    print("1. Visit: http://agents.fel.cvut.cz/boss/index.php?mode=VIEW&tmpl=materials")
    print("2. Register and request download access")
    print("3. Download BOSSbase_1.01.zip")
    print(f"4. Extract to: {output_dir}")
    print()
    print("Due to registration requirements, this cannot be automatically downloaded.")


def setup_coco(output_dir, split='train'):
    """
    Set up COCO dataset.
    Note: COCO is very large (~18GB for train2017)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"COCO Dataset Setup - {split} split")
    print('='*60)

    year = '2017'
    url = f'http://images.cocodataset.org/zips/{split}{year}.zip'
    zip_path = output_dir / f'{split}{year}.zip'

    print(f"Dataset size: ~18GB for train, ~1GB for val")
    print(f"This will take significant time and bandwidth.")
    print()
    response = input("Continue with download? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("Download cancelled.")
        print(f"To manually download: {url}")
        return

    # Download
    if not zip_path.exists():
        try:
            download_url(url, str(zip_path))
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            print(f"Please manually download from: {url}")
            return

    # Extract
    extract_archive(str(zip_path), str(output_dir))

    # Cleanup
    if zip_path.exists():
        zip_path.unlink()
        print(f"✓ Cleaned up {zip_path}")

    images_dir = output_dir / f'{split}{year}'
    print(f"\n✅ COCO {split} dataset ready at: {images_dir}")
    print(f"   Images: {len(list(images_dir.glob('*.jpg')))}")


def download_sample_dataset(output_dir, num_images=1000):
    """
    Download a smaller sample dataset for quick testing.
    Uses Unsplash random images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {num_images} sample images")
    print('='*60)

    try:
        from PIL import Image
        import requests
        from io import BytesIO
    except ImportError:
        print("✗ Required packages: pip install Pillow requests")
        return

    print("Downloading random high-quality images from Unsplash...")

    for i in range(num_images):
        try:
            # Unsplash Source API - random images
            url = f"https://source.unsplash.com/random/512x512?sig={i}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img.save(output_dir / f'image_{i:04d}.jpg', 'JPEG', quality=95)

                if (i + 1) % 10 == 0:
                    print(f"Downloaded {i + 1}/{num_images} images")
            else:
                print(f"⚠ Failed to download image {i}")

        except Exception as e:
            print(f"⚠ Error downloading image {i}: {e}")
            continue

    print(f"\n✅ Sample dataset ready at: {output_dir}")
    print(f"   Images: {len(list(output_dir.glob('*.jpg')))}")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for steganography training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download DIV2K (recommended for training)
    python download_datasets.py --dataset div2k --output data/div2k
    
    # Download COCO train set (large)
    python download_datasets.py --dataset coco --output data/coco --split train
    
    # Download small sample dataset for testing
    python download_datasets.py --dataset sample --output data/sample --num-images 500
    
    # Show BOSSBase download instructions
    python download_datasets.py --dataset bossbase --output data/bossbase
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['div2k', 'coco', 'bossbase', 'sample'],
                        help='Dataset to download')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for dataset')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split (for COCO)')
    parser.add_argument('--num-images', type=int, default=1000,
                        help='Number of images for sample dataset')

    args = parser.parse_args()

    print("="*60)
    print("Dataset Downloader for Steganography Training")
    print("="*60)

    if args.dataset == 'div2k':
        download_div2k(args.output)
    elif args.dataset == 'coco':
        setup_coco(args.output, args.split)
    elif args.dataset == 'bossbase':
        download_bossbase(args.output)
    elif args.dataset == 'sample':
        download_sample_dataset(args.output, args.num_images)

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Verify dataset at: {args.output}")
    print("2. Start training:")
    print(
        f"   python train.py --train_dir {args.output} --num_epochs 50 --batch_size 16")
    print("3. Monitor with TensorBoard:")
    print("   tensorboard --logdir runs/")


if __name__ == '__main__':
    main()
