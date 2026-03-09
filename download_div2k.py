"""
DIV2K Dataset Downloader

Automatically downloads and extracts the DIV2K dataset for training.
Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    print(f"\n[DOWNLOAD] Download started")
    print(f"[DOWNLOAD] URL: {url}")
    print(f"[DOWNLOAD] Destination: {output_path}")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to)

        print(f"\n[SUCCESS] Download completed: {output_path}")
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Network error: {e}")
        print("[ERROR] Please check your internet connection and try again.")
        raise
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        raise


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file with progress bar.

    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to
    """
    print(f"\n[EXTRACT] Extracting dataset...")
    print(f"[EXTRACT] From: {zip_path}")
    print(f"[EXTRACT] To: {extract_dir}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_dir)
                    pbar.update(1)

        print(f"\n[SUCCESS] Extraction completed")
    except zipfile.BadZipFile:
        print(f"\n[ERROR] Invalid or corrupted ZIP file: {zip_path}")
        print("[ERROR] Please delete the ZIP file and try again.")
        raise
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        raise


def verify_dataset(dataset_dir):
    """
    Verify that dataset was downloaded and extracted correctly.

    Args:
        dataset_dir: Directory containing the dataset

    Returns:
        bool: True if dataset is valid
    """
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return False

    # Look for image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(dataset_dir.rglob(f'*{ext}')))

    num_images = len(image_files)

    if num_images == 0:
        print(f"[ERROR] No images found in {dataset_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Dataset verified!")
    print(f"  Location: {dataset_dir}")
    print(f"  Images found: {num_images}")
    print(f"{'='*60}")

    return True


def download_div2k(output_dir="data", keep_zip=False):
    """
    Download and extract DIV2K dataset.

    Args:
        output_dir: Base directory for datasets (default: "data")
        keep_zip: Whether to keep the ZIP file after extraction
    """
    # Setup paths (relative to script location)
    script_dir = Path(__file__).parent
    base_dir = script_dir / output_dir
    div2k_dir = base_dir / "DIV2K"
    train_dir = div2k_dir / "train"
    zip_path = div2k_dir / "DIV2K_train_HR.zip"

    # Create directories
    div2k_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DIV2K Dataset Downloader")
    print("="*60)
    print(f"Dataset will be saved to: {train_dir}")
    print()

    # Check if already downloaded
    if train_dir.exists() and any(train_dir.iterdir()):
        print("\n[INFO] DIV2K dataset already exists!")
        if verify_dataset(train_dir):
            response = input(
                "\nDataset is valid. Re-download? (y/n): ").lower()
            if response != 'y':
                print("\n[INFO] Using existing dataset - skipping download.")
                return str(train_dir.resolve())

    # Download URL (HTTPS for secure connection)
    url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"

    # Download
    if not zip_path.exists():
        try:
            download_url(url, zip_path)
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            print("\n[INFO] Alternative download options:")
            print("  1. Manual download from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
            print(f"  2. Save to: {zip_path}")
            print("  3. Re-run this script")
            sys.exit(1)
    else:
        print(
            f"\n[INFO] ZIP file already exists - skipping download: {zip_path}")

    # Extract
    try:
        extract_zip(zip_path, div2k_dir)
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        sys.exit(1)

    # Find the extracted directory (DIV2K creates DIV2K_train_HR folder)
    extracted_dir = div2k_dir / "DIV2K_train_HR"

    # Move to train directory if needed
    if extracted_dir.exists() and extracted_dir != train_dir:
        print(f"\n[INFO] Moving images to: {train_dir}")
        if train_dir.exists():
            import shutil
            shutil.rmtree(train_dir)
        extracted_dir.rename(train_dir)
        print(f"[SUCCESS] Files moved to {train_dir}")

    # Verify
    if not verify_dataset(train_dir):
        print("\n[ERROR] Dataset verification failed!")
        sys.exit(1)

    # Cleanup
    if not keep_zip and zip_path.exists():
        print(f"\n[CLEANUP] Removing ZIP file: {zip_path}")
        zip_path.unlink()
        print("[SUCCESS] Cleanup complete")

    print("\n" + "="*60)
    print("[SUCCESS] DIV2K dataset ready for training!")
    print("="*60)
    print(f"\nDataset location: {train_dir}")
    print(f"\nTo start training, run:")
    print(f"  python train.py --config config_div2k_balanced.yaml")
    print(f"\nOr manually:")
    print(f"  python train.py --train_dir {train_dir} --dataset_type DIV2K")
    print()

    return str(train_dir.resolve())


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download DIV2K dataset for steganography training')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Base directory for datasets (default: data)')
    parser.add_argument('--keep_zip', action='store_true',
                        help='Keep ZIP file after extraction')

    args = parser.parse_args()

    try:
        train_dir = download_div2k(args.output_dir, args.keep_zip)
        print(f"\n[SUCCESS] Setup complete! Dataset path: {train_dir}")
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Download cancelled by user")
        print("[INFO] Partial files may remain in data/DIV2K/")
        print("[INFO] Run the script again to resume or restart")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
