"""
Training Focused on Combined Attacks
Target: Improve ALL Combined accuracy from 77% to 85%+
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.model import StegoModel
from PIL import Image
import os
import glob


class DIV2KDataset(Dataset):
    def __init__(self, image_dir, message_length, image_size=128, max_images=200):
        self.message_length = message_length
        self.image_size = image_size

        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(
                image_dir, '**', f'*{ext}'), recursive=True))

        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])

        print(f"  Loaded {len(self.image_paths)} DIV2K images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        message = torch.randint(0, 2, (self.message_length,)).float()
        return image, message


def train_combined_attacks(model, dataloader, optimizer, scheduler, device, max_epochs=30):
    """
    Train specifically on combined attacks with increasing difficulty.
    """

    print(f"\n{'='*80}")
    print(f"COMBINED ATTACKS TRAINING")
    print(f"Strategy: High probability of ALL attacks together")
    print(f"Target: 85%+ accuracy on combined attacks")
    print(f"{'='*80}\n")

    best_combined_accuracy = 0

    for epoch in range(max_epochs):
        model.train()

        # Progressive strategy: increase combined attack probability
        # Early epochs: 50% combined, 30% individual
        # Later epochs: 80% combined, 10% individual
        combined_prob = min(0.5 + (epoch / max_epochs) * 0.3, 0.8)

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_pixel_delta = 0
        num_batches = 0

        print(f"\n{'─'*80}")
        print(
            f"Epoch {epoch+1}/{max_epochs} | Combined attacks prob: {combined_prob:.2f}")
        print(f"{'─'*80}")

        # Custom distortion: VERY high probability of combined attacks
        def forward_combined_focus(images, apply_all=False, jpeg_only=False):
            if model.distortions.training:
                import random
                if random.random() < combined_prob:
                    # Apply ALL attacks with high probability
                    images = model.distortions.apply_jpeg_compression(
                        images, probability=0.9)
                    images = model.distortions.apply_gaussian_blur_attack(
                        images, probability=0.9)
                    images = model.distortions.apply_resize_attack(
                        images, probability=0.9)
                    images = model.distortions.apply_color_jitter_attack(
                        images, probability=0.9)
                else:
                    # Occasionally just 2-3 attacks
                    images = model.distortions.apply_jpeg_compression(
                        images, probability=0.5)
                    images = model.distortions.apply_gaussian_blur_attack(
                        images, probability=0.5)
                    images = model.distortions.apply_resize_attack(
                        images, probability=0.5)
                    images = model.distortions.apply_color_jitter_attack(
                        images, probability=0.5)
            return images

        original_forward = model.distortions.forward
        model.distortions.forward = forward_combined_focus

        for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            optimizer.zero_grad()

            # Use higher beta to emphasize message recovery
            loss_dict = model.compute_loss(
                cover_images, binary_messages, alpha=0.8, beta=2.5)

            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                outputs = model(cover_images, binary_messages,
                                apply_distortions=False)
                stego = outputs['stego_image']
                pixel_delta = torch.mean(
                    torch.abs(stego - cover_images)).item()

            accuracy = loss_dict['bit_accuracy'].item()

            epoch_loss += loss_dict['total_loss'].item()
            epoch_accuracy += accuracy
            epoch_pixel_delta += pixel_delta
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:3d}/{len(dataloader):3d} | "
                      f"Loss: {loss_dict['total_loss'].item():.4f} | "
                      f"Acc: {accuracy*100:6.2f}% | "
                      f"Δ: {pixel_delta:.6f}")

        model.distortions.forward = original_forward

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_pixel_delta = epoch_pixel_delta / num_batches

        print(f"\n  {'EPOCH SUMMARY':^76}")
        print(f"  {'-'*76}")
        print(
            f"  Accuracy: {avg_accuracy*100:6.2f}% | Pixel Δ: {avg_pixel_delta:.6f}")
        print(f"  Loss: {avg_loss:.6f}")

        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.2e}")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n  {'EVALUATION':^76}")
            print(f"  {'-'*76}")
            eval_results = quick_eval(model, dataloader, device)
            print(f"  Clean:   {eval_results['clean']*100:.2f}%")
            print(f"  ALL:     {eval_results['all']*100:.2f}% ⭐")

            # Save if improved
            if eval_results['all'] > best_combined_accuracy:
                best_combined_accuracy = eval_results['all']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': avg_accuracy,
                    'pixel_delta': avg_pixel_delta,
                    'combined_accuracy': eval_results['all'],
                    'epoch': epoch
                }, 'checkpoints/model_BEST_COMBINED.pth')
                print(
                    f"  ✓ Saved improved combined model: {eval_results['all']*100:.2f}%")

        # Early stopping if target met
        if epoch > 10 and best_combined_accuracy >= 0.85:
            print(f"\n  ✓ Target achieved: {best_combined_accuracy*100:.2f}%!")
            break

    print(f"\n{'='*80}")
    print(f"COMBINED ATTACKS TRAINING COMPLETE")
    print(f"Best combined accuracy: {best_combined_accuracy*100:.2f}%")
    print(f"{'='*80}\n")


def quick_eval(model, dataloader, device, num_batches=5):
    """Quick evaluation."""
    model.encoder.eval()
    model.decoder.eval()
    model.distortions.train()

    original_forward = model.distortions.forward

    # Test clean and ALL combined
    results = {}

    with torch.no_grad():
        # Clean
        total_acc = 0
        count = 0
        model.distortions.forward = lambda x, apply_all=False, jpeg_only=False: x

        for batch_idx, (images, messages) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            messages = messages.to(device)
            outputs = model(images, messages, apply_distortions=True)
            decoded = outputs['decoded_message']
            decoded_bits = (decoded > 0.5).float()
            acc = (decoded_bits == messages).float().mean().item()
            total_acc += acc
            count += 1

        results['clean'] = total_acc / count if count > 0 else 0

        # ALL combined
        total_acc = 0
        count = 0

        def forward_all(x, apply_all=False, jpeg_only=False):
            x = model.distortions.apply_jpeg_compression(x, 1.0)
            x = model.distortions.apply_gaussian_blur_attack(x, 1.0)
            x = model.distortions.apply_resize_attack(x, 1.0)
            x = model.distortions.apply_color_jitter_attack(x, 1.0)
            return x

        model.distortions.forward = forward_all

        for batch_idx, (images, messages) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            messages = messages.to(device)
            outputs = model(images, messages, apply_distortions=True)
            decoded = outputs['decoded_message']
            decoded_bits = (decoded > 0.5).float()
            acc = (decoded_bits == messages).float().mean().item()
            total_acc += acc
            count += 1

        results['all'] = total_acc / count if count > 0 else 0

    model.distortions.forward = original_forward
    model.train()

    return results


def main():
    print(f"\n{'#'*80}")
    print(f"#{'COMBINED ATTACKS TRAINING - IMPROVE ROBUSTNESS':^78}#")
    print(f"#{'Continue from: model_IMPROVED_BLUR.pth':^78}#")
    print(f"{'#'*80}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Config
    message_length = 16
    image_size = 128
    batch_size = 4
    num_images = 200
    max_epochs = 30

    # Load dataset
    print(f"Loading DIV2K dataset...")
    dataset = DIV2KDataset(
        'data/DIV2K/train', message_length, image_size, num_images)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    print(f"✓ Dataset ready\n")

    # Load existing model
    print(f"Loading model from: checkpoints/model_IMPROVED_BLUR.pth")
    model = StegoModel(message_length, image_size, enable_distortions=True)
    checkpoint = torch.load(
        'checkpoints/model_IMPROVED_BLUR.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Model loaded\n")

    # Very low learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3)

    # Train
    train_combined_attacks(model, dataloader, optimizer,
                           scheduler, device, max_epochs)

    print("Run evaluate_transformations.py for comprehensive results")


if __name__ == "__main__":
    main()
