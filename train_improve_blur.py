"""
Focused Blur Training - Continue from ALL_TRANSFORMATIONS model
Specifically targets Gaussian Blur weakness without hurting other transformations
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
    """DIV2K Dataset loader."""

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


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(torch.tensor(1.0 / (mse ** 0.5))).item()


def train_blur_focused(model, dataloader, optimizer, scheduler, device, max_epochs=50):
    """
    Focused training on Gaussian Blur with gradual increase in difficulty.
    Uses curriculum learning: start with mild blur, increase intensity.
    """

    print(f"\n{'='*80}")
    print(f"FOCUSED GAUSSIAN BLUR TRAINING")
    print(f"Strategy: Curriculum learning with gradual blur intensity increase")
    print(f"Target: 85%+ accuracy on blur while maintaining 84%+ on others")
    print(f"{'='*80}\n")

    best_blur_accuracy = 0
    best_overall_accuracy = 0

    for epoch in range(max_epochs):
        model.train()

        # Curriculum learning: increase blur probability over epochs
        blur_prob = min(0.3 + (epoch / max_epochs) * 0.4, 0.7)  # 0.3 -> 0.7
        other_prob = 0.2  # Keep other transformations active but lower

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_pixel_delta = 0
        num_batches = 0

        print(f"\n{'─'*80}")
        print(f"Epoch {epoch+1}/{max_epochs} | Blur prob: {blur_prob:.2f}")
        print(f"{'─'*80}")

        # Custom distortion function emphasizing blur
        def forward_blur_emphasis(images, apply_all=False, jpeg_only=False):
            if model.distortions.training:
                # High probability for blur
                images = model.distortions.apply_gaussian_blur_attack(
                    images, probability=blur_prob)
                # Lower probability for others (to maintain robustness)
                images = model.distortions.apply_jpeg_compression(
                    images, probability=other_prob)
                images = model.distortions.apply_resize_attack(
                    images, probability=other_prob)
                images = model.distortions.apply_color_jitter_attack(
                    images, probability=other_prob)
            return images

        original_forward = model.distortions.forward
        model.distortions.forward = forward_blur_emphasis

        for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                cover_images, binary_messages, alpha=1.0, beta=2.0)

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

        # Periodic evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n  {'EVALUATION':^76}")
            print(f"  {'-'*76}")
            eval_results = quick_eval(model, dataloader, device)
            print(f"  Clean:   {eval_results['clean']*100:.2f}%")
            print(f"  JPEG:    {eval_results['jpeg']*100:.2f}%")
            print(f"  Blur:    {eval_results['blur']*100:.2f}% ⭐")
            print(f"  Resize:  {eval_results['resize']*100:.2f}%")
            print(f"  Jitter:  {eval_results['jitter']*100:.2f}%")
            print(f"  ALL:     {eval_results['all']*100:.2f}%")

            # Save if blur improved
            if eval_results['blur'] > best_blur_accuracy:
                best_blur_accuracy = eval_results['blur']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': avg_accuracy,
                    'pixel_delta': avg_pixel_delta,
                    'blur_accuracy': eval_results['blur'],
                    'epoch': epoch
                }, 'checkpoints/model_IMPROVED_BLUR.pth')
                print(
                    f"  ✓ Saved improved blur model: {eval_results['blur']*100:.2f}%")

        # Check if pixel delta is drifting
        if avg_pixel_delta > 0.025:
            print(f"  ⚠ Pixel delta increasing: {avg_pixel_delta:.6f}")

        # Early stopping if targets met
        if epoch > 20 and avg_accuracy >= 0.85 and avg_pixel_delta <= 0.022:
            print(f"\n  ✓ Training converged!")
            break

    print(f"\n{'='*80}")
    print(f"BLUR-FOCUSED TRAINING COMPLETE")
    print(f"Best blur accuracy: {best_blur_accuracy*100:.2f}%")
    print(f"{'='*80}\n")


def quick_eval(model, dataloader, device, num_batches=5):
    """Quick evaluation on subset of data."""
    model.encoder.eval()
    model.decoder.eval()
    model.distortions.train()

    results = {}

    # Test each attack (lambdas must accept all forward() parameters)
    attacks = {
        'clean': lambda x, apply_all=False, jpeg_only=False: x,
        'jpeg': lambda x, apply_all=False, jpeg_only=False: model.distortions.apply_jpeg_compression(x, 1.0),
        'blur': lambda x, apply_all=False, jpeg_only=False: model.distortions.apply_gaussian_blur_attack(x, 1.0),
        'resize': lambda x, apply_all=False, jpeg_only=False: model.distortions.apply_resize_attack(x, 1.0),
        'jitter': lambda x, apply_all=False, jpeg_only=False: model.distortions.apply_color_jitter_attack(x, 1.0),
        'all': lambda x, apply_all=False, jpeg_only=False: model.distortions.apply_color_jitter_attack(
            model.distortions.apply_resize_attack(
                model.distortions.apply_gaussian_blur_attack(
                    model.distortions.apply_jpeg_compression(x, 1.0), 1.0), 1.0), 1.0)
    }

    original_forward = model.distortions.forward

    with torch.no_grad():
        for attack_name, attack_fn in attacks.items():
            total_acc = 0
            count = 0

            model.distortions.forward = attack_fn

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

            results[attack_name] = total_acc / count if count > 0 else 0

    model.distortions.forward = original_forward
    model.train()

    return results


def main():
    print(f"\n{'#'*80}")
    print(f"#{'FOCUSED BLUR TRAINING - IMPROVE ROBUSTNESS':^78}#")
    print(f"#{'Continue from: model_ALL_TRANSFORMATIONS_95_DIV2K.pth':^78}#")
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
    max_epochs = 50

    # Load dataset
    print(f"Loading DIV2K dataset...")
    dataset = DIV2KDataset(
        'data/DIV2K/train', message_length, image_size, num_images)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    print(f"✓ Dataset ready\n")

    # Load existing model
    print(f"Loading model from: checkpoints/model_ALL_TRANSFORMATIONS_95_DIV2K.pth")
    model = StegoModel(message_length, image_size, enable_distortions=True)
    checkpoint = torch.load(
        'checkpoints/model_ALL_TRANSFORMATIONS_95_DIV2K.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Model loaded (baseline: {checkpoint['accuracy']*100:.2f}%)\n")

    # Use lower learning rate for fine-tuning
    # Lower LR for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5)

    # Train
    train_blur_focused(model, dataloader, optimizer,
                       scheduler, device, max_epochs)

    # Final evaluation
    print("Running final comprehensive evaluation...")
    print("(Use evaluate_transformations.py for detailed results)")


if __name__ == "__main__":
    main()
