"""
Quick Progressive Training - Trains with new distortions one by one
Shows: Loss, Accuracy, BER, Pixel Delta, PSNR for each batch
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.model import StegoModel
import os


class SimpleDataset(Dataset):
    def __init__(self, num_samples=100, image_size=128, message_length=16):
        self.num_samples = num_samples
        self.image_size = image_size
        self.message_length = message_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cover = torch.rand(3, self.image_size, self.image_size)
        message = torch.randint(0, 2, (self.message_length,)).float()
        return cover, message


def calc_psnr_fast(img1, img2):
    """Fast PSNR calculation."""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(torch.tensor(1.0 / (mse ** 0.5))).item()


def train_quick(model, dataloader, optimizer, device, phase_name, max_epochs=20, target_acc=0.80):
    """Quick training with detailed output."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {phase_name}")
    print(f"{'='*80}\n")

    best_acc = 0

    for epoch in range(max_epochs):
        model.train()
        total_acc = 0
        total_batches = 0

        for batch_idx, (cover, message) in enumerate(dataloader):
            cover, message = cover.to(device), message.to(device)

            # Forward + backward
            optimizer.zero_grad()

            # Get metrics before training step
            with torch.no_grad():
                stego = model.encode(cover, message)
                pixel_delta = torch.abs(stego - cover).mean().item()
                psnr = calc_psnr_fast(cover, stego)

            loss_dict = model.compute_loss(cover, message)
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc = loss_dict['bit_accuracy'].item()
            total_acc += acc
            total_batches += 1

            # Print every batch
            print(f"Epoch {epoch+1:2d} Batch {batch_idx+1:2d}/{len(dataloader)} | "
                  f"Loss: {loss_dict['total_loss'].item():.4f} | "
                  f"Acc: {acc*100:5.2f}% | "
                  f"BER: {loss_dict['ber'].item()*100:5.2f}% | "
                  f"Pixel Δ: {pixel_delta:.6f} | "
                  f"PSNR: {psnr:5.2f} dB")

        avg_acc = total_acc / total_batches
        print(f"\n→ Epoch {epoch+1} Average Accuracy: {avg_acc*100:.2f}%")

        if avg_acc > best_acc:
            best_acc = avg_acc

        if avg_acc >= target_acc:
            print(
                f"✓ TARGET REACHED: {avg_acc*100:.2f}% >= {target_acc*100:.0f}%\n")
            return best_acc

    print(f"✗ Max epochs reached. Best: {best_acc*100:.2f}%\n")
    return best_acc


def main():
    print("="*80)
    print("PROGRESSIVE DISTORTION TRAINING - QUICK VERSION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-detect message_length from checkpoint
    checkpoint_path = 'checkpoints/best_model_local.pth'
    message_length = 16  # default

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in ckpt:
            message_length = ckpt['model_state_dict']['decoder.message_extractor.fc3.bias'].shape[0]
        print(f"\n✓ Loading checkpoint (message_length={message_length})")
        del ckpt

    # Create model with distortions enabled
    model = StegoModel(message_length=message_length,
                       image_size=128, enable_distortions=True).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(
            f"  Loaded from epoch {ckpt['epoch']}, accuracy: {ckpt['metrics']['bit_accuracy']*100:.2f}%")

        # Backup original
        backup = checkpoint_path.replace('.pth', '_BACKUP.pth')
        if not os.path.exists(backup):
            torch.save(ckpt, backup)
            print(f"  ✓ Original backed up to: {os.path.basename(backup)}")

    # Dataset
    train_data = SimpleDataset(
        num_samples=100, image_size=128, message_length=message_length)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    print(
        f"\nConfig: {len(train_data)} samples, batch_size=4, image_size=128x128")
    print(f"Device: {device}")

    # ============================================================================
    # PHASE 1: Add Gaussian Blur
    # ============================================================================
    print(f"\n{'#'*80}")
    print("PHASE 1: ADDING GAUSSIAN BLUR")
    print(f"{'#'*80}")

    def forward_blur_only(self, images, apply_all=False, jpeg_only=False):
        if self.training:
            images = self.apply_gaussian_blur_attack(images, probability=0.5)
        return images

    model.distortions.forward = lambda *args, **kwargs: forward_blur_only(
        model.distortions, *args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    acc1 = train_quick(model, train_loader, optimizer, device,
                       "Gaussian Blur", max_epochs=15, target_acc=0.80)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 'blur',
        'accuracy': acc1
    }, 'checkpoints/model_with_blur.pth')
    print(
        f"✓ Saved: checkpoints/model_with_blur.pth (Accuracy: {acc1*100:.2f}%)\n")

    # ============================================================================
    # PHASE 2: Add Resize Attack
    # ============================================================================
    print(f"\n{'#'*80}")
    print("PHASE 2: ADDING RESIZE ATTACK (+ Blur)")
    print(f"{'#'*80}")

    def forward_blur_resize(self, images, apply_all=False, jpeg_only=False):
        if self.training:
            images = self.apply_gaussian_blur_attack(images, probability=0.4)
            images = self.apply_resize_attack(images, probability=0.4)
        return images

    model.distortions.forward = lambda *args, **kwargs: forward_blur_resize(
        model.distortions, *args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    acc2 = train_quick(model, train_loader, optimizer, device,
                       "Blur + Resize", max_epochs=15, target_acc=0.80)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 'blur_resize',
        'accuracy': acc2
    }, 'checkpoints/model_with_blur_resize.pth')
    print(
        f"✓ Saved: checkpoints/model_with_blur_resize.pth (Accuracy: {acc2*100:.2f}%)\n")

    # ============================================================================
    # PHASE 3: Add Color Jitter
    # ============================================================================
    print(f"\n{'#'*80}")
    print("PHASE 3: ADDING COLOR JITTER (+ Blur + Resize)")
    print(f"{'#'*80}")

    def forward_all(self, images, apply_all=False, jpeg_only=False):
        if self.training:
            images = self.apply_gaussian_blur_attack(images, probability=0.3)
            images = self.apply_resize_attack(images, probability=0.3)
            images = self.apply_color_jitter_attack(images, probability=0.3)
        return images

    model.distortions.forward = lambda *args, **kwargs: forward_all(
        model.distortions, *args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    acc3 = train_quick(model, train_loader, optimizer, device,
                       "Blur + Resize + Color Jitter", max_epochs=20, target_acc=0.80)

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 'all_distortions',
        'accuracy': acc3
    }, 'checkpoints/model_with_all_new_distortions_FINAL.pth')
    print(
        f"✓ Saved: checkpoints/model_with_all_new_distortions_FINAL.pth (Accuracy: {acc3*100:.2f}%)\n")

    # ============================================================================
    # FINAL TEST
    # ============================================================================
    print(f"\n{'='*80}")
    print("FINAL ROBUSTNESS TEST")
    print(f"{'='*80}\n")

    model.eval()
    test_cover = torch.rand(1, 3, 128, 128).to(device)
    test_msg = torch.randint(0, 2, (1, message_length)).float().to(device)

    with torch.no_grad():
        stego = model.encode(test_cover, test_msg)

        # Clean
        dec_clean = model.decode(stego)
        acc_clean = (dec_clean == test_msg).float().mean().item()

        model.distortions.train()

        # Blur
        blurred = model.distortions.apply_gaussian_blur_attack(stego, 1.0)
        acc_blur = (model.decode(blurred) == test_msg).float().mean().item()

        # Resize
        resized = model.distortions.apply_resize_attack(stego, 1.0)
        acc_resize = (model.decode(resized) == test_msg).float().mean().item()

        # Color jitter
        jittered = model.distortions.apply_color_jitter_attack(stego, 1.0)
        acc_jitter = (model.decode(jittered) == test_msg).float().mean().item()

        # All
        attacked = model.distortions.apply_gaussian_blur_attack(stego, 1.0)
        attacked = model.distortions.apply_resize_attack(attacked, 1.0)
        attacked = model.distortions.apply_color_jitter_attack(attacked, 1.0)
        acc_all = (model.decode(attacked) == test_msg).float().mean().item()

    print(f"Attack Type          | Accuracy  | Status")
    print(f"{'-'*80}")
    print(
        f"Clean (no attack)    | {acc_clean*100:5.2f}%   | {'✓ PASS' if acc_clean >= 0.8 else '✗ FAIL'}")
    print(
        f"Gaussian Blur        | {acc_blur*100:5.2f}%   | {'✓ PASS' if acc_blur >= 0.8 else '✗ FAIL'}")
    print(
        f"Resize Attack        | {acc_resize*100:5.2f}%   | {'✓ PASS' if acc_resize >= 0.8 else '✗ FAIL'}")
    print(
        f"Color Jitter         | {acc_jitter*100:5.2f}%   | {'✓ PASS' if acc_jitter >= 0.8 else '✗ FAIL'}")
    print(
        f"All Combined         | {acc_all*100:5.2f}%   | {'✓ PASS' if acc_all >= 0.8 else '✗ FAIL'}")

    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Phase 1 - Blur:          {acc1*100:.2f}%")
    print(f"Phase 2 - + Resize:      {acc2*100:.2f}%")
    print(f"Phase 3 - + Color Jitter: {acc3*100:.2f}%")

    if acc3 >= 0.80:
        print(
            f"\n🎉 SUCCESS! Final accuracy {acc3*100:.2f}% exceeds 80% target!")
    else:
        print(
            f"\n⚠ Continue training to reach 80% target (current: {acc3*100:.2f}%)")

    print(f"\n✓ Original model preserved: checkpoints/best_model_local_BACKUP.pth")
    print("✓ All phase checkpoints saved")


if __name__ == "__main__":
    main()
