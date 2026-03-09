import torch

checkpoint_path = 'checkpoints/best_model_local.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")
