"""
Detailed checkpoint analysis
"""
import torch

checkpoint_path = 'checkpoints/best_model_local.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']

print("=" * 60)
print("Encoder Prep Network Layers:")
print("=" * 60)
for key in sorted(state_dict.keys()):
    if 'encoder.prep_network' in key and ('weight' in key or 'bias' in key):
        print(f"{key}: {state_dict[key].shape}")

print("\n" + "=" * 60)
print("Calculating image_size:")
print("=" * 60)

# fc3 output should be image_size * image_size
fc3_weight_shape = state_dict['encoder.prep_network.fc3.weight'].shape
fc3_output_size = fc3_weight_shape[0]
print(f"fc3 output size: {fc3_output_size}")

import math
image_size = int(math.sqrt(fc3_output_size))
print(f"image_size = sqrt({fc3_output_size}) = {image_size}")
print(f"Verification: {image_size}x{image_size} = {image_size*image_size}")
