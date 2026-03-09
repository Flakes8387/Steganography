"""
Infer all model parameters from checkpoint
"""
import torch
import math

checkpoint_path = 'checkpoints/best_model_local.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 60)
print("Inferring Model Parameters from Checkpoint")
print("=" * 60)

state_dict = checkpoint['model_state_dict']

# Message length from encoder fc1
if 'encoder.prep_network.fc1.weight' in state_dict:
    fc1_shape = state_dict['encoder.prep_network.fc1.weight'].shape
    message_length = fc1_shape[1]
    print(f"\nMessage Length: {message_length}")

# Image size from encoder fc3
if 'encoder.prep_network.fc3.weight' in state_dict:
    fc3_shape = state_dict['encoder.prep_network.fc3.weight'].shape
    # fc3 output is image_size * image_size * 4
    output_size = fc3_shape[0]
    image_size_squared_x4 = output_size
    image_size = int(math.sqrt(image_size_squared_x4 / 4))
    print(f"Image Size: {image_size}")
    print(f"  (from fc3 output size: {output_size} = {image_size}x{image_size}x4)")

print("\n" + "=" * 60)
print("Use these parameters:")
print(f"  --message_length {message_length}")
print(f"  --image_size {image_size}")
print("=" * 60)
