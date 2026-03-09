"""
Quick script to check the configuration of a saved model
"""
import torch

checkpoint_path = 'checkpoints/best_model_local.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 60)
print("Model Checkpoint Information")
print("=" * 60)

# Check available keys
print("\nAvailable keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Check if config exists
if 'config' in checkpoint:
    print("\nModel Configuration:")
    config = checkpoint['config']
    for key, value in config.items():
        print(f"  {key}: {value}")

# Check model state dict to infer message length
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    
    # Check encoder fc1 weight shape
    if 'encoder.prep_network.fc1.weight' in state_dict:
        fc1_shape = state_dict['encoder.prep_network.fc1.weight'].shape
        print(f"\nEncoder fc1 weight shape: {fc1_shape}")
        print(f"Inferred message_length: {fc1_shape[1]}")
    
    # Check decoder fc3 bias shape
    if 'decoder.message_extractor.fc3.bias' in state_dict:
        fc3_bias_shape = state_dict['decoder.message_extractor.fc3.bias'].shape
        print(f"\nDecoder fc3 bias shape: {fc3_bias_shape}")
        print(f"Inferred message_length from decoder: {fc3_bias_shape[0]}")

# Check metrics
if 'metrics' in checkpoint:
    print("\nTraining Metrics:")
    metrics = checkpoint['metrics']
    for key, value in metrics.items():
        print(f"  {key}: {value}")

# Check epoch
if 'epoch' in checkpoint:
    print(f"\nEpoch: {checkpoint['epoch']}")

print("\n" + "=" * 60)
