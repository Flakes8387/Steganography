import torch

ckpt = torch.load('checkpoints/best_model_local.pth', map_location='cpu')
state_dict = ckpt['model_state_dict']

print("Encoder PrepNetwork architecture from checkpoint:")
for key in sorted(state_dict.keys()):
    if 'prep_network' in key:
        print(f"  {key}: {state_dict[key].shape}")
