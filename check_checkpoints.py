import torch
import os

checkpoints = [
    'checkpoints/best_model_local.pth',
    'checkpoints/checkpoint_latest.pth', 
    'checkpoints/checkpoint_epoch_10.pth'
]

for cp in checkpoints:
    if os.path.exists(cp):
        ckpt = torch.load(cp, map_location='cpu')
        print(f'\n{cp}:')
        print(f'  Keys: {list(ckpt.keys())}')
        if 'epoch' in ckpt:
            print(f'  Epoch: {ckpt["epoch"]}')
        if 'metrics' in ckpt:
            print(f'  Metrics: {ckpt["metrics"]}')
        
        # Check model state dict for message_length hints
        state_dict = ckpt['model_state_dict']
        for key in state_dict.keys():
            if 'fc1.weight' in key and 'prep_network' in key:
                shape = state_dict[key].shape
                print(f'  Message length (from {key}): {shape[1]}')
                break
    else:
        print(f'{cp}: not found')
