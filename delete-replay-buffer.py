import torch
# deleting replay buffer before pushing

checkpoint_path = "checkpoints/checkpoint_3500.pth"
new_path = "checkpoints/checkpoint_3500.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

if 'replay_buffer' in checkpoint:
    del checkpoint['replay_buffer']

torch.save(checkpoint, new_path)
print("Completed!")