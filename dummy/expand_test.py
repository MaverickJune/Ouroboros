import torch

tgt_len = 3
past_key_values_length = 2
bsz = 2

mask = torch.full((tgt_len, tgt_len), -float('inf'))  # Shape: (3,3)
mask = mask[None, None, :, :]  # Shape: (1, 1, 3, 3)
expanded_mask = mask.expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

print(expanded_mask.shape)  # Output: torch.Size([2, 1, 3, 5])