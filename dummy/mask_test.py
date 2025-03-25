import torch

# Settings
lguess = 12
dtype = torch.float32
min_val = torch.finfo(dtype).min

# Create the mask
mask = torch.zeros((lguess, lguess), dtype=dtype)

# Create diagonal values
small_m1 = torch.tensor([0, 0, 0, 0, 0, min_val] * (lguess // 6), dtype=dtype)[:-1]
small_m2 = torch.tensor([0, 0, 0, 0, min_val, min_val] * (lguess // 6), dtype=dtype)[:-2]
small_m3 = torch.tensor([0, 0, 0, min_val, min_val, min_val] * (lguess // 6), dtype=dtype)[:-3]
small_m4 = torch.tensor([0, 0, min_val, min_val, min_val, min_val] * (lguess // 6), dtype=dtype)[:-4]
small_m5 = torch.tensor([0, min_val, min_val, min_val, min_val, min_val] * (lguess // 6), dtype=dtype)[:-5]

# Fill in the mask
mask.fill_diagonal_(0)
mask.diagonal(-1).copy_(small_m1)
mask.diagonal(-2).copy_(small_m2)
mask.diagonal(-3).copy_(small_m3)
mask.diagonal(-4).copy_(small_m4)
mask.diagonal(-5).copy_(small_m5)

print(mask)