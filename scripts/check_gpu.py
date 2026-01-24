import torch
import torch.nn as nn

print(f"Cuda is available: {torch.cuda.is_available()}")

# Test a simple attention-like operation
x = torch.randn(1, 10, 512).cuda()
m = nn.Linear(512, 512).cuda()
out = m(x)

print(f"Compute successful! Output shape: {out.shape}")
print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")