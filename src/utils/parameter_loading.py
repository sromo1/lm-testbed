import torch

def assign(left:torch.Tensor, right:torch.Tensor):
    """ Check if two tensors have the same shape and return right tensor as trainable PyTorch parameters"""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         f"Right: {right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right))

