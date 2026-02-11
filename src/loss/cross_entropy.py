import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from typing import Optional

def calc_loss_batch(input_batch:torch.Tensor, target_batch:torch.Tensor, model:nn.Module, device:torch.device):
    """
    Calculate cross entropy loss in a batch.

    Parameters
    ----------
    input_batch (torch.Tensor):
        Tensor containing batched model inputs.
    target_batch (torch.Tensor):
        Tensor containing batched targets.
    model (nn.Module):
        Language model
    device (torch.device):
        Device to calculate loss in. 

    Returns
    -------
    Tensor with the calculated batch loss.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader: DataLoader, model:nn.Module, device: torch.device, num_batches:Optional[int] = None):
    """ 
    Returns the average loss across all batches in a data loader.
    
    Parameters
    ----------
    data_loader (DataLoader):
        Data loader containing model inputs and targets.
    model (nn.Module):
        Language model.
    device (torch.device):
        Device to calculate loss in.
    num_batches (Optional[int]):
        Number of batches in the data. Reduced to number of batches in the data loader if it exceeds the total number of batches in the data loader.

    Returns:
    --------
    float:
        Average loss across batches.
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float ("nan")
    elif num_batches is None:
        num_batches = len(data_loader)      # Itereatives over all batches if no fixed num_batches specified
    else:
        num_batches = min(num_batches, len(data_loader)) # Reduces num_batches to match the total number of batches in the data loader if num_batches exceeds ir

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches