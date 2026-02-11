import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import tiktoken

from src.loss.cross_entropy import calc_loss_batch, calc_loss_loader
from src.utils.token_converter import text_to_token_ids, token_ids_to_text
from src.utils.generate import generate_text_simple


def generate_and_print_sample(model:nn.Module, tokenizer:tiktoken.core.Encoding, device:torch.device, start_context:str):
    """
    Given a start context, generate text using a language model.

    Parameters
    ----------
    model (nn.Module):
        Language model.
    tokenizer (tiktoken.core.Encoding):
        Model tokenizer.
    device (troch.device):
        Device to generate in.
    start_context (str)
        Text to serve as start context for the model.

    Returns
    -------
    None
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, 
            max_new_tokens=50, context_size=context_size
            )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def evaluate_model(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, device:torch.device, eval_iter:int):
    """
    Calculate model training and validation losses.

    Parameters
    ----------
    model (nn.Module):
        Language model.
    train_loader (DataLoader):
        DataLoader with training data.
    val_loader (DataLoader):
        Dataloader with validation data.
    device (torch.device):
        Device to procees training and validation data.
    eval_iter (int):
        Number of batches to calculate the loss on.

    Returns
    -------
    train_loss (float):
        Average training loss across batches.
    val_loss (float):
        Average validation loss across batches.
    """
    model.eval()    # Disable dropout during evaluation for stable, reproducible results
    with torch.no_grad():   # Disable gradient tracking to reduce computational overhead
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, 
                       optimizer:torch.optim.Optimizer, device:torch.device, num_epochs:int, 
                       eval_freq:int, eval_iter:int, start_context:str, tokenizer:tiktoken.core.Encoding):
    """
    Simple model training function for a language model. 

    Parameters
    ----------
    model (nn.Module):
        Language model.
    train_loader (DataLoader):
        DataLoader with training data.
    val_loader (DataLoader):
        Dataloader with validation data.
    optimizer (torch.optim.Optimizer):
        Optimizer used for training.
    device (torch.device):
        Device to procees training and validation data.
    num_epochs (int):
        Number of epochs to train the model for.
    eval_freq (int):
        Epoch frequency in which to evaluate the model.
    eval_iter (int):
        Number of batches to calculate the loss on.
    start_context (str)
        Text to serve as start context for the model.
    tokenizer (tiktoken.core.Encoding):
        Model tokenizer.    
    """
    train_losses, val_losses, track_tokens_seen = [], [], []        # Initialize lists to track losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()       # Resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()             # Calculates loss gradients
            optimizer.step()            # Updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.3f}, "
                      f"Val loss: {val_loss:.3f}"
                )
        
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen