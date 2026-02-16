import torch
import torch.nn as nn
from typing import Optional

def generate_text_simple(model:nn.Module, idx:torch.Tensor, max_new_tokens:int, context_size:int):
    """
    Generate text using a transformers model.
    
    Parameters
    ----------
    model : nn.Module 
        Transformers module whose inputs and outputs are a tensor of embeddings.
    idx : torch.Tensor
        Input tensor with token id's.
    max_new_tokens : int
        Number of new tokens to generate.
    context_size : int
        Context window size to feed into the model

    Returns
    -------
    torch.Tensor
        Tensor containing the idx token id'sequence with the predicted tokens appended.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                       # Crops current context if it exceeds the supported context size,
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]                               # Last token so (batch, n_token, vocab_size) -> (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)                  # Shape (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   # Shape (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)                 # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
    return idx


def generate(model:nn.Module, idx:torch.Tensor, max_new_tokens:int, context_size:int,
             temperature:float=0.0, top_k:Optional[int]=None, eos_id:Optional[int]=None):
    """
    Text generation function with more diversity via temperature and top_k

    Praameters
    ----------
    model : nn.Module 
        Transformers module whose inputs and outputs are a tensor of embeddings.
    idx : torch.Tensor
        Input tensor with token id's.
    max_new_tokesn : int
        Number of new tokens to generate.
    context_size : int
        Context window size to feed into the model
    temperature : float, default: 0.0
        Temperature value for logit scaling
    top_k : optional, int, default: None
        Top_k value for sampling
    eos_id : optional, int, default: None
        End-of-sequence token id

    Returns
    -------
    torch.Tensor
        Tensor containing the idx token id'sequence with the predicted tokens appended.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                       # Crops current context if it exceeds the supported context size
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                               # Last token so (batch, n_token, vocab_size) -> (batch, vocab_size)
    
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)           # Filter logits with top_k sampling
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )
        
        if temperature > 0.0:                                       # Apply temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)                   # Shape (batch, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)   # Greedy next-token selection if temperature is disabled

        if idx_next == eos_id:                                  # Stop generating early if end-of-sequence token is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)                 # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)

    return idx
