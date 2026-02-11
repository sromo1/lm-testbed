import torch
import torch.nn as nn

def generate_text_simple(model:nn.Module, idx:torch.Tensor, max_new_tokens:int, context_size:int):
    """
    Generate text using a transformers model.
    
    Parameters
    ----------
    model (nn.Module): 
        Transformers module whose inputs and outputs are a tensor of embeddings.
    idx (torch.Tensor)
        Input tensor with token id's.
    max_new_tokesn (int):
        Number of new tokens to generate.
    context_size (int):
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