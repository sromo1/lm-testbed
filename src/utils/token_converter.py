import torch
import tiktoken

def text_to_token_ids(text:str, tokenizer:tiktoken.core.Encoding):
    """
    Convert text to token IDs using a tokenizer.

    Parameters
    ----------
    text (str):
        Text to convert.
    tokenizer (tiktoken.core.Encoding):
        Tokenizer.

    Returns
    -------
    torch.Tensor
        Tensor containing token IDs.
    """ 
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)         # Adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids:torch.Tensor, tokenizer:tiktoken.core.Encoding):
    """
    Convert token IDs to text using a tokenizer.

    Parameters
    ----------
    token_ids (torch.Tensor):
        Tensor containing token IDs.
    tokenizer (tiktoken.core.Encoding):
        Tokenizer.
    
    Returns
    -------
    str
        Converted text.
    """
    flat = token_ids.squeeze(0)     # Removes the batch dimension
    decoded = tokenizer.decode(flat.tolist())
    return decoded