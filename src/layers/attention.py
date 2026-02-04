import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    """
    Self attention layer that uses Parameter weight matrices
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key  = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
            )
        context_vec = attn_weights @ values
        return context_vec
    
class SelfAttention_v2(nn.Module):
    """
    Sef attention layer that uses Linear modules for weight matrices
    """
    def __init__(self, d_in:int, d_out:int, qkv_bias:bool=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
            )
        context_vec = attn_weights @ values
        return context_vec
    
class CausalAttention(nn.Module):
    """
    Attention layer with causal attention mask and dropout.
    Parameters
    ----------
    d_in : int
        Input dimension (emmbedding size)
    d_out : int
        Output dimension (query/key/value size)
    context_length : int
        Maximum number of tokens the model can handle (for mask initialization).
    dropout : float
        Fraction of attention weights to set to zero during training.
    qkv_bias : bool
        If True, adds a learnable bias to the query, key, and value projections.
    """
    def __init__(self, d_in:int, d_out:int, context_length:int, dropout:float, qkv_bias:bool=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), 
                                        diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass in the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        torch.Tensor
            Context vectors of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
    
class MultiHeadAttentionWrapper(nn.Module):
    """
    Wrapper to create multiple CausalAttention heads.

    Parameters
    ----------
    num_heads : int
        Number of attention heads to run in parallel.
    """
    def __init__(self, d_in:int, d_out:int, context_length:int,
                 dropout:float, num_heads:int, qkv_bias:bool=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass in multiple CausalAttention heads
        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing tokesns (rows) and embeddings (columns)
        Returns
        -------
        torch.Tensor
            Concatenated context vectors of shape (batch_size, num_tokens, d_out * num_heads).
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Implementation

    Parameters
    ----------
    num_heads : int
        Number of attention heads to run in parallel.
    """
    def __init__(self, d_in:int, d_out:int, context_length:int,
                 dropout:float, num_heads:int, qkv_bias:bool=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads          # Reduces the projection dim to match he desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)     # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length),
                                diagonal=1)
        )

    def forward(self, x:torch.Tensor):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)            #
        queries = self.W_query(x)       # Tensor shape: (b, num_tokens, d_out)
        values = self.W_value(x)        #

        # We implicitly split the matrix by adding a num_heads dimension. 
        # Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys:torch.Tensor = keys.transpose(1, 2)
        queries:torch.Tensor = queries.transpose(1, 2)
        values:torch.Tensor = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2,3)                 # Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]      # Masks truncated to the number of tokens

        attn_scores.masked_fill_(mask_bool, -torch.inf)             # Uses the mask to fill attention scores

        attn_weights:torch.Tensor = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec:torch.Tensor = (attn_weights @ values).transpose(1,2)    # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)       # Adds optional linear projection
        return context_vec

