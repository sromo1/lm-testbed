DUMMY_GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 1024,     # Context length
    "emb_dim": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads
    "n_layers": 12,             # Number of layers
    "drop_rate": 0.1,           # Drop Rate
    "qkv_bias": False           # Query-Key-Value bias
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 1024,      # Context length
    "emb_dim": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads
    "n_layers": 12,             # Number of layers
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,
    "qkv_bias": False           # Query-Key-Value bias
}
