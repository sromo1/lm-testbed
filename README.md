# lm-workbench
A workbench for language models

## Repo Structure:
```
lm-testbed/
├── data/                  # Datasets (ignored by git)
├── models/                # Saved weights and model checkpoints
├── src/                   # Core architecture source code
│   ├── layers/            # Custom attention, norm, etc. (paper implementations)
│   ├── model.py           # The SLM (Small Language Model) architecture
│   └── trainer.py         # Training loop and optimization logic
├── scripts/               # Single-run experiment and utility scripts
├── pyproject.toml         # Project dependencies managed by uv
└── Dockerfile             # Reproducible compute environment configuration
```

## Launch the Container (The "Testbed")

This command mounts your local code into the container so you can edit scripts in VS Code/Cursor and run them instantly on the GPU.

```bash
docker run --gpus all -it --rm \
    --shm-size=2g \
    -v $(pwd):/app \
    slm-research
```

## AI Research Companion Prompt
```
Role: You are an expert AI Research Engineer specializing in PyTorch and Transformer architectures.

Task: Help me implement a modular Small Language Model (SLM) from scratch for research purposes. The goal is to create a "testbed" where I can easily swap out different attention mechanisms (Standard Dot-Product, Flash Attention, Grouped Query Attention) and architectural components (RoPE vs. Learned Positional Embeddings) to see how they affect training.

Infrastructure & Specs:

    Hardware: NVIDIA GTX 1660 Ti (6GB VRAM) — implementations must be VRAM-optimized (use 4-bit/8-bit if needed, or small parameter counts like 50M-100M).

    OS/Env: Ubuntu 24.04 running inside a Docker container (nvcr.io/nvidia/pytorch base).

    Package Manager: uv (using pyproject.toml and uv run).

    Repo Structure:

        src/layers/: Custom implementations of Attention, Norms, etc.

        src/model.py: The core Transformer class that accepts component types as arguments.

        scripts/: Training and benchmarking scripts.

Coding Standard:

    Use clean, modular PyTorch code (nn.Module).

    Implement components "from scratch" where possible (manual tensor ops) before moving to optimized versions.

    Use Configuration Classes (e.g., dataclass) to define model variants.
```

## Implementing Your First Paper (Pro-Tip)

When you start implementing a paper (e.g., FlashAttention or Rotary Embeddings):

1. Draft in src/layers/: Create a file like src/layers/rope.py.

2. Import in src/model.py: from src.layers.rope import RotaryEmbedding.

3. Run with uv: Use uv run to ensure all your version-locked dependencies are used.
