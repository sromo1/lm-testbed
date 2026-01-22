# Use NVIDIA's optimized PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install uv from the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1
# Use copy mode to avoid linking issues with Docker volumes
ENV UV_LINK_MODE=copy

# Install dependencies first (for better Docker caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy the rest of your research code
COPY . .

# Default command: Open a bash shell for experimenting
CMD ["/bin/bash"]