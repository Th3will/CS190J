FROM ubuntu:latest

# If we ever need to change the version just update this to the new URL
# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv .venv
ENV PATH="/.venv/bin:$PATH"
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
RUN pip install torch_geometric
RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html