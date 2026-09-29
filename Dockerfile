# PyTorch cu121 wheels supply their CUDA user-space libraries.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS base

ARG MINIFORGE_VERSION=25.3.0-3
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl bzip2 libgl1 libglib2.0-0 libgomp1 \
        libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH
WORKDIR /opt/MolSteer
COPY . .

FROM base AS agent
RUN conda env create -f environment.agent.yml && conda clean -afy
ENV PATH=/opt/conda/envs/molsteer-agent/bin:/opt/conda/bin:$PATH
CMD ["python", "-m", "molsteer.cli", "--help"]

FROM base AS flowr
RUN test -f flowr_root/flowr/__init__.py && \
    conda env create -f environment.yml && conda clean -afy
ENV PATH=/opt/conda/envs/molsteer-flowr/bin:/opt/conda/bin:$PATH \
    FLOWR_ROOT=/opt/MolSteer/flowr_root \
    PYTHONPATH=/opt/MolSteer/flowr_root
CMD ["python", "-m", "molsteer.cli", "--help"]
