# PyTorch cu121 wheels supply CUDA user-space libraries; Miniforge supplies Conda.
FROM condaforge/miniforge3:25.3.0-3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates libgl1 libglib2.0-0 libgomp1 \
        libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH
WORKDIR /opt/MolSteer
COPY environment.yml environment.agent.yml ./

FROM base AS agent
RUN conda env create -f environment.agent.yml && conda clean -afy
ENV PATH=/opt/conda/envs/molsteer-agent/bin:/opt/conda/bin:$PATH
COPY . .
RUN python -m pip install --no-deps -e .
CMD ["python", "-m", "molsteer.cli", "--help"]

FROM base AS flowr
RUN conda env create -f environment.yml && conda clean -afy
ENV PATH=/opt/conda/envs/molsteer-flowr/bin:/opt/conda/bin:$PATH \
    FLOWR_ROOT=/opt/MolSteer/flowr_root \
    PYTHONPATH=/opt/MolSteer/flowr_root
COPY . .
RUN test -f flowr_root/flowr/__init__.py && python -m pip install --no-deps -e .
CMD ["python", "-m", "molsteer.cli", "--help"]
