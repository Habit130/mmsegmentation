# Linux 4090 Server Runbook

This repository now includes a server bootstrap script at `server/setup.sh`.
It is designed for a single Linux server with one RTX 4090 and follows the repository's documented installation path:

1. create a conda environment
2. install PyTorch
3. install `openmim`, `mmengine`, and `mmcv`
4. install `mmsegmentation` in editable mode
5. verify imports and config loading

## Default behavior

Running `bash server/setup.sh` will:

- bootstrap Miniforge into `$HOME/.local/miniforge3` if `conda` is unavailable
- create a repo-local conda environment at `.conda-env/mmseg-1.2.2-cu121`
- install PyTorch `2.1.2` and torchvision `0.16.2` from the CUDA 12.1 wheel index
- install `mmengine>=0.5.0,<1.0.0`
- install `mmcv>=2.0.0rc4,<2.2.0`
- install this repository with `pip install -e .`
- validate `torch`, `mmcv`, `mmengine`, `mmseg`, and a representative config load

## Usage

Core environment only:

```bash
bash server/setup.sh
```

Include optional dependencies:

```bash
bash server/setup.sh --with-optional
```

Include test dependencies:

```bash
bash server/setup.sh --with-tests
```

Include multimodal dependencies:

```bash
bash server/setup.sh --with-multimodal
```

CPU-only bootstrap:

```bash
bash server/setup.sh --cpu
```

## Environment overrides

You can override the default install locations and versions with environment variables:

```bash
MINIFORGE_DIR=/opt/miniforge3 \
CONDA_ENV_PREFIX=/data/envs/mmseg-cu121 \
PYTHON_VERSION=3.9 \
TORCH_VERSION=2.1.2 \
TORCHVISION_VERSION=0.16.2 \
CUDA_TAG=cu121 \
bash server/setup.sh
```

## Manual prerequisites

- NVIDIA driver must already be installed on the server.
- For GPU mode, the driver must be compatible with CUDA 12.1 wheels.
- If the server lacks `sudo` access, system packages from `apt-get` may need to be installed manually.
- Dataset preparation and custom config selection remain outside this setup script.

## Training and evaluation

After setup:

```bash
source "$HOME/.local/miniforge3/etc/profile.d/conda.sh"
conda activate "$(pwd)/.conda-env/mmseg-1.2.2-cu121"
```

Single-GPU training:

```bash
bash tools/dist_train.sh <CONFIG> 1
```

Single-GPU evaluation:

```bash
bash tools/dist_test.sh <CONFIG> <CHECKPOINT> 1
```
