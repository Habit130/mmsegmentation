#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/.local/miniforge3}"
ENV_PREFIX="${CONDA_ENV_PREFIX:-$REPO_ROOT/.conda-env/mmseg-1.2.2-cu121}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
CUDA_TAG="${CUDA_TAG:-cu121}"
MMENGINE_SPEC="${MMENGINE_SPEC:-mmengine>=0.5.0,<1.0.0}"
MMCV_SPEC="${MMCV_SPEC:-mmcv>=2.0.0rc4,<2.2.0}"
INSTALL_OPTIONAL=0
INSTALL_TESTS=0
INSTALL_MULTIMODAL=0
SKIP_APT=0

usage() {
  cat <<EOF
Usage: bash server/setup.sh [options]

Options:
  --env-prefix PATH       Override the conda env prefix.
  --miniforge-dir PATH    Override the Miniforge install path.
  --python VERSION        Override the Python version. Default: ${PYTHON_VERSION}
  --cpu                   Install CPU-only PyTorch wheels.
  --with-optional         Install requirements/optional.txt.
  --with-tests            Install requirements/tests.txt.
  --with-multimodal       Install requirements/multimodal.txt.
  --skip-apt              Skip apt package installation.
  -h, --help              Show this help message.

Environment overrides:
  MINIFORGE_DIR
  CONDA_ENV_PREFIX
  PYTHON_VERSION
  TORCH_VERSION
  TORCHVISION_VERSION
  CUDA_TAG
  MMENGINE_SPEC
  MMCV_SPEC
EOF
}

log() {
  printf '[server/setup] %s\n' "$*"
}

warn() {
  printf '[server/setup] WARNING: %s\n' "$*" >&2
}

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  if [[ -x "${MINIFORGE_DIR}/bin/conda" ]]; then
    printf '%s\n' "${MINIFORGE_DIR}/bin/conda"
    return 0
  fi
  return 1
}

bootstrap_miniforge() {
  local installer cache_dir
  cache_dir="${REPO_ROOT}/server/.cache"
  installer="${cache_dir}/Miniforge3-Linux-x86_64.sh"

  mkdir -p "${cache_dir}"

  if [[ ! -f "${installer}" ]]; then
    log "Downloading Miniforge installer to ${installer}"
    curl -L \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
      -o "${installer}"
  fi

  log "Installing Miniforge into ${MINIFORGE_DIR}"
  bash "${installer}" -b -p "${MINIFORGE_DIR}"
}

maybe_install_apt_packages() {
  if [[ "${SKIP_APT}" -eq 1 ]]; then
    log "Skipping apt packages by request"
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found, skipping system package installation"
    return 0
  fi

  local runner=()
  if [[ "${EUID}" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      warn "sudo not found, skipping apt packages"
      return 0
    fi
    runner=(sudo)
  fi

  log "Installing Linux runtime packages via apt-get"
  "${runner[@]}" apt-get update
  "${runner[@]}" apt-get install -y \
    git \
    curl \
    wget \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1
}

install_python_packages() {
  local conda_bin="$1"
  local torch_index_url

  if [[ "${CUDA_TAG}" == "cpu" ]]; then
    torch_index_url="https://download.pytorch.org/whl/cpu"
  else
    torch_index_url="https://download.pytorch.org/whl/${CUDA_TAG}"
  fi

  log "Upgrading pip and installing core Python build helpers"
  "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install -U pip setuptools wheel

  log "Installing PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} from ${torch_index_url}"
  "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${torch_index_url}"

  log "Installing OpenMIM, MMEngine and MMCV"
  "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install -U openmim
  "${conda_bin}" run -p "${ENV_PREFIX}" mim install "${MMENGINE_SPEC}"
  "${conda_bin}" run -p "${ENV_PREFIX}" mim install "${MMCV_SPEC}"

  log "Installing MMSegmentation in editable mode"
  "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install -e "${REPO_ROOT}"

  if [[ "${INSTALL_OPTIONAL}" -eq 1 ]]; then
    log "Installing optional dependencies"
    "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install -r "${REPO_ROOT}/requirements/optional.txt"
  fi

  if [[ "${INSTALL_TESTS}" -eq 1 ]]; then
    log "Installing test dependencies"
    "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install -r "${REPO_ROOT}/requirements/tests.txt"
  fi

  if [[ "${INSTALL_MULTIMODAL}" -eq 1 ]]; then
    log "Installing multimodal dependencies"
    "${conda_bin}" run -p "${ENV_PREFIX}" python -m pip install -r "${REPO_ROOT}/requirements/multimodal.txt"
  fi
}

validate_environment() {
  local conda_bin="$1"

  log "Validating imports and config loading"
  "${conda_bin}" run -p "${ENV_PREFIX}" python -c \
    "import torch, mmcv, mmengine, mmseg; \
from mmengine.config import Config; \
cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'); \
print('torch=', torch.__version__); \
print('torch_cuda=', torch.version.cuda); \
print('cuda_available=', torch.cuda.is_available()); \
print('mmcv=', mmcv.__version__); \
print('mmengine=', mmengine.__version__); \
print('mmseg=', mmseg.__version__); \
print('config_dataset_type=', cfg.train_dataloader.dataset.type)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --miniforge-dir)
      MINIFORGE_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --cpu)
      CUDA_TAG="cpu"
      shift
      ;;
    --with-optional)
      INSTALL_OPTIONAL=1
      shift
      ;;
    --with-tests)
      INSTALL_TESTS=1
      shift
      ;;
    --with-multimodal)
      INSTALL_MULTIMODAL=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      warn "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

log "Repository root: ${REPO_ROOT}"
log "Target env prefix: ${ENV_PREFIX}"
log "Target CUDA wheel tag: ${CUDA_TAG}"

maybe_install_apt_packages

CONDA_BIN="$(find_conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  bootstrap_miniforge
  CONDA_BIN="$(find_conda)"
fi

log "Using conda at ${CONDA_BIN}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  log "Creating conda environment"
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
else
  log "Reusing existing conda environment"
fi

install_python_packages "${CONDA_BIN}"
validate_environment "${CONDA_BIN}"

cat <<EOF

[server/setup] Environment is ready.
[server/setup] Activate it with:
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  conda activate "${ENV_PREFIX}"

[server/setup] Batch training entrypoint:
  bash server/train_all.sh

[server/setup] Runbook:
  docs/server-runbook.md
EOF

