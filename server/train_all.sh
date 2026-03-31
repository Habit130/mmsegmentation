#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../IP102}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/work_dirs/ip102}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-3407}"

cd "${REPO_ROOT}"

echo "[0/5] Verifying runtime imports"
python -c "import torch, mmcv, mmengine, mmseg; print('runtime ok')"

echo "[1/5] Generating reproducible train/val splits"
python tools/ip102/generate_splits.py \
  --data-root "${DATA_ROOT}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${SPLIT_SEED}"

echo "[2/5] Validating configs, models, and dataloaders"
python tools/ip102/validate_configs.py

declare -a CONFIGS=(
  "configs/ip102/upernet_swin-t_8xb2-160k_ip102-512x512.py"
  "configs/ip102/ocrnet_hr18_4xb4-160k_ip102-512x512.py"
  "configs/ip102/hrnet_w18_4xb4-160k_ip102-512x512.py"
  "configs/ip102/segmenter_vit-b16_mask_8xb1-160k_ip102-512x512.py"
  "configs/ip102/fcn_r50-d8_4xb4-160k_ip102-512x512.py"
  "configs/ip102/pspnet_r50-d8_4xb4-160k_ip102-512x512.py"
  "configs/ip102/deeplabv3plus_r50-d8_4xb4-160k_ip102-512x512.py"
  "configs/ip102/segformer_mit-b2_8xb2-160k_ip102-512x512.py"
)

echo "[3/5] Running sequential training and final test"
for config in "${CONFIGS[@]}"; do
  config_stem="$(basename "${config}" .py)"
  work_dir="${WORK_ROOT}/${config_stem}"
  metrics_file="${work_dir}/final_test_metrics.json"

  mkdir -p "${work_dir}"

  if [[ -f "${metrics_file}" ]]; then
    echo "Skip ${config_stem}: final metrics already exist."
    continue
  fi

  echo "Training ${config_stem}"
  python tools/train.py "${config}" \
    --work-dir "${work_dir}" \
    --resume \
    --cfg-options \
      default_hooks.checkpoint.save_best=mIoU \
      default_hooks.checkpoint.rule=greater

  echo "Selecting best checkpoint for ${config_stem}"
  best_ckpt="$(python tools/ip102/find_best_checkpoint.py --work-dir "${work_dir}")"

  echo "Testing ${config_stem} with ${best_ckpt}"
  python tools/ip102/test_and_dump.py \
    "${config}" "${best_ckpt}" \
    --work-dir "${work_dir}" \
    --metrics-file "${metrics_file}"
done

echo "[4/5] Collecting summary outputs"
python tools/ip102/collect_results.py \
  --work-root "${WORK_ROOT}" \
  --output-dir "${RESULTS_DIR}"

echo "[5/5] Done"
echo "Summary CSV: ${RESULTS_DIR}/summary.csv"
echo "Summary MD : ${RESULTS_DIR}/summary.md"
