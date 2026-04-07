#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/work_dirs/plantseg_batch}"
RUNS_ROOT="${RUNS_ROOT:-$WORK_ROOT/runs}"
MASK_ROOT="${MASK_ROOT:-$WORK_ROOT/masks}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DO_TTA=0
DO_VAL_EXPORT=0
SKIP_TRAIN=0
SKIP_TEST=0
RESUME_TRAIN=0
KEEP_GOING=0
RUN_ALL=0

declare -A CONFIG_MAP=(
  ["swin_t"]="configs/plantseg/upernet_swin-t_1xb2-160k_plantseg-512x512.py"
  ["ocrnet_hr18"]="configs/plantseg/ocrnet_hr18_1xb4-160k_plantseg-512x512.py"
  ["hrnet_hr18"]="configs/plantseg/fcn_hr18_1xb4-160k_plantseg-512x512.py"
  ["segmenter_vit_b"]="configs/plantseg/segmenter_vit-b_1xb1-160k_plantseg-512x512.py"
  ["fcn_r50"]="configs/plantseg/fcn_r50_1xb4-160k_plantseg-512x512.py"
  ["pspnet_r50"]="configs/plantseg/pspnet_r50_1xb4-160k_plantseg-512x512.py"
  ["deeplabv3plus_r50"]="configs/plantseg/deeplabv3plus_r50_1xb4-160k_plantseg-512x512.py"
  ["segformer_b2"]="configs/plantseg/segformer_mit-b2_1xb2-160k_plantseg-512x512.py"
)

ALL_MODELS=(
  "swin_t"
  "ocrnet_hr18"
  "hrnet_hr18"
  "segmenter_vit_b"
  "fcn_r50"
  "pspnet_r50"
  "deeplabv3plus_r50"
  "segformer_b2"
)

usage() {
  cat <<'EOF'
Usage:
  bash tools/plantseg_batch_runner.sh [options] [model_alias...]

Model aliases:
  swin_t
  ocrnet_hr18
  hrnet_hr18
  segmenter_vit_b
  fcn_r50
  pspnet_r50
  deeplabv3plus_r50
  segformer_b2

Default behavior:
  - Model aliases must be passed explicitly.
  - Use --all only when you really want to run all 8 models.
  - After training, pick the latest best_mIoU checkpoint automatically.
  - Export test split masks to masks/<model>/test as png files.

Options:
  --work-root DIR
      Root output directory. Default: work_dirs/plantseg_batch
  --cuda-visible-devices IDS
      Value for CUDA_VISIBLE_DEVICES. Default: 0
  --tta
      Enable test-time augmentation during export.
  --resume
      Pass --resume to tools/train.py.
  --all
      Run all 8 delivered models.
  --skip-train
      Skip training and only run test export from existing work dirs.
  --skip-test
      Skip mask export.
  --export-val
      Also export masks for the val split.
  --keep-going
      Continue with later models if one model fails.
  -h, --help
      Show this help text.
EOF
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "[ERROR] Missing value for $flag" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root)
      require_value "$1" "${2:-}"
      WORK_ROOT="$2"
      RUNS_ROOT="$WORK_ROOT/runs"
      MASK_ROOT="$WORK_ROOT/masks"
      LOG_ROOT="$WORK_ROOT/logs"
      shift 2
      ;;
    --cuda-visible-devices)
      require_value "$1" "${2:-}"
      CUDA_VISIBLE_DEVICES="$2"
      export CUDA_VISIBLE_DEVICES
      shift 2
      ;;
    --tta)
      DO_TTA=1
      shift
      ;;
    --resume)
      RESUME_TRAIN=1
      shift
      ;;
    --all)
      RUN_ALL=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-test)
      SKIP_TEST=1
      shift
      ;;
    --export-val)
      DO_VAL_EXPORT=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

MODELS=("$@")
if [[ $RUN_ALL -eq 1 ]]; then
  MODELS=("${ALL_MODELS[@]}")
elif [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "[ERROR] No model alias was provided. Pass model aliases explicitly or use --all." >&2
  usage >&2
  exit 1
fi

mkdir -p "$RUNS_ROOT" "$MASK_ROOT" "$LOG_ROOT"

find_best_checkpoint() {
  local work_dir="$1"
  local ckpt

  ckpt="$(find "$work_dir" -maxdepth 1 -type f -name 'best_mIoU*.pth' | sort | tail -n 1 || true)"
  if [[ -n "$ckpt" ]]; then
    printf '%s\n' "$ckpt"
    return 0
  fi

  if [[ -f "$work_dir/last_checkpoint" ]]; then
    ckpt="$(<"$work_dir/last_checkpoint")"
    if [[ -n "$ckpt" && -f "$ckpt" ]]; then
      printf '%s\n' "$ckpt"
      return 0
    fi
    if [[ -n "$ckpt" && -f "$work_dir/$ckpt" ]]; then
      printf '%s\n' "$work_dir/$ckpt"
      return 0
    fi
  fi

  ckpt="$(find "$work_dir" -maxdepth 1 -type f \( -name 'iter_*.pth' -o -name 'epoch_*.pth' -o -name 'latest.pth' \) | sort | tail -n 1 || true)"
  if [[ -n "$ckpt" ]]; then
    printf '%s\n' "$ckpt"
    return 0
  fi

  return 1
}

run_cmd() {
  echo "[RUN] $*"
  "$@"
}

run_test_export() {
  local alias="$1"
  local config="$2"
  local checkpoint="$3"
  local split="$4"
  local work_dir="$5"
  local out_dir="$MASK_ROOT/$alias/$split"
  local eval_dir="$work_dir/eval_$split"
  local cmd=(
    "$PYTHON_BIN" tools/test.py "$config" "$checkpoint"
    --work-dir "$eval_dir"
    --out "$out_dir"
    --cfg-options
    "test_dataloader.dataset.split=$split"
    "model.test_cfg.mode=slide"
    "model.test_cfg.crop_size=(512,512)"
    "model.test_cfg.stride=(384,384)"
  )

  if [[ $DO_TTA -eq 1 ]]; then
    cmd+=(--tta)
  fi

  mkdir -p "$out_dir" "$eval_dir"
  run_cmd "${cmd[@]}" | tee "$LOG_ROOT/${alias}_test_${split}.log"
}

FAILED_MODELS=()
SUMMARY_FILE="$WORK_ROOT/summary.tsv"
printf "model\tconfig\twork_dir\tcheckpoint\ttest_mask_dir\tval_mask_dir\n" > "$SUMMARY_FILE"

for alias in "${MODELS[@]}"; do
  if [[ -z "${CONFIG_MAP[$alias]:-}" ]]; then
    echo "[ERROR] Unknown model alias: $alias" >&2
    exit 1
  fi

  config="${CONFIG_MAP[$alias]}"
  work_dir="$RUNS_ROOT/$alias"
  mkdir -p "$work_dir"

  echo "========== $alias =========="
  echo "config: $config"
  echo "work_dir: $work_dir"

  if [[ $SKIP_TRAIN -eq 0 ]]; then
    train_cmd=(
      "$PYTHON_BIN" tools/train.py "$config"
      --work-dir "$work_dir"
    )
    if [[ $RESUME_TRAIN -eq 1 ]]; then
      train_cmd+=(--resume)
    fi

    if ! run_cmd "${train_cmd[@]}" | tee "$LOG_ROOT/${alias}_train.log"; then
      echo "[ERROR] Training failed: $alias" >&2
      FAILED_MODELS+=("$alias")
      if [[ $KEEP_GOING -eq 1 ]]; then
        continue
      fi
      exit 1
    fi
  fi

  if ! checkpoint="$(find_best_checkpoint "$work_dir")"; then
    echo "[ERROR] Checkpoint not found: $alias ($work_dir)" >&2
    FAILED_MODELS+=("$alias")
    if [[ $KEEP_GOING -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  echo "best checkpoint: $checkpoint"

  if [[ $SKIP_TEST -eq 0 ]]; then
    if ! run_test_export "$alias" "$config" "$checkpoint" "test" "$work_dir"; then
      echo "[ERROR] Test export failed: $alias" >&2
      FAILED_MODELS+=("$alias")
      if [[ $KEEP_GOING -eq 1 ]]; then
        continue
      fi
      exit 1
    fi

    if [[ $DO_VAL_EXPORT -eq 1 ]]; then
      if ! run_test_export "$alias" "$config" "$checkpoint" "val" "$work_dir"; then
        echo "[ERROR] Val export failed: $alias" >&2
        FAILED_MODELS+=("$alias")
        if [[ $KEEP_GOING -eq 1 ]]; then
          continue
        fi
        exit 1
      fi
    fi
  fi

  val_mask_dir="-"
  if [[ $DO_VAL_EXPORT -eq 1 ]]; then
    val_mask_dir="$MASK_ROOT/$alias/val"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$alias" "$config" "$work_dir" "$checkpoint" "$MASK_ROOT/$alias/test" "$val_mask_dir" \
    >> "$SUMMARY_FILE"
done

echo
echo "Batch run finished."
echo "summary: $SUMMARY_FILE"
echo "mask root: $MASK_ROOT"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "failed models: ${FAILED_MODELS[*]}" >&2
  exit 1
fi
