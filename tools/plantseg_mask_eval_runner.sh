#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/work_dirs/plantseg_batch}"
RUNS_ROOT="${RUNS_ROOT:-$WORK_ROOT/runs}"
MASK_ROOT="${MASK_ROOT:-$WORK_ROOT/pred_masks}"
EVAL_ROOT="${EVAL_ROOT:-$WORK_ROOT/eval_metrics}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs}"
GT_ROOT="${GT_ROOT:-$ROOT_DIR/../plantseg}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DO_TTA=0
DO_VAL=0
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
  bash tools/plantseg_mask_eval_runner.sh [options] [model_alias...]

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
  - Use --all only when you really want to process all 8 models.
  - For each selected model:
      1. find the best finished checkpoint
      2. generate prediction masks to MASK_ROOT/<alias>/<split>
      3. evaluate saved masks with eval_ris_metrics.py

Options:
  --work-root DIR
      Root output directory. Default: work_dirs/plantseg_batch
  --runs-root DIR
      Root directory of trained model work dirs. Default: <work-root>/runs
  --mask-root DIR
      Root directory to save predicted masks. Default: <work-root>/pred_masks
  --eval-root DIR
      Root directory to save metric logs. Default: <work-root>/eval_metrics
  --gt-root DIR
      GT root passed to eval_ris_metrics.py. Default: ../plantseg
  --cuda-visible-devices IDS
      Value for CUDA_VISIBLE_DEVICES. Default: 0
  --tta
      Enable test-time augmentation during mask generation.
  --val
      Process val split instead of test split.
  --all
      Process all 8 delivered models.
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
      MASK_ROOT="$WORK_ROOT/pred_masks"
      EVAL_ROOT="$WORK_ROOT/eval_metrics"
      LOG_ROOT="$WORK_ROOT/logs"
      shift 2
      ;;
    --runs-root)
      require_value "$1" "${2:-}"
      RUNS_ROOT="$2"
      shift 2
      ;;
    --mask-root)
      require_value "$1" "${2:-}"
      MASK_ROOT="$2"
      shift 2
      ;;
    --eval-root)
      require_value "$1" "${2:-}"
      EVAL_ROOT="$2"
      shift 2
      ;;
    --gt-root)
      require_value "$1" "${2:-}"
      GT_ROOT="$2"
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
    --val)
      DO_VAL=1
      shift
      ;;
    --all)
      RUN_ALL=1
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

SPLIT="test"
if [[ $DO_VAL -eq 1 ]]; then
  SPLIT="val"
fi

mkdir -p "$RUNS_ROOT" "$MASK_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

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

FAILED_MODELS=()
SUMMARY_FILE="$WORK_ROOT/mask_eval_summary.tsv"
printf "model\tconfig\tcheckpoint\tsplit\tpred_dir\tmetrics_log\n" > "$SUMMARY_FILE"

for alias in "${MODELS[@]}"; do
  if [[ -z "${CONFIG_MAP[$alias]:-}" ]]; then
    echo "[ERROR] Unknown model alias: $alias" >&2
    exit 1
  fi

  config="${CONFIG_MAP[$alias]}"
  work_dir="$RUNS_ROOT/$alias"
  pred_dir="$MASK_ROOT/$alias/$SPLIT"
  eval_log="$EVAL_ROOT/${alias}_${SPLIT}.log"

  echo "========== $alias =========="
  echo "config: $config"
  echo "work_dir: $work_dir"
  echo "pred_dir: $pred_dir"
  echo "gt_root: $GT_ROOT"

  if ! checkpoint="$(find_best_checkpoint "$work_dir")"; then
    echo "[ERROR] Checkpoint not found: $alias ($work_dir)" >&2
    FAILED_MODELS+=("$alias")
    if [[ $KEEP_GOING -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  echo "best checkpoint: $checkpoint"

  test_cmd=(
    "$PYTHON_BIN" tools/test.py "$config" "$checkpoint"
    --work-dir "$work_dir/eval_$SPLIT"
    --save_pred_dir "$pred_dir"
    --cfg-options
    "test_dataloader.dataset.split=$SPLIT"
    "model.test_cfg.mode=slide"
    "model.test_cfg.crop_size=(512,512)"
    "model.test_cfg.stride=(384,384)"
  )

  if [[ $DO_TTA -eq 1 ]]; then
    test_cmd+=(--tta)
  fi

  mkdir -p "$pred_dir" "$(dirname "$eval_log")"

  if ! run_cmd "${test_cmd[@]}" | tee "$LOG_ROOT/${alias}_save_${SPLIT}.log"; then
    echo "[ERROR] Prediction export failed: $alias" >&2
    FAILED_MODELS+=("$alias")
    if [[ $KEEP_GOING -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  if ! run_cmd "$PYTHON_BIN" eval_ris_metrics.py --pred_dir "$pred_dir" --gt_dir "$GT_ROOT" | tee "$eval_log"; then
    echo "[ERROR] Metric evaluation failed: $alias" >&2
    FAILED_MODELS+=("$alias")
    if [[ $KEEP_GOING -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$alias" "$config" "$checkpoint" "$SPLIT" "$pred_dir" "$eval_log" \
    >> "$SUMMARY_FILE"
done

echo
echo "Mask export + evaluation finished."
echo "summary: $SUMMARY_FILE"
echo "mask root: $MASK_ROOT"
echo "eval root: $EVAL_ROOT"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "failed models: ${FAILED_MODELS[*]}" >&2
  exit 1
fi
