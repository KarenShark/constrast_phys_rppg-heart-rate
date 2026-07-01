#!/bin/bash
# UBFC-Phys T1 零样本跨数据集评估（OpenFace 预处理 + native fps + quality-aware eval）
# 用法:
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_ubfc_phys.sh preprocess
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_ubfc_phys.sh full
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_ubfc_phys.sh infer-only
set -e

MODE="${1:-full}"
PYTHON="${PYTHON:-/home/vt_ai_test1/miniconda3/envs/rppg_env/bin/python}"
STRATEGY="${STRATEGY:-curriculum}"
TIME_INTERVAL="${TIME_INTERVAL:-10}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
SAVE_VIZ="${SAVE_VIZ:-1}"

CP="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cd "$CP/.." && pwd)"
cd "$CP"

RAW_ROOT="${RAW_ROOT:-/ssd/UBFC_phy/extracted}"
H5_DIR="${H5_DIR:-/ssd/UBFC_phy/h5}"
OPENFACE_BIN="${OPENFACE_BIN:-$PROJECT/OpenFace/build_openface_env/bin/FeatureExtraction}"
export OPENFACE_BIN

echo "=== UBFC-Phys cross-dataset eval | mode=$MODE | t=${TIME_INTERVAL}s ==="

run_preprocess() {
  local extra=()
  if [ "$FORCE_PREPROCESS" = "1" ]; then
    extra+=(--no-skip-existing)
  fi
  $PYTHON "$PROJECT/prep/preprocess_ubfc_phys_t1.py" \
    --raw-root "$RAW_ROOT" \
    --output-dir "$H5_DIR" \
    --openface-bin "$OPENFACE_BIN" \
    "${extra[@]}"
}

case "$MODE" in
  preprocess)
    run_preprocess
    exit 0
    ;;
  preprocess-force)
    FORCE_PREPROCESS=1 run_preprocess
    exit 0
    ;;
  infer-only)
    ;;
  full)
    run_preprocess
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Modes: preprocess | preprocess-force | full | infer-only"
    exit 1
    ;;
esac

echo "--- Infer (native fps per H5) ---"
$PYTHON EfficientPhysNet/test/infer_cross_dataset_ubfc_phys.py \
  with strategy="$STRATEGY" time_interval="$TIME_INTERVAL"

PRED_ROOT="$CP/results/external_eval/ubfc_phys/$STRATEGY/t${TIME_INTERVAL}"
LATEST_RUN="$(ls -1 "$PRED_ROOT" 2>/dev/null | awk '/^[0-9]+$/ {print $0}' | sort -n | tail -1)"
PRED_DIR="$PRED_ROOT/$LATEST_RUN"

if [ -d "$PRED_DIR" ] && [ -n "$(ls -A "$PRED_DIR"/*.npy 2>/dev/null)" ]; then
  echo "--- Evaluate (UBFC-Phys quality-aware) ---"
  eval_args=()
  if [ "$SAVE_VIZ" = "1" ]; then
    eval_args+=(--save-viz)
  fi
  $PYTHON EfficientPhysNet/evaluation/evaluate_ubfc_phys_quality.py "$PRED_DIR" "${eval_args[@]}"
else
  echo "Pred dir empty: $PRED_DIR"
  exit 1
fi

echo "=== Done ==="
