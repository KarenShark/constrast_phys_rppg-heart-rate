#!/bin/bash
# COHFACE 零样本跨数据集评估编排
# 用法:
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_cohface.sh smoke
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_cohface.sh full
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_cohface.sh infer-only
set -e

MODE="${1:-smoke}"
PYTHON="${PYTHON:-/home/vt_ai_test1/mamba-envs/ml/bin/python}"
STRATEGY="${STRATEGY:-curriculum}"
TIME_INTERVAL="${TIME_INTERVAL:-10}"

CP="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cd "$CP/.." && pwd)"
cd "$CP"

RAW_ROOT="$PROJECT/datasets/COHFACE_raw"
H5_DIR="$PROJECT/datasets/COHFACE_h5"
MANIFEST_DIR="$PROJECT/datasets/COHFACE_manifests"

echo "=== COHFACE cross-dataset eval | mode=$MODE ==="

case "$MODE" in
  smoke-fixture)
    echo "--- Synthetic smoke H5 (no Zenodo data) ---"
    $PYTHON "$PROJECT/prep/create_cohface_smoke_fixture.py"
    ;;
  manifest)
    echo "--- Build manifest ---"
    $PYTHON "$PROJECT/prep/build_manifest_cohface.py" \
      --raw-root "$RAW_ROOT" \
      --manifest-dir "$MANIFEST_DIR"
    ;;
  preprocess-smoke)
    echo "--- Preprocess 1-2 real videos ---"
    $PYTHON "$PROJECT/prep/build_manifest_cohface.py" \
      --raw-root "$RAW_ROOT" --manifest-dir "$MANIFEST_DIR"
    $PYTHON "$PROJECT/prep/preprocess_cohface.py" \
      --subjects 1 --sessions 0 1 --limit 2
    ;;
  preprocess-full)
    echo "--- Preprocess all ok entries ---"
    $PYTHON "$PROJECT/prep/preprocess_cohface.py"
    ;;
  smoke)
    if [ -d "$RAW_ROOT/1/0" ] && [ -f "$RAW_ROOT/1/0/data.avi" ]; then
      echo "--- Real data smoke: manifest + preprocess 1-2 ---"
      $PYTHON "$PROJECT/prep/build_manifest_cohface.py" \
        --raw-root "$RAW_ROOT" --manifest-dir "$MANIFEST_DIR"
      $PYTHON "$PROJECT/prep/preprocess_cohface.py" \
        --subjects 1 --sessions 0 1 --limit 2
    else
      echo "--- No raw COHFACE; using synthetic smoke fixture ---"
      $PYTHON "$PROJECT/prep/create_cohface_smoke_fixture.py"
    fi
    ;;
  full)
    $PYTHON "$PROJECT/prep/build_manifest_cohface.py" \
      --raw-root "$RAW_ROOT" --manifest-dir "$MANIFEST_DIR"
    $PYTHON "$PROJECT/prep/preprocess_cohface.py"
    ;;
  infer-only)
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Modes: smoke | smoke-fixture | manifest | preprocess-smoke | preprocess-full | full | infer-only"
    exit 1
    ;;
esac

if [ "$MODE" != "manifest" ] && [ "$MODE" != "preprocess-smoke" ] && [ "$MODE" != "preprocess-full" ]; then
  echo "--- Infer (native fps per H5) ---"
  $PYTHON EfficientPhysNet/test/infer_cross_dataset_cohface.py \
    with strategy="$STRATEGY" time_interval="$TIME_INTERVAL"

  PRED_ROOT="$CP/results/external_eval/cohface/$STRATEGY/t${TIME_INTERVAL}"
  LATEST_RUN="$(ls -1 "$PRED_ROOT" 2>/dev/null | awk '/^[0-9]+$/ {print $0}' | sort -n | tail -1)"
  PRED_DIR="$PRED_ROOT/$LATEST_RUN"

  if [ -d "$PRED_DIR" ] && [ -n "$(ls -A "$PRED_DIR"/*.npy 2>/dev/null)" ]; then
    echo "--- Evaluate Phase1 + Phase2 ---"
    $PYTHON EfficientPhysNet/evaluation/evaluate_cross_dataset_cohface.py \
      "$PRED_DIR" --h5-dir "$H5_DIR"
  else
    echo "Pred dir empty: $PRED_DIR"
    exit 1
  fi
fi

echo "=== Done ==="
