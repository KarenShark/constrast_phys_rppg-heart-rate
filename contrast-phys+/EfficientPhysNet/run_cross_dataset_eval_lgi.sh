#!/bin/bash
# LGI-PPGI-DB S1 零样本跨数据集评估编排
# 用法:
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_lgi.sh smoke-fixture
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_lgi.sh smoke
#   cd contrast-phys+ && ./EfficientPhysNet/run_cross_dataset_eval_lgi.sh full
set -e

MODE="${1:-smoke-fixture}"
PYTHON="${PYTHON:-/home/vt_ai_test1/mamba-envs/ml/bin/python}"
STRATEGY="${STRATEGY:-curriculum}"
TIME_INTERVAL="${TIME_INTERVAL:-10}"

CP="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cd "$CP/.." && pwd)"
cd "$CP"

RAW_ROOT="$PROJECT/datasets/LGI_raw"
H5_DIR="$PROJECT/datasets/LGI_h5"
MANIFEST_DIR="$PROJECT/datasets/LGI_manifests"

echo "=== LGI S1 cross-dataset eval | mode=$MODE ==="

case "$MODE" in
  smoke-fixture)
    echo "--- Synthetic smoke H5 (no CanControls data) ---"
    $PYTHON "$PROJECT/prep/create_lgi_smoke_fixture.py"
    ;;
  manifest)
    echo "--- Build manifest (S1 only) ---"
    $PYTHON "$PROJECT/prep/build_manifest_lgi.py" \
      --raw-root "$RAW_ROOT" \
      --manifest-dir "$MANIFEST_DIR"
    ;;
  preprocess-smoke)
    echo "--- Preprocess 1 S1 clip ---"
    $PYTHON "$PROJECT/prep/build_manifest_lgi.py" \
      --raw-root "$RAW_ROOT" --manifest-dir "$MANIFEST_DIR"
    $PYTHON "$PROJECT/prep/preprocess_lgi.py" --limit 1
    ;;
  preprocess-full)
    echo "--- Preprocess all S1 ok entries ---"
    $PYTHON "$PROJECT/prep/preprocess_lgi.py"
    ;;
  smoke)
    if find "$RAW_ROOT" -name "*.avi" 2>/dev/null | head -1 | grep -q .; then
      echo "--- Real data smoke: manifest + preprocess 1 ---"
      $PYTHON "$PROJECT/prep/build_manifest_lgi.py" \
        --raw-root "$RAW_ROOT" --manifest-dir "$MANIFEST_DIR"
      $PYTHON "$PROJECT/prep/preprocess_lgi.py" --limit 1
    else
      echo "--- No raw LGI; using synthetic smoke fixture ---"
      $PYTHON "$PROJECT/prep/create_lgi_smoke_fixture.py"
    fi
    ;;
  full)
    $PYTHON "$PROJECT/prep/build_manifest_lgi.py" \
      --raw-root "$RAW_ROOT" --manifest-dir "$MANIFEST_DIR"
    $PYTHON "$PROJECT/prep/preprocess_lgi.py"
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
  $PYTHON EfficientPhysNet/test/infer_cross_dataset_lgi.py \
    with strategy="$STRATEGY" time_interval="$TIME_INTERVAL"

  PRED_ROOT="$CP/results/external_eval/lgi/$STRATEGY/t${TIME_INTERVAL}"
  LATEST_RUN="$(ls -1 "$PRED_ROOT" 2>/dev/null | awk '/^[0-9]+$/ {print $0}' | sort -n | tail -1)"
  PRED_DIR="$PRED_ROOT/$LATEST_RUN"

  if [ -d "$PRED_DIR" ] && [ -n "$(ls -A "$PRED_DIR"/*.npy 2>/dev/null)" ]; then
    echo "--- Evaluate Phase1 + Phase2 ---"
    $PYTHON EfficientPhysNet/evaluation/evaluate_cross_dataset_lgi.py \
      "$PRED_DIR" --h5-dir "$H5_DIR"
  else
    echo "Pred dir empty: $PRED_DIR"
    exit 1
  fi
fi

echo "=== Done ==="
