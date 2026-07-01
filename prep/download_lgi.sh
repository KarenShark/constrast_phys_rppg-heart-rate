#!/bin/bash
# LGI-PPGI-DB 下载（仅 Session1）
#   ./prep/download_lgi.sh daemon 1
#   ./prep/download_lgi.sh daemon all
#   ./prep/download_lgi.sh status
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="$PROJECT/prep/lgi_download.py"
MODE="${1:-status}"
ARG2="${2:-}"

mkdir -p "$PROJECT/datasets/LGI_staging/archives" "$PROJECT/datasets/LGI_raw" \
  "$PROJECT/datasets/LGI_manifests"

case "$MODE" in
  probe)
    exec "$PY" "$PROJECT/prep/probe_lgi_download.py"
    ;;
  probe-data)
    exec "$PY" "$PROJECT/prep/probe_lgi_dataset.py"
    ;;
  daemon|status|stop)
    exec "$PY" "$SCRIPT" "$MODE" ${ARG2:+"$ARG2"}
    ;;
  subject)
    [[ -n "${ARG2:-}" ]] || { echo "usage: subject <1-6|all>"; exit 1; }
    if [[ "$ARG2" == "all" ]]; then
      exec "$PY" "$SCRIPT" run 1 2 3 4 5 6
    else
      exec "$PY" "$SCRIPT" run "$ARG2"
    fi
    ;;
  *)
    echo "Modes: probe | probe-data | daemon <1-6|all> | status | stop | subject <1-6|all>"
    exit 1
    ;;
esac
