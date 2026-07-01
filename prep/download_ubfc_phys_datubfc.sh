#!/bin/bash
# UBFC-Phys 下载入口 — 逻辑在 ubfc_phys_download.py
# 用法:
#   ./prep/download_ubfc_phys_datubfc.sh daemon all   # 后台串行 6 包
#   ./prep/download_ubfc_phys_datubfc.sh daemon 1     # 仅 bundle 1
#   ./prep/download_ubfc_phys_datubfc.sh status       # 进度条
#   ./prep/download_ubfc_phys_datubfc.sh stop
#   ./prep/download_ubfc_phys_datubfc.sh bundle 3     # 前台单包
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="$PROJECT/prep/ubfc_phys_download.py"

MODE="${1:-status}"
ARG2="${2:-}"

case "$MODE" in
  daemon|status|stop)
    exec "$PY" "$SCRIPT" "$MODE" ${ARG2:+"$ARG2"}
    ;;
  bundle)
    [[ -n "${ARG2:-}" ]] || { echo "usage: bundle <1-6|all>"; exit 1; }
    if [[ "$ARG2" == "all" ]]; then
      exec "$PY" "$SCRIPT" run 1 2 3 4 5 6
    else
      exec "$PY" "$SCRIPT" run "$ARG2"
    fi
    ;;
  readme|probe|head)
    echo "deprecated: use daemon/status; probe=head bundle 6"
    ;;
  *)
    echo "Modes: daemon <1-6|all> | status | stop | bundle <1-6|all>"
    exit 1
    ;;
esac
