#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 LGI_raw 中 Session1 clip：fps、文件布局。"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lgi_paths import (
    BVP_FS_HZ,
    LGI_DATASET_PROBE,
    LGI_MANIFEST_DIR,
    LGI_RAW,
    PAPER_VIDEO_FPS_REF,
    SESSION_S1,
)

# 排除 session 2/3/4 误匹配
_S_EXCLUDE = re.compile(
    r"session[_\s]?[234]|session[234]|/s[234]/|_s[234]_|ergometer|urban|rotation",
    re.I,
)
_S1_HINT = re.compile(
    r"session[_\s]?1|session1|head[_\s]?rest|resting|/1/|_s1_|_1\.(avi|txt|csv|xml)",
    re.I,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_s1_path(path: Path) -> bool:
    s = str(path).replace("\\", "/")
    if _S_EXCLUDE.search(s):
        return False
    if _S1_HINT.search(s):
        return True
    # 单 session 包内可能无 session 字样；含 avi 且非排除
    return path.suffix.lower() == ".avi" and "session" not in s.lower()


def _probe_video(p: Path) -> dict | None:
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1 or fps > 120:
        fps_source = "invalid_cv2"
    else:
        fps_source = "cv2_metadata"
    return {
        "path": str(p.relative_to(LGI_RAW)),
        "fps": float(fps) if fps > 1 else None,
        "frame_count": n,
        "fps_source": fps_source,
    }


def main() -> int:
    raw = Path(LGI_RAW)
    Path(LGI_MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
    avis = [p for p in raw.rglob("*.avi") if _is_s1_path(p)]
    others = [str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file() and p not in avis]

    probes = []
    for avi in sorted(avis):
        info = _probe_video(avi)
        if info:
            probes.append(info)

    fps_vals = [p["fps"] for p in probes if p.get("fps")]
    report = {
        "created_at": _utc_now(),
        "raw_root": str(raw.resolve()),
        "session_filter": SESSION_S1,
        "n_s1_avi": len(avis),
        "paper_video_fps_ref": PAPER_VIDEO_FPS_REF,
        "bvp_fs_hz": BVP_FS_HZ,
        "fps_summary": {
            "min": min(fps_vals) if fps_vals else None,
            "max": max(fps_vals) if fps_vals else None,
            "mean": sum(fps_vals) / len(fps_vals) if fps_vals else None,
            "unique": sorted(set(fps_vals)),
        },
        "probes": probes,
        "other_files_sample": others[:40],
    }
    with open(LGI_DATASET_PROBE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"S1 avi={len(avis)}  wrote {LGI_DATASET_PROBE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
