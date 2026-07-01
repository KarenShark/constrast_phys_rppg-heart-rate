#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 LGI_raw，生成 S1-only manifest 与 eval list。"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lgi_paths import (
    BVP_FS_HZ,
    LGI_EVAL_LIST,
    LGI_H5,
    LGI_MANIFEST_DIR,
    LGI_RAW,
    PAPER_VIDEO_FPS_REF,
    SESSION_S1,
)
from lgi_s1_filter import infer_subject_id, is_s1_path


def _probe_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, "video_open_failed"
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps_meta <= 1.0 or fps_meta > 120:
        return None, None, f"invalid_cv2_fps={fps_meta}"
    if frame_count <= 0:
        return None, None, "invalid_frame_count"
    native_fps = float(fps_meta)
    return native_fps, {
        "frame_count": frame_count,
        "duration_sec": round(frame_count / native_fps, 3),
        "fps_source": "cv2_metadata",
    }, None


def _find_ppg(avi_path: Path) -> Path | None:
    ppg_names = (
        "pulse", "ppg", "bvp", "oximeter", "pleth", "hr",
    )
    search_dirs = [avi_path.parent, avi_path.parent.parent]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() not in (
                ".txt", ".csv", ".dat", ".mat",
            ):
                continue
            low = f.name.lower()
            if any(k in low for k in ppg_names):
                return f
        for f in sorted(d.glob("*.txt")):
            if f != avi_path and "timestamp" not in f.name.lower():
                return f
    return None


def scan_lgi(raw_root: str):
    raw_root = Path(raw_root)
    rows = []
    if not raw_root.is_dir():
        return rows, [f"raw_root not found: {raw_root}"]

    seen = set()
    for avi in sorted(raw_root.rglob("*.avi")):
        if not is_s1_path(avi):
            continue
        subj = infer_subject_id(avi, raw_root)
        if subj is None:
            continue
        key = (subj, str(avi.resolve()))
        if key in seen:
            continue
        seen.add(key)

        ppg = _find_ppg(avi)
        row = {
            "subject_id": subj,
            "session": SESSION_S1,
            "video_path": str(avi.resolve()),
            "ppg_path": str(ppg.resolve()) if ppg else "",
            "native_fps": None,
            "frame_count": None,
            "duration_sec": None,
            "fps_source": None,
            "status": "ok",
            "skip_reason": "",
        }
        if not ppg or not ppg.is_file():
            row["status"] = "skip"
            row["skip_reason"] = "missing_ppg"
        else:
            fps, probe, err = _probe_video(avi)
            if err:
                row["status"] = "skip"
                row["skip_reason"] = err
            else:
                row["native_fps"] = fps
                row["frame_count"] = probe["frame_count"]
                row["duration_sec"] = probe["duration_sec"]
                row["fps_source"] = probe["fps_source"]
        rows.append(row)
    return rows, []


def write_manifests(rows, manifest_dir):
    os.makedirs(manifest_dir, exist_ok=True)
    csv_path = os.path.join(manifest_dir, "lgi_manifest.csv")
    json_path = os.path.join(manifest_dir, "lgi_manifest.json")
    fields = [
        "subject_id", "session", "video_path", "ppg_path",
        "native_fps", "frame_count", "duration_sec", "fps_source",
        "status", "skip_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "LGI-PPGI-DB",
        "session_filter": SESSION_S1,
        "ppg_sample_rate_hz": BVP_FS_HZ,
        "paper_video_fps_ref": PAPER_VIDEO_FPS_REF,
        "entries_total": len(rows),
        "entries_ok": len(ok_rows),
        "entries_skip": len(rows) - len(ok_rows),
        "entries": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    eval_paths = [
        os.path.join(LGI_H5, f"id{r['subject_id']}_s{SESSION_S1}.h5")
        for r in ok_rows
    ]
    np.save(LGI_EVAL_LIST, np.array(eval_paths, dtype=object))
    return csv_path, json_path, LGI_EVAL_LIST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=LGI_RAW)
    ap.add_argument("--manifest-dir", default=LGI_MANIFEST_DIR)
    args = ap.parse_args()

    rows, issues = scan_lgi(args.raw_root)
    csv_p, json_p, eval_p = write_manifests(rows, args.manifest_dir)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Scanned S1: {len(rows)} | ok={ok} skip={len(rows)-ok}")
    for iss in issues:
        print(f"  issue: {iss}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  skip id{r['subject_id']}: {r['skip_reason']}")
    print(f"CSV: {csv_p}\nJSON: {json_p}\neval: {eval_p}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
