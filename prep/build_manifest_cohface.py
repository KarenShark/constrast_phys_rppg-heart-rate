#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 COHFACE raw 目录，生成 manifest CSV/JSON 与 cohface_eval_list.npy。

期望结构: datasets/COHFACE_raw/{subject}/{session}/data.avi + data.hdf5
下载: https://zenodo.org/records/4081054 （需机构邮箱申请）
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import h5py
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "prep"))
from cohface_paths import (  # noqa: E402
    COHFACE_DEFAULT_FPS,
    COHFACE_EVAL_LIST,
    COHFACE_H5,
    COHFACE_MANIFEST_DIR,
    COHFACE_PULSE_FS,
    COHFACE_RAW,
)

DEFAULT_RAW = COHFACE_RAW
DEFAULT_MANIFEST_DIR = COHFACE_MANIFEST_DIR


def _read_illumination(hdf5_path):
    try:
        with h5py.File(hdf5_path, "r") as f:
            val = f.attrs.get("illumination", "")
            if hasattr(val, "decode"):
                val = val.decode()
            return str(val) if val else ""
    except Exception:
        return ""


def _probe_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "video_open_failed"
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps_meta <= 1.0 or fps_meta > 120:
        native_fps = COHFACE_DEFAULT_FPS
        fps_source = "default_20"
    else:
        native_fps = float(fps_meta)
        fps_source = "cv2_metadata"
    if frame_count <= 0:
        return native_fps, None, "invalid_frame_count"
    duration = frame_count / native_fps
    return native_fps, {
        "frame_count": frame_count,
        "duration_sec": round(duration, 3),
        "fps_source": fps_source,
    }, None


def _validate_hdf5(gt_path):
    try:
        with h5py.File(gt_path, "r") as f:
            if "pulse" not in f:
                return "missing_pulse_dataset"
            n = len(f["pulse"])
            if n < 10:
                return "pulse_too_short"
        return None
    except Exception as exc:
        return f"hdf5_error:{exc}"


def scan_cohface(raw_root):
    raw_root = os.path.abspath(raw_root)
    rows = []
    if not os.path.isdir(raw_root):
        return rows, [f"raw_root not found: {raw_root}"]

    issues = []
    for subject in sorted(os.listdir(raw_root), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        sub_path = os.path.join(raw_root, subject)
        if not os.path.isdir(sub_path) or not subject.isdigit():
            continue
        for session in sorted(os.listdir(sub_path), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            sess_path = os.path.join(sub_path, session)
            if not os.path.isdir(sess_path) or not session.isdigit():
                continue
            video_path = os.path.join(sess_path, "data.avi")
            gt_path = os.path.join(sess_path, "data.hdf5")
            row = {
                "subject": int(subject),
                "session": int(session),
                "video_path": video_path,
                "gt_path": gt_path,
                "native_fps": None,
                "frame_count": None,
                "duration_sec": None,
                "fps_source": None,
                "illumination": "",
                "status": "ok",
                "skip_reason": "",
            }
            if not os.path.isfile(video_path):
                row["status"] = "skip"
                row["skip_reason"] = "missing_data.avi"
            elif not os.path.isfile(gt_path):
                row["status"] = "skip"
                row["skip_reason"] = "missing_data.hdf5"
            else:
                h5_err = _validate_hdf5(gt_path)
                if h5_err:
                    row["status"] = "skip"
                    row["skip_reason"] = h5_err
                else:
                    native_fps, probe, vid_err = _probe_video(video_path)
                    row["illumination"] = _read_illumination(gt_path)
                    if vid_err:
                        row["status"] = "skip"
                        row["skip_reason"] = vid_err
                    else:
                        row["native_fps"] = native_fps
                        row["frame_count"] = probe["frame_count"]
                        row["duration_sec"] = probe["duration_sec"]
                        row["fps_source"] = probe["fps_source"]
            rows.append(row)
    return rows, issues


def write_manifests(rows, manifest_dir):
    os.makedirs(manifest_dir, exist_ok=True)
    csv_path = os.path.join(manifest_dir, "cohface_manifest.csv")
    json_path = os.path.join(manifest_dir, "cohface_manifest.json")
    fieldnames = [
        "subject", "session", "video_path", "gt_path",
        "native_fps", "frame_count", "duration_sec", "fps_source",
        "illumination", "status", "skip_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    skip_rows = [r for r in rows if r["status"] != "ok"]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "COHFACE",
        "pulse_sample_rate_hz": COHFACE_PULSE_FS,
        "entries_total": len(rows),
        "entries_ok": len(ok_rows),
        "entries_skip": len(skip_rows),
        "entries": rows,
        "skipped": skip_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    eval_list_path = COHFACE_EVAL_LIST
    h5_dir = COHFACE_H5
    eval_paths = [
        os.path.join(h5_dir, f"{r['subject']}_{r['session']}.h5")
        for r in ok_rows
    ]
    np.save(eval_list_path, np.array(eval_paths, dtype=object))

    return csv_path, json_path, eval_list_path


def main():
    ap = argparse.ArgumentParser(description="Build COHFACE manifest")
    ap.add_argument("--raw-root", default=DEFAULT_RAW)
    ap.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    args = ap.parse_args()

    rows, issues = scan_cohface(args.raw_root)
    csv_p, json_p, eval_p = write_manifests(rows, args.manifest_dir)

    ok = sum(1 for r in rows if r["status"] == "ok")
    skip = len(rows) - ok
    print(f"Scanned: {len(rows)} entries | ok={ok} skip={skip}")
    for iss in issues:
        print(f"  issue: {iss}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  skip subj={r['subject']} sess={r['session']}: {r['skip_reason']}")
    print(f"CSV:  {csv_p}")
    print(f"JSON: {json_p}")
    print(f"eval list: {eval_p}")
    if ok == 0:
        print(
            "\nNo ok entries. Download COHFACE from Zenodo and extract to:",
            args.raw_root,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
