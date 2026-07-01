#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LGI Session1 -> H5（native fps, 96x96）。PPG @ 60Hz 插值到视频帧轴。
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import cv2
import h5py
import numpy as np
from scipy import interpolate

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "prep"))

from lgi_paths import (  # noqa: E402
    BVP_FS_HZ,
    LGI_EVAL_LIST,
    LGI_H5,
    LGI_MANIFEST_JSON,
    SESSION_S1,
    STORE_SIZE,
)
from preprocess_ubfc import openface_h5_with_ppg  # noqa: E402

PPG_FS = BVP_FS_HZ


def _find_openface_bin():
    candidates = [
        "/home/vt_ai_test1/mamba-envs/ml/local/bin/FeatureExtraction",
        os.path.join(PROJECT_DIR, "OpenFace", "build", "bin", "FeatureExtraction"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_lgi_ppg(ppg_path, target_length, video_fps):
    """CMS50E waveform @ 60Hz -> 每视频帧一个 BVP 样本。"""
    path = ppg_path
    if path.lower().endswith(".mat"):
        from scipy.io import loadmat
        m = loadmat(path)
        for k in ("ppg", "pulse", "bvp", "P", "signal"):
            if k in m:
                pulse = np.asarray(m[k], dtype=np.float64).reshape(-1)
                break
        else:
            raise ValueError(f"no ppg in mat: {path}")
    else:
        try:
            pulse = np.loadtxt(path, comments="#")
        except Exception:
            pulse = np.genfromtxt(path, comments="#")
        pulse = np.asarray(pulse, dtype=np.float64).reshape(-1)
        if pulse.ndim == 2:
            pulse = pulse[:, -1]

    if len(pulse) < 2:
        raise ValueError(f"ppg too short: {path}")
    t_pulse = np.arange(len(pulse)) / PPG_FS
    t_video = np.arange(target_length) / float(video_fps)
    t_video = np.clip(t_video, t_pulse[0], t_pulse[-1])
    fn = interpolate.interp1d(
        t_pulse, pulse, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return fn(t_video).astype(np.float32)


def run_openface(video_path, landmark_csv, openface_bin, landmarks_dir):
    os.makedirs(landmarks_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(landmark_csv))[0]
    if os.path.isfile(landmark_csv):
        return landmark_csv
    cmd = [
        openface_bin, "-f", video_path,
        "-out_dir", landmarks_dir, "-of", stem, "-2Dfp",
    ]
    subprocess.run(cmd, check=True)
    if not os.path.isfile(landmark_csv):
        raise FileNotFoundError(f"OpenFace output missing: {landmark_csv}")
    return landmark_csv


def process_one(entry, output_dir, landmarks_dir, openface_bin, skip_existing=True):
    sid = entry["subject_id"]
    video_path = entry["video_path"]
    ppg_path = entry["ppg_path"]
    video_fps = float(entry["native_fps"])

    h5_name = f"id{sid}_s{SESSION_S1}.h5"
    h5_path = os.path.join(output_dir, h5_name)
    landmark_csv = os.path.join(landmarks_dir, h5_name.replace(".h5", ".csv"))

    log = {
        "subject_id": sid,
        "session": SESSION_S1,
        "status": "ok",
        "skip_reason": "",
        "h5_path": h5_path,
        "video_fps": video_fps,
    }

    if skip_existing and os.path.isfile(h5_path):
        log["status"] = "skip"
        log["skip_reason"] = "h5_exists"
        return log

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n_frames < 30:
        log["status"] = "skip"
        log["skip_reason"] = "too_few_frames"
        return log

    bvp = load_lgi_ppg(ppg_path, n_frames, video_fps)
    run_openface(video_path, landmark_csv, openface_bin, landmarks_dir)
    openface_h5_with_ppg(
        video_path, landmark_csv, h5_path, bvp, video_fps,
        store_size=STORE_SIZE,
        h5_attrs={
            "dataset": "LGI-PPGI-DB",
            "subject": str(sid),
            "session": str(SESSION_S1),
            "source_fps": float(video_fps),
            "bvp_fs": float(PPG_FS),
            "resampled": False,
        },
    )
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=LGI_MANIFEST_JSON)
    ap.add_argument("--output-dir", default=LGI_H5)
    ap.add_argument("--landmarks-dir", default=os.path.join(LGI_H5, "_landmarks"))
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.manifest):
        print(f"manifest missing: {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    entries = [e for e in manifest["entries"] if e["status"] == "ok"]
    if args.subjects:
        entries = [e for e in entries if e["subject_id"] in args.subjects]
    if args.limit:
        entries = entries[: args.limit]

    openface_bin = _find_openface_bin()
    if not openface_bin:
        print("OpenFace FeatureExtraction not found", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    logs = []
    for e in entries:
        print(f"Processing id{e['subject_id']} S1 ...")
        logs.append(
            process_one(
                e, args.output_dir, args.landmarks_dir, openface_bin,
                skip_existing=not args.no_skip_existing,
            )
        )

    ok = sum(1 for x in logs if x["status"] == "ok")
    skip = sum(1 for x in logs if x["status"] == "skip")
    print(f"Done ok={ok} skip={skip}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
