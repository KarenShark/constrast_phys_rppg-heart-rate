#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COHFACE -> H5（native fps, 96x96），裁脸流程与 UBFC preprocess_ubfc 一致。

BVP: data.hdf5['pulse'] @ 256Hz -> 插值到视频帧轴（不重采样视频）
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

from cohface_paths import (  # noqa: E402
    COHFACE_DEFAULT_FPS,
    COHFACE_EVAL_LIST,
    COHFACE_H5,
    COHFACE_MANIFEST_JSON,
    COHFACE_PULSE_FS,
    COHFACE_STORE_SIZE,
)
from preprocess_ubfc import openface_h5_with_ppg  # noqa: E402

DEFAULT_OUT = COHFACE_H5
DEFAULT_MANIFEST = COHFACE_MANIFEST_JSON
PULSE_FS = COHFACE_PULSE_FS
STORE_SIZE = COHFACE_STORE_SIZE


def _find_openface_bin():
    candidates = [
        "/home/vt_ai_test1/mamba-envs/ml/local/bin/FeatureExtraction",
        os.path.join(PROJECT_DIR, "OpenFace", "build", "bin", "FeatureExtraction"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_cohface_bvp(hdf5_path, target_length, video_fps):
    """pulse @ 256Hz -> 每视频帧一个 BVP 样本。"""
    with h5py.File(hdf5_path, "r") as f:
        pulse = np.asarray(f["pulse"], dtype=np.float64).reshape(-1)
    if len(pulse) < 2:
        raise ValueError(f"pulse too short: {hdf5_path}")
    t_pulse = np.arange(len(pulse)) / PULSE_FS
    t_video = np.arange(target_length) / float(video_fps)
    t_video = np.clip(t_video, t_pulse[0], t_pulse[-1])
    fn = interpolate.interp1d(
        t_pulse, pulse, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return fn(t_video).astype(np.float32)


def read_hdf5_meta(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        attrs = dict(f.attrs)

    def _dec(v):
        return v.decode() if hasattr(v, "decode") else v

    return {k: _dec(v) for k, v in attrs.items()}


def run_openface(video_path, landmark_csv, openface_bin, landmarks_dir):
    os.makedirs(landmarks_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(landmark_csv))[0]
    if os.path.isfile(landmark_csv):
        return landmark_csv
    cmd = [
        openface_bin,
        "-f", video_path,
        "-out_dir", landmarks_dir,
        "-of", stem,
        "-2Dfp",
    ]
    subprocess.run(cmd, check=True)
    if not os.path.isfile(landmark_csv):
        alt = os.path.join(landmarks_dir, "data.csv")
        if os.path.isfile(alt):
            os.rename(alt, landmark_csv)
    if not os.path.isfile(landmark_csv):
        raise FileNotFoundError(f"OpenFace output missing: {landmark_csv}")
    return landmark_csv


def process_one(entry, output_dir, landmarks_dir, openface_bin, skip_existing=True):
    subject = entry["subject"]
    session = entry["session"]
    video_path = entry["video_path"]
    gt_path = entry["gt_path"]
    video_fps = float(entry["native_fps"])

    h5_name = f"{subject}_{session}.h5"
    h5_path = os.path.join(output_dir, h5_name)
    landmark_csv = os.path.join(landmarks_dir, f"{subject}_{session}.csv")

    log = {
        "subject": subject,
        "session": session,
        "status": "ok",
        "skip_reason": "",
        "h5_path": h5_path,
        "video_fps": video_fps,
    }

    if skip_existing and os.path.isfile(h5_path):
        log["status"] = "skip"
        log["skip_reason"] = "h5_exists"
        return log

    if not os.path.isfile(video_path) or not os.path.isfile(gt_path):
        log["status"] = "fail"
        log["skip_reason"] = "missing_raw_files"
        return log

    try:
        run_openface(video_path, landmark_csv, openface_bin, landmarks_dir)
        import pandas as pd
        n_frames = len(pd.read_csv(landmark_csv))
        bvp = load_cohface_bvp(gt_path, n_frames, video_fps)
        h5_meta_raw = read_hdf5_meta(gt_path)
        illumination = str(h5_meta_raw.get("illumination", ""))

        attrs = {
            "fps": float(video_fps),
            "source_fps": float(video_fps),
            "dataset": "COHFACE",
            "resampled": False,
            "subject": str(subject),
            "session": str(session),
            "illumination": illumination,
            "n_frames": int(n_frames),
            "pulse_fs": PULSE_FS,
        }

        ok = openface_h5_with_ppg(
            video_path,
            landmark_csv,
            h5_path,
            ppg_path=None,
            store_size=STORE_SIZE,
            video_fps=video_fps,
            bvp_array=bvp,
            h5_attrs=attrs,
        )
        if not ok:
            log["status"] = "fail"
            log["skip_reason"] = "openface_h5_failed"
            if os.path.isfile(h5_path):
                os.remove(h5_path)
        else:
            with h5py.File(h5_path, "r") as f:
                log["n_frames"] = int(f["imgs"].shape[0])
                log["fps_attr"] = float(f.attrs["fps"])
    except Exception as exc:
        log["status"] = "fail"
        log["skip_reason"] = str(exc)
        if os.path.isfile(h5_path):
            os.remove(h5_path)

    return log


def load_manifest_entries(manifest_path, only_ok=True):
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", data)
    if only_ok:
        entries = [e for e in entries if e.get("status") == "ok"]
    return entries


def filter_entries(entries, subjects=None, sessions=None, limit=None):
    out = entries
    if subjects is not None:
        sub_set = {int(s) for s in subjects}
        out = [e for e in out if int(e["subject"]) in sub_set]
    if sessions is not None:
        sess_set = {int(s) for s in sessions}
        out = [e for e in out if int(e["session"]) in sess_set]
    if limit is not None:
        out = out[: int(limit)]
    return out


def main():
    ap = argparse.ArgumentParser(description="Preprocess COHFACE to native-fps H5")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--landmarks-dir", default=None)
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--sessions", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--openface-bin", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.manifest):
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run: python prep/build_manifest_cohface.py", file=sys.stderr)
        return 1

    openface_bin = args.openface_bin or _find_openface_bin()
    if not openface_bin:
        print("OpenFace FeatureExtraction not found", file=sys.stderr)
        return 1

    output_dir = os.path.abspath(args.output_dir)
    landmarks_dir = args.landmarks_dir or os.path.join(output_dir, "landmarks")
    os.makedirs(output_dir, exist_ok=True)

    entries = filter_entries(
        load_manifest_entries(args.manifest),
        subjects=args.subjects,
        sessions=args.sessions,
        limit=args.limit,
    )
    if not entries:
        print("No entries to process", file=sys.stderr)
        return 1

    log_path = os.path.join(output_dir, "preprocess_log.jsonl")
    ok_n = fail_n = skip_n = 0
    with open(log_path, "a", encoding="utf-8") as log_f:
        for entry in entries:
            rec = process_one(
                entry,
                output_dir,
                landmarks_dir,
                openface_bin,
                skip_existing=not args.no_skip_existing,
            )
            rec["ts"] = datetime.now().isoformat(timespec="seconds")
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            st = rec["status"]
            if st == "ok":
                ok_n += 1
                print(f"  ok {rec['subject']}_{rec['session']} -> {rec['h5_path']}")
            elif st == "skip":
                skip_n += 1
                print(f"  skip {rec['subject']}_{rec['session']}: {rec['skip_reason']}")
            else:
                fail_n += 1
                print(f"  fail {rec['subject']}_{rec['session']}: {rec['skip_reason']}")

    eval_list = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".h5")
    )
    np.save(COHFACE_EVAL_LIST, np.array(eval_list, dtype=object))

    print(f"Done: ok={ok_n} skip={skip_n} fail={fail_n} | log={log_path}")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
