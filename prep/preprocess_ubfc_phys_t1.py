#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UBFC-Phys T1 -> H5（OpenFace 裁脸、native fps、96×96 RGB）。

BVP: Empatica csv @64Hz -> 线性插值到视频帧轴（均匀 t=frame/fps，无 clock offset）
裁脸: 与 preprocess_ubfc / live_recorded_infer 一致（OpenFace 68 点）

运行:
  cd /home/vt_ai_test1/KarenHE/contrast-phys
  python prep/preprocess_ubfc_phys_t1.py [--subjects 1 2 4 5 6 7 10]
  已有 Haar H5 需重跑: --no-skip-existing
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import interpolate

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "prep"))

from ubfc_phys_paths import (  # noqa: E402
    BVP_FS_HZ,
    EXCLUDE_S1_TO_S10,
    STORE_SIZE,
    TASK_T1,
    UBFC_PHYS_EVAL_LIST,
    UBFC_PHYS_H5,
    UBFC_PHYS_RAW_SSD,
)
from preprocess_ubfc import openface_h5_with_ppg  # noqa: E402

DEFAULT_RAW_ROOT = UBFC_PHYS_RAW_SSD
DATASET = "UBFC-Phys"
DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 10]


def _find_openface_bin(explicit=None):
    """优先 openface_env 构建（libavcodec 兼容），须从 exe 目录 cwd 运行。"""
    if explicit and os.path.isfile(explicit):
        return explicit
    env_bin = os.environ.get("OPENFACE_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    candidates = [
        os.path.join(PROJECT_DIR, "OpenFace", "build_openface_env", "bin", "FeatureExtraction"),
        "/home/vt_ai_test1/mamba-envs/ml/local/bin/FeatureExtraction",
        os.path.join(PROJECT_DIR, "OpenFace", "build", "bin", "FeatureExtraction"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _read_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise ValueError(f"invalid fps={fps} for {video_path}")
    return fps


def load_ubfc_phys_bvp(bvp_csv, n_frames, video_fps):
    """64Hz BVP -> 每视频帧一个样本，帧轴 t=i/video_fps。"""
    pulse = np.loadtxt(bvp_csv, delimiter=",", dtype=np.float64).reshape(-1)
    if len(pulse) < 2:
        raise ValueError(f"BVP too short: {bvp_csv}")
    t_pulse = np.arange(len(pulse)) / float(BVP_FS_HZ)
    t_video = np.arange(n_frames) / float(video_fps)
    t_video = np.clip(t_video, t_pulse[0], t_pulse[-1])
    fn = interpolate.interp1d(
        t_pulse, pulse, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    return fn(t_video).astype(np.float32)


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
    # OpenFace 模型相对路径，须从 bin 目录启动
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(openface_bin)))
    if not os.path.isfile(landmark_csv):
        alt = os.path.join(landmarks_dir, "data.csv")
        if os.path.isfile(alt):
            os.rename(alt, landmark_csv)
    if not os.path.isfile(landmark_csv):
        raise FileNotFoundError(f"OpenFace output missing: {landmark_csv}")
    return landmark_csv


def process_subject(
    sid_int,
    output_dir,
    raw_root,
    landmarks_dir,
    openface_bin,
    skip_existing=True,
):
    sid = str(sid_int)
    name = f"s{sid}"
    subj_dir = os.path.join(raw_root, name)
    vid_path = os.path.join(subj_dir, f"vid_{name}_{TASK_T1}.avi")
    bvp_path = os.path.join(subj_dir, f"bvp_{name}_{TASK_T1}.csv")
    h5_path = os.path.join(output_dir, f"{name}_{TASK_T1}.h5")
    landmark_csv = os.path.join(landmarks_dir, f"{name}_{TASK_T1}.csv")

    log = {
        "subject": sid,
        "session": TASK_T1,
        "status": "ok",
        "skip_reason": "",
        "h5_path": h5_path,
        "face_method": "openface",
    }

    if skip_existing and os.path.isfile(h5_path):
        log["status"] = "skip"
        log["skip_reason"] = "h5_exists"
        print(f"  skip s{sid} (h5 already exists)")
        return log

    if not os.path.isfile(vid_path):
        log["status"] = "fail"
        log["skip_reason"] = f"video not found: {vid_path}"
        print(f"  fail s{sid}: {log['skip_reason']}")
        return log

    if not os.path.isfile(bvp_path):
        log["status"] = "fail"
        log["skip_reason"] = f"bvp not found: {bvp_path}"
        print(f"  fail s{sid}: {log['skip_reason']}")
        return log

    if not openface_bin:
        log["status"] = "fail"
        log["skip_reason"] = "OpenFace FeatureExtraction not found"
        print(f"  fail s{sid}: {log['skip_reason']}")
        return log

    try:
        video_fps = _read_video_fps(vid_path)
        print(f"  processing s{sid}: fps={video_fps:.3f} -> {h5_path}")

        run_openface(vid_path, landmark_csv, openface_bin, landmarks_dir)
        n_frames = len(pd.read_csv(landmark_csv))
        bvp = load_ubfc_phys_bvp(bvp_path, n_frames, video_fps)

        attrs = {
            "fps": float(video_fps),
            "source_fps": float(video_fps),
            "dataset": DATASET,
            "subject": sid,
            "session": TASK_T1,
            "n_frames": int(n_frames),
            "bvp_fs": float(BVP_FS_HZ),
            "resampled": False,
            "face_method": "openface",
        }

        ok = openface_h5_with_ppg(
            vid_path,
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
            return log

        with h5py.File(h5_path, "r") as f:
            log["n_frames"] = int(f["imgs"].shape[0])
            log["fps"] = float(f.attrs["fps"])
            log["bvp_n"] = int(f["bvp"].shape[0])
        print(f"    OK  n_frames={log['n_frames']}  fps={log['fps']:.3f}")
    except Exception as exc:
        log["status"] = "fail"
        log["skip_reason"] = str(exc)
        if os.path.isfile(h5_path):
            os.remove(h5_path)
        print(f"  fail s{sid}: {exc}")

    return log


def main():
    ap = argparse.ArgumentParser(description="UBFC-Phys T1 -> H5 (OpenFace, native fps)")
    ap.add_argument("--subjects", type=int, nargs="*", default=DEFAULT_SUBJECTS)
    ap.add_argument("--output-dir", default=UBFC_PHYS_H5)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument(
        "--landmarks-dir",
        default=None,
        help="OpenFace csv 目录，默认 <output-dir>/landmarks",
    )
    ap.add_argument("--openface-bin", default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    raw_root = os.path.abspath(args.raw_root)
    landmarks_dir = os.path.abspath(
        args.landmarks_dir or os.path.join(output_dir, "landmarks")
    )
    openface_bin = _find_openface_bin(args.openface_bin)
    os.makedirs(output_dir, exist_ok=True)

    subjects = [s for s in args.subjects if s not in EXCLUDE_S1_TO_S10]
    print(f"UBFC-Phys T1 preprocess | OpenFace | subjects={subjects}")
    print(f"RAW_ROOT   = {raw_root}")
    print(f"OUTPUT_DIR = {output_dir}")
    print(f"STORE_SIZE = {STORE_SIZE}  BVP_FS={BVP_FS_HZ}Hz")
    print("-" * 60)

    log_path = os.path.join(output_dir, "preprocess_log.jsonl")
    ok_n = fail_n = skip_n = 0
    h5_paths = []

    with open(log_path, "a", encoding="utf-8") as log_f:
        for sid in subjects:
            rec = process_subject(
                sid,
                output_dir,
                raw_root,
                landmarks_dir,
                openface_bin,
                skip_existing=not args.no_skip_existing,
            )
            rec["ts"] = datetime.now().isoformat(timespec="seconds")
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            if rec["status"] == "ok":
                ok_n += 1
                h5_paths.append(rec["h5_path"])
            elif rec["status"] == "skip":
                skip_n += 1
                h5_paths.append(rec["h5_path"])
            else:
                fail_n += 1

    # 扫描 output_dir 所有 H5（分批跑也不丢之前的 subjects）
    all_h5 = sorted(
        glob.glob(os.path.join(output_dir, f"s*_{TASK_T1}.h5")),
        key=lambda p: int(os.path.basename(p).split("_")[0][1:]),
    )
    eval_list_path = (
        UBFC_PHYS_EVAL_LIST
        if output_dir == os.path.abspath(UBFC_PHYS_H5)
        else os.path.join(output_dir, "ubfc_phys_t1_eval_list.npy")
    )
    np.save(eval_list_path, np.array(all_h5, dtype=object))

    print("-" * 60)
    print(f"Done: ok={ok_n}  skip={skip_n}  fail={fail_n}")
    print(f"eval_list ({len(all_h5)}): {eval_list_path}")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
