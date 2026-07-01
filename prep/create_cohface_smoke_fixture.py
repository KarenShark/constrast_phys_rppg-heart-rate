#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成 COHFACE smoke H5（无 Zenodo 数据时验证 infer/eval 链路）。

输出 native fps=20, 96x96, 含 pulse 对齐的 bvp。
"""
import argparse
import os
import sys

import h5py
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "prep"))
from cohface_paths import COHFACE_EVAL_LIST, COHFACE_H5  # noqa: E402

DEFAULT_OUT = COHFACE_H5


def make_smoke_h5(out_dir, subject, session, n_frames=400, fps=20.0, hr_bpm=72.0):
    """400 frames @ 20fps = 20s, 2 clips of 10s."""
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{subject}_{session}.h5")
    t = np.arange(n_frames) / fps
    hr_hz = hr_bpm / 60.0
    bvp = np.sin(2 * np.pi * hr_hz * t).astype(np.float32)
    bvp += 0.05 * np.sin(2 * np.pi * 2 * hr_hz * t)
    rng = np.random.RandomState(subject * 10 + session)
    imgs = rng.randint(80, 180, (n_frames, 96, 96, 3), dtype=np.uint8)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("imgs", data=imgs, compression="gzip")
        f.create_dataset("bvp", data=bvp, compression="gzip")
        f.attrs["fps"] = float(fps)
        f.attrs["source_fps"] = float(fps)
        f.attrs["dataset"] = "COHFACE"
        f.attrs["resampled"] = False
        f.attrs["subject"] = str(subject)
        f.attrs["session"] = str(session)
        f.attrs["illumination"] = "lamp" if session < 2 else "natural"
        f.attrs["n_frames"] = int(n_frames)
        f.attrs["smoke_fixture"] = True
    return h5_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = [
        make_smoke_h5(args.output_dir, 1, 0),
        make_smoke_h5(args.output_dir, 1, 1),
    ]
    eval_list = COHFACE_EVAL_LIST
    os.makedirs(os.path.dirname(eval_list), exist_ok=True)
    np.save(eval_list, np.array(paths, dtype=object))
    for p in paths:
        print(f"  {p}")
    print(f"eval list: {eval_list}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
