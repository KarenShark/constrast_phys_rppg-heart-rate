#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无 CanControls 数据时验证 LGI infer/eval 链路。@ 25fps S1。"""
import os
import sys

import h5py
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "prep"))
from lgi_paths import LGI_EVAL_LIST, LGI_H5, SESSION_S1  # noqa: E402


def make_smoke_h5(out_dir, subject_id, n_frames=500, fps=25.0, hr_bpm=70.0):
    """500 frames @ 25fps = 20s, 2 clips of 10s."""
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"id{subject_id}_s{SESSION_S1}.h5")
    t = np.arange(n_frames) / fps
    hr_hz = hr_bpm / 60.0
    bvp = np.sin(2 * np.pi * hr_hz * t).astype(np.float32)
    bvp += 0.05 * np.sin(2 * np.pi * 2 * hr_hz * t)
    rng = np.random.RandomState(subject_id)
    imgs = rng.randint(80, 180, (n_frames, 96, 96, 3), dtype=np.uint8)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("imgs", data=imgs, compression="gzip")
        f.create_dataset("bvp", data=bvp, compression="gzip")
        f.attrs["fps"] = float(fps)
        f.attrs["source_fps"] = float(fps)
        f.attrs["bvp_fs"] = 60.0
        f.attrs["dataset"] = "LGI-PPGI-DB"
        f.attrs["subject"] = str(subject_id)
        f.attrs["session"] = str(SESSION_S1)
        f.attrs["resampled"] = False
        f.attrs["smoke_fixture"] = True
    return h5_path


def main():
    paths = [make_smoke_h5(LGI_H5, 1)]
    os.makedirs(os.path.dirname(LGI_EVAL_LIST), exist_ok=True)
    np.save(LGI_EVAL_LIST, np.array(paths, dtype=object))
    for p in paths:
        print(p)
    print(f"eval list: {LGI_EVAL_LIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
