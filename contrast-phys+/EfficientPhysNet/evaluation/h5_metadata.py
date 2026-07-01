# -*- coding: utf-8 -*-
"""H5 元数据读取 — 跨数据集评估统一 fps 来源。"""
import os

import h5py


def read_h5_fps(h5_path):
    """读 H5 attr fps；缺失则报错，不 fallback 到 30。"""
    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "fps" not in f.attrs:
            raise KeyError(f"H5 missing attr 'fps': {h5_path}")
        fps = float(f.attrs["fps"])
    if fps <= 0 or fps > 240:
        raise ValueError(f"Invalid fps={fps} in {h5_path}")
    return fps


def read_h5_eval_meta(h5_path):
    """评估用元数据：fps / dataset / subject / session / resampled。"""
    h5_path = os.path.abspath(h5_path)
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
    fps = read_h5_fps(h5_path)
    def _attr(key, default=None):
        v = attrs.get(key, default)
        if hasattr(v, "decode"):
            return v.decode()
        return v

    return {
        "h5_path": h5_path,
        "fps": fps,
        "source_fps": float(_attr("source_fps", fps)),
        "dataset": str(_attr("dataset", "")),
        "subject": str(_attr("subject", "")),
        "session": str(_attr("session", "")),
        "illumination": str(_attr("illumination", "")),
        "resampled": bool(_attr("resampled", False)),
        "n_frames": int(attrs.get("n_frames", 0)) or None,
    }


def h5_stem_to_path(h5_dir, stem):
    """pred .npy stem -> H5 路径。"""
    return os.path.join(os.path.abspath(h5_dir), f"{stem}.h5")
