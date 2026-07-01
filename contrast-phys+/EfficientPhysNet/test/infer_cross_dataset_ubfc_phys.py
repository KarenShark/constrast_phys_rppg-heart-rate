# -*- coding: utf-8 -*-
"""
UBFC-Phys T1 零样本跨数据集推理 — per-H5 native fps。

运行:
  cd /home/vt_ai_test1/KarenHE/contrast-phys/contrast-phys+
  /home/vt_ai_test1/miniconda3/envs/rppg_env/bin/python \
      EfficientPhysNet/test/infer_cross_dataset_ubfc_phys.py \
      with strategy=curriculum time_interval=10
"""
import json
import os
import sys
import time

_EPN      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP       = os.path.dirname(_EPN)
_PROJECT  = os.path.dirname(_CP)
_PRETRAINED = os.path.join(_CP, "pretrained", "EfficientPhysNet")
sys.path.insert(0, _CP)
sys.path.insert(0, os.path.join(_CP, "PhysNet_2D"))

import h5py
import numpy as np
import torch
from sacred import Experiment

from EfficientPhysNet import EfficientPhysNet
from utils_inference import dl_model

import importlib.util
_h5_spec = importlib.util.spec_from_file_location(
    "h5_metadata",
    os.path.join(_EPN, "evaluation", "h5_metadata.py"),
)
_h5_meta = importlib.util.module_from_spec(_h5_spec)
_h5_spec.loader.exec_module(_h5_meta)
read_h5_fps      = _h5_meta.read_h5_fps
read_h5_eval_meta = _h5_meta.read_h5_eval_meta

ex = Experiment("infer_cross_dataset_ubfc_phys", save_git_info=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

DEFAULT_H5_DIR    = "/ssd/UBFC_phy/h5"
DEFAULT_EVAL_LIST = os.path.join(DEFAULT_H5_DIR, "ubfc_phys_t1_eval_list.npy")


@ex.config
def my_config():
    strategy      = "curriculum"
    time_interval = 10          # 每 clip 10 秒 × ~35 fps ≈ 351 帧
    e             = None        # None → best_model.pt
    pretrained_dir = os.path.join(_PRETRAINED, strategy)
    h5_list_path   = DEFAULT_EVAL_LIST


@ex.automain
def my_main(_run, strategy, time_interval, e, pretrained_dir, h5_list_path):
    infer_t0 = time.perf_counter()

    # ── 加载 config & 权重 ─────────────────────────────────────────────────
    config_path = os.path.join(pretrained_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    train_fs   = float(config.get("fs", 30))
    input_size = int(config.get("input_size", 96))
    weight_path = (
        os.path.join(pretrained_dir, "best_model.pt")
        if e is None
        else os.path.join(pretrained_dir, f"epoch{e}.pt")
    )
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(weight_path)

    # ── 加载 eval list ─────────────────────────────────────────────────────
    if not os.path.isfile(h5_list_path):
        raise FileNotFoundError(
            f"Eval list not found: {h5_list_path}\n"
            "先运行: prep/preprocess_ubfc_phys_t1.py"
        )
    h5_list = []
    for p in np.load(h5_list_path, allow_pickle=True):
        p = str(p)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_PROJECT, p))
        h5_list.append(p)
    print(f"Eval list: {len(h5_list)} H5 files")

    # ── 加载模型 ────────────────────────────────────────────────────────────
    model = (
        EfficientPhysNet(config["S"], in_ch=config["in_ch"], input_size=input_size)
        .to(device).eval()
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"Model loaded: {weight_path}  input_size={input_size}  device={device}")

    # ── 输出目录 ────────────────────────────────────────────────────────────
    run_id = int(_run._id) if _run._id is not None else 1
    pred_dir = os.path.join(
        _CP, "results", "external_eval", "ubfc_phys",
        strategy, f"t{time_interval}", str(run_id),
    )
    os.makedirs(pred_dir, exist_ok=True)

    # ── 逐文件推理 ──────────────────────────────────────────────────────────
    samples_meta  = []
    eval_fps_vals = []
    n_saved = n_clips = 0

    for h5_path in h5_list:
        if not os.path.isfile(h5_path):
            print(f"  Skip (not found): {h5_path}")
            continue

        # 从 H5 attrs 读 native fps — 不 fallback，确保正确
        fs_sample = read_h5_fps(h5_path)
        eval_meta = read_h5_eval_meta(h5_path)
        eval_fps_vals.append(fs_sample)

        with h5py.File(h5_path, "r") as f:
            imgs = f["imgs"]
            bvp  = f["bvp"]
            n    = int(min(imgs.shape[0], bvp.shape[0]))

            # 按 time_interval 秒切片
            num_blocks = int((n / fs_sample) // time_interval)
            rppg_list, bvp_list, clip_fs = [], [], []

            for b in range(num_blocks):
                s     = int(b       * time_interval * fs_sample)
                e_idx = int((b + 1) * time_interval * fs_sample)
                imgs_clip = np.array(imgs[s:e_idx])
                # 丢弃不足 90% 长度的尾部片段
                if len(imgs_clip) < int(time_interval * fs_sample * 0.9):
                    break
                rppg_list.append(dl_model(model, imgs_clip, device))
                bvp_list.append(np.array(bvp[s:e_idx]))
                clip_fs.append(fs_sample)

        if not rppg_list:
            print(f"  Skip (no clips): {h5_path}")
            continue

        stem = os.path.basename(h5_path).replace(".h5", "")
        np.save(
            os.path.join(pred_dir, stem),
            {
                "rppg_list": np.array(rppg_list, dtype=object),
                "bvp_list":  np.array(bvp_list,  dtype=object),
                "fs":        fs_sample,
                "h5_path":   h5_path,
                "clip_fs":   np.array(clip_fs, dtype=np.float64),
            },
        )
        n_saved += 1
        n_clips += len(rppg_list)
        samples_meta.append({
            "stem":    stem,
            "h5_path": h5_path,
            "fs":      fs_sample,
            "n_clips": len(rppg_list),
            "subject": eval_meta.get("subject", ""),
            "session": eval_meta.get("session", ""),
        })
        print(f"  Saved: {stem}.npy ({len(rppg_list)} clips @ fs={fs_sample:.3f})")

    # ── 保存 meta ────────────────────────────────────────────────────────────
    infer_sec = time.perf_counter() - infer_t0
    meta = {
        "dataset":           "UBFC-Phys T1",
        "native_fps_policy": True,
        "train_fs":          train_fs,
        "eval_fps_values":   eval_fps_vals,
        "strategy":          strategy,
        "time_interval_sec": int(time_interval),
        "input_size":        input_size,
        "run_id":            run_id,
        "weight_path":       weight_path,
        "device":            str(device),
        "subjects_saved":    n_saved,
        "clips_total":       n_clips,
        "inference_seconds": float(infer_sec),
        "created_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples":           samples_meta,
    }
    with open(os.path.join(pred_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nPred saved to: {pred_dir}")
    print(f"Subjects: {n_saved}  Clips: {n_clips}  Time: {infer_sec:.1f}s")
    print(f"fps: {sorted(set(round(v,3) for v in eval_fps_vals))}  (train_fs={train_fs})")
    print(f"\nNext:")
    print(f"  python EfficientPhysNet/evaluation/evaluate.py {pred_dir} --save-viz")
