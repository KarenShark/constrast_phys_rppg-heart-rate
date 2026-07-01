# -*- coding: utf-8 -*-
"""
COHFACE 零样本跨数据集推理 — per-sample native fps，不读 config.fs。

运行:
  cd contrast-phys+
  python EfficientPhysNet/test/infer_cross_dataset_cohface.py with strategy=curriculum time_interval=10
"""
import json
import os
import sys
import time

_EPN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP = os.path.dirname(_EPN)
_PROJECT = os.path.dirname(_CP)
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
read_h5_fps = _h5_meta.read_h5_fps
read_h5_eval_meta = _h5_meta.read_h5_eval_meta

ex = Experiment("infer_cross_dataset_cohface", save_git_info=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

DEFAULT_EVAL_LIST = os.path.join(
    _PROJECT, "datasets", "COHFACE_manifests", "cohface_eval_list.npy"
)


@ex.config
def my_config():
    strategy = "curriculum"
    time_interval = 10
    e = None
    pretrained_dir = os.path.join(_PRETRAINED, strategy)
    h5_list_path = DEFAULT_EVAL_LIST


@ex.automain
def my_main(_run, strategy, time_interval, e, pretrained_dir, h5_list_path):
    infer_t0 = time.perf_counter()
    config_path = os.path.join(pretrained_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    train_fs = float(config.get("fs", 30))
    input_size = int(config.get("input_size", 96))
    weight_path = (
        os.path.join(pretrained_dir, "best_model.pt")
        if e is None
        else os.path.join(pretrained_dir, f"epoch{e}.pt")
    )
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(weight_path)

    if not os.path.isfile(h5_list_path):
        raise FileNotFoundError(
            f"H5 list not found: {h5_list_path}\n"
            "Run manifest + preprocess, or prep/create_cohface_smoke_fixture.py"
        )

    h5_list_raw = np.load(h5_list_path, allow_pickle=True)
    h5_list = []
    for p in h5_list_raw:
        p = str(p)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_PROJECT, p))
        h5_list.append(p)

    model = (
        EfficientPhysNet(config["S"], in_ch=config["in_ch"], input_size=input_size)
        .to(device)
        .eval()
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))

    run_id = int(_run._id) if _run._id is not None else 1
    pred_dir = os.path.join(
        _CP,
        "results",
        "external_eval",
        "cohface",
        strategy,
        f"t{time_interval}",
        str(run_id),
    )
    os.makedirs(pred_dir, exist_ok=True)

    samples_meta = []
    n_saved = 0
    n_clips = 0
    eval_fps_values = []

    for h5_path in h5_list:
        if not os.path.isfile(h5_path):
            print(f"  Skip (not found): {h5_path}")
            continue
        fs_sample = read_h5_fps(h5_path)
        eval_meta = read_h5_eval_meta(h5_path)
        eval_fps_values.append(fs_sample)

        with h5py.File(h5_path, "r") as f:
            imgs = f["imgs"]
            bvp = f["bvp"]
            n = int(min(imgs.shape[0], bvp.shape[0]))
            duration = n / fs_sample
            num_blocks = int(duration // time_interval)
            rppg_list, bvp_list = [], []
            clip_fs = []
            for b in range(num_blocks):
                s = int(b * time_interval * fs_sample)
                e_idx = int((b + 1) * time_interval * fs_sample)
                imgs_clip = np.array(imgs[s:e_idx])
                if len(imgs_clip) < int(time_interval * fs_sample * 0.9):
                    break
                rppg_clip = dl_model(model, imgs_clip, device)
                rppg_list.append(rppg_clip)
                bvp_list.append(np.array(bvp[s:e_idx]))
                clip_fs.append(fs_sample)

            if not rppg_list:
                print(f"  Skip (no clips): {h5_path}")
                continue

            stem = os.path.basename(h5_path).replace(".h5", "")
            results = {
                "rppg_list": np.array(rppg_list, dtype=object),
                "bvp_list": np.array(bvp_list, dtype=object),
                "fs": fs_sample,
                "h5_path": h5_path,
                "clip_fs": np.array(clip_fs, dtype=np.float64),
            }
            np.save(os.path.join(pred_dir, stem), results)
            print(
                f"  Saved: {stem}.npy ({len(rppg_list)} clips @ fs={fs_sample})"
            )
            n_saved += 1
            n_clips += len(rppg_list)
            samples_meta.append(
                {
                    "stem": stem,
                    "h5_path": h5_path,
                    "fs": fs_sample,
                    "n_clips": len(rppg_list),
                    "subject": eval_meta.get("subject", ""),
                    "session": eval_meta.get("session", ""),
                    "illumination": eval_meta.get("illumination", ""),
                }
            )

    infer_sec = time.perf_counter() - infer_t0
    meta = {
        "dataset": "COHFACE",
        "native_fps_policy": True,
        "train_fs": train_fs,
        "eval_fps_values": eval_fps_values,
        "strategy": strategy,
        "time_interval_sec": int(time_interval),
        "input_size": input_size,
        "run_id": run_id,
        "weight_path": weight_path,
        "device": str(device),
        "subjects_saved": n_saved,
        "clips_total": n_clips,
        "inference_seconds": float(infer_sec),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": samples_meta,
    }
    with open(os.path.join(pred_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nPred saved to: {pred_dir}")
    print(f"fs used: {sorted(set(eval_fps_values))} (train_fs={train_fs})")
    print(
        f"Next: python EfficientPhysNet/evaluation/evaluate_cross_dataset_cohface.py {pred_dir}"
    )
