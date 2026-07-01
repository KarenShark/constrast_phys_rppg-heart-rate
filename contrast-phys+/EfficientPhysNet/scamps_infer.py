#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCAMPS inference — EPN + POS
Loads SCAMPS .mat clips, runs inference, saves .npy in evaluate.py-compatible format.

Usage (cd contrast-phys+ first):
  python EfficientPhysNet/scamps_infer.py \
      --data-dir /path/to/scamps_videos_example \
      --out-dir  results/scamps_eval \
      --strategy curriculum

Then evaluate:
  python EfficientPhysNet/evaluation/evaluate.py results/scamps_eval/epn
  python EfficientPhysNet/evaluation/evaluate.py results/scamps_eval/pos
"""

import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
_EPN = os.path.dirname(os.path.abspath(__file__))
_CP  = os.path.dirname(_EPN)                                           # contrast-phys+
_VHR = os.path.join(os.path.dirname(_CP), "heart_breathing_rate")     # local pyVHR
sys.path.insert(0, _CP)
sys.path.insert(0, os.path.join(_CP, "PhysNet_2D"))
sys.path.insert(0, _VHR)

from EfficientPhysNet import EfficientPhysNet
from utils_inference import dl_model
from pyVHR.BVP.BVP import signals_to_bvps_cpu
from pyVHR.BVP.methods import cpu_POS

FS = 30   # SCAMPS is 30 fps
T  = 300  # 10s window, same as test_epn.py


# ── data loading ──────────────────────────────────────────────────────────────

def load_scamps_mat(mat_path, input_size=96):
    """
    Load one SCAMPS .mat file.

    Returns:
      frames : (600, input_size, input_size, 3)  float32  [0, 255]
      bvp    : (600,)  float64  — d_ppg derivative PPG (FFT peak freq unchanged)
    """
    with h5py.File(mat_path, 'r') as f:
        # Xsub stored as (3, 240, 240, 600) float64 [0, 1]
        xsub = np.array(f['Xsub'])                              # (3, 240, 240, 600)
        xsub = np.transpose(xsub, (3, 2, 1, 0))                # (600, 240, 240, 3)
        xsub = (xsub * 255).astype(np.float32)                  # [0, 255], same as H5Dataset
        bvp  = np.array(f['d_ppg']).flatten().astype(np.float64)  # (600,)

    n_frames, H, W, C = xsub.shape
    if H != input_size or W != input_size:
        resized = np.empty((n_frames, input_size, input_size, C), dtype=np.float32)
        for i in range(n_frames):
            resized[i] = cv2.resize(xsub[i], (input_size, input_size),
                                    interpolation=cv2.INTER_AREA)
        xsub = resized

    return xsub, bvp


# ── POS inference ─────────────────────────────────────────────────────────────

def run_pos(frames_clip):
    """
    frames_clip : (T, H, W, 3) float32 [0, 255]
    Returns rppg : (T,) float32
    """
    # spatial mean over face crop → (T, 3), then (1, 3, T) for pyVHR cpu_POS
    rgb_mean = frames_clip.reshape(len(frames_clip), -1, 3).mean(axis=1)  # (T, 3)
    pos_in   = rgb_mean.T[np.newaxis].astype(np.float32)                   # (1, 3, T)
    bvp_out  = signals_to_bvps_cpu(pos_in, cpu_POS, params={"fps": float(FS)})
    if bvp_out.ndim > 1:
        bvp_out = np.mean(bvp_out, axis=0)
    return bvp_out.astype(np.float32)


# ── per-clip inference ────────────────────────────────────────────────────────

def infer_mat(mat_path, model, device, input_size, run_epn, run_pos_flag):
    """
    Run EPN and/or POS on one .mat file.

    Returns dict with keys 'epn' and/or 'pos', each:
      {'rppg_list': (n_clips, T), 'bvp_list': (n_clips, T), 'fs': 30.0}
    """
    frames, bvp_full = load_scamps_mat(mat_path, input_size=input_size)
    n_clips = len(frames) // T

    epn_rppg, epn_bvp = [], []
    pos_rppg, pos_bvp = [], []

    for b in range(n_clips):
        sl     = slice(b * T, (b + 1) * T)
        clip   = frames[sl]     # (T, H, W, 3)
        bvp_c  = bvp_full[sl]   # (T,)

        if run_epn:
            epn_rppg.append(dl_model(model, clip, device))
            epn_bvp.append(bvp_c)

        if run_pos_flag:
            pos_rppg.append(run_pos(clip))
            pos_bvp.append(bvp_c)

    results = {}
    if run_epn:
        results['epn'] = {
            'rppg_list': np.array(epn_rppg),
            'bvp_list':  np.array(epn_bvp),
            'fs':        float(FS),
        }
    if run_pos_flag:
        results['pos'] = {
            'rppg_list': np.array(pos_rppg),
            'bvp_list':  np.array(pos_bvp),
            'fs':        float(FS),
        }
    return results, n_clips


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SCAMPS inference (EPN + POS) → evaluate.py-compatible .npy")
    ap.add_argument('--data-dir',  required=True,       help='Folder with SCAMPS .mat files')
    ap.add_argument('--out-dir',   required=True,       help='Output root (epn/ and pos/ created inside)')
    ap.add_argument('--strategy',  default='curriculum',help='EPN pretrained strategy (default: curriculum)')
    ap.add_argument('--method',    default='both',      choices=['epn', 'pos', 'both'])
    args = ap.parse_args()

    run_epn      = args.method in ('epn', 'both')
    run_pos_flag = args.method in ('pos', 'both')

    # ── load EPN model ────────────────────────────────────────────────────────
    pretrained_dir = os.path.join(_CP, 'pretrained', 'EfficientPhysNet', args.strategy)
    with open(os.path.join(pretrained_dir, 'config.json')) as f:
        config = json.load(f)
    input_size  = config.get('input_size', 96)
    weight_path = os.path.join(pretrained_dir, 'best_model.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = None
    if run_epn:
        model = EfficientPhysNet(config['S'], in_ch=config['in_ch'],
                                 input_size=input_size).to(device).eval()
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"EPN model: {args.strategy}  input_size={input_size}  device={device}")

    # ── output dirs ───────────────────────────────────────────────────────────
    epn_dir = os.path.join(args.out_dir, 'epn')
    pos_dir = os.path.join(args.out_dir, 'pos')
    if run_epn:
        os.makedirs(epn_dir, exist_ok=True)
    if run_pos_flag:
        os.makedirs(pos_dir, exist_ok=True)

    # ── inference loop ────────────────────────────────────────────────────────
    mat_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.mat'))
    print(f"\nFound {len(mat_files)} .mat files  →  {T/FS:.0f}s clips, {FS} fps")
    print("-" * 56)

    t0 = time.perf_counter()
    total_clips = 0

    for mat_file in mat_files:
        mat_path = os.path.join(args.data_dir, mat_file)
        name     = mat_file.replace('.mat', '')

        results, n = infer_mat(mat_path, model, device, input_size,
                               run_epn=run_epn, run_pos_flag=run_pos_flag)
        total_clips += n

        if run_epn:
            np.save(os.path.join(epn_dir, name + '.npy'), results['epn'])
        if run_pos_flag:
            np.save(os.path.join(pos_dir, name + '.npy'), results['pos'])

        print(f"  {name}  {n} clips  ({n * T / FS:.0f}s)")

    elapsed = time.perf_counter() - t0
    print("-" * 56)
    print(f"Done: {len(mat_files)} clips  {total_clips} windows  {elapsed:.1f}s\n")

    print("Next — run evaluate.py:")
    if run_epn:
        print(f"  cd {_CP}")
        print(f"  python EfficientPhysNet/evaluation/evaluate.py {epn_dir}")
    if run_pos_flag:
        print(f"  python EfficientPhysNet/evaluation/evaluate.py {pos_dir}")


if __name__ == '__main__':
    main()
