# -*- coding: utf-8 -*-
"""
离线 POS rPPG — 输出与 live_recorded_infer / evaluate.py 兼容的 eval_bundle。

roi_mode:
  openface_crop  96×96 OpenFace crop 每帧 RGB 均值 → POS (E=1)
  skin_patch     同 bbox 内 4×4 grid patch RGB 均值 → POS (E=16)，全分辨率

用法 (contrast-phys+):
  python EfficientPhysNet/pos_recorded_infer.py \\
    --video .../android_*.avi --session-dir .../karen \\
    --roi-mode openface_crop --landmarks ...csv
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

_EPN = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_EPN)
_HBR = os.path.join(os.path.dirname(_CP), "heart_breathing_rate")
_EPN2D = os.path.join(_CP, "PhysNet_2D")
for _p in (_CP, _EPN, _EPN2D, _HBR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_predict_webcam_EPN as live_epn
import live_recorded_infer as lri
from pyVHR.BVP.BVP import signals_to_bvps_cpu
from pyVHR.BVP.methods import cpu_POS
from utils_paths import format_label_ratio


def _openface_cols(landmark_df):
    success_col = "success" if "success" in landmark_df.columns else " success"
    x_prefix = "x_" if "x_0" in landmark_df.columns else " x_"
    y_prefix = "y_" if "y_0" in landmark_df.columns else " y_"
    return success_col, x_prefix, y_prefix


def load_openface_video_bundle(video_path, landmark_path, store_size=96, want_full=False):
    """与 live_epn.load_frames_from_video_with_openface 同索引；可选返回全分辨率 BGR。"""
    landmark = pd.read_csv(landmark_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    success_col, x_prefix, y_prefix = _openface_cols(landmark)
    total_num_frame = len(landmark)

    bbox_size = None
    for frame_num in range(total_num_frame):
        if landmark[success_col][frame_num]:
            lm_x = [landmark[f"{x_prefix}{i}"][frame_num] for i in range(68)]
            lm_y = [landmark[f"{y_prefix}{i}"][frame_num] for i in range(68)]
            miny, maxy = np.min(lm_y), np.max(lm_y)
            y_range_ext = (maxy - miny) * 0.2
            miny = miny - y_range_ext
            bbox_size = int(np.round(1.5 * (maxy - miny)))
            break
    if bbox_size is None:
        raise ValueError("landmark 中无有效帧")

    crops, full_bgr, frame_nums, bbox_meta = [], [], [], []
    lm_x_prev = lm_y_prev = None
    cnt_x = cnt_y = None

    for frame_num in range(total_num_frame):
        if landmark[success_col][frame_num]:
            lm_x_ = np.array([landmark[f"{x_prefix}{i}"][frame_num] for i in range(68)])
            lm_y_ = np.array([landmark[f"{y_prefix}{i}"][frame_num] for i in range(68)])
            lm_x = 0.9 * lm_x_prev + 0.1 * lm_x_ if lm_x_prev is not None else lm_x_
            lm_y = 0.9 * lm_y_prev + 0.1 * lm_y_ if lm_y_prev is not None else lm_y_
            lm_x_prev, lm_y_prev = lm_x, lm_y
            minx, maxx = np.min(lm_x), np.max(lm_x)
            miny, maxy = np.min(lm_y), np.max(lm_y)
            y_range_ext = (maxy - miny) * 0.2
            miny = miny - y_range_ext
            cnt_x = int(np.round((minx + maxx) / 2))
            cnt_y = int(np.round((maxy + miny) / 2))

        ret, frame = cap.read()
        if not ret:
            break
        if cnt_x is None:
            continue

        if want_full:
            full_bgr.append(frame.copy())

        bbox_half = int(bbox_size / 2)
        face = np.take(
            frame,
            range(cnt_y - bbox_half, cnt_y - bbox_half + bbox_size),
            0,
            mode="clip",
        )
        face = np.take(
            face,
            range(cnt_x - bbox_half, cnt_x - bbox_half + bbox_size),
            1,
            mode="clip",
        )
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if store_size != bbox_size:
            face_rgb = cv2.resize(face_rgb, (store_size, store_size))
        crops.append(face_rgb)
        frame_nums.append(frame_num)
        bbox_meta.append((int(cnt_x), int(cnt_y), int(bbox_size)))

    cap.release()
    out = {
        "crops_rgb": np.asarray(crops, dtype=np.uint8),
        "frame_nums": np.asarray(frame_nums, dtype=np.int64),
        "bbox_meta": bbox_meta,
    }
    if want_full:
        out["full_bgr"] = full_bgr
    return out


def _rgb_to_pos_input(rgb_t_e3):
    """[T,E,3] float → POS [E,3,T]."""
    x = np.asarray(rgb_t_e3, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, np.newaxis, :]
    return np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)


def _run_pos_waveform(rgb_t_e3, fs):
    pos_in = _rgb_to_pos_input(rgb_t_e3)
    bvp = signals_to_bvps_cpu(pos_in, cpu_POS, params={"fps": float(fs)})
    if bvp.ndim == 1:
        return bvp.astype(np.float32)
    return np.mean(bvp, axis=0).astype(np.float32)


def _openface_crop_rgb(crops_rgb):
    # crops_rgb [T,H,W,3] RGB
    return crops_rgb.reshape(len(crops_rgb), -1, 3).mean(axis=1)


def _grid_patch_rgb(full_bgr, bbox_meta, grid=4, rgb_low=75, rgb_high=230):
    """bbox 内 grid×grid patch 均值；过滤极端 RGB 像素。"""
    t = len(full_bgr)
    e = grid * grid
    sig = np.zeros((t, e, 3), dtype=np.float32)
    for ti, (frame, (cnt_x, cnt_y, bbox_size)) in enumerate(zip(full_bgr, bbox_meta)):
        half = bbox_size // 2
        y0, y1 = cnt_y - half, cnt_y - half + bbox_size
        x0, x1 = cnt_x - half, cnt_x - half + bbox_size
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        rh, rw = roi.shape[:2]
        ps = max(4, int(min(rh, rw) / (grid * 2)))
        if ps % 2:
            ps += 1
        half_p = ps // 2
        for gi in range(grid):
            for gj in range(grid):
                idx = gi * grid + gj
                cy = int((gi + 0.5) * rh / grid)
                cx = int((gj + 0.5) * rw / grid)
                y_a, y_b = max(0, cy - half_p), min(rh, cy + half_p + 1)
                x_a, x_b = max(0, cx - half_p), min(rw, cx + half_p + 1)
                patch = roi[y_a:y_b, x_a:x_b].reshape(-1, 3).astype(np.float32)
                if patch.size == 0:
                    continue
                keep = ~(
                    ((patch[:, 0] <= rgb_low) & (patch[:, 1] <= rgb_low) & (patch[:, 2] <= rgb_low))
                    | ((patch[:, 0] >= rgb_high) & (patch[:, 1] >= rgb_high) & (patch[:, 2] >= rgb_high))
                )
                patch = patch[keep]
                if len(patch):
                    sig[ti, idx] = patch.mean(axis=0)
    return sig


def _build_clip_waveforms(bundle, roi_mode, fs, sec, grid=4):
    crops = bundle["crops_rgb"]
    fns = bundle["frame_nums"]
    ti = round(sec * fs)
    n = len(crops) // ti
    waves, gt_waves, clip_meta = [], [], []

    for b in range(n):
        sl = slice(b * ti, (b + 1) * ti)
        if roi_mode == "openface_crop":
            rgb = _openface_crop_rgb(crops[sl])
        elif roi_mode == "skin_patch":
            rgb = _grid_patch_rgb(
                bundle["full_bgr"][sl], bundle["bbox_meta"][sl], grid=grid
            )
        else:
            raise ValueError(f"unknown roi_mode: {roi_mode}")

        wave = _run_pos_waveform(rgb, fs)
        waves.append(wave)
        clip_meta.append(
            {
                "clip_idx": b + 1,
                "frame_num_start": int(fns[sl][0]),
                "frame_num_end": int(fns[sl][-1]),
            }
        )
    return waves, n, clip_meta, fns, ti


def main():
    ap = argparse.ArgumentParser(description="Offline POS → eval_bundle (evaluate.py compatible)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--openface-dir", default=os.path.join(os.path.dirname(_CP), "OpenFace"))
    ap.add_argument(
        "--roi-mode",
        choices=("openface_crop", "skin_patch"),
        required=True,
    )
    ap.add_argument("--scales", default="10")
    ap.add_argument("--label-ratio", type=float, default=0.0)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--fs", type=float, default=None,
                        help="采样率 (fps)；默认从视频元数据自动读取")
    ap.add_argument("--input-size", type=int, default=96)
    ap.add_argument("--patch-grid", type=int, default=4)
    ap.add_argument("--skip-eval-npy", action="store_true")
    args = ap.parse_args()

    video_path = os.path.abspath(args.video)
    session_dir = os.path.abspath(args.session_dir or os.path.dirname(video_path))

    # Read actual fps from video metadata; --fs overrides only if explicitly given
    _cap_check = cv2.VideoCapture(video_path)
    _video_fps = _cap_check.get(cv2.CAP_PROP_FPS)
    _cap_check.release()
    if args.fs is not None:
        fs = float(args.fs)
        if abs(fs - _video_fps) > 1.0:
            print(f"警告: --fs={fs} 与视频实际 fps={_video_fps:.2f} 不符，使用 --fs 指定值")
    elif _video_fps > 0 and not (isinstance(_video_fps, float) and _video_fps != _video_fps):
        fs = _video_fps
    else:
        print("警告: 视频 fps 读取失败，使用默认 30")
        fs = 30.0
    print(f"FPS: {fs:.2f} (来自{'参数' if args.fs is not None else '视频元数据'})")

    store_size = int(args.input_size)

    lr_tag = format_label_ratio(args.label_ratio)
    if args.output_root:
        base = os.path.join(args.output_root, "live_runs")
    else:
        base = os.path.join(
            _CP,
            "results",
            "EfficientPhysNet",
            lr_tag,
            f"pos_{args.roi_mode}",
            "live_runs",
        )
    run_dir = os.path.join(base, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    landmark_path = args.landmarks
    if not landmark_path and os.path.isdir(args.openface_dir):
        lm_dir = os.path.join(run_dir, "landmarks")
        landmark_path = live_epn.run_openface(video_path, lm_dir, args.openface_dir)
        print(f"OpenFace -> {landmark_path}")
    if not landmark_path or not os.path.isfile(landmark_path):
        raise FileNotFoundError("需要 --landmarks 或 --openface-dir")

    want_full = args.roi_mode == "skin_patch"
    bundle = load_openface_video_bundle(
        video_path, landmark_path, store_size=store_size, want_full=want_full
    )
    print(
        f"加载 {len(bundle['crops_rgb'])} 帧 | roi_mode={args.roi_mode} | fs={fs}"
    )

    gt_pack = lri._try_load_session_gt(session_dir, video_path=video_path)
    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]

    meta = {
        "video": video_path,
        "session_dir": session_dir,
        "landmarks": os.path.abspath(landmark_path),
        "method": "POS",
        "roi_mode": args.roi_mode,
        "patch_grid": args.patch_grid if args.roi_mode == "skin_patch" else None,
        "fs": fs,
        "input_size": store_size,
        "scales_sec": scales,
        "has_gt": bool(gt_pack),
        "source": "pos_recorded_infer",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.skip_eval_npy or not gt_pack:
        print("skip eval_bundle (no GT or --skip-eval-npy)")
        return

    for sec in scales:
        waves, n_clips, clip_meta, fns, ti = _build_clip_waveforms(
            bundle, args.roi_mode, fs, sec, grid=args.patch_grid
        )
        if n_clips == 0:
            print(f"  [t{sec}s] 跳过: 帧不足")
            continue
        rppg_list, bvp_list = [], []
        for b in range(n_clips):
            sl = slice(b * ti, (b + 1) * ti)
            rppg_list.append(waves[b])
            bvp_list.append(lri._bvp_clip(gt_pack, fns[sl], fs))

        sub = os.path.join(run_dir, f"eval_bundle_t{sec}s")
        os.makedirs(sub, exist_ok=True)
        meta_eval = {
            **meta,
            "time_interval_sec": sec,
            "clips_total": n_clips,
            "clip_meta": clip_meta,
        }
        lri._save_pred_npy(sub, "session", rppg_list, bvp_list, meta_eval)
        print(f"  已写 eval_bundle: {sub}/session.npy ({n_clips} clips)")

    print(f"完成: {run_dir}")
    print(f"评估: python EfficientPhysNet/evaluation/evaluate.py {run_dir}/eval_bundle_t10s")


if __name__ == "__main__":
    main()
