#!/usr/bin/env python3
"""Per-clip OpenFace ROI audit vs Layer2 eval metrics."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CP = Path(__file__).resolve().parents[2]
FS = 30
FRAMES_PER_CLIP = 10 * FS

VIDEO_WH = {
    "video_RAW_YUV420": (640, 480),
    "android_KB2505160252": (480, 640),
    "ios_00008110_000A55643AEB801E": (480, 640),
}


def find_landmark_csv(live_run_dir):
    d = Path(live_run_dir) / "landmarks"
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.csv"))
    return hits[0] if hits else None


def load_landmark_table(path):
    df = pd.read_csv(path)
    success_col = "success" if "success" in df.columns else " success"
    x_prefix = "x_" if "x_0" in df.columns else " x_"
    y_prefix = "y_" if "y_0" in df.columns else " y_"
    xs = df[[f"{x_prefix}{i}" for i in range(68)]].to_numpy(dtype=float)
    ys = df[[f"{y_prefix}{i}" for i in range(68)]].to_numpy(dtype=float)
    frame_ids = df["frame"].to_numpy(dtype=int) if "frame" in df.columns else np.arange(1, len(df) + 1)
    success = df[success_col].astype(int).to_numpy()
    conf = df["confidence"].to_numpy(dtype=float) if "confidence" in df.columns else np.full(len(df), np.nan)
    return frame_ids, success, conf, xs, ys


def pipeline_trace(xs, ys, success):
    """Mirror live_predict_webcam_EPN crop: fixed bbox_size, smoothed center."""
    n = len(success)
    bbox_size = None
    for i in range(n):
        if success[i]:
            h = np.max(ys[i]) - np.min(ys[i])
            bbox_size = int(np.round(1.5 * h))
            break
    if bbox_size is None:
        return None

    cnt_x = np.full(n, np.nan)
    cnt_y = np.full(n, np.nan)
    face_h = np.full(n, np.nan)
    lm_x_prev = lm_y_prev = None
    for i in range(n):
        if success[i]:
            lm_x_ = xs[i]
            lm_y_ = ys[i]
            lm_x = 0.9 * lm_x_prev + 0.1 * lm_x_ if lm_x_prev is not None else lm_x_
            lm_y = 0.9 * lm_y_prev + 0.1 * lm_y_ if lm_y_prev is not None else lm_y_
            lm_x_prev, lm_y_prev = lm_x, lm_y
            minx, maxx = np.min(lm_x), np.max(lm_x)
            miny, maxy = np.min(lm_y), np.max(lm_y)
            y_ext = (maxy - miny) * 0.2
            miny = miny - y_ext
            cnt_x[i] = (minx + maxx) / 2.0
            cnt_y[i] = (miny + maxy) / 2.0
            face_h[i] = maxy - miny
        elif lm_x_prev is not None:
            minx, maxx = np.min(lm_x_prev), np.max(lm_x_prev)
            miny, maxy = np.min(lm_y_prev), np.max(lm_y_prev)
            y_ext = (maxy - miny) * 0.2
            miny = miny - y_ext
            cnt_x[i] = (minx + maxx) / 2.0
            cnt_y[i] = (miny + maxy) / 2.0
            face_h[i] = maxy - miny

    return {
        "bbox_size_fixed": float(bbox_size),
        "cnt_x": cnt_x,
        "cnt_y": cnt_y,
        "face_h": face_h,
    }


def clip_metrics(trace, success, conf, frame_lo, frame_hi, vw, vh):
    """Metrics on inclusive frame index range (0-based, matches inference frame_num)."""
    sl = slice(frame_lo, frame_hi + 1)
    s = success[sl]
    c = conf[sl]
    cx = trace["cnt_x"][sl]
    cy = trace["cnt_y"][sl]
    fh = trace["face_h"][sl]
    bbox = trace["bbox_size_fixed"]
    valid = np.isfinite(cx)

    if valid.sum() < 10:
        return None

    cx_v, cy_v, fh_v = cx[valid], cy[valid], fh[valid]
    half = bbox / 2.0
    edge = (
        (cx_v - half < 0)
        | (cx_v + half > vw)
        | (cy_v - half < 0)
        | (cy_v + half > vh)
    )

    # scale mismatch: current face vs fixed bbox from clip1 frame0
    scale_ratio = fh_v / (bbox / 1.5 + 1e-6)

    fail_carry = int(np.sum((s == 0) & valid))

    return {
        "n_frames": int(frame_hi - frame_lo + 1),
        "success_rate": float(np.mean(s)),
        "conf_mean": float(np.nanmean(c)),
        "conf_min": float(np.nanmin(c)) if np.any(np.isfinite(c)) else np.nan,
        "center_std_px": float(np.sqrt(np.nanvar(cx_v) + np.nanvar(cy_v))),
        "center_max_step_px": float(
            np.nanmax(np.sqrt(np.diff(cx_v) ** 2 + np.diff(cy_v) ** 2))
            if len(cx_v) > 1
            else 0.0
        ),
        "face_height_std_px": float(np.nanstd(fh_v)),
        "face_height_range_px": float(np.nanmax(fh_v) - np.nanmin(fh_v)),
        "scale_ratio_mean": float(np.mean(scale_ratio)),
        "scale_ratio_std": float(np.std(scale_ratio)),
        "bbox_fixed_px": bbox,
        "edge_clip_frac": float(np.mean(edge)),
        "fail_carry_frames": fail_carry,
    }


def load_clip_ranges(live_run_dir):
    p = Path(live_run_dir) / "inference_by_scale.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text())["t10s"]["clips"]
    out = {}
    for c in data:
        out[int(c["clip_idx"])] = (int(c["frame_num_start"]), int(c["frame_num_end"]))
    return out


def main():
    run_root = Path(sys.argv[1]) if len(sys.argv) > 1 else CP / "results/EfficientPhysNet/label_ratio_0/camera_compare_self_recorded/20260622_101519"
    clips_eval = list(csv.DictReader(open(run_root / "all_clips_layer12.csv")))
    sessions = list(csv.DictReader(open(run_root / "camera_session_results.csv")))

    live_by_key = {}
    for row in sessions:
        if row.get("status") != "ok":
            continue
        live_by_key[(row["subject"], row["camera_key"])] = row["live_run_dir"]

    rows = []
    for ev in clips_eval:
        key = (ev["subject"], ev["camera_key"])
        live_dir = live_by_key.get(key)
        if not live_dir:
            continue
        lm_path = find_landmark_csv(live_dir)
        if not lm_path:
            continue
        clip_idx = int(ev["clip_idx"])
        ranges = load_clip_ranges(live_dir)
        if clip_idx not in ranges:
            continue
        frame_lo, frame_hi = ranges[clip_idx]

        frame_ids, success, conf, xs, ys = load_landmark_table(lm_path)
        # inference frame_num is 0-based index into landmark rows order
        if frame_hi >= len(success):
            frame_hi = len(success) - 1

        trace = pipeline_trace(xs, ys, success)
        if trace is None:
            continue
        vw, vh = VIDEO_WH.get(ev["camera_key"], (640, 480))
        m = clip_metrics(trace, success, conf, frame_lo, frame_hi, vw, vh)
        if m is None:
            continue

        err = float(ev["err_bpm"])
        tier = "good" if err <= 2 else ("bad" if err > 5 else "mid")
        rows.append(
            {
                "subject": ev["subject"],
                "camera_key": ev["camera_key"],
                "clip_idx": clip_idx,
                "err_bpm": err,
                "psd_corr": float(ev["psd_corr"]),
                "hr_pred_bpm": float(ev["hr_pred_bpm"]),
                "hr_gt_bpm": float(ev["hr_gt_bpm"]),
                "tier": tier,
                **m,
            }
        )

    out_csv = run_root / "landmark_audit_clips.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary stats
    print(f"wrote {len(rows)} rows -> {out_csv}\n")
    tiers = ["good", "mid", "bad"]
    metrics = [
        "success_rate",
        "conf_mean",
        "center_std_px",
        "center_max_step_px",
        "face_height_std_px",
        "scale_ratio_mean",
        "edge_clip_frac",
        "fail_carry_frames",
    ]
    print("=== tier means (all cameras) ===")
    print("tier\tn\t" + "\t".join(metrics))
    for t in tiers:
        sub = [r for r in rows if r["tier"] == t]
        if not sub:
            continue
        vals = [str(len(sub))]
        for k in metrics:
            vals.append(f"{np.mean([float(r[k]) for r in sub]):.3f}")
        print(t + "\t" + "\t".join(vals))

    print("\n=== corr with err_bpm (all clips) ===")
    errs = np.array([float(r["err_bpm"]) for r in rows])
    for k in metrics:
        x = np.array([float(r[k]) for r in rows])
        if np.std(x) > 0:
            print(f"  {k}: r={np.corrcoef(x, errs)[0,1]:.3f}")

    print("\n=== corr with psd_corr ===")
    psd = np.array([float(r["psd_corr"]) for r in rows])
    for k in metrics:
        x = np.array([float(r[k]) for r in rows])
        if np.std(x) > 0:
            print(f"  {k}: r={np.corrcoef(x, psd)[0,1]:.3f}")

    # Hugo/Karen android bad vs good subjects android
    print("\n=== android only: good-subject vs Hugo/Karen ===")
    good_subs = {"Baldwin", "Benny", "Eric", "Valetta"}
    for label, filt in [
        ("good_subj_android", lambda r: "android" in r["camera_key"] and r["subject"] in good_subs),
        ("Hugo_android", lambda r: r["subject"] == "Hugo" and "android" in r["camera_key"]),
        ("Karen_android", lambda r: r["subject"] == "Karen" and "android" in r["camera_key"]),
    ]:
        sub = [r for r in rows if filt(r)]
        if not sub:
            continue
        print(f"\n{label} n={len(sub)} err_mean={np.mean([float(r['err_bpm']) for r in sub]):.2f}")
        for k in metrics:
            print(f"  {k}: {np.mean([float(r[k]) for r in sub]):.3f}")

    print("\n=== worst 10 clips by err ===")
    for r in sorted(rows, key=lambda x: -float(x["err_bpm"]))[:10]:
        print(
            f"{r['subject']} {r['camera_key']} c{r['clip_idx']}: err={float(r['err_bpm']):.1f} "
            f"psd={float(r['psd_corr']):.2f} succ={float(r['success_rate']):.2f} "
            f"ctr_std={float(r['center_std_px']):.1f} fh_std={float(r['face_height_std_px']):.1f} "
            f"scale={float(r['scale_ratio_mean']):.2f} edge={float(r['edge_clip_frac']):.2f}"
        )


if __name__ == "__main__":
    main()
