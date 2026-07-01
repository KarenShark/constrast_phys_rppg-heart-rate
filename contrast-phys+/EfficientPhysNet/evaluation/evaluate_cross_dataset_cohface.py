# -*- coding: utf-8 -*-
"""
COHFACE 零样本跨数据集评估 — 两层结构，per-clip native fps。

Phase 1: 波形/频域信号质量 + viz_signal/ 三 panel 图
Phase 2: 后处理 HR vs GT（MAE/RMSE/Pearson）

运行:
  cd contrast-phys+
  python EfficientPhysNet/evaluation/evaluate_cross_dataset_cohface.py \\
    results/external_eval/cohface/curriculum/t10/1 \\
    --h5-dir ../datasets/COHFACE_h5
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

_EPN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP = os.path.dirname(_EPN)
_PROJECT = os.path.dirname(_CP)
if _CP not in sys.path:
    sys.path.insert(0, _CP)
if _EPN not in sys.path:
    sys.path.insert(0, _EPN)

from evaluation import evaluate as ev
from evaluation.h5_metadata import (
    h5_stem_to_path,
    read_h5_eval_meta,
    read_h5_fps,
)
from utils_sig import butter_bandpass


def _resolve_fs(stem, pred_data, h5_dir, meta_json):
    if "fs" in pred_data and pred_data["fs"] is not None:
        return float(pred_data["fs"])
    if h5_dir:
        h5_path = h5_stem_to_path(h5_dir, stem)
        if os.path.isfile(h5_path):
            return read_h5_fps(h5_path)
    for s in meta_json.get("samples", []):
        if s.get("stem") == stem:
            return float(s["fs"])
    raise KeyError(f"Cannot resolve fs for {stem}")


def _resolve_eval_meta(stem, h5_dir, pred_data):
    h5_path = pred_data.get("h5_path")
    if h5_path and os.path.isfile(h5_path):
        return read_h5_eval_meta(h5_path)
    if h5_dir:
        p = h5_stem_to_path(h5_dir, stem)
        if os.path.isfile(p):
            return read_h5_eval_meta(p)
    return {"subject": "", "session": "", "illumination": ""}


def _save_phase1_viz(
    viz_dir, stem, clip_i, fs_hz, rppg_f, bvp_f, hr_pred, hr_gt, eval_meta
):
    os.makedirs(viz_dir, exist_ok=True)
    t = np.arange(len(rppg_f)) / fs_hz
    fbpm_p, psd_p = ev._compute_norm_psd(rppg_f, fs_hz)
    fbpm_g, psd_g = ev._compute_norm_psd(bvp_f, fs_hz)
    acf_lag = int(4.0 * fs_hz)
    acf_p = ev._acf(rppg_f, acf_lag)
    acf_g = ev._acf(bvp_f, acf_lag)
    lag_sec = np.arange(acf_lag + 1) / fs_hz

    rppg_plot = (rppg_f - np.mean(rppg_f)) / (np.std(rppg_f) + 1e-12)
    bvp_plot = (bvp_f - np.mean(bvp_f)) / (np.std(bvp_f) + 1e-12)

    subj = eval_meta.get("subject", "?")
    sess = eval_meta.get("session", "?")
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    ax[0].plot(t, rppg_plot, "b-", label="rppg (pred)", linewidth=0.8)
    ax[0].plot(t, bvp_plot, "r-", alpha=0.7, label="bvp (GT)", linewidth=0.8)
    ax[0].set_ylabel("Amplitude (z-score)")
    ax[0].set_title(
        f"COHFACE | subj={subj} sess={sess} | fs={fs_hz:g} | clip={clip_i + 1}"
    )
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    if fbpm_p is not None and fbpm_g is not None:
        ax[1].plot(fbpm_p, psd_p, "b-", linewidth=1.0, label="pred PSD")
        ax[1].plot(fbpm_g, psd_g, "r-", linewidth=1.0, alpha=0.8, label="GT PSD")
        if hr_pred is not None and not np.isnan(hr_pred):
            ax[1].axvline(hr_pred, color="b", ls="--", alpha=0.6, label=f"HR pred {hr_pred:.1f}")
        if hr_gt is not None and not np.isnan(hr_gt):
            ax[1].axvline(hr_gt, color="r", ls="--", alpha=0.6, label=f"HR GT {hr_gt:.1f}")
        ax[1].set_xlim(40, 250)
        ax[1].set_title("Normalized PSD (40-250 BPM)")
        ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    if acf_p is not None and acf_g is not None:
        ax[2].plot(lag_sec, acf_p, "b-", linewidth=1.0, label="pred ACF")
        ax[2].plot(lag_sec, acf_g, "r-", linewidth=1.0, alpha=0.8, label="GT ACF")
        ax[2].set_title("Autocorrelation (0-4s)")
        ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    ax[2].set_xlabel("Time (s) / lag")

    out_name = f"{stem}_clip{clip_i + 1}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, out_name), dpi=120)
    plt.close()


def _percentile_err(errors, p):
    if not errors:
        return np.nan
    return float(np.percentile(np.abs(errors), p))


def run_evaluation(pred_dir, h5_dir, save_viz=True):
    pred_dir = os.path.abspath(pred_dir)
    eval_out = os.path.join(pred_dir, "eval")
    os.makedirs(eval_out, exist_ok=True)
    viz_dir = os.path.join(eval_out, "viz_signal")

    meta_path = os.path.join(pred_dir, "meta.json")
    meta_json = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta_json = json.load(f)

    npy_files = sorted(
        f for f in os.listdir(pred_dir) if f.endswith(".npy")
    )
    if not npy_files:
        raise FileNotFoundError(f"No .npy in {pred_dir}")

    phase1_clips = []
    phase2_hrs = []
    hr_by_illum = {"lamp": [], "natural": [], "unknown": []}

    print("=" * 68)
    print("COHFACE cross-dataset eval | Phase1 signal + Phase2 HR")
    print("=" * 68)
    print(f"pred_dir: {pred_dir}")
    if meta_json.get("native_fps_policy"):
        print(
            f"native_fps_policy=True | train_fs={meta_json.get('train_fs')} | "
            f"eval_fps={meta_json.get('eval_fps_values')}"
        )

    for npy_name in npy_files:
        stem = npy_name.replace(".npy", "")
        if stem.startswith("pred_"):
            continue
        data = np.load(os.path.join(pred_dir, npy_name), allow_pickle=True).item()
        fs_hz = _resolve_fs(stem, data, h5_dir, meta_json)
        eval_meta = _resolve_eval_meta(stem, h5_dir, data)
        illum = eval_meta.get("illumination") or "unknown"
        if illum not in hr_by_illum:
            hr_by_illum[illum] = []

        rppg_list = data["rppg_list"]
        bvp_list = data["bvp_list"]

        for i, (rppg, bvp) in enumerate(zip(rppg_list, bvp_list)):
            rppg = np.asarray(rppg).reshape(-1)
            bvp = np.asarray(bvp).reshape(-1)
            n = min(len(rppg), len(bvp))
            rppg, bvp = rppg[:n], bvp[:n]

            rppg_f = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fs_hz)
            bvp_f = butter_bandpass(bvp, lowcut=0.6, highcut=4, fs=fs_hz)

            psd_corr, psd_mse = ev._psd_metrics(rppg_f, bvp_f, fs_hz)
            w0 = ev._safe_pearson(rppg_f, bvp_f)
            w_lag = ev._max_lag_corr(rppg_f, bvp_f, fs_hz, max_lag_s=2.0)
            acf_c = ev._acf_corr(rppg_f, bvp_f, fs_hz)
            ipr = ev._ipr_numpy(rppg, fs_hz)

            hr_pred = ev._estimate_hr(
                rppg_f, fs_hz, "parabolic", True, True
            )
            hr_gt = ev._estimate_hr(bvp_f, fs_hz, "parabolic", True, True)
            cap_snr = ev._capture_snr_db(rppg_f, hr_gt, fs_hz)
            hr_err = (
                abs(hr_pred - hr_gt)
                if not (np.isnan(hr_pred) or np.isnan(hr_gt))
                else np.nan
            )

            rec1 = {
                "stem": stem,
                "clip_idx": i,
                "fs": fs_hz,
                "subject": eval_meta.get("subject"),
                "session": eval_meta.get("session"),
                "illumination": illum,
                "psd_corr": float(psd_corr) if not np.isnan(psd_corr) else None,
                "psd_mse": float(psd_mse) if not np.isnan(psd_mse) else None,
                "wave_corr_0": float(w0) if not np.isnan(w0) else None,
                "wave_corr_lag": float(w_lag) if not np.isnan(w_lag) else None,
                "acf_corr": float(acf_c) if not np.isnan(acf_c) else None,
                "ipr": float(ipr) if not np.isnan(ipr) else None,
                "capture_snr_db": float(cap_snr) if not np.isnan(cap_snr) else None,
            }
            phase1_clips.append(rec1)

            rec2 = {
                **rec1,
                "hr_pred": float(hr_pred) if not np.isnan(hr_pred) else None,
                "hr_gt": float(hr_gt) if not np.isnan(hr_gt) else None,
                "hr_error": float(hr_err) if not np.isnan(hr_err) else None,
            }
            phase2_hrs.append(rec2)
            if rec2["hr_error"] is not None:
                hr_by_illum.setdefault(illum, []).append(rec2["hr_error"])

            if save_viz:
                _save_phase1_viz(
                    viz_dir, stem, i, fs_hz, rppg_f, bvp_f, hr_pred, hr_gt, eval_meta
                )

            print(
                f"  {stem} clip{i + 1} fs={fs_hz:g} | "
                f"PSD={ev._fmt(psd_corr)} | HR err={ev._fmt(hr_err)}"
            )

    def _nanmean(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else np.nan

    hr_errors = [r["hr_error"] for r in phase2_hrs if r.get("hr_error") is not None]
    hr_pred = [r["hr_pred"] for r in phase2_hrs if r.get("hr_pred") is not None]
    hr_gt = [r["hr_gt"] for r in phase2_hrs if r.get("hr_gt") is not None]

    phase2_summary = {
        "n_clips": len(phase2_hrs),
        "mae": float(np.mean(hr_errors)) if hr_errors else None,
        "rmse": float(np.sqrt(np.mean(np.array(hr_errors) ** 2))) if hr_errors else None,
        "p5": _percentile_err(hr_errors, 5),
        "p10": _percentile_err(hr_errors, 10),
        "pearson_hr": ev._safe_pearson(hr_pred, hr_gt) if len(hr_pred) >= 5 else None,
        "by_illumination": {},
    }
    for illum, errs in hr_by_illum.items():
        if errs:
            phase2_summary["by_illumination"][illum] = {
                "n": len(errs),
                "mae": float(np.mean(errs)),
            }

    phase1_summary = {
        "n_clips": len(phase1_clips),
        "psd_corr_mean": _nanmean("psd_corr", phase1_clips),
        "psd_mse_mean": _nanmean("psd_mse", phase1_clips),
        "wave_corr_0_mean": _nanmean("wave_corr_0", phase1_clips),
        "wave_corr_lag_mean": _nanmean("wave_corr_lag", phase1_clips),
        "acf_corr_mean": _nanmean("acf_corr", phase1_clips),
        "capture_snr_db_mean": _nanmean("capture_snr_db", phase1_clips),
    }

    with open(os.path.join(eval_out, "phase1_signal_quality.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": phase1_summary, "clips": phase1_clips}, f, indent=2)
    with open(os.path.join(eval_out, "phase2_hr_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": phase2_summary, "clips": phase2_hrs}, f, indent=2)

    lines = [
        "COHFACE Cross-Dataset Evaluation",
        f"pred_dir: {pred_dir}",
        "",
        "=== Phase 1: Signal Quality ===",
        f"  PSD Pearson:  {ev._fmt(phase1_summary['psd_corr_mean'])}",
        f"  PSD MSE:      {ev._fmt(phase1_summary['psd_mse_mean'], 6)}",
        f"  WaveCorr@0:   {ev._fmt(phase1_summary['wave_corr_0_mean'])}",
        f"  WaveCorr@lag: {ev._fmt(phase1_summary['wave_corr_lag_mean'])}",
        f"  ACF Corr:     {ev._fmt(phase1_summary['acf_corr_mean'])}",
        f"  Capture-SNR:  {ev._fmt(phase1_summary['capture_snr_db_mean'], 2)} dB",
        "",
        "=== Phase 2: HR Device-level ===",
        f"  MAE:     {ev._fmt(phase2_summary['mae'])} BPM",
        f"  RMSE:    {ev._fmt(phase2_summary['rmse'])} BPM",
        f"  P5/P10:  {ev._fmt(phase2_summary['p5'])} / {ev._fmt(phase2_summary['p10'])} BPM",
        f"  Pearson: {ev._fmt(phase2_summary['pearson_hr'])}",
    ]
    for illum, st in phase2_summary.get("by_illumination", {}).items():
        lines.append(f"  MAE [{illum}]: {ev._fmt(st['mae'])} (n={st['n']})")
    if save_viz:
        lines.append(f"\n  viz: {viz_dir}/")

    summary_path = os.path.join(eval_out, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {summary_path}")
    return phase1_summary, phase2_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir", help="推理输出目录")
    ap.add_argument(
        "--h5-dir",
        default=os.path.join(_PROJECT, "datasets", "COHFACE_h5"),
        help="H5 目录，用于回读 native fps / metadata",
    )
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()
    run_evaluation(args.pred_dir, args.h5_dir, save_viz=not args.no_viz)


if __name__ == "__main__":
    main()
