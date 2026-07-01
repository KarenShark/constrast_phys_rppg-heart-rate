# -*- coding: utf-8 -*-
"""
UBFC-Phys quality-aware evaluation.

Primary: signal capture on all clips.
Secondary: HR accuracy on clips with reliable GT BVP.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_EPN = os.path.dirname(_EVAL_DIR)
_CP = os.path.dirname(_EPN)
if _CP not in sys.path:
    sys.path.insert(0, _CP)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import evaluate as ev  # noqa: E402
from utils_sig import butter_bandpass, compute_fft_peaks, hr_fft_parabolic  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(
        description="UBFC-Phys GT-quality gated evaluation"
    )
    p.add_argument("pred_dir", help="Directory with UBFC-Phys prediction .npy files")
    p.add_argument("--save-viz", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--multi-peak-ratio", type=float, default=0.70)
    p.add_argument("--gt-half-diff-bpm", type=float, default=15.0)
    p.add_argument("--gt-snr-db", type=float, default=0.0)
    p.add_argument("--parent30-diff-bpm", type=float, default=12.0)
    p.add_argument("--no-parent30-gate", action="store_true")
    p.add_argument("--max-lag-sec", type=float, default=2.0)
    return p.parse_args()


def _fmt(v, nd=3):
    if v is None:
        return "N/A"
    try:
        if np.isnan(v):
            return "N/A"
    except TypeError:
        pass
    return f"{float(v):.{nd}f}"


def _nanmean(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def _nanstd(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.std(vals)) if vals else np.nan


def _hr_stats(rows):
    errs = [
        r["hr_err"]
        for r in rows
        if r.get("hr_err") is not None and not np.isnan(r["hr_err"])
    ]
    preds = [
        r["hr_pred"]
        for r in rows
        if r.get("hr_pred") is not None and not np.isnan(r["hr_pred"])
    ]
    gts = [
        r["hr_gt"]
        for r in rows
        if r.get("hr_gt") is not None and not np.isnan(r["hr_gt"])
    ]
    if not errs:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "p5": np.nan, "p10": np.nan,
                "pearson": np.nan}
    e = np.asarray(errs, dtype=float)
    return {
        "n": len(errs),
        "mae": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "p5": float(np.mean(e <= 5) * 100),
        "p10": float(np.mean(e <= 10) * 100),
        "pearson": ev._safe_pearson(preds, gts) if len(preds) == len(gts) else np.nan,
    }


def _is_harmonic_pair(hr_a, hr_b, fs_hz, n_samples):
    lo = min(hr_a, hr_b)
    hi = max(hr_a, hr_b)
    if lo <= 0:
        return False
    resolution_bpm = fs_hz / n_samples * 60.0
    tol_bpm = max(6.0, 1.5 * resolution_bpm)
    return abs(hi - 2.0 * lo) <= tol_bpm or abs(hi - 3.0 * lo) <= tol_bpm


def _gt_peak_conflict(sig_f, fs_hz):
    peaks = compute_fft_peaks(sig_f, fs_hz)
    if not peaks:
        return {
            "gt_peak1_bpm": np.nan,
            "gt_peak2_bpm": np.nan,
            "gt_peak_ratio": np.nan,
        }
    p1 = peaks[0]
    for p2 in peaks[1:]:
        if not _is_harmonic_pair(p1["hr_bpm"], p2["hr_bpm"], fs_hz, len(sig_f)):
            return {
                "gt_peak1_bpm": float(p1["hr_bpm"]),
                "gt_peak2_bpm": float(p2["hr_bpm"]),
                "gt_peak_ratio": float(p2["power"] / (p1["power"] + 1e-12)),
            }
    return {
        "gt_peak1_bpm": float(p1["hr_bpm"]),
        "gt_peak2_bpm": np.nan,
        "gt_peak_ratio": np.nan,
    }


def _band_snr_db(sig_f, fs_hz, hr_bpm):
    if hr_bpm is None or np.isnan(hr_bpm):
        return np.nan
    x = np.asarray(sig_f, dtype=np.float64).reshape(-1)
    if len(x) < 8:
        return np.nan
    x = x - np.mean(x)
    n = len(x)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
    band = (freqs >= 0.6) & (freqs <= 4.0)
    if not np.any(band):
        return np.nan
    f0 = hr_bpm / 60.0
    width_hz = max(0.15, 1.5 * fs_hz / n)
    signal_band = np.abs(freqs - f0) <= width_hz
    if 2.0 * f0 <= 4.0:
        signal_band |= np.abs(freqs - 2.0 * f0) <= width_hz
    signal_band &= band
    signal_power = float(np.sum(spec[signal_band]))
    total_power = float(np.sum(spec[band]))
    noise_power = max(total_power - signal_power, 1e-12)
    if signal_power <= 1e-12:
        return -np.inf
    return float(10.0 * np.log10(signal_power / noise_power))


def _half_hr_diff(sig, fs_hz):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    half = len(sig) // 2
    if half < int(4.0 * fs_hz):
        return np.nan
    a = sig[:half]
    b = sig[half:]
    h1 = ev._estimate_hr(a, fs_hz, "parabolic", True, True)
    h2 = ev._estimate_hr(b, fs_hz, "parabolic", True, True)
    if np.isnan(h1) or np.isnan(h2):
        return np.nan
    return float(abs(h1 - h2))


def _estimate_gt_hr(sig_f, fs_hz):
    return ev._estimate_hr(sig_f, fs_hz, "parabolic", True, True)


def _build_parent30_refs(bvp_list, fs_hz):
    if len(bvp_list) == 0:
        return {}
    clip_dur = len(bvp_list[0]) / fs_hz
    if clip_dur <= 0 or clip_dur > 12.5:
        return {}
    group = int(round(30.0 / clip_dur))
    if group < 2 or abs(group * clip_dur - 30.0) > 2.5:
        return {}
    refs = {}
    for start in range(0, len(bvp_list) - group + 1, group):
        concat = np.concatenate([
            np.asarray(x, dtype=np.float64).reshape(-1)
            for x in bvp_list[start:start + group]
        ])
        filt = butter_bandpass(concat, 0.6, 4, fs=fs_hz)
        hr = _estimate_gt_hr(filt, fs_hz)
        for idx in range(start, start + group):
            refs[idx] = hr
    return refs


def _assess_gt_quality(sig_f, fs_hz, hr_gt, parent_hr, args):
    peak = _gt_peak_conflict(sig_f, fs_hz)
    snr = _band_snr_db(sig_f, fs_hz, hr_gt)
    half_diff = _half_hr_diff(sig_f, fs_hz)
    parent_diff = (
        abs(hr_gt - parent_hr)
        if parent_hr is not None and not np.isnan(parent_hr)
        else np.nan
    )

    reasons = []
    ratio = peak["gt_peak_ratio"]
    if not np.isnan(ratio) and ratio > args.multi_peak_ratio:
        reasons.append("multi_peak")
    if not np.isnan(half_diff) and half_diff > args.gt_half_diff_bpm:
        reasons.append("unstable_half_hr")
    if not np.isnan(snr) and snr < args.gt_snr_db:
        reasons.append("low_gt_snr")
    if (
        not args.no_parent30_gate
        and not np.isnan(parent_diff)
        and parent_diff > args.parent30_diff_bpm
    ):
        reasons.append("parent30_mismatch")

    return {
        **peak,
        "gt_snr_db": snr,
        "gt_half_hr_diff": half_diff,
        "gt_parent30_hr": parent_hr if parent_hr is not None else np.nan,
        "gt_parent30_diff": parent_diff,
        "gt_reliable": len(reasons) == 0,
        "reject_reasons": reasons,
    }


def _save_viz(row, rppg_f, bvp_f, fs_hz, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(len(rppg_f)) / fs_hz
    acf_lag = int(4.0 * fs_hz)
    acf_p = ev._acf(rppg_f, acf_lag)
    acf_g = ev._acf(bvp_f, acf_lag)
    lag_sec = np.arange(acf_lag + 1) / fs_hz

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(t, (rppg_f - np.mean(rppg_f)) / (np.std(rppg_f) + 1e-8),
               "b-", linewidth=1.0, label="pred rPPG")
    ax[0].plot(t, (bvp_f - np.mean(bvp_f)) / (np.std(bvp_f) + 1e-8),
               "r-", linewidth=1.0, alpha=0.75, label="GT BVP")
    status = "GT reliable" if row["gt_reliable"] else "GT unreliable"
    reason = ",".join(row["reject_reasons"]) if row["reject_reasons"] else "ok"
    ax[0].set_title(
        f"{row['subject']} clip {row['clip_idx'] + 1} | {status} | {reason}"
    )
    ax[0].set_ylabel("z-score")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    def _band_amp(sig_f):
        _, amp_full, _ = hr_fft_parabolic(sig_f, fs=fs_hz, harmonics_removal=True)
        xb = np.arange(len(amp_full)) / len(amp_full) * fs_hz * 60.0
        m = (xb >= 40) & (xb <= 250)
        amp = amp_full[m].astype(np.float64)
        amp = amp / (np.max(amp) + 1e-12)
        return xb[m], amp

    xb_p, ap = _band_amp(rppg_f)
    xb_g, ag = _band_amp(bvp_f)
    ax[1].plot(xb_p, ap, "b-", linewidth=1.0, label="pred |FFT|")
    ax[1].plot(xb_g, ag, "r-", linewidth=1.0, alpha=0.8, label="GT |FFT|")
    ax[1].axvline(row["hr_pred"], color="b", ls="--", alpha=0.7,
                  label=f"pred {row['hr_pred']:.1f}")
    ax[1].axvline(row["hr_gt"], color="r", ls="--", alpha=0.7,
                  label=f"GT {row['hr_gt']:.1f}")
    if not np.isnan(row["gt_peak2_bpm"]):
        ax[1].axvline(row["gt_peak2_bpm"], color="r", ls=":", alpha=0.5,
                      label=f"GT peak2 {row['gt_peak2_bpm']:.1f}")
    ax[1].set_xlim(40, 250)
    ax[1].set_xlabel("Heart Rate (bpm)")
    ax[1].set_title(
        "Amplitude spectrum | "
        f"GT peak ratio={_fmt(row['gt_peak_ratio'], 2)}, "
        f"SNR={_fmt(row['gt_snr_db'], 1)}dB"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    if acf_p is not None and acf_g is not None:
        ax[2].plot(lag_sec, acf_p, "b-", linewidth=1.0, label="pred ACF")
        ax[2].plot(lag_sec, acf_g, "r-", linewidth=1.0, alpha=0.8, label="GT ACF")
        ax[2].legend()
    ax[2].set_title("Autocorrelation (0-4s)")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{row['subject']}_{row['clip_idx'] + 1}.png")
    plt.savefig(path, dpi=120)
    plt.close()


def _write_csv(rows, out_path):
    keys = [
        "subject", "clip_idx", "fs", "duration_s",
        "psd_corr", "psd_mse", "wave_corr0", "wave_lag_corr", "acf_corr",
        "pred_ipr", "pred_half_hr_std",
        "hr_pred", "hr_gt", "hr_err", "hr_err_reliable",
        "gt_reliable", "reject_reasons",
        "gt_peak1_bpm", "gt_peak2_bpm", "gt_peak_ratio",
        "gt_snr_db", "gt_half_hr_diff", "gt_parent30_hr", "gt_parent30_diff",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            r = dict(row)
            r["reject_reasons"] = "|".join(row["reject_reasons"])
            writer.writerow({k: r.get(k, "") for k in keys})


def _write_summary(rows, out_dir, pred_dir, meta):
    signal_rows = rows
    reliable_rows = [r for r in rows if r["gt_reliable"]]
    unreliable_rows = [r for r in rows if not r["gt_reliable"]]
    all_hr = _hr_stats(rows)
    rel_hr = _hr_stats(reliable_rows)
    reason_counts = Counter()
    for row in unreliable_rows:
        reason_counts.update(row["reject_reasons"])

    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)

    lines = []
    lines.append("UBFC-Phys Quality-Aware Evaluation")
    lines.append("=" * 72)
    lines.append(f"pred_dir: {pred_dir}")
    if meta:
        lines.append(
            "meta: input_size={}, clips_total={}, native_fps={}, train_fs={}".format(
                meta.get("input_size", "N/A"),
                meta.get("clips_total", "N/A"),
                meta.get("native_fps_policy", "N/A"),
                meta.get("train_fs", "N/A"),
            )
        )
    lines.append("")

    lines.append("A. Signal Capture on All Clips")
    lines.append(
        "  PSD_corr {:.4f}±{:.4f}, PSD_MSE {:.6f}±{:.6f}, n={}".format(
            _nanmean([r["psd_corr"] for r in signal_rows]),
            _nanstd([r["psd_corr"] for r in signal_rows]),
            _nanmean([r["psd_mse"] for r in signal_rows]),
            _nanstd([r["psd_mse"] for r in signal_rows]),
            len(signal_rows),
        )
    )
    lines.append(
        "  WaveCorr@0 {:.4f}, WaveCorr@lag {:.4f}, ACF {:.4f}".format(
            _nanmean([r["wave_corr0"] for r in signal_rows]),
            _nanmean([r["wave_lag_corr"] for r in signal_rows]),
            _nanmean([r["acf_corr"] for r in signal_rows]),
        )
    )
    lines.append(
        "  Pred quality: IPR {:.4f}, pred_HR_half_std {:.2f} BPM".format(
            _nanmean([r["pred_ipr"] for r in signal_rows]),
            _nanmean([r["pred_half_hr_std"] for r in signal_rows]),
        )
    )
    lines.append("")

    total = len(rows)
    rel_n = len(reliable_rows)
    unrel_n = len(unreliable_rows)
    lines.append("B. GT Reliability Gate")
    lines.append(
        "  reliable {}/{} ({:.1f}%), unreliable {} ({:.1f}%)".format(
            rel_n, total, rel_n / max(total, 1) * 100,
            unrel_n, unrel_n / max(total, 1) * 100,
        )
    )
    if reason_counts:
        lines.append("  reject reasons:")
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"    {reason}: {count}")
    else:
        lines.append("  reject reasons: none")
    lines.append("")

    lines.append("C. HR Accuracy")
    lines.append(
        "  All clips (reference only): MAE {:.2f}, RMSE {:.2f}, "
        "P5 {:.1f}%, P10 {:.1f}%, Pearson {:.4f}, n={}".format(
            all_hr["mae"], all_hr["rmse"], all_hr["p5"], all_hr["p10"],
            all_hr["pearson"], all_hr["n"],
        )
    )
    lines.append(
        "  GT-reliable clips: MAE {:.2f}, RMSE {:.2f}, "
        "P5 {:.1f}%, P10 {:.1f}%, Pearson {:.4f}, n={}".format(
            rel_hr["mae"], rel_hr["rmse"], rel_hr["p5"], rel_hr["p10"],
            rel_hr["pearson"], rel_hr["n"],
        )
    )
    lines.append("")

    lines.append("D. Per-subject")
    for subject, sub_rows in sorted(by_subject.items()):
        sub_rel = [r for r in sub_rows if r["gt_reliable"]]
        sub_hr_all = _hr_stats(sub_rows)
        sub_hr_rel = _hr_stats(sub_rel)
        lines.append(
            "  {}: signal PSD={:.4f} ACF={:.4f} WLag={:.3f} | "
            "GT reliable {}/{} ({:.1f}%) | HR all MAE={:.2f} | "
            "HR reliable MAE={:.2f}".format(
                subject,
                _nanmean([r["psd_corr"] for r in sub_rows]),
                _nanmean([r["acf_corr"] for r in sub_rows]),
                _nanmean([r["wave_lag_corr"] for r in sub_rows]),
                len(sub_rel),
                len(sub_rows),
                len(sub_rel) / max(len(sub_rows), 1) * 100,
                sub_hr_all["mae"],
                sub_hr_rel["mae"],
            )
        )
    lines.append("")

    lines.append("E. Per-clip")
    for row in rows:
        status = "OK" if row["gt_reliable"] else "UNRELIABLE"
        reason = ",".join(row["reject_reasons"]) if row["reject_reasons"] else "-"
        lines.append(
            "  {} clip{:2d}: {} | pred={:.1f} gt={:.1f} err={:.1f} | "
            "PSD={:.3f} ACF={:.3f} | {}".format(
                row["subject"], row["clip_idx"] + 1, status,
                row["hr_pred"], row["hr_gt"], row["hr_err"],
                row["psd_corr"], row["acf_corr"], reason,
            )
        )

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = _parse_args()
    pred_dir = os.path.abspath(args.pred_dir)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(pred_dir)
    out_dir = args.out_dir or os.path.join(pred_dir, "eval_ubfc_phys_quality")
    os.makedirs(out_dir, exist_ok=True)

    meta = ev._load_meta(pred_dir)
    npy_files = [f for f in sorted(os.listdir(pred_dir)) if f.endswith(".npy")]
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")

    rows = []
    print("=" * 72)
    print("UBFC-Phys quality-aware evaluation")
    print("=" * 72)
    print(f"pred_dir: {pred_dir}")
    print(f"out_dir:  {out_dir}")

    for fname in npy_files:
        data = np.load(os.path.join(pred_dir, fname), allow_pickle=True).item()
        subject = fname.replace(".npy", "")
        fs_hz = float(data.get("fs", 30.0))
        parent_refs = _build_parent30_refs(data["bvp_list"], fs_hz)
        print(f"\n{subject} | fs={fs_hz:.3f} | clips={len(data['rppg_list'])}")

        for idx, (rppg, bvp) in enumerate(zip(data["rppg_list"], data["bvp_list"])):
            rppg = np.asarray(rppg, dtype=np.float64).reshape(-1)
            bvp = np.asarray(bvp, dtype=np.float64).reshape(-1)
            rppg_f = butter_bandpass(rppg, 0.6, 4, fs=fs_hz)
            bvp_f = butter_bandpass(bvp, 0.6, 4, fs=fs_hz)

            psd_corr, psd_mse = ev._psd_metrics(rppg_f, bvp_f, fs_hz)
            hr_pred = ev._estimate_hr(rppg_f, fs_hz, "parabolic", True, True)
            hr_gt = _estimate_gt_hr(bvp_f, fs_hz)
            hr_err = abs(hr_pred - hr_gt)
            quality = _assess_gt_quality(
                bvp_f,
                fs_hz,
                hr_gt,
                parent_refs.get(idx),
                args,
            )

            row = {
                "subject": subject,
                "clip_idx": idx,
                "fs": fs_hz,
                "duration_s": len(rppg) / fs_hz,
                "psd_corr": psd_corr,
                "psd_mse": psd_mse,
                "wave_corr0": ev._safe_pearson(rppg_f, bvp_f),
                "wave_lag_corr": ev._max_lag_corr(
                    rppg_f, bvp_f, fs_hz=fs_hz, max_lag_s=args.max_lag_sec
                ),
                "acf_corr": ev._acf_corr(rppg_f, bvp_f, fs_hz=fs_hz, acf_lag_s=4.0),
                "pred_ipr": ev._ipr_numpy(rppg, fs_hz),
                "pred_half_hr_std": ev._hr_subwindow_stability(
                    rppg, fs_hz, "parabolic", True, True
                ),
                "hr_pred": hr_pred,
                "hr_gt": hr_gt,
                "hr_err": hr_err,
                "hr_err_reliable": hr_err if quality["gt_reliable"] else np.nan,
                **quality,
            }
            rows.append(row)

            status = "OK" if row["gt_reliable"] else "UNRELIABLE"
            reason = ",".join(row["reject_reasons"]) if row["reject_reasons"] else "-"
            print(
                "  clip{:2d}: {} | HR pred={:.1f} gt={:.1f} err={:.1f} | "
                "PSD={:.3f} ACF={:.3f} | {}".format(
                    idx + 1, status, hr_pred, hr_gt, hr_err,
                    psd_corr, row["acf_corr"], reason,
                )
            )

            if args.save_viz:
                _save_viz(row, rppg_f, bvp_f, fs_hz, os.path.join(out_dir, "viz"))

    _write_csv(rows, os.path.join(out_dir, "clip_quality.csv"))
    _write_summary(rows, out_dir, pred_dir, meta)

    print("\nSaved:")
    print(f"  {os.path.join(out_dir, 'summary.txt')}")
    print(f"  {os.path.join(out_dir, 'clip_quality.csv')}")
    if args.save_viz:
        print(f"  {os.path.join(out_dir, 'viz')}")
    print("Done.")


if __name__ == "__main__":
    main()
