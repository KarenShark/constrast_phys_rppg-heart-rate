#!/usr/bin/env python3
"""Karen Android 3-way viz: A old sess/old phone, B new sess/old phone, C new sess/new phone."""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CP))

from EfficientPhysNet.evaluation.evaluate import _compute_norm_psd, _estimate_hr
from utils_sig import hr_fft_parabolic

FS = 30
GROUPS = {
    "A: old session\nold phone (KB25)": {
        "bundle": CP
        / "results/EfficientPhysNet/label_ratio_0/camera_compare_self_recorded/20260622_101519"
        / "per_camera/Karen/v01/android_KB2505160252/live_runs/2026-06-22_10-19-13/eval_bundle_t10s",
        "color": "#1f77b4",
    },
    "B: new session\nold phone (KB25)": {
        "bundle": CP
        / "results/EfficientPhysNet/label_ratio_0/camera_compare_android/20260624_karen_full"
        / "per_camera/Karen/karen/android_KB2505160252/live_runs/2026-06-24_10-08-12/eval_bundle_t10s",
        "color": "#ff7f0e",
    },
    "C: new session\nnew phone": {
        "bundle": CP
        / "results/EfficientPhysNet/label_ratio_0/camera_compare_android/20260624_karen_full"
        / "per_camera/Karen/karen/android_NEWPHONE/live_runs/2026-06-24_10-08-29/eval_bundle_t10s",
        "color": "#2ca02c",
    },
}
OUT = (
    CP
    / "results/EfficientPhysNet/label_ratio_0/camera_compare_android/20260624_karen_full/viz"
)


def load_group(bundle_dir):
    bundle_dir = Path(bundle_dir)
    data = np.load(bundle_dir / "session.npy", allow_pickle=True).item()
    clips = list(csv.DictReader(open(bundle_dir / "eval/clip_details.csv")))
    return data["rppg_list"], data["bvp_list"], clips


def band_spectrum(sig, fs=FS):
    _, amp, _ = hr_fft_parabolic(sig, fs=fs, harmonics_removal=True)
    n = len(sig)
    freqs_bpm = np.arange(n) / n * fs * 60.0
    m = (freqs_bpm >= 40) & (freqs_bpm <= 250)
    a = amp[m].astype(np.float64)
    s = a.max() if a.size and a.max() > 0 else 1.0
    return freqs_bpm[m], a / s


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def plot_summary_bars(all_data, out_path):
    names = list(GROUPS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("MAE (drop clip1)", "mae", "BPM"),
        ("Mean PSD corr (drop clip1)", "psd", ""),
        ("P5 |err|≤5 (drop clip1)", "p5", "%"),
    ]
    for ax, (title, key, unit) in zip(axes, metrics):
        vals = [all_data[n][key] for n in names]
        cols = [GROUPS[n]["color"] for n in names]
        bars = ax.bar(range(len(names)), vals, color=cols, edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.1f}{unit}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Karen Android 3-way comparison (EfficientPhysNet t10s)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_clip_errors(all_data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(GROUPS.keys())
    xlabels = sorted(
        {c["clip_idx"] for n in names for c in all_data[n]["per_clip"] if c["clip_idx"] != 1}
    )

    for ax, metric, ylabel in zip(
        axes, ["abs_err", "psd_corr"], ["|HR error| (BPM)", "PSD corr"]
    ):
        w = 0.25
        xs = np.arange(len(xlabels))
        for i, name in enumerate(names):
            lookup = {c["clip_idx"]: c for c in all_data[name]["per_clip"]}
            ys = [lookup.get(ci, {}).get(metric, np.nan) for ci in xlabels]
            ax.bar(xs + (i - 1) * w, ys, width=w, label=name.split("\n")[0], color=GROUPS[name]["color"])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"clip {c}" for c in xlabels])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if metric == "abs_err":
            ax.axhline(5, color="gray", ls="--", alpha=0.5, label="5 BPM")

    fig.suptitle("Per-clip metrics (clip1 excluded from bars)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hr_pred_gt(all_data, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(GROUPS.keys())
    clips = sorted(
        {c["clip_idx"] for n in names for c in all_data[n]["per_clip"] if c["clip_idx"] != 1}
    )
    w = 0.12
    for ci, clip_idx in enumerate(clips):
        for j, name in enumerate(names):
            lookup = {c["clip_idx"]: c for c in all_data[name]["per_clip"]}
            if clip_idx not in lookup:
                continue
            c = lookup[clip_idx]
            x0 = ci + (j - 1) * w
            ax.bar(x0, c["hr_gt"], width=w * 0.9, color="lightgray", edgecolor="k", linewidth=0.3)
            ax.bar(
                x0,
                c["hr_pred"],
                width=w * 0.5,
                color=GROUPS[name]["color"],
                alpha=0.85,
                label=name.split("\n")[0] if ci == 0 and j == 0 else None,
            )
    ax.set_xticks(range(len(clips)))
    ax.set_xticklabels([f"clip {c}" for c in clips])
    ax.set_ylabel("HR (BPM)")
    ax.set_title("HR GT (wide gray) vs pred (color) per clip — 3 groups overlaid")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clip_3way(all_data, clip_idx, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    t = np.arange(300) / FS
    for col, (name, pack) in enumerate(all_data.items()):
        if clip_idx > len(pack["rppg"]) or clip_idx < 1:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            axes[2, col].set_visible(False)
            continue
        i = clip_idx - 1
        rppg = pack["rppg"][i]
        bvp = pack["bvp"][i]
        clip = pack["per_clip"][i]
        err = clip["abs_err"]
        psd = clip["psd_corr"]
        hr_p, hr_g = clip["hr_pred"], clip["hr_gt"]

        axes[0, col].plot(t, zscore(rppg), "b-", lw=0.8, label="rPPG pred")
        axes[0, col].plot(t, zscore(bvp), "r-", lw=0.8, alpha=0.7, label="BVP GT")
        axes[0, col].set_title(f"{name.split(chr(10))[0]}\nerr={err:.1f} PSD={psd:.2f}")
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.3)

        xb_p, ap = band_spectrum(rppg)
        xb_g, ag = band_spectrum(bvp)
        axes[1, col].plot(xb_p, ap, "b-", label="pred |FFT|")
        axes[1, col].plot(xb_g, ag, "r-", alpha=0.8, label="GT |FFT|")
        axes[1, col].axvline(hr_p, color="b", ls="--", alpha=0.6)
        axes[1, col].axvline(hr_g, color="r", ls="--", alpha=0.6)
        axes[1, col].set_xlim(40, 180)
        axes[1, col].set_xlabel("BPM")
        axes[1, col].set_title(f"HR pred={hr_p:.0f} GT={hr_g:.0f}")
        axes[1, col].legend(fontsize=7)
        axes[1, col].grid(True, alpha=0.3)

        fb, pr = _compute_norm_psd(rppg, FS)
        _, pb = _compute_norm_psd(bvp, FS)
        if pr is not None and pb is not None:
            n = min(len(pr), len(pb))
            axes[2, col].plot(fb[:n], pr[:n], "b-", label="pred PSD")
            axes[2, col].plot(fb[:n], pb[:n], "r-", alpha=0.8, label="GT PSD")
            axes[2, col].legend(fontsize=7)
        axes[2, col].set_xlabel("BPM")
        axes[2, col].set_title("Normalized PSD")
        axes[2, col].grid(True, alpha=0.3)

    fig.suptitle(f"Karen clip {clip_idx} — waveform / amplitude spectrum / PSD (3 groups)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for name, cfg in GROUPS.items():
        rppg, bvp, clips = load_group(cfg["bundle"])
        per = []
        for c in clips:
            per.append(
                {
                    "clip_idx": int(c["clip_idx"]),
                    "hr_pred": float(c["hr_pred"]),
                    "hr_gt": float(c["hr_gt"]),
                    "abs_err": abs(float(c["error"])),
                    "psd_corr": float(c["psd_corr"]),
                }
            )
        sel = [p for p in per if p["clip_idx"] != 1]
        all_data[name] = {
            "rppg": rppg,
            "bvp": bvp,
            "per_clip": per,
            "mae": float(np.mean([p["abs_err"] for p in sel])) if sel else np.nan,
            "psd": float(np.mean([p["psd_corr"] for p in sel])) if sel else np.nan,
            "p5": 100 * float(np.mean([p["abs_err"] <= 5 for p in sel])) if sel else np.nan,
        }

    plot_summary_bars(all_data, OUT / "01_summary_mae_psd_p5.png")
    plot_per_clip_errors(all_data, OUT / "02_per_clip_err_psd.png")
    plot_hr_pred_gt(all_data, OUT / "03_hr_pred_vs_gt_by_clip.png")

    for clip_idx in [2, 3, 4, 5, 6]:
        if any(clip_idx <= len(all_data[n]["rppg"]) for n in all_data):
            plot_clip_3way(all_data, clip_idx, OUT / f"04_clip{clip_idx}_waveform_psd_3way.png")

    # export viz index
    with open(OUT / "README.txt", "w", encoding="utf-8") as f:
        f.write("Karen Android 3-way visualization outputs\n")
        f.write("A = old session 2026-06-17, KB2505160252\n")
        f.write("B = new session 2026-06-24, KB2505160252\n")
        f.write("C = new session 2026-06-24, NEWPHONE\n")
        for name, d in all_data.items():
            f.write(f"\n{name}: MAE={d['mae']:.2f} PSD={d['psd']:.3f} P5={d['p5']:.0f}%\n")

    print(f"Viz saved to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
