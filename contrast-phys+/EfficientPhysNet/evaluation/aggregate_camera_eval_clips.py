#!/usr/bin/env python3
"""Re-run evaluate --save-viz and aggregate per-clip Layer1+2 CSV for a camera benchmark run."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

CP = Path(__file__).resolve().parents[2]
EVAL_PY = CP / "EfficientPhysNet" / "evaluation" / "evaluate.py"

CLIP_FIELDS = [
    "subject",
    "session",
    "camera_key",
    "camera_label",
    "clip_idx",
    "clip_dur_s",
    "psd_corr",
    "psd_mse",
    "capture_snr_db",
    "ipr",
    "wave_corr0",
    "wave_corr_lag",
    "acf_corr",
    "hr_half_std",
    "hr_pred_bpm",
    "hr_gt_bpm",
    "err_bpm",
    "viz_png",
    "eval_bundle_dir",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_evaluate(eval_bundle, save_viz, python_bin):
    cmd = [python_bin, str(EVAL_PY), str(eval_bundle)]
    if save_viz:
        cmd.append("--save-viz")
    subprocess.run(cmd, cwd=str(CP), check=True)


def collect_clip_row(session_row, clip_row):
    eval_bundle = Path(session_row["eval_bundle_dir"])
    clip_idx = int(clip_row["clip_idx"])
    viz_png = eval_bundle / "eval" / "viz_waveform" / f"{clip_row['subject']}_{clip_idx}.png"
    return {
        "subject": session_row["subject"],
        "session": session_row["session"],
        "camera_key": session_row["camera_key"],
        "camera_label": session_row["camera_label"],
        "clip_idx": clip_idx,
        "clip_dur_s": clip_row.get("clip_dur_s"),
        "psd_corr": clip_row.get("psd_corr"),
        "psd_mse": clip_row.get("psd_mse"),
        "capture_snr_db": clip_row.get("capture_snr_db"),
        "ipr": clip_row.get("ipr"),
        "wave_corr0": clip_row.get("wave_corr0"),
        "wave_corr_lag": clip_row.get("wave_corr_lag"),
        "acf_corr": clip_row.get("acf_corr"),
        "hr_half_std": clip_row.get("hr_half_std"),
        "hr_pred_bpm": clip_row.get("hr_pred"),
        "hr_gt_bpm": clip_row.get("hr_gt"),
        "err_bpm": clip_row.get("error"),
        "viz_png": str(viz_png) if viz_png.is_file() else "",
        "eval_bundle_dir": str(eval_bundle),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_root", type=Path, help="camera_compare run directory")
    p.add_argument("--save-viz", action="store_true", help="re-run evaluate.py with --save-viz")
    p.add_argument("--skip-eval", action="store_true", help="only aggregate existing clip_details.csv")
    p.add_argument("--python-bin", default=sys.executable)
    args = p.parse_args()

    run_root = args.run_root.resolve()
    session_csv = run_root / "camera_session_results.csv"
    if not session_csv.is_file():
        raise SystemExit(f"missing {session_csv}")

    sessions = [r for r in read_csv(session_csv) if r.get("status") == "ok"]
    all_clips = []
    for row in sessions:
        eval_bundle = Path(row["eval_bundle_dir"])
        clip_csv = eval_bundle / "eval" / "clip_details.csv"
        if args.save_viz and not args.skip_eval:
            print(f"[evaluate] {row['subject']}/{row['camera_key']}")
            run_evaluate(eval_bundle, save_viz=True, python_bin=args.python_bin)
        elif not args.skip_eval and not clip_csv.is_file():
            print(f"[evaluate] {row['subject']}/{row['camera_key']} (no clip_details)")
            run_evaluate(eval_bundle, save_viz=args.save_viz, python_bin=args.python_bin)

        if not clip_csv.is_file():
            raise SystemExit(f"missing {clip_csv} after evaluate")
        for clip in read_csv(clip_csv):
            all_clips.append(collect_clip_row(row, clip))

    out_path = run_root / "all_clips_layer12.csv"
    write_csv(all_clips, out_path, CLIP_FIELDS)
    print(f"wrote {len(all_clips)} clips -> {out_path}")


if __name__ == "__main__":
    main()
