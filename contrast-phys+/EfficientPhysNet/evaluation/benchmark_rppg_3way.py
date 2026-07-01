#!/usr/bin/env python3
"""
三条 rPPG 线对比：EfficientPhysNet | POS(openface_crop) | POS(skin_patch)

控制变量：同一 video / session GT / 10s clip / evaluate.py 指标。
默认 skip clip1；EPN 可复用已有 live_run。

用法:
  cd contrast-phys+
  python EfficientPhysNet/evaluation/benchmark_rppg_3way.py \\
    --manifest EfficientPhysNet/evaluation/configs/rppg_3way_karen_android.json
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CP = Path(__file__).resolve().parents[2]
_EPN = CP / "EfficientPhysNet"
EVAL_PY = CP / "EfficientPhysNet" / "evaluation" / "evaluate.py"
EPN_INFER = CP / "EfficientPhysNet" / "live_recorded_infer.py"
POS_INFER = CP / "EfficientPhysNet" / "pos_recorded_infer.py"

DEFAULT_MANIFEST = _EPN / "evaluation" / "configs" / "rppg_3way_karen_android.json"


def parse_args():
    p = argparse.ArgumentParser(description="3-way rPPG benchmark: EPN vs POS×2")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--results-root", default="")
    p.add_argument("--strategy", default="curriculum")
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--label-ratio", type=float, default=0.0)
    p.add_argument("--openface-dir", default=str(CP.parent / "OpenFace"))
    p.add_argument("--skip-clips", default="1", help="Comma clip_idx to exclude in summary")
    p.add_argument("--run-epn", action="store_true", help="Run EPN infer (default: reuse epn_run_dir)")
    p.add_argument("--run-pos", action="store_true", help="Run both POS modes")
    p.add_argument("--run-eval", action="store_true", help="Run evaluate.py on bundles")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd, log_path, cwd, dry_run=False):
    text = " ".join(str(c) for c in cmd)
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"[dry-run] {text}\n", encoding="utf-8")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    body = [f"$ {text}"]
    if r.stdout:
        body.append("[stdout]\n" + r.stdout)
    if r.stderr:
        body.append("[stderr]\n" + r.stderr)
    log_path.write_text("\n".join(body), encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {text}")
    return r.returncode


def latest_live_run(output_root):
    live = Path(output_root) / "live_runs"
    cands = sorted((p for p in live.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no live_runs under {live}")
    return cands[-1]


def eval_bundle_dir(live_run, scale):
    return live_run / f"eval_bundle_t{scale}s"


def run_evaluate(bundle_dir, python_bin, log_path, dry_run=False):
    run_cmd([python_bin, str(EVAL_PY), str(bundle_dir), "--save-viz"], log_path, CP, dry_run)


def read_clip_details(bundle_dir):
    p = Path(bundle_dir) / "eval" / "clip_details.csv"
    if not p.is_file():
        return []
    rows = []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_summary(bundle_dir):
    p = Path(bundle_dir) / "eval" / "summary.txt"
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    out = {}
    m = re.search(
        r"MAE\s+([-0-9.]+) BPM, RMSE\s+([-0-9.]+) BPM, P5\s+([-0-9.]+)%, P10\s+([-0-9.]+)%",
        text,
    )
    if m:
        out["mae_bpm"] = float(m.group(1))
        out["rmse_bpm"] = float(m.group(2))
        out["p5_pct"] = float(m.group(3))
        out["p10_pct"] = float(m.group(4))
    m2 = re.search(r"PSD: corr\s+([-0-9.]+)", text)
    if m2:
        out["psd_corr_mean"] = float(m2.group(1))
    return out


def clip_key(row):
    return (
        row.get("subject", ""),
        row.get("camera_key", ""),
        str(row.get("clip_idx", "")),
    )


def aggregate_3way(all_clip_rows, skip_clips):
    skip = {str(x) for x in skip_clips}
    by_key = {}
    for r in all_clip_rows:
        if str(r.get("clip_idx", "")) in skip:
            continue
        k = (
            r.get("job_id", ""),
            r.get("subject", ""),
            r.get("session", ""),
            r.get("camera_key", ""),
            str(r.get("clip_idx", "")),
        )
        if k not in by_key:
            by_key[k] = {
                "job_id": r.get("job_id"),
                "subject": r.get("subject"),
                "session": r.get("session"),
                "camera_key": r.get("camera_key"),
                "camera_label": r.get("camera_label"),
                "clip_idx": r.get("clip_idx"),
                "hr_gt": r.get("hr_gt"),
            }
        method = r.get("method")
        err = r.get("error") or r.get("hr_err_bpm")
        by_key[k][f"hr_pred_{method}"] = r.get("hr_pred")
        by_key[k][f"hr_error_{method}"] = err
        by_key[k][f"psd_corr_{method}"] = r.get("psd_corr")
    return list(by_key.values())


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    skip_clips = [x.strip() for x in args.skip_clips.split(",") if x.strip()]

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.results_root:
        run_root = Path(args.results_root)
    else:
        run_root = (
            CP
            / "results"
            / "EfficientPhysNet"
            / "label_ratio_0"
            / "rppg_3way"
            / ts
        )
    run_root.mkdir(parents=True, exist_ok=True)
    per_job = run_root / "per_job"
    per_job.mkdir(exist_ok=True)

    jobs = manifest.get("jobs", [])
    print(f"Jobs: {len(jobs)} | run_root: {run_root}")

    manifest_log = []
    all_clip_rows = []

    for ji, job in enumerate(jobs, 1):
        job_id = job.get("id") or f"{job['subject']}_{job['camera_key']}"
        print(f"\n[{ji}/{len(jobs)}] {job_id}")
        out_dir = per_job / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        video = Path(job["video"])
        session_dir = Path(job["session_dir"])
        scale = int(job.get("scale", args.scale))
        bundle_key = f"eval_bundle_t{scale}s"

        row_base = {
            "job_id": job_id,
            "subject": job.get("subject"),
            "session": job.get("session"),
            "camera_key": job.get("camera_key"),
            "camera_label": job.get("camera_label", job.get("camera_key")),
            "video": str(video),
        }

        methods = {}

        # --- EPN ---
        if args.run_epn:
            epn_out = out_dir / "epn"
            cmd = [
                args.python_bin,
                str(EPN_INFER),
                "--video",
                str(video),
                "--session-dir",
                str(session_dir),
                "--strategy",
                args.strategy,
                "--scales",
                str(scale),
                "--label-ratio",
                str(args.label_ratio),
                "--output-root",
                str(epn_out),
                "--openface-dir",
                args.openface_dir,
            ]
            if job.get("landmarks"):
                cmd.extend(["--landmarks", job["landmarks"]])
            run_cmd(cmd, out_dir / "epn_infer.log", CP, args.dry_run)
            epn_live = latest_live_run(epn_out)
        elif job.get("epn_run_dir"):
            epn_live = Path(job["epn_run_dir"])
        else:
            epn_live = None

        if epn_live:
            methods["epn"] = eval_bundle_dir(epn_live, scale)

        # --- POS modes ---
        if args.run_pos:
            for roi_mode, tag in (
                ("openface_crop", "pos_openface"),
                ("skin_patch", "pos_skin"),
            ):
                pos_out = out_dir / tag
                cmd = [
                    args.python_bin,
                    str(POS_INFER),
                    "--video",
                    str(video),
                    "--session-dir",
                    str(session_dir),
                    "--roi-mode",
                    roi_mode,
                    "--scales",
                    str(scale),
                    "--label-ratio",
                    str(args.label_ratio),
                    "--output-root",
                    str(pos_out),
                    "--openface-dir",
                    args.openface_dir,
                ]
                if job.get("landmarks"):
                    cmd.extend(["--landmarks", job["landmarks"]])
                run_cmd(cmd, out_dir / f"{tag}_infer.log", CP, args.dry_run)
                methods[tag] = eval_bundle_dir(latest_live_run(pos_out), scale)
        else:
            for tag, key in (
                ("pos_openface", "pos_openface_run_dir"),
                ("pos_skin", "pos_skin_run_dir"),
            ):
                if job.get(key):
                    methods[tag] = eval_bundle_dir(Path(job[key]), scale)

        # --- evaluate ---
        method_summaries = {}
        for method, bundle in methods.items():
            if bundle is None or not Path(bundle).is_dir():
                continue
            if args.run_eval:
                run_evaluate(
                    bundle,
                    args.python_bin,
                    out_dir / f"{method}_eval.log",
                    args.dry_run,
                )
            method_summaries[method] = parse_summary(bundle)
            for cr in read_clip_details(bundle):
                all_clip_rows.append(
                    {
                        **row_base,
                        "method": method,
                        "clip_idx": cr.get("clip_idx"),
                        "hr_pred": cr.get("hr_pred"),
                        "hr_gt": cr.get("hr_gt"),
                        "error": cr.get("error"),
                        "psd_corr": cr.get("psd_corr"),
                        "psd_mse": cr.get("psd_mse"),
                        "capture_snr_db": cr.get("capture_snr_db"),
                        "eval_bundle": str(bundle),
                    }
                )

        manifest_log.append(
            {
                **row_base,
                "methods": {m: str(p) for m, p in methods.items()},
                "summaries": method_summaries,
            }
        )

    # write outputs
    with open(run_root / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_log, f, ensure_ascii=False, indent=2)

    clip_csv = run_root / "all_methods_clip_details.csv"
    if all_clip_rows:
        fields = sorted({k for r in all_clip_rows for k in r})
        with clip_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_clip_rows)

    agg = aggregate_3way(all_clip_rows, skip_clips)
    agg_csv = run_root / "rppg_3way_comparison_no_clip1.csv"
    if agg:
        fields = sorted({k for r in agg for k in r})
        with agg_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(agg)

    # session-level summary
    summary_rows = []
    for entry in manifest_log:
        for method, sm in entry.get("summaries", {}).items():
            summary_rows.append(
                {
                    "job_id": entry["job_id"],
                    "subject": entry["subject"],
                    "camera_key": entry["camera_key"],
                    "method": method,
                    **sm,
                }
            )
    if summary_rows:
        with open(run_root / "session_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in summary_rows for k in r}), extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)

    print(f"\n完成: {run_root}")
    print(f"  clip log: {clip_csv.name}")
    print(f"  3-way table: {agg_csv.name}")


if __name__ == "__main__":
    main()
