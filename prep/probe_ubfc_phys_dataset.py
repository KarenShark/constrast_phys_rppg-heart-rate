#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UBFC-Phys Phase 0：探测视频/BVP/info，禁止假设 fps=20/30/35。
输出 datasets/UBFC-Phys_manifests/dataset_probe_report.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from ubfc_phys_paths import (  # noqa: E402
    BVP_FS_HZ,
    PAPER_VIDEO_FPS_REF,
    UBFC_PHYS_MANIFEST_DIR,
    UBFC_PHYS_PROBE_REPORT,
    UBFC_PHYS_RAW,
)


def _read_info(info_path):
    out = {}
    if not os.path.isfile(info_path):
        return out
    with open(info_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _probe_bvp(bvp_path):
    import pandas as pd
    arr = pd.read_csv(bvp_path, header=None).iloc[:, 0].astype(float).values
    n = len(arr)
    dur_64 = n / BVP_FS_HZ if n else 0.0
    return {
        "n_samples": int(n),
        "bvp_fs_hz": BVP_FS_HZ,
        "duration_sec_at_64hz": round(dur_64, 3),
    }


def probe_video_bvp_pair(video_path, bvp_path=None, info_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "video_open_failed", "video_path": video_path}

    fps_meta = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()

    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

    rec = {
        "video_path": os.path.abspath(video_path),
        "cap_prop_fps": fps_meta if fps_meta > 0 else None,
        "frame_count": n_frames,
        "width": w,
        "height": h,
        "fourcc": fourcc_str,
        "paper_fps_ref": PAPER_VIDEO_FPS_REF,
    }

    if fps_meta > 1 and n_frames > 0:
        dur_meta = n_frames / fps_meta
        rec["duration_sec_from_cap_fps"] = round(dur_meta, 3)
        rec["fps_for_pipeline"] = fps_meta
        rec["fps_source"] = "cv2_metadata"
    else:
        rec["fps_for_pipeline"] = None
        rec["fps_source"] = "needs_review"

    if bvp_path and os.path.isfile(bvp_path):
        bvp_info = _probe_bvp(bvp_path)
        rec["bvp_path"] = os.path.abspath(bvp_path)
        rec.update(bvp_info)
        if rec.get("duration_sec_from_cap_fps") and bvp_info["duration_sec_at_64hz"]:
            rec["bvp_video_duration_ratio"] = round(
                bvp_info["duration_sec_at_64hz"] / rec["duration_sec_from_cap_fps"], 4
            )

    if info_path and os.path.isfile(info_path):
        rec["info"] = _read_info(info_path)
        rec["info_path"] = os.path.abspath(info_path)

    return rec


def discover_t1_clips(raw_root):
    clips = []
    raw_root = os.path.abspath(raw_root)
    if not os.path.isdir(raw_root):
        return clips
    for name in sorted(os.listdir(raw_root)):
        if not name.startswith("s") or not name[1:].isdigit():
            continue
        subj_dir = os.path.join(raw_root, name)
        if not os.path.isdir(subj_dir):
            continue
        sid = name[1:]
        vid = os.path.join(subj_dir, f"vid_s{sid}_T1.avi")
        bvp = os.path.join(subj_dir, f"bvp_s{sid}_T1.csv")
        info = os.path.join(subj_dir, f"info_s{sid}.txt")
        if os.path.isfile(vid):
            clips.append((vid, bvp if os.path.isfile(bvp) else None, info))
    return clips


def main():
    ap = argparse.ArgumentParser(description="Probe UBFC-Phys T1 clips")
    ap.add_argument("--raw-root", default=UBFC_PHYS_RAW)
    ap.add_argument("--video", default=None, help="单文件探测")
    ap.add_argument("--bvp", default=None)
    ap.add_argument("--info", default=None)
    ap.add_argument("--max-clips", type=int, default=0, help="0=全部发现的 T1")
    args = ap.parse_args()

    os.makedirs(UBFC_PHYS_MANIFEST_DIR, exist_ok=True)
    probes = []

    if args.video:
        probes.append(probe_video_bvp_pair(args.video, args.bvp, args.info))
    else:
        clips = discover_t1_clips(args.raw_root)
        if args.max_clips > 0:
            clips = clips[: args.max_clips]
        for vid, bvp, info in clips:
            probes.append(probe_video_bvp_pair(vid, bvp, info))

    fps_vals = [
        p["fps_for_pipeline"] for p in probes
        if p.get("fps_for_pipeline") is not None
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_root": os.path.abspath(args.raw_root),
        "n_clips_probed": len(probes),
        "paper_video_fps_ref": PAPER_VIDEO_FPS_REF,
        "bvp_fs_hz": BVP_FS_HZ,
        "note": "fps_for_pipeline 来自 cv2 元数据；禁止默认 20/30。无 raw 时仅 README 探测。",
        "fps_summary": {
            "min": float(np.min(fps_vals)) if fps_vals else None,
            "max": float(np.max(fps_vals)) if fps_vals else None,
            "mean": float(np.mean(fps_vals)) if fps_vals else None,
            "unique": sorted(set(round(x, 3) for x in fps_vals)),
        },
        "probes": probes,
    }

    with open(UBFC_PHYS_PROBE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Probed {len(probes)} clip(s)")
    if fps_vals:
        print(f"  fps_for_pipeline: unique={report['fps_summary']['unique']}")
    else:
        print("  无 T1 视频可探测（需先解压 raw）")
    print(f"  report: {UBFC_PHYS_PROBE_REPORT}")
    return 0 if probes else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
