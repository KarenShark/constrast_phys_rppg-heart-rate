"""GT / frame timestamp helpers for self-recorded multi-camera sessions."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

RAW_CAMERA_STEM = "video_RAW_YUV420"
ANDROID_OLD_RE = re.compile(r"^(android_[^_]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")
ANDROID_NEW_RE = re.compile(r"^(android_[^_]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_30fps$")


def detect_camera_from_video(video_path):
    """Return dict(camera_key, platform, device_prefix) or None."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    session_dir = os.path.dirname(os.path.abspath(video_path))
    if stem == RAW_CAMERA_STEM:
        return {
            "camera_key": RAW_CAMERA_STEM,
            "platform": "webcam",
            "device_prefix": None,
            "timestamp_path": os.path.join(session_dir, "frames_timestamp.csv"),
        }
    m = ANDROID_OLD_RE.match(stem)
    if m:
        dev = m.group(1)
        return {
            "camera_key": dev,
            "platform": "android",
            "device_prefix": dev,
            "timestamp_path": os.path.join(session_dir, f"{dev}_frames_timestamp.csv"),
        }
    m = ANDROID_NEW_RE.match(stem)
    if m:
        dev = m.group(1)
        ts = _pick_android_ts_csv(session_dir, dev, video_path)
        return {
            "camera_key": dev,
            "platform": "android",
            "device_prefix": dev,
            "timestamp_path": ts,
        }
    if re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_30fps$", stem):
        ts = _find_capture_timestamp(session_dir)
        dev = "android_NEWPHONE"
        if ts:
            base = os.path.splitext(os.path.basename(ts))[0]
            m2 = re.match(r"^(android_[^_]+)_", base)
            if m2:
                dev = m2.group(1)
        return {
            "camera_key": dev,
            "platform": "android",
            "device_prefix": dev,
            "timestamp_path": ts,
        }
    if stem.startswith("ios_") and stem.endswith("_60_30_raw_yuv") and "_ios_" in stem:
        dev = stem.split("_ios_")[0]
        ts = _find_one(session_dir, f"{dev}_*_raw_yuv.csv")
        return {
            "camera_key": dev,
            "platform": "ios",
            "device_prefix": dev,
            "timestamp_path": ts,
        }
    return None


def _find_one(directory, pattern):
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    return hits[0] if hits else None


def _find_capture_timestamp(session_dir):
    cap = sorted(
        glob.glob(os.path.join(session_dir, "rppg_captures", "capture_*", "android_frames_timestamp.csv"))
    )
    return cap[0] if cap else None


def _count_ts_rows(ts_path):
    return len(pd.read_csv(ts_path, comment="#"))


def _video_frame_count(video_path):
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode == 0 and out.stdout.strip().isdigit():
        return int(out.stdout.strip())
    return None


def _pick_android_ts_csv(session_dir, dev, video_path):
    hits = sorted(glob.glob(os.path.join(session_dir, f"{dev}_*_raw_yuv.csv")))
    if not hits:
        return _find_capture_timestamp(session_dir)
    if len(hits) == 1:
        return hits[0]
    manifest = _load_sync_manifest(session_dir)
    target_n = _video_frame_count(video_path) if video_path else None
    if target_n is not None:
        best, best_diff = None, 10**9
        for path in hits:
            stem = os.path.splitext(os.path.basename(path))[0]
            entry = manifest.get(stem)
            n = entry.get("actual_num_frames") if entry else _count_ts_rows(path)
            diff = abs(int(n) - int(target_n))
            if diff < best_diff:
                best_diff = diff
                best = path
        if best is not None:
            return best
    return max(hits, key=_count_ts_rows)


def _load_sync_manifest(session_dir):
    path = os.path.join(session_dir, "sync_manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _manifest_broadcast_s(session_dir, device_prefix, timestamp_path=None):
    """broadcast_unix_ms; prefer exact manifest key matching timestamp CSV stem."""
    manifest = _load_sync_manifest(session_dir)
    if timestamp_path:
        stem = os.path.splitext(os.path.basename(timestamp_path))[0]
        entry = manifest.get(stem)
        if entry and entry.get("broadcast_unix_ms") is not None:
            return float(entry["broadcast_unix_ms"]) / 1e3
    if not device_prefix:
        return None
    for key, entry in manifest.items():
        if key.startswith(device_prefix):
            ms = entry.get("broadcast_unix_ms")
            return float(ms) / 1e3 if ms is not None else None
    return None


def _read_timestamp_csv(ts_path):
    df = pd.read_csv(ts_path, comment="#")
    frame_col = df.columns[0]
    ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
    ts_col = ts_cols[0] if ts_cols else df.columns[1]
    frames = df[frame_col].to_numpy(dtype=np.int64)
    ts = df[ts_col].to_numpy(dtype=np.float64)
    return frames, ts


def host_aligned_timestamps(session_dir, video_path):
    """Per-frame unix timestamps on host/GT clock. shape (n_frames,)."""
    cam = detect_camera_from_video(video_path)
    if cam is None or not cam["timestamp_path"] or not os.path.isfile(cam["timestamp_path"]):
        return None, cam
    frames, ts = _read_timestamp_csv(cam["timestamp_path"])
    broadcast_s = _manifest_broadcast_s(
        session_dir, cam["device_prefix"], cam.get("timestamp_path")
    )
    if broadcast_s is not None and cam["platform"] in ("android", "ios"):
        ts = broadcast_s + (ts - ts[0])
    return (frames, ts), cam


def build_ts_lut(session_dir, video_path, bvp_times=None):
    """frame_id -> host unix timestamp lookup + optional clock_offset fix."""
    out = host_aligned_timestamps(session_dir, video_path)
    if out[0] is None:
        return None
    (frame_ids, ts_wall), cam = out
    mx = int(frame_ids.max()) + 1 if len(frame_ids) else 0
    ts_lut = np.full(mx, np.nan, dtype=np.float64)
    for fid, tw in zip(frame_ids, ts_wall):
        if 0 <= fid < mx:
            ts_lut[fid] = tw

    clock_offset = 0.0
    valid = ts_lut[~np.isnan(ts_lut)]
    if bvp_times is not None and len(valid) > 1 and len(bvp_times) > 1:
        cam_start, cam_end = float(valid[0]), float(valid[-1])
        bvp_start, bvp_end = float(bvp_times[0]), float(bvp_times[-1])
        overlap = max(0.0, min(cam_end, bvp_end) - max(cam_start, bvp_start))
        cam_dur = cam_end - cam_start
        if cam_dur > 0 and overlap < cam_dur * 0.5:
            clock_offset = cam_start - bvp_start
            ts_lut = ts_lut - clock_offset
    return {
        "ts_lut": ts_lut,
        "clock_offset": clock_offset,
        "camera_key": cam["camera_key"],
        "timestamp_path": cam["timestamp_path"],
    }


def write_host_frames_timestamp_csv(out_path, session_dir, video_path):
    """Write frame,timestamp CSV on host clock for gt_proxy / infer."""
    out = host_aligned_timestamps(session_dir, video_path)
    if out[0] is None:
        raise FileNotFoundError(f"No timestamp for {video_path}")
    frames, ts = out[0]
    pd.DataFrame({"frame": frames, "timestamp": ts}).to_csv(out_path, index=False)
    return out_path
