# -*- coding: utf-8 -*-
"""UBFC-Phys dat@UBFC 串行下载：断点续传、T1 选择性解压、state/progress。"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ubfc_phys_paths import (
    DATUBFC_BASE,
    PROJECT_DIR,
    UBFC_PHYS_ARCHIVES,
    UBFC_PHYS_RAW,
    UBFC_PHYS_README,
    UBFC_PHYS_STAGING,
)

_DEFAULT_7ZZ = Path(PROJECT_DIR) / "prep" / "tools" / "7zz"
SEVENZIP = Path(os.environ.get("SEVENZIP", str(_DEFAULT_7ZZ)))
STATE_PATH = Path(UBFC_PHYS_STAGING) / "state.json"
PROGRESS_PATH = Path(UBFC_PHYS_STAGING) / "progress.json"
PID_PATH = Path(UBFC_PHYS_STAGING) / "download.pid"
LOG_PATH = Path(UBFC_PHYS_STAGING) / "download.log"
COOKIE_FILE = os.environ.get("UBFC_PHYS_COOKIE_FILE", "")

MAX_RETRIES = 30
RETRY_SLEEP_BASE = 30
RETRY_SLEEP_MAX = 300

BUNDLES = [
    {"n": 1, "name": "s1_to_s10.7z", "file_id": 140, "subjects": list(range(1, 11))},
    {"n": 2, "name": "s11_to_s20.7z", "file_id": 141, "subjects": list(range(11, 21))},
    {"n": 3, "name": "s21_to_s30.7z", "file_id": 142, "subjects": list(range(21, 31))},
    {"n": 4, "name": "s31_to_s40.7z", "file_id": 220, "subjects": list(range(31, 41))},
    {"n": 5, "name": "s41_to_s50.7z", "file_id": 143, "subjects": list(range(41, 51))},
    {"n": 6, "name": "s51_to_s56.7z", "file_id": 144, "subjects": list(range(51, 57))},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _curl_extra() -> list[str]:
    if COOKIE_FILE and Path(COOKIE_FILE).is_file():
        return ["--cookie", Path(COOKIE_FILE).read_text(encoding="utf-8").strip()]
    return []


def _remote_size(file_id: int) -> int | None:
    # S3 预签名 URL 拒绝 HEAD；用 Range GET 取 Content-Range 总长
    url = f"{DATUBFC_BASE}?file={file_id}"
    try:
        r = subprocess.run(
            ["curl", "-s", "-L", "-r", "0-0", "-o", "/dev/null", "-D", "-", url],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        for line in r.stdout.splitlines():
            low = line.lower()
            if low.startswith("content-range:") and "/" in line:
                return int(line.split("/")[-1].strip())
        _log(f"Content-Range missing file_id={file_id}")
        return None
    except (ValueError, subprocess.TimeoutExpired, OSError) as e:
        _log(f"size probe failed file_id={file_id}: {e}")
        return None


def _local_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _archive_ok(path: Path) -> bool:
    if not SEVENZIP.is_file():
        _log(f"missing 7zz: {SEVENZIP}")
        return False
    r = subprocess.run(
        [str(SEVENZIP), "t", str(path)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _bundle_t1_in_raw(subjects: list[int]) -> bool:
    raw = Path(UBFC_PHYS_RAW)
    if not raw.is_dir():
        return False
    for s in subjects:
        hits = list(raw.rglob(f"vid_s{s}_T1.avi"))
        if not hits:
            return False
    return True


def _init_state() -> dict[str, Any]:
    state = _read_json(STATE_PATH)
    if state.get("bundles"):
        return state
    bundles_state = {}
    for b in BUNDLES:
        n = b["n"]
        arc = Path(UBFC_PHYS_ARCHIVES) / b["name"]
        entry: dict[str, Any] = {
            "name": b["name"],
            "file_id": b["file_id"],
            "status": "pending",
            "bytes_downloaded": 0,
            "bytes_total": None,
            "updated_at": _utc_now(),
        }
        if _bundle_t1_in_raw(b["subjects"]):
            entry["status"] = "done"
        elif arc.is_file():
            entry["bytes_downloaded"] = _local_size(arc)
            entry["status"] = "downloading"
        bundles_state[str(n)] = entry
    state = {"version": 1, "updated_at": _utc_now(), "bundles": bundles_state}
    _write_json(STATE_PATH, state)
    return state


def _set_bundle_state(state: dict, n: int, **kwargs: Any) -> None:
    key = str(n)
    entry = state["bundles"].setdefault(key, {})
    entry.update(kwargs)
    entry["updated_at"] = _utc_now()
    state["updated_at"] = _utc_now()
    _write_json(STATE_PATH, state)


def _write_progress(
    *,
    phase: str,
    bundle_n: int,
    bundle_name: str,
    bytes_done: int,
    bytes_total: int | None,
    speed_bps: float = 0.0,
    message: str = "",
) -> None:
    pct = None
    if bytes_total and bytes_total > 0:
        pct = min(100.0, 100.0 * bytes_done / bytes_total)
    eta_s = None
    if speed_bps > 0 and bytes_total and bytes_done < bytes_total:
        eta_s = int((bytes_total - bytes_done) / speed_bps)
    data = {
        "updated_at": _utc_now(),
        "phase": phase,
        "bundle_n": bundle_n,
        "bundle_total": len(BUNDLES),
        "bundle_name": bundle_name,
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "pct": pct,
        "speed_bps": speed_bps,
        "eta_s": eta_s,
        "message": message,
    }
    _write_json(PROGRESS_PATH, data)


def _download_with_resume(
    file_id: int,
    dest: Path,
    bundle_n: int,
    bundle_name: str,
    state: dict,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATUBFC_BASE}?file={file_id}"
    expected = _remote_size(file_id)
    _set_bundle_state(
        state, bundle_n, status="downloading", bytes_total=expected
    )

    for attempt in range(1, MAX_RETRIES + 1):
        local = _local_size(dest)
        if expected and local >= expected and _archive_ok(dest):
            _log(f"bundle {bundle_n}: archive complete ({local} bytes)")
            _set_bundle_state(
                state, bundle_n, status="downloaded", bytes_downloaded=local
            )
            return True

        _log(
            f"bundle {bundle_n}: download attempt {attempt}/{MAX_RETRIES} "
            f"resume={local} expected={expected}"
        )
        _write_progress(
            phase="download",
            bundle_n=bundle_n,
            bundle_name=bundle_name,
            bytes_done=local,
            bytes_total=expected,
            message=f"attempt {attempt}",
        )

        t0 = time.monotonic()
        last_size = local
        last_t = t0

        proc = subprocess.Popen(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "--fail",
                "--connect-timeout",
                "60",
                # 5min 平均 <10KB/s 则中断，触发 attempt 续传
                "--speed-time",
                "300",
                "--speed-limit",
                "10000",
                "--max-time",
                "14400",
                "--retry",
                "3",
                "--retry-delay",
                "5",
                *_curl_extra(),
                "-o",
                str(dest),
                url,
            ],
            stdout=subprocess.DEVNULL,
            # stderr=PIPE 且不读会填满缓冲区，curl 卡在 pipe_write
            stderr=subprocess.DEVNULL,
        )

        while proc.poll() is None:
            time.sleep(2)
            now = time.monotonic()
            cur = _local_size(dest)
            dt = max(now - last_t, 1e-6)
            speed = (cur - last_size) / dt if cur >= last_size else 0.0
            last_size, last_t = cur, now
            _write_progress(
                phase="download",
                bundle_n=bundle_n,
                bundle_name=bundle_name,
                bytes_done=cur,
                bytes_total=expected,
                speed_bps=speed,
                message=f"attempt {attempt}",
            )
            _set_bundle_state(
                state, bundle_n, bytes_downloaded=cur, bytes_total=expected
            )

        local = _local_size(dest)

        if proc.returncode == 0 and expected and local >= expected and _archive_ok(dest):
            _log(f"bundle {bundle_n}: download OK")
            _set_bundle_state(
                state, bundle_n, status="downloaded", bytes_downloaded=local
            )
            return True

        if proc.returncode == 0 and expected is None:
            if _archive_ok(dest):
                _log(f"bundle {bundle_n}: download OK (unknown size, 7z ok)")
                _set_bundle_state(
                    state, bundle_n, status="downloaded", bytes_downloaded=local
                )
                return True

        _log(
            f"bundle {bundle_n}: curl exit {proc.returncode} local={local}"
        )
        sleep_s = min(RETRY_SLEEP_BASE * attempt, RETRY_SLEEP_MAX)
        _write_progress(
            phase="download_retry",
            bundle_n=bundle_n,
            bundle_name=bundle_name,
            bytes_done=local,
            bytes_total=expected,
            message=f"retry in {sleep_s}s",
        )
        time.sleep(sleep_s)

    return False


def _extract_t1(archive: Path, bundle_n: int, bundle_name: str) -> bool:
    _write_progress(
        phase="extract",
        bundle_n=bundle_n,
        bundle_name=bundle_name,
        bytes_done=0,
        bytes_total=None,
        message="7zz T1 only",
    )
    Path(UBFC_PHYS_RAW).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            str(SEVENZIP),
            "x",
            str(archive),
            f"-o{UBFC_PHYS_RAW}",
            r"-ir!*vid_s*_T1.avi",
            r"-ir!*bvp_s*_T1.csv",
            r"-ir!*info_s*.txt",
            "-y",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        _log(f"bundle {bundle_n}: extract failed: {r.stderr[:300]}")
        return False
    _log(f"bundle {bundle_n}: T1 extract OK")
    return True


def _download_readme() -> None:
    if Path(UBFC_PHYS_README).is_file():
        return
    dest = Path(UBFC_PHYS_README)
    url = f"{DATUBFC_BASE}?file=108"
    subprocess.run(
        ["curl", "-L", "-f", "-o", str(dest), url, *_curl_extra()],
        check=True,
    )
    _log("README downloaded")


def run_pipeline(bundle_nums: list[int]) -> int:
    state = _init_state()
    _download_readme()

    for b in BUNDLES:
        if b["n"] not in bundle_nums:
            continue
        n, name, fid = b["n"], b["name"], b["file_id"]
        arc = Path(UBFC_PHYS_ARCHIVES) / name

        if state["bundles"].get(str(n), {}).get("status") == "done":
            _log(f"bundle {n}: skip (state=done)")
            continue
        if _bundle_t1_in_raw(b["subjects"]):
            _set_bundle_state(state, n, status="done")
            _log(f"bundle {n}: skip (T1 already in raw)")
            continue

        if arc.is_file() and _archive_ok(arc):
            _log(f"bundle {n}: archive complete, skip download")
            _set_bundle_state(state, n, status="downloaded", bytes_downloaded=_local_size(arc))
        else:
            if not _download_with_resume(fid, arc, n, name, state):
                _set_bundle_state(state, n, status="failed")
                _log(f"bundle {n}: download failed after retries")
                return 1

        _set_bundle_state(state, n, status="extracting")
        if not _extract_t1(arc, n, name):
            _set_bundle_state(state, n, status="failed")
            return 1

        arc.unlink(missing_ok=True)
        _log(f"bundle {n}: removed archive {name}")
        _set_bundle_state(state, n, status="done", bytes_downloaded=0)
        _write_progress(
            phase="bundle_done",
            bundle_n=n,
            bundle_name=name,
            bytes_done=0,
            bytes_total=None,
            message="archive deleted",
        )

    done = sum(1 for b in BUNDLES if state["bundles"].get(str(b["n"]), {}).get("status") == "done")
    _write_progress(
        phase="all_done" if done == len(BUNDLES) else "idle",
        bundle_n=done,
        bundle_name="",
        bytes_done=done,
        bytes_total=len(BUNDLES),
        message=f"{done}/{len(BUNDLES)} bundles done",
    )
    _log(f"pipeline finished: {done}/{len(BUNDLES)} bundles done")
    return 0 if done == len(bundle_nums) else 0


def cmd_status() -> int:
    prog = _read_json(PROGRESS_PATH)
    state = _read_json(STATE_PATH)
    if not prog:
        print("no progress yet (run: daemon all)")
        return 0

    pct = prog.get("pct")
    bar_w = 40
    if pct is not None:
        filled = int(bar_w * pct / 100)
        bar = "#" * filled + "-" * (bar_w - filled)
        pct_s = f"{pct:5.1f}%"
    else:
        bar = "?" * bar_w
        pct_s = "  n/a"

    def _fmt_bytes(n: float | int | None) -> str:
        if n is None:
            return "?"
        n = float(n)
        for u in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or u == "TB":
                return f"{n:.2f}{u}"
            n /= 1024
        return f"{n:.2f}TB"

    speed = prog.get("speed_bps") or 0
    eta = prog.get("eta_s")
    eta_s = f"{eta // 3600}h{(eta % 3600) // 60}m" if eta else "?"

    print(
        f"[{bar}] {pct_s}  "
        f"bundle {prog.get('bundle_n')}/{prog.get('bundle_total')} "
        f"{prog.get('bundle_name', '')}  "
        f"phase={prog.get('phase')}  "
        f"{_fmt_bytes(prog.get('bytes_done'))}/{_fmt_bytes(prog.get('bytes_total'))}  "
        f"{_fmt_bytes(speed)}/s  ETA {eta_s}"
    )
    if prog.get("message"):
        print(f"  {prog['message']}")
    print(f"  updated: {prog.get('updated_at')}")

    if state.get("bundles"):
        print("  bundle status:", end="")
        for i in range(1, 7):
            st = state["bundles"].get(str(i), {}).get("status", "?")
            print(f" b{i}={st}", end="")
        print()
    return 0


def cmd_daemon(bundle_arg: str) -> int:
    if PID_PATH.is_file():
        pid = int(PID_PATH.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"already running pid={pid}")
            return 1
        except OSError:
            PID_PATH.unlink(missing_ok=True)

    nums = list(range(1, 7)) if bundle_arg == "all" else [int(bundle_arg)]
    log_f = open(LOG_PATH, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run", *[str(n) for n in nums]],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_PATH.write_text(str(proc.pid), encoding="utf-8")
    print(f"started pid={proc.pid}  log={LOG_PATH}")
    print(f"status: {sys.executable} {Path(__file__).resolve()} status")
    return 0


def cmd_stop() -> int:
    if not PID_PATH.is_file():
        print("not running")
        return 0
    pid = int(PID_PATH.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"sent SIGTERM to {pid}")
    except OSError as e:
        print(f"kill failed: {e}")
    PID_PATH.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="UBFC-Phys dat@UBFC downloader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help=argparse.SUPPRESS)
    p_run.add_argument("bundles", nargs="*", type=int)

    p_daemon = sub.add_parser("daemon")
    p_daemon.add_argument("target", choices=[str(i) for i in range(1, 7)] + ["all"])

    sub.add_parser("status")
    sub.add_parser("stop")

    args = parser.parse_args()
    Path(UBFC_PHYS_STAGING).mkdir(parents=True, exist_ok=True)
    Path(UBFC_PHYS_ARCHIVES).mkdir(parents=True, exist_ok=True)

    if args.cmd == "run":
        nums = args.bundles or list(range(1, 7))
        rc = run_pipeline(nums)
        PID_PATH.unlink(missing_ok=True)
        return rc
    if args.cmd == "daemon":
        return cmd_daemon(args.target)
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "stop":
        return cmd_stop()
    return 1


if __name__ == "__main__":
    sys.exit(main())
