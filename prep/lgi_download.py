#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LGI-PPGI-DB 串行下载：断点续传、仅解压 Session1、删 zip。"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lgi_paths import (
    LGI_ARCHIVES,
    LGI_BASE,
    LGI_DOWNLOAD_LOG,
    LGI_DOWNLOAD_PROGRESS,
    LGI_DOWNLOAD_STATE,
    LGI_RAW,
    LGI_STAGING,
    LGI_SUBJECTS,
    PROJECT_DIR,
)

_DEFAULT_7ZZ = Path(PROJECT_DIR) / "prep" / "tools" / "7zz"
SEVENZIP = Path(os.environ.get("SEVENZIP", str(_DEFAULT_7ZZ)))
STATE_PATH = Path(LGI_DOWNLOAD_STATE)
PROGRESS_PATH = Path(LGI_DOWNLOAD_PROGRESS)
PID_PATH = Path(LGI_STAGING) / "download.pid"
LOG_PATH = Path(LGI_DOWNLOAD_LOG)

MAX_RETRIES = 30
RETRY_SLEEP_BASE = 30
RETRY_SLEEP_MAX = 300

_S1_RE = re.compile(
    r"session[_\s]?1|session1|head[_\s]?rest|resting|_s1_|[_/]1/|_1\.(avi|txt|csv|xml|mat)",
    re.I,
)
_S_EXCLUDE = re.compile(
    r"session[_\s]?[234]|session[234]|ergometer|urban|rotation|/s[234]/|_s[234]_",
    re.I,
)


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


def _local_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _remote_size(url: str, timeout_s: int = 120) -> int | None:
    try:
        r = subprocess.run(
            [
                "curl",
                "-s",
                "-L",
                "-r",
                "0-0",
                "-o",
                "/dev/null",
                "-D",
                "-",
                "--max-time",
                str(timeout_s),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s + 15,
            check=False,
        )
        for line in r.stdout.splitlines():
            low = line.lower()
            if low.startswith("content-range:") and "/" in line:
                return int(line.split("/")[-1].strip())
        return None
    except (ValueError, subprocess.TimeoutExpired, OSError) as e:
        _log(f"size probe failed: {e}")
        return None


def _archive_ok(path: Path) -> bool:
    if not SEVENZIP.is_file():
        return False
    r = subprocess.run(
        [str(SEVENZIP), "t", str(path)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _list_members(archive: Path) -> list[str]:
    r = subprocess.run(
        [str(SEVENZIP), "l", "-ba", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return []
    return [ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip()]


def _is_s1_member(name: str) -> bool:
    n = name.replace("\\", "/")
    if _S_EXCLUDE.search(n):
        return False
    if _S1_RE.search(n):
        return True
    low = n.lower()
    if low.endswith((".avi", ".txt", ".csv", ".xml", ".mat")):
        if "session" not in low and "ergometer" not in low and "urban" not in low:
            return True
    return False


def _s1_in_raw(subject_n: int) -> bool:
    raw = Path(LGI_RAW)
    if not raw.is_dir():
        return False
    pat = re.compile(rf"id{subject_n}|/s?{subject_n}/|subject{subject_n}", re.I)
    for avi in raw.rglob("*.avi"):
        if pat.search(str(avi)) and _is_s1_member(str(avi)):
            return True
    # idN 解压后可能顶层即 subject 目录
    for avi in raw.rglob("*.avi"):
        if _is_s1_member(str(avi)):
            rel = avi.relative_to(raw)
            parts = rel.parts
            if parts and parts[0].lower() in {f"id{subject_n}", str(subject_n)}:
                return True
    return False


def _extract_s1(archive: Path, subject_n: int) -> bool:
    members = _list_members(archive)
    s1 = [m for m in members if _is_s1_member(m) and not m.endswith("/")]
    if not s1:
        _log(f"id{subject_n}: no S1 members matched; listing sample: {members[:8]}")
        return False

    out = Path(LGI_RAW) / f"id{subject_n}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(SEVENZIP), "x", str(archive), f"-o{out}", "-y"]
    for m in s1:
        cmd.append(f"-ir!{m}")
    _log(f"id{subject_n}: extract S1 n_files={len(s1)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _log(f"id{subject_n}: 7zz extract failed: {r.stderr[:400]}")
        return False
    return True


def _write_progress(**kwargs: Any) -> None:
    _write_json(PROGRESS_PATH, {"updated_at": _utc_now(), **kwargs})


def _download_with_resume(
    url: str,
    dest: Path,
    subject_n: int,
    state: dict,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)
    key = str(subject_n)
    state.setdefault("subjects", {})[key] = state.get("subjects", {}).get(key, {})
    state["subjects"][key].update(
        {"status": "downloading", "bytes_total": expected, "name": dest.name}
    )
    _write_json(STATE_PATH, state)

    for attempt in range(1, MAX_RETRIES + 1):
        local = _local_size(dest)
        if expected and local >= expected and _archive_ok(dest):
            _log(f"id{subject_n}: archive complete")
            state["subjects"][key]["status"] = "downloaded"
            state["subjects"][key]["bytes_downloaded"] = local
            _write_json(STATE_PATH, state)
            return True

        _log(f"id{subject_n}: download attempt {attempt}/{MAX_RETRIES} at {local}")
        _write_progress(
            phase="download",
            subject_n=subject_n,
            subject_total=len(LGI_SUBJECTS),
            name=dest.name,
            bytes_done=local,
            bytes_total=expected,
            pct=(100.0 * local / expected) if expected else None,
            message=f"attempt {attempt}",
        )

        t0 = time.monotonic()
        last_size = local
        proc = subprocess.Popen(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "--fail",
                "--max-time",
                "0",
                "-o",
                str(dest),
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        while proc.poll() is None:
            time.sleep(2)
            cur = _local_size(dest)
            dt = max(time.monotonic() - t0, 1e-6)
            speed = (cur - last_size) / dt if cur >= last_size else 0.0
            last_size = cur
            t0 = time.monotonic()
            _write_progress(
                phase="download",
                subject_n=subject_n,
                subject_total=len(LGI_SUBJECTS),
                name=dest.name,
                bytes_done=cur,
                bytes_total=expected,
                pct=(100.0 * cur / expected) if expected else None,
                speed_bps=speed,
                message=f"attempt {attempt}",
            )
            state["subjects"][key]["bytes_downloaded"] = cur
            _write_json(STATE_PATH, state)

        err = proc.stderr.read() if proc.stderr else ""
        local = _local_size(dest)
        if proc.returncode == 0 and _archive_ok(dest):
            if expected is None or local >= expected:
                state["subjects"][key]["status"] = "downloaded"
                _write_json(STATE_PATH, state)
                return True

        _log(f"id{subject_n}: curl rc={proc.returncode} local={local} {err[:200]}")
        sleep_s = min(RETRY_SLEEP_BASE * attempt, RETRY_SLEEP_MAX)
        time.sleep(sleep_s)

    state["subjects"][key]["status"] = "failed"
    _write_json(STATE_PATH, state)
    return False


def _download_subject(n: int, state: dict) -> bool:
    sub = next((s for s in LGI_SUBJECTS if s["n"] == n), None)
    if not sub:
        return False
    url = f"{LGI_BASE}/{sub['name']}"
    arc = Path(LGI_ARCHIVES) / sub["name"]

    if state.get("subjects", {}).get(str(n), {}).get("status") == "done":
        _log(f"id{n}: skip done")
        return True
    if _s1_in_raw(n):
        state.setdefault("subjects", {})[str(n)] = {"status": "done", "name": sub["name"]}
        _write_json(STATE_PATH, state)
        _log(f"id{n}: skip S1 already in raw")
        return True

    if not (arc.is_file() and _archive_ok(arc)):
        if not _download_with_resume(url, arc, n, state):
            return False
    else:
        _log(f"id{n}: using existing archive")

    state["subjects"][str(n)]["status"] = "extracting"
    _write_json(STATE_PATH, state)
    if not _extract_s1(arc, n):
        return False

    arc.unlink(missing_ok=True)
    _log(f"id{n}: removed {arc.name}")
    state["subjects"][str(n)]["status"] = "done"
    _write_json(STATE_PATH, state)
    return True


def run_pipeline(subject_nums: list[int]) -> int:
    Path(LGI_STAGING).mkdir(parents=True, exist_ok=True)
    Path(LGI_ARCHIVES).mkdir(parents=True, exist_ok=True)
    state = _read_json(STATE_PATH) or {"subjects": {}}

    ok = 0
    for n in subject_nums:
        if _download_subject(n, state):
            ok += 1
        else:
            _log(f"id{n}: pipeline failed")
            return 1

    _write_progress(
        phase="all_done" if ok == len(subject_nums) else "idle",
        subject_n=ok,
        subject_total=len(LGI_SUBJECTS),
        message=f"{ok} subjects S1 done",
    )
    _log(f"finished {ok}/{len(subject_nums)}")
    return 0


def cmd_status() -> int:
    prog = _read_json(PROGRESS_PATH)
    if not prog:
        print("no progress; run: download_lgi.sh daemon 1")
        return 0
    pct = prog.get("pct")
    bar_w = 40
    if pct is not None:
        filled = int(bar_w * float(pct) / 100)
        bar = "#" * filled + "-" * (bar_w - filled)
        pct_s = f"{float(pct):5.1f}%"
    else:
        bar, pct_s = "?" * bar_w, "  n/a"

    def _gb(x: Any) -> str:
        if x is None:
            return "?"
        return f"{float(x) / 1e9:.2f}GB"

    print(
        f"[{bar}] {pct_s}  id{prog.get('subject_n')}/{prog.get('subject_total')} "
        f"{prog.get('name', '')}  phase={prog.get('phase')}  "
        f"{_gb(prog.get('bytes_done'))}/{_gb(prog.get('bytes_total'))}"
    )
    st = _read_json(STATE_PATH)
    if st.get("subjects"):
        print(" ", end="")
        for i in range(1, 7):
            print(f"id{i}={st['subjects'].get(str(i), {}).get('status', '?')}", end=" ")
        print()
    return 0


def cmd_daemon(target: str) -> int:
    if PID_PATH.is_file():
        pid = int(PID_PATH.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"already running pid={pid}")
            return 1
        except OSError:
            PID_PATH.unlink(missing_ok=True)

    nums = list(range(1, 7)) if target == "all" else [int(target)]
    log_f = open(LOG_PATH, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run", *[str(n) for n in nums]],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_PATH.write_text(str(proc.pid), encoding="utf-8")
    print(f"started pid={proc.pid}  log={LOG_PATH}")
    return 0


def cmd_stop() -> int:
    if not PID_PATH.is_file():
        print("not running")
        return 0
    pid = int(PID_PATH.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"kill: {e}")
    PID_PATH.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("subjects", nargs="*", type=int)
    p_d = sub.add_parser("daemon")
    p_d.add_argument("target", choices=[str(i) for i in range(1, 7)] + ["all"])
    sub.add_parser("status")
    sub.add_parser("stop")

    args = parser.parse_args()
    if args.cmd == "run":
        nums = args.subjects or [1]
        PID_PATH.unlink(missing_ok=True)
        return run_pipeline(nums)
    if args.cmd == "daemon":
        return cmd_daemon(args.target)
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "stop":
        return cmd_stop()
    return 1


if __name__ == "__main__":
    sys.exit(main())
