#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测 CanControls LGI zip 链接可达性与 Content-Length。"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lgi_paths import LGI_BASE, LGI_DOWNLOAD_PROBE, LGI_MANIFEST_DIR, LGI_SUBJECTS


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _probe_url(url: str, timeout_s: int = 90) -> dict:
    t0 = time.monotonic()
    entry = {
        "url": url,
        "ok": False,
        "bytes_total": None,
        "http_status": None,
        "error": None,
        "latency_s": None,
        "looks_like_html": False,
    }
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
            timeout=timeout_s + 10,
            check=False,
        )
        entry["latency_s"] = round(time.monotonic() - t0, 2)
        headers = r.stdout
        if re.search(r"content-type:\s*text/html", headers, re.I):
            entry["looks_like_html"] = True
        for line in headers.splitlines():
            low = line.lower()
            if low.startswith("http/"):
                entry["http_status"] = line.split()[1] if len(line.split()) > 1 else None
            if low.startswith("content-range:") and "/" in line:
                entry["bytes_total"] = int(line.split("/")[-1].strip())
                entry["ok"] = True
        if r.returncode != 0 and not entry["ok"]:
            entry["error"] = (r.stderr or f"curl_exit_{r.returncode}").strip()[:300]
    except subprocess.TimeoutExpired:
        entry["error"] = "timeout"
        entry["latency_s"] = round(time.monotonic() - t0, 2)
    except OSError as e:
        entry["error"] = str(e)
        entry["latency_s"] = round(time.monotonic() - t0, 2)
    return entry


def main() -> int:
    Path(LGI_MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
    probes = []
    any_ok = False
    for sub in LGI_SUBJECTS:
        url = f"{LGI_BASE}/{sub['name']}"
        p = _probe_url(url)
        p["subject_n"] = sub["n"]
        p["name"] = sub["name"]
        p["size_gb_ref"] = sub["size_gb_ref"]
        probes.append(p)
        if p["ok"]:
            any_ok = True
        print(
            f"id{sub['n']}: ok={p['ok']} bytes={p['bytes_total']} "
            f"latency={p['latency_s']}s err={p.get('error')}"
        )

    report = {
        "created_at": _utc_now(),
        "base": LGI_BASE,
        "any_reachable": any_ok,
        "note": "S1-only eval; 仅 ID1-6 有链接",
        "probes": probes,
    }
    with open(LGI_DOWNLOAD_PROBE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {LGI_DOWNLOAD_PROBE}")
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
