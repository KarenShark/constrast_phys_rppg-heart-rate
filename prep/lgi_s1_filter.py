# -*- coding: utf-8 -*-
"""LGI Session1 路径过滤 — manifest / download / probe 共用。"""
import re
from pathlib import Path

SESSION_S1 = 1

_S_EXCLUDE = re.compile(
    r"session[_\s]?[234]|session[234]|/s[234]/|_s[234]_|ergometer|urban|rotation",
    re.I,
)
_S1_HINT = re.compile(
    r"session[_\s]?1|session1|head[_\s]?rest|resting|/1/|_s1_|_1\.(avi|txt|csv|xml|mat)",
    re.I,
)


def is_s1_path(path: Path | str) -> bool:
    s = str(path).replace("\\", "/")
    if _S_EXCLUDE.search(s):
        return False
    if _S1_HINT.search(s):
        return True
    low = s.lower()
    if low.endswith(".avi") and "session" not in low:
        return True
    return False


def infer_subject_id(path: Path, raw_root: Path) -> int | None:
    try:
        rel = path.relative_to(raw_root)
    except ValueError:
        rel = path
    parts = rel.parts
    for p in parts:
        m = re.match(r"id(\d+)$", p, re.I)
        if m:
            return int(m.group(1))
        if p.isdigit():
            return int(p)
    m = re.search(r"id(\d+)", str(path), re.I)
    return int(m.group(1)) if m else None
