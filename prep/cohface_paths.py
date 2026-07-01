# -*- coding: utf-8 -*-
"""COHFACE 目录约定 — 与 datasets/UBFC_* 平行。"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COHFACE_RAW = os.path.join(PROJECT_DIR, "datasets", "COHFACE_raw")
COHFACE_H5 = os.path.join(PROJECT_DIR, "datasets", "COHFACE_h5")
COHFACE_MANIFEST_DIR = os.path.join(PROJECT_DIR, "datasets", "COHFACE_manifests")

COHFACE_MANIFEST_CSV = os.path.join(COHFACE_MANIFEST_DIR, "cohface_manifest.csv")
COHFACE_MANIFEST_JSON = os.path.join(COHFACE_MANIFEST_DIR, "cohface_manifest.json")
COHFACE_EVAL_LIST = os.path.join(COHFACE_MANIFEST_DIR, "cohface_eval_list.npy")

# native fps=20, face crop 96x96（attrs 写在 H5 内）
COHFACE_STORE_SIZE = 96
COHFACE_DEFAULT_FPS = 20.0
COHFACE_PULSE_FS = 256.0
