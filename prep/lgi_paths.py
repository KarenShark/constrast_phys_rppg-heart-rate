# -*- coding: utf-8 -*-
"""LGI-PPGI-DB 目录与常量 — 与 datasets/UBFC_*、COHFACE_* 平行。仅 Session1。"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LGI_RAW = os.path.join(PROJECT_DIR, "datasets", "LGI_raw")
LGI_H5 = os.path.join(PROJECT_DIR, "datasets", "LGI_h5")
LGI_STAGING = os.path.join(PROJECT_DIR, "datasets", "LGI_staging")
LGI_ARCHIVES = os.path.join(LGI_STAGING, "archives")
LGI_MANIFEST_DIR = os.path.join(PROJECT_DIR, "datasets", "LGI_manifests")
LGI_META = os.path.join(PROJECT_DIR, "prep", "vendor", "LGI-PPGI-DB")

LGI_MANIFEST_JSON = os.path.join(LGI_MANIFEST_DIR, "lgi_manifest.json")
LGI_MANIFEST_CSV = os.path.join(LGI_MANIFEST_DIR, "lgi_manifest.csv")
LGI_EVAL_LIST = os.path.join(LGI_MANIFEST_DIR, "lgi_eval_list.npy")

LGI_DOWNLOAD_PROBE = os.path.join(LGI_MANIFEST_DIR, "download_probe_report.json")
LGI_DATASET_PROBE = os.path.join(LGI_MANIFEST_DIR, "dataset_probe_report.json")
LGI_DOWNLOAD_STATE = os.path.join(LGI_STAGING, "state.json")
LGI_DOWNLOAD_PROGRESS = os.path.join(LGI_STAGING, "progress.json")
LGI_DOWNLOAD_LOG = os.path.join(LGI_STAGING, "download.log")

# 文档参考；manifest/H5 必须用 cv2 实测 fps
PAPER_VIDEO_FPS_REF = 25.0
BVP_FS_HZ = 60.0
STORE_SIZE = 96
SESSION_S1 = 1

# id1.zip 内 S1 路径模式（id1 实测后可收紧）
S1_7Z_INCLUDES = [
    r"-ir!*session1*",
    r"-ir!*Session1*",
    r"-ir!*session_1*",
    r"-ir!*S1*",
    r"-ir!*_1.*",
]

LGI_BASE = "https://gw.cancontrols.com/LGI_DATABASE"
LGI_SUBJECTS = [
    {"n": 1, "name": "id1.zip", "size_gb_ref": 7.57},
    {"n": 2, "name": "id2.zip", "size_gb_ref": 5.49},
    {"n": 3, "name": "id3.zip", "size_gb_ref": 4.21},
    {"n": 4, "name": "id4.zip", "size_gb_ref": 5.36},
    {"n": 5, "name": "id5.zip", "size_gb_ref": 6.41},
    {"n": 6, "name": "id6.zip", "size_gb_ref": 6.55},
]
