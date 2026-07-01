# -*- coding: utf-8 -*-
"""UBFC-Phys 目录与常量 — 与 datasets/UBFC_*、COHFACE_* 平行。运行时不使用固定 fps。"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UBFC_PHYS_RAW = os.path.join(PROJECT_DIR, "datasets", "UBFC-Phys_raw")
# 预处理 H5 放 /ssd，避免占 home 盘
UBFC_PHYS_H5 = "/ssd/UBFC_phy/h5"
UBFC_PHYS_RAW_SSD = "/ssd/UBFC_phy/extracted"
UBFC_PHYS_STAGING = os.path.join(PROJECT_DIR, "datasets", "UBFC-Phys_staging")
UBFC_PHYS_ARCHIVES = os.path.join(UBFC_PHYS_STAGING, "archives")
UBFC_PHYS_DOWNLOAD_STATE = os.path.join(UBFC_PHYS_STAGING, "state.json")
UBFC_PHYS_DOWNLOAD_PROGRESS = os.path.join(UBFC_PHYS_STAGING, "progress.json")
UBFC_PHYS_DOWNLOAD_LOG = os.path.join(UBFC_PHYS_STAGING, "download.log")
UBFC_PHYS_MANIFEST_DIR = os.path.join(PROJECT_DIR, "datasets", "UBFC-Phys_manifests")

UBFC_PHYS_README = os.path.join(PROJECT_DIR, "datasets", "UBFC-Phys_README.pdf")
UBFC_PHYS_PROBE_REPORT = os.path.join(
    UBFC_PHYS_MANIFEST_DIR, "dataset_probe_report.json"
)
UBFC_PHYS_MANIFEST_JSON = os.path.join(
    UBFC_PHYS_MANIFEST_DIR, "ubfc_phys_manifest.json"
)
UBFC_PHYS_MANIFEST_CSV = os.path.join(
    UBFC_PHYS_MANIFEST_DIR, "ubfc_phys_manifest.csv"
)
UBFC_PHYS_EVAL_LIST = os.path.join(UBFC_PHYS_H5, "ubfc_phys_t1_eval_list.npy")

# s1-s10 子集评估排除
EXCLUDE_S1_TO_S10 = {3, 8, 9}

# 文档参考值；预处理/评估必须用 manifest 或 H5 实测 fps
BVP_FS_HZ = 64.0
PAPER_VIDEO_FPS_REF = 35.0
STORE_SIZE = 96
TASK_T1 = "T1"

# rPPG-Toolbox / 论文补充材料 T1 排除
EXCLUSION_LIST_T1 = [
    "s3_T1", "s8_T1", "s9_T1", "s26_T1", "s28_T1", "s30_T1", "s31_T1",
    "s32_T1", "s33_T1", "s40_T1", "s52_T1", "s53_T1", "s54_T1", "s56_T1",
]

DATUBFC_BASE = "https://search-data.ubfc.fr/dl_data.php"
DATUBFC_FILES = {
    "readme": 108,
    "s1_to_s10": 140,
    "s11_to_s20": 141,
    "s21_to_s30": 142,
    "s31_to_s40": 220,
    "s41_to_s50": 143,
    "s51_to_s56": 144,
}
