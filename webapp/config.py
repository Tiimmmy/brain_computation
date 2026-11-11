#!/usr/bin/env python
# coding=utf-8
"""全局配置：路径、软件信息、默认算法参数。"""

import json
import os
from pathlib import Path

SOFTWARE_NAME = "斑马鱼全脑钙成像神经计算分析系统"
SOFTWARE_SHORT = "BrainComp"
VERSION = "V1.0"

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEBAPP_DIR.parent
WORKSPACE = Path(os.environ.get("BRAINCOMP_WORKSPACE", PROJECT_DIR / "workspace"))
DATA_DIR = Path(os.environ.get("BRAINCOMP_DATA", PROJECT_DIR / "data"))

EXPERIMENT_DIR = WORKSPACE / "experiments"
LIBRARY_DIR = WORKSPACE / "library"
DATASET_DIR = WORKSPACE / "datasets"
MODEL_DIR = WORKSPACE / "models"
TEMPLATE_DIR = WORKSPACE / "templates"
CACHE_DIR = WORKSPACE / "cache"
SETTINGS_FILE = WORKSPACE / "settings.json"
USERS_FILE = WORKSPACE / "users.json"

for _d in (WORKSPACE, EXPERIMENT_DIR, LIBRARY_DIR, DATASET_DIR, MODEL_DIR, TEMPLATE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.environ.get("BRAINCOMP_SECRET", "braincomp-zebrafish-secret-key")

# 特征类型（与 feature_generation/key.py 中 features_dict 保持一致）
SCALAR_FEATURES = [
    ("num_peaks", "钙峰数量"),
    ("areas_sum", "峰面积总和"),
    ("areas_mean", "平均峰面积"),
    ("duration_mean", "平均持续时间"),
    ("dr", "占空比"),
    ("height_max", "最大峰高"),
    ("height_mean", "平均峰高"),
    ("ap_mean", "平均动作电位强度"),
    ("interval_mean", "平均峰间间隔"),
    ("separation_mean", "平均峰距"),
]
MATRIX_FEATURES = [
    ("connection_corr", "放电同步连接矩阵"),
    ("value_corr", "峰值强度相关矩阵"),
    ("active_regions", "逐帧活跃脑区数"),
]

DEFAULT_SETTINGS = {
    "preprocess": {"rotate_left": 26, "rotate_right": -26, "crop_h": 450, "crop_w": 250,
                   "ref_frame": 500, "threshold": 3},
    "registration": {"transform": "ElasticSyN", "resample_h": 100, "resample_w": 80,
                     "workers": 48},
    "signal": {"lowpass_order": 30, "lowpass_cutoff": 0.3, "poly_degree": 6,
               "trim": 10, "outlier_range": 500},
    "kernel": {"band_width": 0.5},
    "training": {"k_folds": 5, "epochs": 100, "lr": 0.01, "momentum": 0.9,
                 "weight_decay": 1e-5, "optimizer": "SGD", "device": "cuda"},
}


def load_settings():
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if isinstance(v, dict) and k in settings:
                    settings[k].update(v)
        except Exception:
            pass
    return settings


def save_settings(settings):
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
