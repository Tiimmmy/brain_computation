#!/usr/bin/env python
# coding=utf-8
"""加载项目原有算法模块（align / feature_generation / feature_modify / Models），
供 Web 服务层统一调用，避免复制算法实现。"""

import importlib.util
import sys
from pathlib import Path

from config import PROJECT_DIR

ALIGN_DIR = PROJECT_DIR / "align"
FEATURE_DIR = PROJECT_DIR / "feature_generation"
KERNEL_DIR = PROJECT_DIR / "feature_modify" / "gaussian_kernel"
FC_DIR = PROJECT_DIR / "Models" / "FC"

# feature_extraction.py 中使用 `from signal_prep import Peak`，需将目录加入搜索路径
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))


def _load(alias, path):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def align_tp():
    """align/align_tp_double.py：旋转、拆分、居中裁剪与 ANTs 配准。"""
    return _load("bc_align_tp_double", ALIGN_DIR / "align_tp_double.py")


def fish_to_mat():
    """align/fish_to_mat_double.py：双鱼图像预处理与对齐矩阵生成。"""
    return _load("bc_fish_to_mat_double", ALIGN_DIR / "fish_to_mat_double.py")


def signal_prep():
    return _load("signal_prep", FEATURE_DIR / "signal_prep.py")


def feature_extraction():
    return _load("feature_extraction", FEATURE_DIR / "feature_extraction.py")


def ood():
    return _load("ood", FEATURE_DIR / "ood.py")


def dist_kernel():
    return _load("bc_dist_kernel", KERNEL_DIR / "dist_kernel.py")


def fc_params():
    return _load("FC_params", FC_DIR / "FC_params.py")


def fc_train():
    return _load("bc_fc_train", FC_DIR / "train.py")


def source_statistics():
    """统计项目源程序量（行数），用于系统信息展示。"""
    stats = []
    exts = {".py": "Python", ".js": "JavaScript", ".html": "HTML", ".css": "CSS"}
    skip = {"envs", "workspace", "data", "vendor", "__pycache__", ".git", "docs"}
    for p in sorted(PROJECT_DIR.rglob("*")):
        if p.is_file() and p.suffix in exts and not (set(p.relative_to(PROJECT_DIR).parts) & skip):
            try:
                n = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
            except Exception:
                n = 0
            stats.append({"file": str(p.relative_to(PROJECT_DIR)), "lang": exts[p.suffix], "lines": n})
    return stats
