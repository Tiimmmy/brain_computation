#!/usr/bin/env python
# coding=utf-8
"""建模数据集构建：特征选择、高斯核空间平滑（feature_modify/gaussian_kernel/dist_kernel.py）、
多通道归一化（min-max / z-score，参照 feature_generation/data_prep.ipynb）与数据集统计。"""

import json
import time

import numpy as np

from config import DATASET_DIR, SCALAR_FEATURES, CACHE_DIR
from services import imaging, legacy, library
from services.pipeline import template_files

CHANNELS = {
    "raw": "原始特征值",
    "minmax": "Min-Max 归一化",
    "zscore": "Z-Score 标准化",
    "kernel_raw": "高斯核平滑",
    "kernel_minmax": "高斯核平滑 + Min-Max",
    "kernel_zscore": "高斯核平滑 + Z-Score",
}


def gaussian_kernel(centers_path, band_width):
    """与 dist_kernel.dist_simi 相同的距离高斯核，区域中心坐标文件可配置。"""
    dk = legacy.dist_kernel()
    X = np.loadtxt(centers_path)
    dist = dk.EuclideanDistances(X)
    dist = np.clip(dist, 0, None)
    d = np.sort(np.sqrt(dist), axis=1).T[1]
    sigma = band_width * d
    kernel = np.exp(-dist / (2 * sigma ** 2))
    return kernel / np.sum(kernel, axis=0)


def kernel_smooth(X, centers_path, band_width):
    """X: (样本, 特征类型, 脑区)。调用 dist_kernel.final 完成核平滑（替换其固定路径的核函数）。"""
    dk = legacy.dist_kernel()
    orig = dk.dist_simi
    dk.dist_simi = lambda num, a: gaussian_kernel(centers_path, a)
    try:
        tmp = CACHE_DIR / f"kernel_input_{int(time.time() * 1000)}.npy"
        np.save(tmp, X[None, ...])                    # (1, N, F, R)，与 final 的拼接/拆分约定一致
        out = dk.final(str(tmp), X.shape[2], band_width)[0]
        tmp.unlink(missing_ok=True)
    finally:
        dk.dist_simi = orig
    return out.astype(np.float32)


def kernel_preview(template_id, band_width, region=25):
    _, atlas_p, centers_p = template_files(template_id)
    import cv2
    atlas = cv2.imread(str(atlas_p), 0)
    K = gaussian_kernel(centers_p, band_width)
    path = CACHE_DIR / f"kernel_{template_id}_{band_width}_{region}.png"
    imaging.save(imaging.region_value_map(atlas, K[:, region - 1], "reds", 0, float(K[:, region - 1].max()),
                                          bg=(255, 255, 255)), path)
    eff = 1.0 / np.sum(K ** 2, axis=0)              # 有效参与脑区数
    return {"matrix": np.round(K, 4).tolist(), "map": f"/files/cache/{path.name}?t={int(time.time())}",
            "effective_regions": round(float(eff.mean()), 2), "self_weight": round(float(np.diag(K).mean()), 4)}


def smoothing_demo(fish_id, feature_key, template_id, band_width):
    import cv2
    _, atlas_p, centers_p = template_files(template_id)
    atlas = cv2.imread(str(atlas_p), 0)
    X, _ = library.feature_matrix([fish_id], [feature_key])
    Xs = kernel_smooth(X, centers_p, band_width)
    vmin, vmax = float(X.min()), float(X.max())
    a = CACHE_DIR / f"smooth_before_{fish_id}_{feature_key}.png"
    b = CACHE_DIR / f"smooth_after_{fish_id}_{feature_key}_{band_width}.png"
    imaging.save(imaging.region_value_map(atlas, X[0, 0], "turbo", vmin, vmax), a)
    imaging.save(imaging.region_value_map(atlas, Xs[0, 0], "turbo", vmin, vmax), b)
    t = int(time.time())
    return {"before": f"/files/cache/{a.name}?t={t}", "after": f"/files/cache/{b.name}?t={t}",
            "values_before": np.round(X[0, 0], 4).tolist(), "values_after": np.round(Xs[0, 0], 4).tolist()}


def _minmax(A):
    mx, mn = A.max(axis=0), A.min(axis=0)
    return (A - mn) / np.where(mx - mn == 0, 1, mx - mn)


def _zscore(A):
    mu, sd = A.mean(axis=0), A.std(axis=0)
    return (A - mu) / np.where(sd == 0, 1, sd)


def list_datasets():
    items = []
    for p in sorted(DATASET_DIR.glob("*/meta.json")):
        items.append(json.loads(p.read_text(encoding="utf-8")))
    return sorted(items, key=lambda x: x["created"], reverse=True)


def get_dataset(did):
    return json.loads((DATASET_DIR / did / "meta.json").read_text(encoding="utf-8"))


def run_build(ctx, name, ids, feature_keys, channels, band_width, template_id="zebrafish_std_100", description=""):
    _, _, centers_p = template_files(template_id)
    ctx.log(f"样本 {len(ids)} 个，特征类型 {len(feature_keys)} 种，通道 {channels}，带宽系数 H={band_width}")
    X, y = library.feature_matrix(ids, feature_keys)          # (N, F, R)
    ctx.progress(0.3, "特征矩阵已加载")
    Xs = kernel_smooth(X, centers_p, band_width) if any(c.startswith("kernel") for c in channels) else None
    ctx.progress(0.6, "高斯核平滑完成")
    N, F, R = X.shape
    flat, flat_s = X.reshape(N, F * R), (Xs.reshape(N, F * R) if Xs is not None else None)
    chans = []
    for c in channels:
        src = flat_s if c.startswith("kernel") else flat
        kind = c.replace("kernel_", "")
        chans.append(src if kind == "raw" else _minmax(src) if kind == "minmax" else _zscore(src))
    data = np.stack(chans, axis=-1).astype(np.float32)        # (N, F*R, C)
    data[~np.isfinite(data)] = 0
    did = time.strftime("DS%Y%m%d%H%M%S")
    d = DATASET_DIR / did
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "features.npy", data)
    np.save(d / "label.npy", y)
    # PCA 二维投影
    A = _zscore(flat)
    U, S, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    pc = (A - A.mean(0)) @ Vt[:2].T
    var = (S ** 2 / np.sum(S ** 2))[:2]
    meta = {"id": did, "name": name, "description": description, "samples": int(N), "ids": ids,
            "labels": {"0": int((y == 0).sum()), "1": int((y == 1).sum())}, "feature_keys": feature_keys,
            "feature_names": [dict(SCALAR_FEATURES)[k] for k in feature_keys], "regions": int(R),
            "channels": channels, "shape": list(data.shape), "band_width": band_width, "template": template_id,
            "pca": {"x": np.round(pc[:, 0], 4).tolist(), "y": np.round(pc[:, 1], 4).tolist(),
                    "label": y.tolist(), "var": np.round(var, 4).tolist()},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb": round(data.nbytes / 1024 ** 2, 2)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx.log(f"数据集 {did} 构建完成，张量形状 {data.shape}")
    return {"id": did, "shape": list(data.shape)}


def delete_dataset(did):
    import shutil
    shutil.rmtree(DATASET_DIR / did, ignore_errors=True)
