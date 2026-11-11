#!/usr/bin/env python
# coding=utf-8
"""图像渲染工具：16 位荧光图像显示映射、伪彩色、脑区叠加、脑区数值图。"""

import colorsys
from pathlib import Path

import cv2
import numpy as np

CMAPS = {
    "gray": None, "inferno": cv2.COLORMAP_INFERNO, "viridis": cv2.COLORMAP_VIRIDIS,
    "jet": cv2.COLORMAP_JET, "turbo": cv2.COLORMAP_TURBO, "hot": cv2.COLORMAP_HOT,
    "magma": cv2.COLORMAP_MAGMA, "plasma": cv2.COLORMAP_PLASMA, "cool": cv2.COLORMAP_COOL,
}


def _gradient_lut(stops):
    """由颜色节点（RGB）生成 OpenCV 自定义 256 级色表（BGR）。"""
    xs = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    rgb = np.stack([np.interp(t, xs, [c[i] for c in stops]) for i in range(3)], axis=1)
    return rgb[:, ::-1].astype(np.uint8).reshape(256, 1, 3)


CUSTOM = {
    "reds": _gradient_lut([(255, 255, 255), (253, 224, 197), (240, 140, 43), (176, 58, 10), (74, 18, 5)]),
    "blues": _gradient_lut([(255, 255, 255), (198, 219, 239), (66, 146, 198), (8, 69, 148)]),
}


def auto_range(arr, low=0.5, high=99.8):
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(a, [low, high])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def to_uint8(arr, vmin=None, vmax=None):
    a = np.asarray(arr, dtype=np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = auto_range(a)
    return np.clip((a - vmin) / (vmax - vmin + 1e-9) * 255, 0, 255).astype(np.uint8)


def colorize(arr, cmap="gray", vmin=None, vmax=None):
    u8 = to_uint8(arr, vmin, vmax)
    if cmap in CUSTOM:
        return cv2.applyColorMap(u8, CUSTOM[cmap])
    if CMAPS.get(cmap) is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(u8, CMAPS[cmap])


def region_palette(n=100, seed=3):
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.618033988749895) % 1.0
    pal = np.zeros((n + 1, 3), np.uint8)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.55 + 0.35 * rng.random(), 0.85 + 0.15 * rng.random())
        pal[i + 1] = [int(b * 255), int(g * 255), int(r * 255)]
    return pal


def label_boundaries(labels):
    lab = labels.astype(np.int32)
    edge = np.zeros(lab.shape, bool)
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return edge


def atlas_color(labels, background=None, alpha=0.55):
    pal = region_palette(int(labels.max()) if labels.max() > 0 else 100)
    color = pal[labels.astype(np.int32)]
    if background is not None:
        base = colorize(background, "gray")
        mask = labels > 0
        out = base.copy()
        out[mask] = (base[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    else:
        out = color.copy()
        out[labels == 0] = (250, 250, 250)
    out[label_boundaries(labels)] = (40, 40, 40)
    return out


def region_value_map(labels, values, cmap="turbo", vmin=None, vmax=None, bg=(255, 255, 255)):
    vals = np.asarray(values, np.float32)
    if vmin is None:
        vmin = float(np.nanmin(vals))
    if vmax is None:
        vmax = float(np.nanmax(vals))
    lut = np.concatenate([[vmin], vals])
    img = lut[labels.astype(np.int32)]
    out = colorize(img, cmap, vmin, vmax)
    out[labels == 0] = bg
    out[label_boundaries(labels) & (labels > 0)] = (90, 90, 90)
    return out


def overlay_contour(gray, mask_a, mask_b):
    """两组掩膜轮廓叠加：A 为青色，B 为品红色。"""
    base = colorize(gray, "gray")
    for m, col in ((mask_a, (255, 220, 0)), (mask_b, (255, 0, 220))):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, cnts, -1, col, 1)
    return base


def checkerboard(a, b, tiles=8):
    a8, b8 = to_uint8(a), to_uint8(b)
    h, w = a8.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sel = ((yy // (h // tiles + 1)) + (xx // (w // tiles + 1))) % 2 == 0
    return cv2.cvtColor(np.where(sel, a8, b8).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def blend_rg(a, b):
    """红绿通道融合：a→品红，b→绿，重合区域呈白/灰。"""
    a8, b8 = to_uint8(a), to_uint8(b)
    return np.dstack([a8, b8, a8])


def save(img, path, scale=1.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), img)
    return path
