#!/usr/bin/env python
# coding=utf-8
"""实验数据管理：实验登记、原始 TIFF 序列扫描、元数据读写、帧预览。"""

import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from config import EXPERIMENT_DIR, load_settings
from services import imaging

FRAME_RE = re.compile(r"img_(\d+)_.*\.tif{1,2}$", re.I)
STEPS = [("imported", "数据导入"), ("preprocess", "图像预处理"), ("register", "运动校正配准"),
         ("atlas", "脑区模板映射"), ("signal", "钙信号提取"), ("feature", "特征计算")]
GROUPS = {0: "对照组", 1: "药物处理组"}


def exp_dir(exp_id):
    return EXPERIMENT_DIR / exp_id


def scan_raw(raw_dir):
    """扫描原始目录，返回包含 TIFF 序列（>800 帧判定为完整视频，与原脚本一致）的子目录。"""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"目录不存在：{raw_dir}")
    found = []
    for d in sorted([raw_dir] + [p for p in raw_dir.rglob("*") if p.is_dir()]):
        files = [f for f in d.iterdir() if f.is_file() and FRAME_RE.match(f.name)]
        if files:
            found.append({"path": str(d), "frames": len(files), "complete": len(files) > 800})
    return found


def list_frames(seq_dir):
    files = []
    for f in Path(seq_dir).iterdir():
        m = FRAME_RE.match(f.name)
        if m:
            files.append((int(m.group(1)), str(f)))
    return sorted(files)


def read_meta(exp_id):
    p = exp_dir(exp_id) / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"实验不存在：{exp_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_meta(meta):
    d = exp_dir(meta["id"])
    d.mkdir(parents=True, exist_ok=True)
    meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def mark_step(exp_id, step, done=True, **info):
    meta = read_meta(exp_id)
    meta["status"][step] = done
    if info:
        meta.setdefault("results", {}).setdefault(step, {}).update(info)
    return write_meta(meta)


def list_experiments():
    items = []
    for p in sorted(EXPERIMENT_DIR.glob("*/meta.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            m["progress"] = sum(1 for k, _ in STEPS if m["status"].get(k)) / len(STEPS)
            items.append(m)
        except Exception:
            continue
    return sorted(items, key=lambda x: x.get("created", ""), reverse=True)


def create_experiment(name, seq_dir, layout="double", groups=(0, 1), description="", operator="admin",
                      species="斑马鱼幼鱼 (7 dpf)", indicator="GCaMP6s", frame_rate=2.0):
    frames = list_frames(seq_dir)
    if not frames:
        raise ValueError("所选目录中未找到 img_xxx_*.tif 帧序列")
    exp_id = re.sub(r"[^\w\-]", "_", name).strip("_") or time.strftime("EXP%Y%m%d%H%M%S")
    if exp_dir(exp_id).exists():
        raise ValueError(f"实验编号已存在：{exp_id}")
    img = cv2.imread(frames[0][1], 2)
    acq = Path(seq_dir).parent / "acquisition.json"
    if acq.exists():
        try:
            info = json.loads(acq.read_text(encoding="utf-8"))
            groups = tuple(f["group"] for f in info.get("fish", [])) or groups
        except Exception:
            pass
    sides = ["l", "r"] if layout == "double" else ["l"]
    settings = load_settings()
    meta = {
        "id": exp_id, "name": name, "seq_dir": str(seq_dir), "layout": layout, "description": description,
        "operator": operator, "species": species, "indicator": indicator, "frame_rate": frame_rate,
        "frames": len(frames), "first_frame": frames[0][0], "last_frame": frames[-1][0],
        "image_size": list(img.shape[:2]), "bit_depth": int(img.dtype.itemsize * 8),
        "fish": [{"side": s, "fish_id": f"{exp_id}-{s.upper()}", "group": int(groups[i] if i < len(groups) else 0)}
                 for i, s in enumerate(sides)],
        "params": {"preprocess": settings["preprocess"], "registration": settings["registration"]},
        "status": {k: False for k, _ in STEPS}, "results": {},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta["status"]["imported"] = True
    meta["results"]["imported"] = {"frames": len(frames), "size_mb": round(
        sum(Path(p).stat().st_size for _, p in frames) / 1024 ** 2, 1)}
    write_meta(meta)
    return meta


def update_experiment(exp_id, data):
    meta = read_meta(exp_id)
    for key in ("name", "description", "operator", "species", "indicator", "frame_rate"):
        if key in data:
            meta[key] = data[key]
    if "fish" in data:
        for f, new in zip(meta["fish"], data["fish"]):
            f["group"] = int(new.get("group", f["group"]))
            f["fish_id"] = new.get("fish_id", f["fish_id"])
    if "params" in data:
        for k, v in data["params"].items():
            meta["params"].setdefault(k, {}).update(v)
    return write_meta(meta)


def delete_experiment(exp_id):
    d = exp_dir(exp_id)
    if d.exists():
        shutil.rmtree(d)


def frame_path(meta, frame_idx):
    frames = dict(list_frames(meta["seq_dir"]))
    if frame_idx not in frames:
        frame_idx = min(frames, key=lambda k: abs(k - frame_idx))
    return frame_idx, frames[frame_idx]


def frame_preview(exp_id, frame_idx, cmap="gray", vmin=None, vmax=None):
    meta = read_meta(exp_id)
    frame_idx, path = frame_path(meta, frame_idx)
    img = cv2.imread(path, 2)
    if vmin is None or vmax is None:
        vmin, vmax = imaging.auto_range(img)
    out = exp_dir(exp_id) / "cache" / f"frame_{frame_idx}_{cmap}_{int(vmin)}_{int(vmax)}.png"
    if not out.exists():
        imaging.save(imaging.colorize(img, cmap, vmin, vmax), out)
    hist, edges = np.histogram(img, bins=64)
    return {"frame": frame_idx, "url": f"/files/experiments/{exp_id}/cache/{out.name}",
            "min": int(img.min()), "max": int(img.max()), "mean": round(float(img.mean()), 1),
            "std": round(float(img.std()), 1), "vmin": vmin, "vmax": vmax,
            "hist": {"counts": hist.tolist(), "edges": [round(float(e)) for e in edges]},
            "shape": list(img.shape), "dtype": str(img.dtype), "file": Path(path).name}


def intensity_profile(exp_id, step=5):
    """整段视频的平均荧光强度曲线（降采样读取），用于识别漂移与异常帧。"""
    meta = read_meta(exp_id)
    cache = exp_dir(exp_id) / "cache" / "intensity_profile.json"
    if cache.exists():
        return json.loads(cache.read_text())
    frames = list_frames(meta["seq_dir"])
    xs, left, right = [], [], []
    for idx, p in frames[::step]:
        img = cv2.imread(p, 2).astype(np.float32)
        h, w = img.shape
        xs.append(idx)
        if meta["layout"] == "double":
            left.append(round(float(img[:, : w // 2].mean()), 2))
            right.append(round(float(img[:, w // 2:].mean()), 2))
        else:
            left.append(round(float(img.mean()), 2))
    data = {"frames": xs, "l": left, "r": right}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data
