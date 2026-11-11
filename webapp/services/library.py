#!/usr/bin/env python
# coding=utf-8
"""个体特征库：汇总来自实验流水线或批量导入的脑区信号矩阵的单鱼特征，作为建模样本来源。"""

import csv
import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import LIBRARY_DIR, SCALAR_FEATURES
from services import analysis
from services import experiments as ex

GROUP_NAME = {0: "对照组", 1: "药物处理组"}


def _meta_path(fid):
    return LIBRARY_DIR / fid / "meta.json"


def list_entries():
    items = []
    for p in sorted(LIBRARY_DIR.glob("*/meta.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return items


def get_entry(fid):
    return json.loads(_meta_path(fid).read_text(encoding="utf-8"))


def write_entry(meta):
    _meta_path(meta["fish_id"]).parent.mkdir(parents=True, exist_ok=True)
    _meta_path(meta["fish_id"]).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_entries(ids):
    for fid in ids:
        d = LIBRARY_DIR / fid
        if d.exists():
            shutil.rmtree(d)


def add_from_experiment(exp_id):
    meta = ex.read_meta(exp_id)
    added = []
    for f in meta["fish"]:
        src = ex.exp_dir(exp_id) / "features" / f"{f['side']}.pkl"
        if not src.exists():
            continue
        d = LIBRARY_DIR / f["fish_id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / "features.pkl")
        sig = np.load(ex.exp_dir(exp_id) / "signals" / f"{f['side']}.npz")
        np.savez_compressed(d / "signal.npz", raw=sig["raw"], dff=sig["dff"], baseline=sig["baseline"],
                            interp=sig["interp"], need_move=sig["need_move"])
        feats = analysis.load_features(d / "features.pkl")
        write_entry({"fish_id": f["fish_id"], "label": f["group"], "group": GROUP_NAME[f["group"]],
                     "source": "实验流水线", "exp_id": exp_id, "side": f["side"],
                     "fly_frames": sig["need_move"].tolist(), "summary": analysis.feature_summary(feats),
                     "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        added.append(f["fish_id"])
    return added


def scan_signal_folder(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.npy"))
    labels = {}
    lab = folder / "labels.csv"
    if lab.exists():
        with lab.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                labels[row["fish_id"]] = int(row["label"])
    preview = []
    for f in files[:500]:
        arr = np.load(f, mmap_mode="r")
        preview.append({"file": f.name, "fish_id": f.stem, "shape": list(arr.shape),
                        "label": labels.get(f.stem), "group": GROUP_NAME.get(labels.get(f.stem), "未标注")})
    return {"folder": str(folder), "count": len(files), "has_labels": lab.exists(), "files": preview}


def run_import(ctx, folder, workers=48, overwrite=False):
    info = scan_signal_folder(folder)
    folder = Path(folder)
    jobs = []
    for item in info["files"]:
        if not overwrite and _meta_path(item["fish_id"]).exists():
            continue
        jobs.append(item)
    ctx.log(f"待分析信号矩阵 {len(jobs)} 个（已存在 {info['count'] - len(jobs)} 个跳过），并行进程 {workers}")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(analysis.library_worker, (str(folder / it["file"]), str(LIBRARY_DIR / it["fish_id"]))): it
                for it in jobs}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                _, summ, need_move = fut.result()
                label = it["label"] if it["label"] is not None else 0
                write_entry({"fish_id": it["fish_id"], "label": label, "group": GROUP_NAME[label],
                             "source": "信号矩阵导入", "exp_id": None, "side": None,
                             "file": str(folder / it["file"]), "fly_frames": need_move, "summary": summ,
                             "created": time.strftime("%Y-%m-%d %H:%M:%S")})
            except Exception as e:
                ctx.log(f"{it['fish_id']} 分析失败：{e}")
            done += 1
            ctx.progress(done / max(1, len(jobs)), f"已分析 {done}/{len(jobs)}")
            if done % 20 == 0:
                ctx.log(f"已完成 {done} 个样本")
    return {"imported": done}


def entry_detail(fid, regions=None):
    meta = get_entry(fid)
    d = LIBRARY_DIR / fid
    feats = analysis.load_features(d / "features.pkl")
    sig = np.load(d / "signal.npz")
    regions = regions or [1, 20, 45, 70, 95]
    r = lambda a: np.round(np.asarray(a, float), 4).tolist()  # noqa
    return {"meta": meta, "table": {k: r(feats[k]) for k, _ in SCALAR_FEATURES},
            "connection_corr": r(feats["connection_corr"]), "regions": regions,
            "dff": r(sig["dff"][[x - 1 for x in regions]]), "raw": r(sig["raw"][[x - 1 for x in regions]])}


def feature_matrix(ids, keys=None):
    keys = keys or [k for k, _ in SCALAR_FEATURES]
    X = np.zeros((len(ids), len(keys), 100), np.float32)
    y = np.zeros(len(ids), np.int64)
    for i, fid in enumerate(ids):
        feats = analysis.load_features(LIBRARY_DIR / fid / "features.pkl")
        for j, k in enumerate(keys):
            X[i, j] = np.asarray(feats[k], np.float32)[:100]
        y[i] = get_entry(fid)["label"]
    X[~np.isfinite(X)] = 0
    return X, y


def group_statistics():
    from scipy import stats
    entries = list_entries()
    if not entries:
        return {"count": 0}
    ids = [e["fish_id"] for e in entries]
    X, y = feature_matrix(ids)
    per_fish = X.mean(axis=2)
    out = []
    for j, (k, name) in enumerate(SCALAR_FEATURES):
        a, b = per_fish[y == 0, j], per_fish[y == 1, j]
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else None
        q = lambda v: [round(float(x), 4) for x in np.percentile(v, [0, 25, 50, 75, 100])] if len(v) else []  # noqa
        out.append({"key": k, "name": name, "ctl_mean": round(float(a.mean()), 4) if len(a) else None,
                    "drug_mean": round(float(b.mean()), 4) if len(b) else None, "p": p,
                    "ctl_box": q(a), "drug_box": q(b)})
    return {"count": len(ids), "ctl": int((y == 0).sum()), "drug": int((y == 1).sum()), "features": out,
            "sources": {s: sum(1 for e in entries if e["source"] == s) for s in {e["source"] for e in entries}}}
