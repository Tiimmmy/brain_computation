#!/usr/bin/env python
# coding=utf-8
"""单个实验的分析流水线：图像预处理 → 运动校正配准 → 脑区模板映射 → 钙信号提取 → 特征计算。"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from config import TEMPLATE_DIR, SCALAR_FEATURES
from services import analysis, imaging, legacy
from services import experiments as ex


def _url(exp_id, rel):
    return f"/files/experiments/{exp_id}/{rel}?t={int(time.time())}"


def _sides(meta):
    return [f["side"] for f in meta["fish"]]


def split_rotate(img, meta):
    atp = legacy.align_tp()
    p = meta["params"]["preprocess"]
    if meta["layout"] == "single":
        return [atp.rotate_img(img, p["rotate_left"]) if p["rotate_left"] else img]
    h, w = img.shape[:2]
    return [atp.rotate_img(img[0:h, 0:int(w / 2)], p["rotate_left"]),
            atp.rotate_img(img[0:h, int(w / 2):int(w)], p["rotate_right"])]


# ----------------------------------------------------------------------------
# 1. 图像预处理
# ----------------------------------------------------------------------------
def preprocess_preview(exp_id, frame_idx=None, params=None):
    meta = ex.read_meta(exp_id)
    if params:
        meta["params"]["preprocess"].update({k: float(v) if k.startswith("rotate") else int(v)
                                             for k, v in params.items()})
    p = meta["params"]["preprocess"]
    frame_idx = p["ref_frame"] if frame_idx is None else frame_idx
    frame_idx, path = ex.frame_path(meta, frame_idx)
    ftm = legacy.fish_to_mat()
    img = cv2.imread(path, 2)
    vmin, vmax = imaging.auto_range(img)
    d = ex.exp_dir(exp_id) / "preprocess"
    imaging.save(imaging.colorize(img, "gray", vmin, vmax), d / "raw.png")
    out = {"frame": frame_idx, "raw": _url(exp_id, "preprocess/raw.png"), "fish": []}
    h, w = img.shape[:2]
    if meta["layout"] == "double":
        raw_split = imaging.colorize(img, "gray", vmin, vmax)
        cv2.line(raw_split, (w // 2, 0), (w // 2, h), (0, 200, 255), 2)
        imaging.save(raw_split, d / "split.png")
        out["split"] = _url(exp_id, "preprocess/split.png")
    for side, rot in zip(_sides(meta), split_rotate(img, meta)):
        cx, cy = ftm.crop_center(rot)
        ch, cw = p["crop_h"], p["crop_w"]
        vis = imaging.colorize(rot, "gray", vmin, vmax)
        mask = cv2.convertScaleAbs(rot, alpha=(255.0 / 65535.0)) > p["threshold"]
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 0), 1)
        x0, y0 = int(cx - 0.5 * cw), int(cy - 0.5 * ch)
        cv2.rectangle(vis, (x0, y0), (x0 + cw, y0 + ch), (0, 160, 255), 2)
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        crop = ftm.crop_img(rot.astype(np.uint16), (cx, cy), ch, cw)
        imaging.save(vis, d / f"rot_{side}.png")
        imaging.save(imaging.colorize(crop, "inferno", vmin, vmax), d / f"crop_{side}.png")
        out["fish"].append({"side": side, "rotated": _url(exp_id, f"preprocess/rot_{side}.png"),
                            "crop": _url(exp_id, f"preprocess/crop_{side}.png"),
                            "rotated_shape": list(rot.shape), "center": [round(cx, 1), round(cy, 1)],
                            "crop_shape": list(crop.shape), "fish_pixels": int(mask.sum()),
                            "crop_box": [x0, y0, x0 + cw, y0 + ch]})
    out["params"] = p
    return out


def apply_preprocess(exp_id, params):
    meta = ex.read_meta(exp_id)
    meta["params"]["preprocess"].update({k: float(v) if k.startswith("rotate") else int(v)
                                         for k, v in params.items()})
    ex.write_meta(meta)
    res = preprocess_preview(exp_id)
    ex.mark_step(exp_id, "preprocess", True, centers={f["side"]: f["center"] for f in res["fish"]},
                 crop=[meta["params"]["preprocess"]["crop_h"], meta["params"]["preprocess"]["crop_w"]])
    return res


# ----------------------------------------------------------------------------
# 2. 运动校正配准（ANTs）
# ----------------------------------------------------------------------------
def run_register(ctx, exp_id, transform=None, workers=None):
    from services import reg_worker
    meta = ex.read_meta(exp_id)
    pp, rp = meta["params"]["preprocess"], meta["params"]["registration"]
    if transform:
        rp["transform"] = transform
    if workers:
        rp["workers"] = int(workers)
    meta["params"]["registration"] = rp
    ex.write_meta(meta)
    frames = ex.list_frames(meta["seq_dir"])
    ref_idx, ref_path = ex.frame_path(meta, pp["ref_frame"])
    d = ex.exp_dir(exp_id)
    tdir, adir = d / "transforms", d / "aligned"
    tdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    sides = _sides(meta)
    ch, cw = pp["crop_h"], pp["crop_w"]
    n = max(i for i, _ in frames) + 1
    mats = {s: np.lib.format.open_memmap(adir / f"{s}.npy", mode="w+", dtype=np.uint16, shape=(n, ch * cw))
            for s in sides}
    metrics = {s: [None] * n for s in sides}
    ctx.log(f"参考帧 #{ref_idx}，配准算法 {rp['transform']}，降采样 {rp['resample_h']}×{rp['resample_w']}，"
            f"并行进程 {rp['workers']}，共 {len(frames)} 帧 × {len(sides)} 条鱼")
    jobs = []
    for idx, path in frames:
        name = Path(path).stem
        jobs.append((idx, path, [str(tdir / f"{s}{name}") for s in sides]))
    t0 = time.time()
    ctxmp = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(rp["workers"]), mp_context=ctxmp,
                             initializer=reg_worker.init_worker,
                             initargs=(ref_path, meta["layout"], [pp["rotate_left"], pp["rotate_right"]],
                                       (ch, cw), rp["transform"], (rp["resample_h"], rp["resample_w"]))) as pool:
        futs = [pool.submit(reg_worker.process_frame, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs)):
            idx, arrs, mets = fut.result()
            for s, a, m in zip(sides, arrs, mets):
                mats[s][idx] = a
                metrics[s][idx] = m
            if k % 25 == 0 or k == len(futs) - 1:
                ctx.progress((k + 1) / len(futs), f"已配准 {k + 1}/{len(futs)} 帧")
            if k % 200 == 0 and k:
                ctx.log(f"已完成 {k} 帧，平均 {(time.time() - t0) / k * int(rp['workers']):.2f} 秒/帧/进程")
    for m in mats.values():
        m.flush()
    summary = {}
    for s in sides:
        before = [m["before"] for m in metrics[s] if m]
        after = [m["after"] for m in metrics[s] if m]
        summary[s] = {"corr_before": round(float(np.mean(before)), 4), "corr_after": round(float(np.mean(after)), 4),
                      "failed": int(sum(1 for m in metrics[s] if m and not m["ok"]))}
        ctx.log(f"鱼 {s.upper()}：平均相关系数 {summary[s]['corr_before']:.4f} → {summary[s]['corr_after']:.4f}")
    (d / "aligned" / "metrics.json").write_text(json.dumps(metrics))
    _projection_images(exp_id, meta, mats)
    ex.mark_step(exp_id, "register", True, summary=summary, ref_frame=ref_idx, transform=rp["transform"],
                 seconds=round(time.time() - t0, 1), frames=len(frames))
    return {"summary": summary, "seconds": round(time.time() - t0, 1)}


def _projection_images(exp_id, meta, mats):
    """配准后时间平均 / 标准差投影图，以及配准前对应投影（用于对比运动伪影）。"""
    pp = meta["params"]["preprocess"]
    ch, cw = pp["crop_h"], pp["crop_w"]
    ftm = legacy.fish_to_mat()
    frames = ex.list_frames(meta["seq_dir"])[::10]
    ref_img = cv2.imread(ex.frame_path(meta, pp["ref_frame"])[1], 2)
    centers = [ftm.crop_center(r) for r in split_rotate(ref_img, meta)]
    raw_stack = {s: [] for s in _sides(meta)}
    for _, p in frames:
        for s, rot, c in zip(_sides(meta), split_rotate(cv2.imread(p, 2), meta), centers):
            raw_stack[s].append(ftm.crop_img(rot.astype(np.float32), c, ch, cw))
    d = ex.exp_dir(exp_id) / "aligned"
    for s, m in mats.items():
        sub = np.asarray(m[[i for i, _ in frames]], dtype=np.float32).reshape(-1, ch, cw)
        raw = np.stack(raw_stack[s])
        lo, hi = imaging.auto_range(sub.mean(0))
        imaging.save(imaging.colorize(raw.mean(0), "gray", lo, hi), d / f"mean_before_{s}.png")
        imaging.save(imaging.colorize(sub.mean(0), "gray", lo, hi), d / f"mean_after_{s}.png")
        slo, shi = imaging.auto_range(sub.std(0))
        imaging.save(imaging.colorize(raw.std(0), "inferno", slo, shi * 2.5), d / f"std_before_{s}.png")
        imaging.save(imaging.colorize(sub.std(0), "inferno", slo, shi * 2.5), d / f"std_after_{s}.png")


def register_result(exp_id, frame_idx=None):
    meta = ex.read_meta(exp_id)
    d = ex.exp_dir(exp_id) / "aligned"
    if not (d / "metrics.json").exists():
        return {"ready": False}
    metrics = json.loads((d / "metrics.json").read_text())
    pp = meta["params"]["preprocess"]
    ch, cw = pp["crop_h"], pp["crop_w"]
    frame_idx = frame_idx if frame_idx is not None else 100
    frame_idx, path = ex.frame_path(meta, frame_idx)
    ftm = legacy.fish_to_mat()
    ref_img = cv2.imread(ex.frame_path(meta, pp["ref_frame"])[1], 2)
    refs = split_rotate(ref_img, meta)
    cur = split_rotate(cv2.imread(path, 2), meta)
    out = {"ready": True, "frame": frame_idx, "fish": [], "summary": meta["results"].get("register", {})}
    for s, r, c in zip(_sides(meta), refs, cur):
        center = ftm.crop_center(r)
        ref_crop = ftm.crop_img(r.astype(np.float32), center, ch, cw)
        raw_crop = ftm.crop_img(c.astype(np.float32), center, ch, cw)
        mat = np.load(d / f"{s}.npy", mmap_mode="r")
        aligned = np.asarray(mat[frame_idx], np.float32).reshape(ch, cw)
        imaging.save(imaging.blend_rg(ref_crop, raw_crop), d / f"cmp_before_{s}.png")
        imaging.save(imaging.blend_rg(ref_crop, aligned), d / f"cmp_after_{s}.png")
        imaging.save(imaging.checkerboard(ref_crop, aligned), d / f"checker_{s}.png")
        m = metrics[s]
        out["fish"].append({
            "side": s,
            "before": _url(exp_id, f"aligned/cmp_before_{s}.png"), "after": _url(exp_id, f"aligned/cmp_after_{s}.png"),
            "checker": _url(exp_id, f"aligned/checker_{s}.png"),
            "mean_before": _url(exp_id, f"aligned/mean_before_{s}.png"),
            "mean_after": _url(exp_id, f"aligned/mean_after_{s}.png"),
            "std_before": _url(exp_id, f"aligned/std_before_{s}.png"),
            "std_after": _url(exp_id, f"aligned/std_after_{s}.png"),
            "curve": {"frames": [i for i, v in enumerate(m) if v],
                      "before": [round(v["before"], 4) for v in m if v],
                      "after": [round(v["after"], 4) for v in m if v]},
            "frame_metric": m[frame_idx],
        })
    return out


# ----------------------------------------------------------------------------
# 3. 脑区模板映射
# ----------------------------------------------------------------------------
def list_templates():
    items = []
    for p in sorted(TEMPLATE_DIR.glob("*/template.json")):
        info = json.loads(p.read_text(encoding="utf-8"))
        info["id"] = p.parent.name
        items.append(info)
    return items


def template_files(tid):
    d = TEMPLATE_DIR / tid
    return d / "template.tif", d / "atlas_100.png", d / "atlas_100.txt"


def template_preview(tid):
    tif, atlas_p, centers_p = template_files(tid)
    d = TEMPLATE_DIR / tid
    tpl = cv2.imread(str(tif), 2)
    atlas = cv2.imread(str(atlas_p), 0)
    centers = np.loadtxt(centers_p)
    imaging.save(imaging.colorize(tpl, "gray"), d / "preview_template.png")
    imaging.save(imaging.atlas_color(atlas), d / "preview_atlas.png")
    imaging.save(imaging.atlas_color(atlas, tpl, 0.45), d / "preview_overlay.png")
    info = json.loads((d / "template.json").read_text(encoding="utf-8"))
    regions = []
    for r in range(1, int(atlas.max()) + 1):
        regions.append({"id": r, "name": f"{'L' if r <= 50 else 'R'}-{(r - 1) % 50 + 1:02d}",
                        "zone": info.get("zones", [""] * 100)[r - 1], "pixels": int((atlas == r).sum()),
                        "cx": round(float(centers[r - 1][0]), 1), "cy": round(float(centers[r - 1][1]), 1)})
    t = int(time.time())
    return {"id": tid, "info": info, "regions": regions,
            "template": f"/files/templates/{tid}/preview_template.png?t={t}",
            "atlas": f"/files/templates/{tid}/preview_atlas.png?t={t}",
            "overlay": f"/files/templates/{tid}/preview_overlay.png?t={t}"}


def run_atlas(ctx, exp_id, template_id="zebrafish_std_100"):
    import ants
    atp = legacy.align_tp()
    meta = ex.read_meta(exp_id)
    pp = meta["params"]["preprocess"]
    ch, cw = pp["crop_h"], pp["crop_w"]
    tif, atlas_p, _ = template_files(template_id)
    arr_fi = cv2.imread(str(tif), 2)
    fi = ants.from_numpy(arr_fi.astype(np.uint32), has_components=False)
    atlas_arr = cv2.imread(str(atlas_p), 0)
    tp = ants.from_numpy(atlas_arr, has_components=False)
    d = ex.exp_dir(exp_id) / "tp"
    d.mkdir(parents=True, exist_ok=True)
    summary = {}
    for k, s in enumerate(_sides(meta)):
        ctx.progress(k / len(_sides(meta)), f"鱼 {s.upper()} 模板配准中")
        mat = np.load(ex.exp_dir(exp_id) / "aligned" / f"{s}.npy", mmap_mode="r")
        fish_ref = np.asarray(mat[pp["ref_frame"]]).reshape(ch, cw)
        mi = ants.from_numpy(fish_ref.astype(np.uint32), has_components=False)
        prefix = str(d / s)
        atp.registration(fi, mi, prefix)                     # 模板(fixed) ← 鱼(moving)，同 align_tp_double.py
        ctx.log(f"鱼 {s.upper()}：生成 {s}Composite.h5 / {s}InverseComposite.h5")
        mi_view = ants.from_numpy(np.array(fish_ref, dtype=np.uint32), has_components=False)
        labels = ants.apply_transforms(fixed=mi_view, moving=tp, transformlist=prefix + "InverseComposite.h5",
                                       interpolator="nearestNeighbor").numpy().astype(np.uint8)
        warped_tpl = ants.apply_transforms(fixed=mi_view, moving=fi, transformlist=prefix + "InverseComposite.h5",
                                           interpolator="linear").numpy()
        np.save(d / f"labels_{s}.npy", labels)
        fish_mask = cv2.convertScaleAbs(fish_ref, alpha=255.0 / 65535.0) > pp["threshold"]
        tpl_mask_raw = np.zeros_like(fish_mask)
        tpl_mask_raw[:atlas_arr.shape[0], :atlas_arr.shape[1]] = atlas_arr[:ch, :cw] > 0
        dice_before = float(2 * (fish_mask & tpl_mask_raw).sum() / (fish_mask.sum() + tpl_mask_raw.sum()))
        dice_after = float(2 * (fish_mask & (labels > 0)).sum() / (fish_mask.sum() + (labels > 0).sum()))
        covered = int(len(np.unique(labels)) - 1)
        imaging.save(imaging.colorize(fish_ref, "gray"), d / f"fish_{s}.png")
        imaging.save(imaging.overlay_contour(fish_ref, fish_mask, tpl_mask_raw), d / f"contour_before_{s}.png")
        imaging.save(imaging.overlay_contour(fish_ref, fish_mask, labels > 0), d / f"contour_after_{s}.png")
        imaging.save(imaging.atlas_color(labels, fish_ref, 0.5), d / f"labels_{s}.png")
        imaging.save(imaging.colorize(warped_tpl, "gray"), d / f"warped_tpl_{s}.png")
        summary[s] = {"dice_before": round(dice_before, 4), "dice_after": round(dice_after, 4),
                      "regions": covered}
        ctx.log(f"鱼 {s.upper()}：脑轮廓 Dice {dice_before:.3f} → {dice_after:.3f}，覆盖脑区 {covered}/100")
    ex.mark_step(exp_id, "atlas", True, summary=summary, template=template_id)
    return summary


def atlas_result(exp_id):
    meta = ex.read_meta(exp_id)
    d = ex.exp_dir(exp_id) / "tp"
    if not meta["status"].get("atlas"):
        return {"ready": False}
    out = {"ready": True, "summary": meta["results"]["atlas"], "fish": []}
    for s in _sides(meta):
        labels = np.load(d / f"labels_{s}.npy")
        sizes = np.bincount(labels.ravel(), minlength=101)[1:101]
        out["fish"].append({"side": s, "fish": _url(exp_id, f"tp/fish_{s}.png"),
                            "before": _url(exp_id, f"tp/contour_before_{s}.png"),
                            "after": _url(exp_id, f"tp/contour_after_{s}.png"),
                            "labels": _url(exp_id, f"tp/labels_{s}.png"),
                            "warped": _url(exp_id, f"tp/warped_tpl_{s}.png"),
                            "sizes": sizes.tolist()})
    return out


# ----------------------------------------------------------------------------
# 4. 钙信号提取
# ----------------------------------------------------------------------------
def run_signal(ctx, exp_id):
    ood, sp = legacy.ood(), legacy.signal_prep()
    meta = ex.read_meta(exp_id)
    pp = meta["params"]["preprocess"]
    d = ex.exp_dir(exp_id) / "signals"
    d.mkdir(parents=True, exist_ok=True)
    summary = {}
    sides = _sides(meta)
    for k, s in enumerate(sides):
        ctx.progress(k / len(sides), f"鱼 {s.upper()} 信号提取")
        fish_mat = np.load(ex.exp_dir(exp_id) / "aligned" / f"{s}.npy")
        fish_value = np.load(ex.exp_dir(exp_id) / "tp" / f"labels_{s}.npy").astype(np.float32)
        need_move = ood.Frame_outliers(fish_mat, fish_value)            # 甩动帧
        need_move = sorted(int(i) for i in need_move)
        ctx.log(f"鱼 {s.upper()}：检测到异常(甩动)帧 {len(need_move)} 个：{need_move}")
        fish_mat_preped = np.delete(fish_mat, need_move, axis=0).astype(np.float32)
        c = sp.Sparse_tp(fish_value)
        time_range = ood.Time_range(fish_mat_preped)
        dead = bool(ood.Dead_fish(time_range))
        outliers = np.where(time_range > 500)
        fish_mat_preped.T[outliers] = 0
        ctx.log(f"鱼 {s.upper()}：异常像素 {len(outliers[0])} 个，死鱼判定 {'是' if dead else '否'}")
        raw_time = sp.Time(fish_mat_preped, c)
        res = analysis.process_traces(raw_time, need_move)
        frame_mean = fish_mat.reshape(fish_mat.shape[0], -1)[:, fish_value.ravel() > 0].mean(axis=1)
        np.savez_compressed(d / f"{s}.npz", raw=raw_time.astype(np.float32), filtered=res["filtered"],
                            interp=res["interp"], baseline=res["baseline"], dff=res["dff"].astype(np.float32),
                            need_move=np.array(need_move, int), frame_mean=frame_mean.astype(np.float32),
                            time_range=np.histogram(time_range, bins=60, range=(0, 1500))[0])
        imaging.save(imaging.colorize(res["dff"], "inferno", 0, np.percentile(res["dff"], 99.5)),
                     d / f"raster_{s}.png")
        summary[s] = {"fly_frames": need_move, "dead": dead, "outlier_pixels": int(len(outliers[0])),
                      "frames_used": int(res["dff"].shape[1]), "dff_max": round(float(res["dff"].max()), 4),
                      "dff_mean": round(float(res["dff"].mean()), 4)}
        ctx.progress((k + 1) / len(sides))
    ex.mark_step(exp_id, "signal", True, summary=summary)
    return summary


def signal_result(exp_id, side="l", regions=None):
    meta = ex.read_meta(exp_id)
    f = ex.exp_dir(exp_id) / "signals" / f"{side}.npz"
    if not f.exists():
        return {"ready": False}
    z = np.load(f)
    regions = regions or [1, 12, 35, 60, 88]
    r = lambda a: np.round(a, 4).tolist()  # noqa
    sel = [x - 1 for x in regions]
    return {"ready": True, "side": side, "regions": regions,
            "raw": r(z["raw"][sel]), "filtered": r(z["filtered"][sel]), "interp": r(z["interp"][sel]),
            "baseline": r(z["baseline"][sel]), "dff": r(z["dff"][sel]),
            "need_move": z["need_move"].tolist(), "frame_mean": r(z["frame_mean"]),
            "time_range": z["time_range"].tolist(),
            "raster": _url(exp_id, f"signals/raster_{side}.png"),
            "summary": meta["results"].get("signal", {}).get("summary", {}).get(side, {}),
            "region_mean_dff": r(z["dff"].mean(axis=1)), "region_max_dff": r(z["dff"].max(axis=1))}


# ----------------------------------------------------------------------------
# 5. 特征计算
# ----------------------------------------------------------------------------
def run_feature(ctx, exp_id):
    meta = ex.read_meta(exp_id)
    d = ex.exp_dir(exp_id) / "features"
    d.mkdir(parents=True, exist_ok=True)
    summary = {}
    for k, f in enumerate(meta["fish"]):
        s = f["side"]
        ctx.progress(k / len(meta["fish"]), f"鱼 {s.upper()} 特征计算")
        sig = np.load(ex.exp_dir(exp_id) / "signals" / f"{s}.npz")
        if meta["results"].get("signal", {}).get("summary", {}).get(s, {}).get("dead"):
            (d / f"{s}_Dead").touch()
            ctx.log(f"鱼 {s.upper()} 判定为死鱼，跳过特征计算")
            continue
        feats = analysis.compute_features(sig["dff"].astype(np.float64))
        analysis.save_features(feats, d / f"{s}.pkl")
        (d / f"{s}_peaks.json").write_text(json.dumps(feats["_peaks"]))
        summary[s] = analysis.feature_summary(feats)
        ctx.log(f"鱼 {s.upper()}：钙峰总数 {summary[s]['total_peaks']}，同步指数 {summary[s]['sync_index']}")
    ex.mark_step(exp_id, "feature", True, summary=summary)
    return summary


def feature_result(exp_id, side="l"):
    meta = ex.read_meta(exp_id)
    f = ex.exp_dir(exp_id) / "features" / f"{side}.pkl"
    if not f.exists():
        return {"ready": False}
    feats = analysis.load_features(f)
    labels = np.load(ex.exp_dir(exp_id) / "tp" / f"labels_{side}.npy")
    d = ex.exp_dir(exp_id) / "features"
    maps = {}
    for key, name in SCALAR_FEATURES:
        imaging.save(imaging.region_value_map(labels, feats[key], "turbo"), d / f"map_{side}_{key}.png")
        maps[key] = _url(exp_id, f"features/map_{side}_{key}.png")
    r = lambda a: np.round(np.asarray(a, float), 4).tolist()  # noqa
    peaks = json.loads((d / f"{side}_peaks.json").read_text()) if (d / f"{side}_peaks.json").exists() else []
    return {"ready": True, "side": side, "maps": maps,
            "table": {k: r(feats[k]) for k, _ in SCALAR_FEATURES},
            "connection_corr": r(feats["connection_corr"]), "value_corr": r(feats["value_corr"]),
            "active_regions": r(feats["active_regions"]), "peaks": peaks,
            "summary": meta["results"].get("feature", {}).get("summary", {}).get(side, {})}


def run_all(ctx, exp_id, template_id="zebrafish_std_100"):
    meta = ex.read_meta(exp_id)
    if not meta["status"].get("preprocess"):
        apply_preprocess(exp_id, {})
    steps = [("register", run_register, {}), ("atlas", run_atlas, {"template_id": template_id}),
             ("signal", run_signal, {}), ("feature", run_feature, {})]

    class Sub:
        def __init__(self, i):
            self.i = i

        def progress(self, v, msg=None):
            ctx.progress((self.i + v) / len(steps), msg)

        def log(self, t):
            ctx.log(t)

    for i, (name, fn, kw) in enumerate(steps):
        ctx.log(f"===== 步骤 {i + 1}/{len(steps)}：{dict(ex.STEPS)[name]} =====")
        fn(Sub(i), exp_id, **kw)
    return {"done": True}
