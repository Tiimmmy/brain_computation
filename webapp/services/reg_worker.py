#!/usr/bin/env python
# coding=utf-8
"""逐帧运动校正配准的子进程工作函数（ProcessPoolExecutor 中执行）。

流程与 align/align_double.py + align/fish_to_mat_double.py 一致：
  拆分双鱼 → 旋转摆正 → 以参考帧为 fixed 做 ANTs 配准（输出 Composite.h5）
  → apply_transforms 得到校正后图像 → 以参考帧中心裁剪 → 展平写入对齐矩阵。
"""

import os

import numpy as np

_STATE = {}


def init_worker(ref_path, layout, angles, crop, transform, resample, threads=1):
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import cv2
    import ants
    from services import legacy
    atp = legacy.align_tp()
    ftm = legacy.fish_to_mat()
    _STATE.update(cv2=cv2, ants=ants, atp=atp, ftm=ftm, layout=layout, angles=angles,
                  crop=crop, transform=transform, resample=tuple(resample))
    refs = split_rotate(cv2.imread(ref_path, 2))
    _STATE["ref_arr"] = refs
    _STATE["ref_img"] = [ants.from_numpy(r.astype(np.uint32), has_components=False) for r in refs]
    _STATE["centers"] = [ftm.crop_center(r) for r in refs]


def split_rotate(img):
    atp = _STATE["atp"]
    if _STATE["layout"] == "single":
        return [atp.rotate_img(img, _STATE["angles"][0]) if _STATE["angles"][0] else img]
    h, w = img.shape[:2]
    left, right = img[0:h, 0:int(w / 2)], img[0:h, int(w / 2):int(w)]
    return [atp.rotate_img(left, _STATE["angles"][0]), atp.rotate_img(right, _STATE["angles"][1])]


def register(fi, mi, prefix):
    ants, atp = _STATE["ants"], _STATE["atp"]
    if _STATE["transform"] == "ElasticSyN" and _STATE["resample"] == (100, 80):
        atp.registration(fi, mi, prefix)            # 原始实现
        return
    fixed = ants.resample_image(fi, _STATE["resample"], 1, 0)
    moving = ants.resample_image(mi, _STATE["resample"], 1, 0)
    ants.registration(fixed=fixed, moving=moving, type_of_transform=_STATE["transform"], outprefix=prefix,
                      aff_random_sampling_rate=0.1, random_seed=1, smoothing_sigmas=(4, 3, 2, 1),
                      aff_shrink_factors=(8, 4, 2, 1), write_composite_transform=True, verbose=False)


def _corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def process_frame(args):
    frame_idx, path, prefixes = args
    cv2, ants, ftm = _STATE["cv2"], _STATE["ants"], _STATE["ftm"]
    ch, cw = _STATE["crop"]
    img = cv2.imread(path, 2)
    out, metrics = [], []
    for k, arr in enumerate(split_rotate(img)):
        fi = _STATE["ref_img"][k]
        mi = ants.from_numpy(arr.astype(np.uint32), has_components=False)
        center = _STATE["centers"][k]
        ref_crop = ftm.crop_img(_STATE["ref_arr"][k].astype(np.uint16), center, ch, cw)
        raw_crop = ftm.crop_img(arr.astype(np.uint16), center, ch, cw)
        try:
            register(fi, mi, prefixes[k])
            warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=prefixes[k] + "Composite.h5",
                                           interpolator="gaussian")
            aligned = ftm.crop_img(warped.numpy().astype(np.uint16), center, ch, cw)
            ok = True
        except Exception:
            aligned = raw_crop
            ok = False
        out.append(aligned.flatten())
        metrics.append({"before": _corr(ref_crop, raw_crop), "after": _corr(ref_crop, aligned), "ok": ok,
                        "mean": float(raw_crop.mean())})
    return frame_idx, out, metrics
