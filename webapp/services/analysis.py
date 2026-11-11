#!/usr/bin/env python
# coding=utf-8
"""钙信号预处理与神经活动特征计算（调用 feature_generation 下的原始算法）。

信号预处理链：异常帧(甩动帧)识别 → 脑区平均荧光 → Butterworth 低通滤波
→ 异常帧线性插值 → IModPoly 多项式基线估计 → ΔF/F 归一化。
特征：钙峰数量、峰面积、持续时间、占空比、峰高、动作电位强度、峰间间隔、
      峰距、放电同步连接矩阵、峰值相关矩阵、逐帧活跃脑区数。
"""

import pickle
import warnings
from collections import Counter

import numpy as np

from services import legacy

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------------------------------------------------------------------
# 信号预处理
# ----------------------------------------------------------------------------
def region_fly_frames(traces, min_regions=None):
    """脑区信号矩阵（脑区×时间）的甩动帧识别：复用 ood.Fly_frame 的分位数围栏判据。"""
    ood = legacy.ood()
    n_regions = traces.shape[0]
    min_regions = min_regions or max(5, n_regions // 3)
    fly = ood.Fly_frame(traces.T)                  # 按列(脑区)计算 80/20 分位围栏
    counts = Counter(fly[0])
    return sorted(int(k) for k, v in counts.items() if v > min_regions)


def process_traces(time_mat, need_move):
    """time_mat: 去除异常帧后的脑区平均荧光（脑区×时间）。返回各阶段结果，与 signal_prep.Norm_time 等价。"""
    sp = legacy.signal_prep()
    filtered = sp.Filter(time_mat)
    interp = sp.Interplot(filtered, np.array(need_move, dtype=int)) if len(need_move) else filtered.copy()
    interp[np.isnan(interp)] = 0
    interp[np.isinf(interp)] = 0
    trimmed = interp.T[10:-10].T
    baseline = sp.Baseline(trimmed)
    dff = (trimmed.T[10:-10].T - baseline.T[10:-10].T) / baseline.T[10:-10].T
    dff[~np.isfinite(dff)] = 0
    return {"filtered": filtered, "interp": interp, "trimmed": trimmed, "baseline": baseline, "dff": dff}


def process_region_matrix(raw):
    """直接导入的脑区荧光矩阵（脑区×时间）处理流程。"""
    need_move = region_fly_frames(raw)
    kept = np.delete(raw, need_move, axis=1)
    res = process_traces(kept, need_move)
    res.update(raw=raw, need_move=need_move, kept=kept)
    return res


# ----------------------------------------------------------------------------
# 特征计算
# ----------------------------------------------------------------------------
def _objarr(seq):
    o = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        o[i] = np.asarray(v, dtype=np.float64)
    return o


def _areas_safe(fe, wl, wr, wp, vl, vr, vp):
    try:
        return fe.Areas(wl, wr, wp, vl, vr, vp)
    except Exception:
        n = len(wp)
        s, m = np.zeros(n), np.zeros(n)
        for i in range(n):
            try:
                a, b = fe.Areas(wl[i:i + 1], wr[i:i + 1], wp[i:i + 1], vl[i:i + 1], vr[i:i + 1], vp[i:i + 1])
                s[i], m[i] = a[0], b[0]
            except Exception:
                pass
        return s, m


def compute_features(s):
    """s: ΔF/F 矩阵（100 脑区 × 960 帧）。返回与 key.py features_dict 相同结构的特征字典。"""
    fe = legacy.feature_extraction()
    tp = fe.Find_turning_points(s)
    whole_left, whole_right, whole_peaks, value_left, value_right, value_peaks = [_objarr(x) for x in tp]

    num_peaks = fe.Peak_num(s)
    areas_sum, areas_mean = _areas_safe(fe, whole_left, whole_right, whole_peaks, value_left, value_right,
                                        value_peaks)
    spans = fe.Span(whole_left, whole_right)
    duration_mean, dr = fe.Duration(spans)
    duration_mean[np.isnan(duration_mean)] = 0
    dr[np.isnan(dr)] = 0
    height_max, height_mean = fe.Height(value_peaks)
    ap_mean = fe.AP(spans, value_peaks)
    ap_mean[~np.isfinite(ap_mean)] = 0
    interval_mean, separation_mean = fe.Interval(whole_left, whole_right, whole_peaks)
    where_peaks = fe.Neural_connection(s)
    peak_indexs = sorted(np.array(list(set(where_peaks))))
    connection_corr, corr_value, active_regions = fe.Connection_corr(peak_indexs, whole_peaks, s)
    connection_corr[np.isnan(connection_corr)] = 0
    corr_value[np.isnan(corr_value)] = 0
    peaks_pos = [np.asarray(p, dtype=int).tolist() for p in whole_peaks]
    return {
        "num_peaks": np.asarray(num_peaks, dtype=float), "areas_sum": areas_sum, "areas_mean": areas_mean,
        "duration_mean": duration_mean, "dr": dr, "height_max": height_max, "height_mean": height_mean,
        "ap_mean": ap_mean, "interval_mean": interval_mean, "separation_mean": separation_mean,
        "connection_corr": connection_corr, "value_corr": corr_value, "active_regions": active_regions,
        "_peaks": peaks_pos,
    }


def save_features(features, path):
    with open(path, "wb") as f:
        pickle.dump({k: v for k, v in features.items() if not k.startswith("_")}, f)


def load_features(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def feature_summary(features):
    from config import SCALAR_FEATURES
    out = {}
    for key, name in SCALAR_FEATURES:
        v = np.asarray(features[key], dtype=float)
        out[key] = {"name": name, "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                    "max": round(float(v.max()), 4), "min": round(float(v.min()), 4)}
    cc = features["connection_corr"]
    iu = np.triu_indices_from(cc, 1)
    out["sync_index"] = round(float(np.mean(cc[iu])), 4)
    out["total_peaks"] = int(np.sum(features["num_peaks"]))
    out["active_peak_frames"] = int(np.sum(np.asarray(features["active_regions"]) > 0))
    return out


def library_worker(args):
    """信号库批量分析子进程函数：读取脑区信号矩阵 → 预处理 → 特征 → 保存。"""
    src, out_dir = args
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(src).astype(np.float64)
    res = process_region_matrix(raw)
    feats = compute_features(res["dff"])
    np.savez_compressed(out_dir / "signal.npz", raw=raw.astype(np.float32), dff=res["dff"].astype(np.float32),
                        baseline=res["baseline"].astype(np.float32), interp=res["interp"].astype(np.float32),
                        need_move=np.array(res["need_move"], dtype=int))
    save_features(feats, out_dir / "features.pkl")
    return str(out_dir), feature_summary(feats), res["need_move"]
