#!/usr/bin/env python
# coding=utf-8
"""示例数据生成模块。

生成与真实实验数据格式一致的演示数据，用于软件功能验证：
  1. 标准脑模板图像 template.tif 与 100 脑区分区图谱 atlas_100.png / 区域中心坐标 atlas_100.txt；
  2. 双鱼并排成像的钙荧光 TIFF 序列（每帧 img_000000xxx_Default_000.tif，1000 帧）；
  3. 已提取的脑区荧光信号矩阵库（脑区 × 时间），用于批量特征计算与模型训练。
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

TP_H, TP_W = 450, 250            # 模板 / 裁剪尺寸（与 feature_generation/data_prep.py 一致）
HALF_H, HALF_W = 512, 400        # 单鱼半幅原始图像尺寸
N_REGIONS = 100
N_FRAMES = 1000
ROT_ANGLE = 26


# ----------------------------------------------------------------------------
# 模板与图谱
# ----------------------------------------------------------------------------
def brain_mask(h=TP_H, w=TP_W):
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2.0

    def ell(x0, y0, ax, ay):
        return ((xx - x0) / ax) ** 2 + ((yy - y0) / ay) ** 2 <= 1.0

    m = ell(cx, 85, 52, 45)                                   # 端脑
    m |= ell(cx - 40, 175, 50, 62) | ell(cx + 40, 175, 50, 62)  # 视顶盖
    m |= ell(cx, 165, 38, 75)                                 # 间脑
    m |= ell(cx, 262, 82, 30)                                 # 小脑
    m |= ell(cx, 335, 52, 75)                                 # 后脑
    m |= (np.abs(xx - cx) < 22) & (yy > 330) & (yy < 415)     # 脊髓起始段
    return m


def build_template(seed=7):
    rng = np.random.default_rng(seed)
    mask = brain_mask()
    soft = gaussian_filter(mask.astype(np.float32), 2.0)
    texture = gaussian_filter(rng.random((TP_H, TP_W)).astype(np.float32), 8)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-9)
    cells = np.zeros((TP_H, TP_W), np.float32)
    idx = rng.integers(0, TP_H * TP_W, 2200)
    cells.flat[idx] = 1.0
    cells = gaussian_filter(cells, 1.4)
    cells = cells / cells.max()
    yy, xx = np.mgrid[0:TP_H, 0:TP_W]
    neuropil = np.exp(-(((xx - 125) / 70.0) ** 2 + ((yy - 180) / 40.0) ** 2))
    img = (680 + 300 * texture + 180 * cells + 120 * neuropil) * soft
    return img.astype(np.float32), mask


def build_atlas(mask, n_regions=N_REGIONS, seed=0):
    """左半脑 KMeans 分区并镜像到右半脑，生成左右对称的 100 脑区图谱。"""
    half = n_regions // 2
    ys, xs = np.where(mask & (np.arange(TP_W)[None, :] < TP_W // 2))
    km = KMeans(n_clusters=half, random_state=seed, n_init=4).fit(np.column_stack([xs, ys]))
    centers = km.cluster_centers_
    order = np.lexsort((centers[:, 0], centers[:, 1]))     # 先按 y 再按 x 排序
    remap = np.zeros(half, int)
    remap[order] = np.arange(half)
    atlas = np.zeros((TP_H, TP_W), np.uint8)
    atlas[ys, xs] = remap[km.labels_] + 1
    ys_r, xs_r = np.where(mask & (np.arange(TP_W)[None, :] >= TP_W // 2))
    mirror = atlas[ys_r, TP_W - 1 - xs_r]
    atlas[ys_r, xs_r] = np.where(mirror > 0, mirror + half, 0)
    centers_xy = np.zeros((n_regions, 2))
    for r in range(1, n_regions + 1):
        yy, xx = np.where(atlas == r)
        if len(xx):
            centers_xy[r - 1] = [xx.mean(), yy.mean()]
    return atlas, centers_xy


def region_names(n_regions=N_REGIONS):
    """依据区域中心位置给出解剖分区名称。"""
    return [f"{'L' if r <= n_regions // 2 else 'R'}-{(r - 1) % (n_regions // 2) + 1:02d}"
            for r in range(1, n_regions + 1)]


def anatomical_zone(y):
    if y < 130:
        return "端脑"
    if y < 235:
        return "视顶盖/间脑"
    if y < 292:
        return "小脑"
    return "后脑"


def write_template_files(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img, mask = build_template()
    atlas, centers = build_atlas(mask)
    cv2.imwrite(str(out_dir / "template.tif"), img.astype(np.uint16))
    cv2.imwrite(str(out_dir / "atlas_100.png"), atlas)
    np.savetxt(out_dir / "atlas_100.txt", centers, fmt="%.3f")
    info = {"name": "斑马鱼幼鱼标准脑模板（背侧视角）", "height": TP_H, "width": TP_W,
            "regions": N_REGIONS, "hemisphere": "左右对称，各 50 区",
            "zones": [anatomical_zone(c[1]) for c in centers]}
    (out_dir / "template.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


# ----------------------------------------------------------------------------
# 神经活动模拟
# ----------------------------------------------------------------------------
def calcium_kernel(length=80, tau_rise=1.5, tau_decay=12.0):
    t = np.arange(length, dtype=np.float32)
    k = (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    return k / k.max()


def simulate_dff(centers, group, rng, n_frames=N_FRAMES):
    """模拟各脑区 ΔF/F。group=1 表示药物处理（成瘾）组，放电更频繁且同步性更高。"""
    n = len(centers)
    drug = group == 1
    effect = rng.uniform(1.05, 1.85) if drug else rng.uniform(0.8, 1.35)   # 个体差异
    base_rate = rng.uniform(0.003, 0.007, n) * effect
    if drug:
        hot = (centers[:, 1] < 130) | (centers[:, 1] > 300)   # 端脑与后脑活动增强
        base_rate[hot] *= rng.uniform(1.0, 1.5)
    spikes = (rng.random((n, n_frames)) < base_rate[:, None]).astype(np.float32)
    spikes *= rng.lognormal(np.log(rng.uniform(0.2, 0.26) if not drug else rng.uniform(0.23, 0.31)), 0.25, (n, n_frames))
    # 网络同步事件：以随机种子脑区为中心、按距离衰减招募周边脑区
    ev_rate = rng.uniform(0.003, 0.007) if drug else rng.uniform(0.002, 0.005)
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    for t in np.where(rng.random(n_frames) < ev_rate)[0]:
        seed = rng.integers(n)
        p = np.exp(-dist[seed] ** 2 / (2 * (75 if drug else 45) ** 2))
        recruited = rng.random(n) < p
        spikes[recruited, t] += rng.lognormal(np.log(0.28), 0.2, recruited.sum())
    k = calcium_kernel()
    dff = np.stack([np.convolve(s, k)[:n_frames] for s in spikes])
    return np.clip(dff, 0, 0.35).astype(np.float32)


def bleaching(n_frames=N_FRAMES, strength=0.08):
    t = np.arange(n_frames, dtype=np.float32)
    return np.exp(-strength * t / n_frames)


def pick_fly_frames(rng, n_frames=N_FRAMES, k=None):
    k = k if k is not None else int(rng.integers(5, 10))
    cand = np.setdiff1d(np.arange(30, n_frames - 30), np.arange(470, 531))
    return np.sort(rng.choice(cand, k, replace=False))


# ----------------------------------------------------------------------------
# 双鱼 TIFF 序列
# ----------------------------------------------------------------------------
def deform_fish(template, atlas, rng):
    """对模板施加随机仿射 + 弹性形变，得到个体鱼脑图像与脑区标签。"""
    h, w = template.shape
    ang = rng.uniform(-4, 4)
    s = rng.uniform(0.94, 1.04)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    tex = cv2.warpAffine(template, M, (w, h), flags=cv2.INTER_LINEAR)
    lab = cv2.warpAffine(atlas, M, (w, h), flags=cv2.INTER_NEAREST)
    dx = gaussian_filter(rng.standard_normal((h, w)), 25)
    dy = gaussian_filter(rng.standard_normal((h, w)), 25)
    dx *= 5.0 / (np.abs(dx).max() + 1e-9)
    dy *= 5.0 / (np.abs(dy).max() + 1e-9)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx, my = (gx + dx).astype(np.float32), (gy + dy).astype(np.float32)
    tex = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
    lab = cv2.remap(lab, mx, my, cv2.INTER_NEAREST)
    return tex, lab


def straighten_matrix(angle, h=HALF_H, w=HALF_W):
    """与 align/fish_to_mat_double.rotate_img 完全一致的旋转矩阵（半幅图像 → 摆正画布）。"""
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return M, new_w, new_h


def _h(M):
    return np.vstack([M, [0, 0, 1]])


def render_half(tex_frame, angle, offset, jitter):
    M, cw, ch = straighten_matrix(angle)
    Minv = _h(cv2.invertAffineTransform(M))
    ox, oy = offset
    P = _h(np.array([[1, 0, cw / 2 - TP_W / 2 + ox], [0, 1, ch / 2 - TP_H / 2 + oy]], np.float64))
    jr, jx, jy = jitter
    J = _h(cv2.getRotationMatrix2D((cw / 2 + ox, ch / 2 + oy), jr, 1.0))
    J[0, 2] += jx
    J[1, 2] += jy
    A = (Minv @ J @ P)[:2]
    return cv2.warpAffine(tex_frame, A, (HALF_W, HALF_H), flags=cv2.INTER_LINEAR, borderValue=0)


def generate_experiment(raw_root, name, groups=(0, 1), seed=1, n_frames=N_FRAMES,
                        template_dir=None, progress=None):
    """生成一个双鱼实验的 TIFF 序列。groups=(左鱼组别, 右鱼组别)。"""
    rng = np.random.default_rng(seed)
    template = cv2.imread(str(Path(template_dir) / "template.tif"), 2).astype(np.float32)
    atlas = cv2.imread(str(Path(template_dir) / "atlas_100.png"), 0)
    centers = np.loadtxt(Path(template_dir) / "atlas_100.txt")
    out = Path(raw_root) / name / "Pos0"
    out.mkdir(parents=True, exist_ok=True)

    fish = []
    for side, angle in ((0, ROT_ANGLE), (1, -ROT_ANGLE)):   # 与预处理摆正角度一致（左 +26°，右 -26°）
        tex, lab = deform_fish(template, atlas, rng)
        dff = simulate_dff(centers, groups[side], rng, n_frames)
        lut = np.vstack([np.zeros((1, n_frames), np.float32), dff])
        walk = np.cumsum(rng.normal(0, 0.25, (n_frames, 3)), axis=0)
        walk = gaussian_filter(walk, (6, 0)) * np.array([0.3, 1.0, 1.0])
        walk -= walk[min(500, n_frames - 1)]
        fish.append(dict(tex=tex, lab=lab, lut=lut, angle=angle, walk=walk,
                         offset=rng.uniform(-12, 12, 2), fly=pick_fly_frames(rng, n_frames)))
    bleach = bleaching(n_frames)

    for t in range(n_frames):
        halves = []
        for f in fish:
            act = cv2.GaussianBlur(f["lut"][f["lab"], t], (0, 0), 2.5)
            frame_tex = f["tex"] * bleach[t] * (1.0 + act)
            jitter = tuple(f["walk"][t])
            if t in f["fly"]:
                jitter = (rng.uniform(-6, 6), rng.uniform(-25, 25), rng.uniform(-25, 25))
                frame_tex = cv2.GaussianBlur(frame_tex, (0, 0), 4) * 0.55
            halves.append(render_half(frame_tex, f["angle"], f["offset"], jitter))
        img = np.hstack(halves) + 180.0
        img += rng.normal(0, 1, img.shape) * np.sqrt(img) * 0.45
        cv2.imwrite(str(out / f"img_{t:09d}_Default_000.tif"), np.clip(img, 0, 65535).astype(np.uint16))
        if progress and t % 20 == 0:
            progress(t / n_frames)

    meta = {"experiment": name, "frames": n_frames, "frame_rate": "2 Hz", "layout": "双鱼并排（左/右）",
            "fish": [{"side": "l", "group": int(groups[0]), "fly_frames": fish[0]["fly"].tolist()},
                     {"side": "r", "group": int(groups[1]), "fly_frames": fish[1]["fly"].tolist()}],
            "image_size": [HALF_H, HALF_W * 2], "bit_depth": 16}
    (out.parent / "acquisition.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out.parent


# ----------------------------------------------------------------------------
# 脑区荧光信号矩阵库
# ----------------------------------------------------------------------------
def generate_signal_library(out_dir, template_dir, n_per_group=80, seed=11, n_frames=N_FRAMES):
    rng = np.random.default_rng(seed)
    centers = np.loadtxt(Path(template_dir) / "atlas_100.txt")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    bleach = bleaching(n_frames)
    for g in (0, 1):
        for i in range(n_per_group):
            dff = simulate_dff(centers, g, rng, n_frames)
            base = rng.uniform(1000, 1500, (len(centers), 1))
            trace = base * bleach[None, :] * (1 + dff)
            trace += rng.normal(0, 3.0, trace.shape)
            fly = pick_fly_frames(rng, n_frames)
            trace[:, fly] *= rng.uniform(0.5, 0.7, (1, len(fly)))
            fid = f"{'DRG' if g else 'CTL'}-{i + 1:03d}"
            np.save(out_dir / f"{fid}.npy", trace.astype(np.float32))
            rows.append(f"{fid},{g},{'药物处理组' if g else '对照组'}")
    (out_dir / "labels.csv").write_text("fish_id,label,group\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return out_dir


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import DATA_DIR, TEMPLATE_DIR

    ap = argparse.ArgumentParser(description="生成演示数据")
    ap.add_argument("--experiments", type=int, default=2)
    ap.add_argument("--library", type=int, default=80, help="信号库每组样本数")
    args = ap.parse_args()
    tdir = write_template_files(DATA_DIR / "demo" / "template")
    write_template_files(TEMPLATE_DIR / "zebrafish_std_100")
    plans = [("EXP20260315_A", (0, 1)), ("EXP20260315_B", (1, 0)), ("EXP20260322_C", (0, 0))]
    for k, (name, groups) in enumerate(plans[:args.experiments]):
        print("generating", name)
        generate_experiment(DATA_DIR / "demo" / "raw", name, groups, seed=100 + k, template_dir=tdir)
    generate_signal_library(DATA_DIR / "demo" / "signal_library", tdir, args.library)
    print("done")
