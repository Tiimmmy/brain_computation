#!/usr/bin/env python
# coding=utf-8
"""深度学习建模：基于 Models/FC 中 BrainNet（逐特征单元子网络 Cell + 深层全连接分类头），
K 折交叉验证训练、实时曲线记录、混淆矩阵与 ROC 评估、模型保存、梯度显著性脑区分析与预测。"""

import copy
import json
import shutil
import time

import numpy as np

from config import MODEL_DIR, DATASET_DIR, SCALAR_FEATURES
from services import legacy, library


def _torch():
    import torch
    return torch


def device_info():
    try:
        torch = _torch()
        cuda = torch.cuda.is_available()
        return {"torch": torch.__version__, "cuda": cuda,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []}
    except Exception as e:
        return {"torch": None, "cuda": False, "gpus": [], "error": str(e)}


def list_models():
    items = []
    for p in sorted(MODEL_DIR.glob("*/meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        m.pop("history", None)
        items.append(m)
    return sorted(items, key=lambda x: x["created"], reverse=True)


def get_model(mid):
    return json.loads((MODEL_DIR / mid / "meta.json").read_text(encoding="utf-8"))


def delete_model(mid):
    shutil.rmtree(MODEL_DIR / mid, ignore_errors=True)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score, roc_curve
    if len(set(y)) < 2:
        return None, [], []
    fpr, tpr, _ = roc_curve(y, p)
    return float(roc_auc_score(y, p)), np.round(fpr, 4).tolist(), np.round(tpr, 4).tolist()


def run_train(ctx, dataset_id, name, k_folds=5, epochs=100, lr=0.01, momentum=0.9, weight_decay=1e-5,
              optimizer="SGD", device="cuda", seed=42, description=""):
    torch = _torch()
    from torch import nn, optim
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from sklearn.model_selection import KFold
    FC, TR = legacy.fc_params(), legacy.fc_train()

    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = json.loads((DATASET_DIR / dataset_id / "meta.json").read_text(encoding="utf-8"))
    final_dataset = TR.data_loader(str(DATASET_DIR / dataset_id / "features.npy"),
                                   str(DATASET_DIR / dataset_id / "label.npy"))       # 原始数据加载函数
    N, num_features, in_dim = ds["shape"]
    dev = torch.device("cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu")
    net = FC.BrainNet(in_dim=in_dim, num_features=num_features).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    ctx.log(f"数据集 {dataset_id}：样本 {N}，特征单元 {num_features}，通道 {in_dim}；设备 {dev}；参数量 {n_params:,}")
    criterion = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    history = []
    fold_results = []
    best_overall = {"acc": -1}
    all_prob, all_true = [], []
    t0 = time.time()
    total_steps = k_folds * epochs
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(N))):
        train_loader = DataLoader(final_dataset, batch_size=len(train_ids), sampler=SubsetRandomSampler(train_ids))
        test_loader = DataLoader(final_dataset, batch_size=len(val_ids), sampler=SubsetRandomSampler(val_ids))
        net.apply(TR.reset_weights)                                                   # 原始权重重置函数
        if optimizer == "Adam":
            opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        hist = {"fold": fold + 1, "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best = {"acc": -1}
        for epoch in range(1, epochs + 1):
            net.train()
            tl, tc, tn = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                out = net(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                tl += loss.item()
                tc += out.argmax(1).eq(yb).sum().item()
                tn += yb.size(0)
            net.eval()
            vl, vc, vn, probs, trues, preds = 0.0, 0, 0, [], [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    out = net(xb)
                    vl += criterion(out, yb).item()
                    pr = torch.softmax(out, 1)[:, 1]
                    vc += out.argmax(1).eq(yb).sum().item()
                    vn += yb.size(0)
                    probs += pr.cpu().tolist()
                    trues += yb.cpu().tolist()
                    preds += out.argmax(1).cpu().tolist()
            tr_acc, te_acc = 100.0 * tc / tn, 100.0 * vc / vn
            hist["train_loss"].append(round(tl / len(train_loader), 4))
            hist["train_acc"].append(round(tr_acc, 2))
            hist["test_loss"].append(round(vl / len(test_loader), 4))
            hist["test_acc"].append(round(te_acc, 2))
            if te_acc > best["acc"]:
                best = {"acc": te_acc, "epoch": epoch, "probs": probs, "trues": trues, "preds": preds,
                        "state": copy.deepcopy(net.state_dict())}
            step = fold * epochs + epoch
            if epoch % 5 == 0 or epoch == epochs:
                ctx.progress(step / total_steps, f"第 {fold + 1}/{k_folds} 折 · Epoch {epoch}/{epochs} · "
                                                 f"训练准确率 {tr_acc:.1f}% · 验证准确率 {te_acc:.1f}%")
                ctx.extra(live=history + [hist])
            if epoch % 25 == 0 or epoch == epochs:
                ctx.log(f"Fold {fold + 1} Epoch {epoch}: Train Loss {hist['train_loss'][-1]:.3f} | "
                        f"Acc {tr_acc:.2f} ; Test Loss {hist['test_loss'][-1]:.3f} | Acc {te_acc:.2f}")
        history.append(hist)
        pr_t, lab_t = torch.tensor(best["preds"]), torch.tensor(best["trues"])
        tp_, fp_, tn_, fn_ = [int(v[0]) for v in TR.Confusion([pr_t], [lab_t])]       # 原始混淆矩阵函数
        fold_results.append({"fold": fold + 1, "best_epoch": best["epoch"], "acc": round(best["acc"], 2),
                             "tp": tp_, "fp": fp_, "tn": tn_, "fn": fn_, "n_val": len(val_ids)})
        all_prob += best["probs"]
        all_true += best["trues"]
        ctx.log(f"第 {fold + 1} 折最佳验证准确率 {best['acc']:.2f}%（Epoch {best['epoch']}）")
        if best["acc"] > best_overall["acc"]:
            best_overall = best
    TP = sum(f["tp"] for f in fold_results)
    FP = sum(f["fp"] for f in fold_results)
    TN = sum(f["tn"] for f in fold_results)
    FN = sum(f["fn"] for f in fold_results)
    acc = float(np.mean([f["acc"] for f in fold_results]))
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc, fpr, tpr = _auc(all_true, all_prob)
    mid = time.strftime("M%Y%m%d%H%M%S")
    d = MODEL_DIR / mid
    d.mkdir(parents=True, exist_ok=True)
    state = recalibrate_bn(net, best_overall["state"], final_dataset, dev)
    ctx.log("已完成 BatchNorm 统计量重校准（关闭 Dropout 后以全量样本重估均值/方差）")
    torch.save(state, d / "model.pt")
    importance = saliency(net, state, final_dataset, dev, ds)
    meta = {"id": mid, "name": name, "description": description, "dataset": dataset_id,
            "dataset_name": ds["name"], "arch": "BrainNet", "in_dim": in_dim, "num_features": num_features,
            "params": n_params, "hyper": {"k_folds": k_folds, "epochs": epochs, "lr": lr, "momentum": momentum,
                                          "weight_decay": weight_decay, "optimizer": optimizer,
                                          "device": str(dev), "seed": seed},
            "metrics": {"accuracy": round(acc, 2), "precision": round(precision * 100, 2),
                        "recall": round(recall * 100, 2), "f1": round(f1 * 100, 2),
                        "auc": round(auc, 4) if auc is not None else None,
                        "confusion": {"tp": TP, "fp": FP, "tn": TN, "fn": FN}},
            "roc": {"fpr": fpr, "tpr": tpr}, "folds": fold_results, "history": history,
            "importance": importance, "seconds": round(time.time() - t0, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_keys": ds["feature_keys"], "channels": ds["channels"], "band_width": ds["band_width"],
            "norm_ids": ds["ids"]}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    ctx.log(f"交叉验证平均准确率 {acc:.2f}%，精确率 {precision * 100:.2f}%，召回率 {recall * 100:.2f}%，"
            f"F1 {f1 * 100:.2f}%，AUC {auc if auc is None else round(auc, 4)}")
    return {"model_id": mid, "accuracy": round(acc, 2)}


def recalibrate_bn(net, state, dataset, dev):
    """消除 Dropout 与 BatchNorm 串联造成的方差偏移：关闭 Dropout，以全量样本累积重估 BN 统计量。"""
    torch = _torch()
    net.load_state_dict(state)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    with torch.no_grad():
        net(dataset.tensors[0].to(dev))
    net.eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = 0.1
    return copy.deepcopy(net.state_dict())


def saliency(net, state, dataset, dev, ds):
    """输入梯度显著性：|∂P(药物组)/∂x| 在样本与通道维平均，得到 (特征类型 × 脑区) 重要性。"""
    torch = _torch()
    net.load_state_dict(state)
    net.eval()
    X = dataset.tensors[0].to(dev).clone().requires_grad_(True)
    prob = torch.softmax(net(X), 1)[:, 1].sum()
    prob.backward()
    g = X.grad.abs().mean(dim=(0, 2)).detach().cpu().numpy()
    F = len(ds["feature_keys"])
    imp = g.reshape(F, -1)
    imp = imp / (imp.max() + 1e-12)
    return {"matrix": np.round(imp, 4).tolist(), "by_feature": np.round(imp.mean(1), 4).tolist(),
            "by_region": np.round(imp.mean(0), 4).tolist()}


def importance_map(mid, template_id="zebrafish_std_100"):
    import cv2
    from config import CACHE_DIR
    from services import imaging
    from services.pipeline import template_files
    meta = get_model(mid)
    atlas = cv2.imread(str(template_files(template_id)[1]), 0)
    vals = np.array(meta["importance"]["by_region"])
    path = CACHE_DIR / f"importance_{mid}.png"
    imaging.save(imaging.region_value_map(atlas, vals, "reds", 0, float(vals.max()), bg=(255, 255, 255)), path,
                 scale=1.0)
    return f"/files/cache/{path.name}?t={int(time.time())}"


def _build_inputs(meta, ids):
    """按照模型训练数据集的构建方式（同一归一化参考样本）为新样本生成输入张量。"""
    from services import datasets
    from services.pipeline import template_files
    ref_ids = meta["norm_ids"]
    all_ids = list(ref_ids) + [i for i in ids if i not in ref_ids]
    X, _ = library.feature_matrix(all_ids, meta["feature_keys"])
    N, F, R = X.shape
    flat = X.reshape(N, F * R)
    need_k = any(c.startswith("kernel") for c in meta["channels"])
    flat_s = datasets.kernel_smooth(X, template_files("zebrafish_std_100")[2], meta["band_width"]).reshape(N, F * R) \
        if need_k else None
    nref = len(ref_ids)
    chans = []
    for c in meta["channels"]:
        src = flat_s if c.startswith("kernel") else flat
        kind = c.replace("kernel_", "")
        ref = src[:nref]
        if kind == "minmax":
            mx, mn = ref.max(0), ref.min(0)
            src = (src - mn) / np.where(mx - mn == 0, 1, mx - mn)
        elif kind == "zscore":
            mu, sd = ref.mean(0), ref.std(0)
            src = (src - mu) / np.where(sd == 0, 1, sd)
        chans.append(src)
    data = np.stack(chans, -1).astype(np.float32)
    index = {fid: k for k, fid in enumerate(all_ids)}
    return data[[index[i] for i in ids]]


def predict(mid, ids):
    torch = _torch()
    FC = legacy.fc_params()
    meta = get_model(mid)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = FC.BrainNet(in_dim=meta["in_dim"], num_features=meta["num_features"]).to(dev)
    net.load_state_dict(torch.load(MODEL_DIR / mid / "model.pt", map_location=dev, weights_only=True))
    net.eval()
    data = _build_inputs(meta, ids)
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(data).to(dev)), 1)[:, 1].cpu().numpy()
    results = []
    for fid, p in zip(ids, prob):
        e = library.get_entry(fid)
        pred = int(p >= 0.5)
        results.append({"fish_id": fid, "true": e["label"], "true_name": e["group"], "prob": round(float(p), 4),
                        "pred": pred, "pred_name": library.GROUP_NAME[pred], "correct": pred == e["label"],
                        "in_training": fid in meta["norm_ids"], "source": e["source"]})
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": mid, "count": len(results), "results": results}
    hist_p = MODEL_DIR / mid / "predictions.json"
    hist = json.loads(hist_p.read_text(encoding="utf-8")) if hist_p.exists() else []
    hist.insert(0, rec)
    hist_p.write_text(json.dumps(hist[:50], ensure_ascii=False), encoding="utf-8")
    return rec


def feature_names(keys):
    return [dict(SCALAR_FEATURES)[k] for k in keys]
