#!/usr/bin/env python
# coding=utf-8
"""斑马鱼全脑钙成像神经计算分析系统 —— Web 服务入口（Flask）。

启动：  python app.py --host 0.0.0.0 --port 8765
"""

import argparse
import csv
import io
import json
import os
import platform
import sys
import time
from functools import wraps
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session  # noqa: E402
from werkzeug.security import check_password_hash, generate_password_hash  # noqa: E402

import config  # noqa: E402
from config import (SCALAR_FEATURES, SOFTWARE_NAME, VERSION, WORKSPACE, load_settings,  # noqa: E402
                    save_settings)
from services import experiments as ex  # noqa: E402
from services import datasets, legacy, library, modeling, pipeline  # noqa: E402
from services.tasks import manager  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.secret_key = config.SECRET_KEY
app.config["JSON_AS_ASCII"] = False
app.json.sort_keys = False


# ----------------------------------------------------------------------------
# 用户与鉴权
# ----------------------------------------------------------------------------
def load_users():
    if not config.USERS_FILE.exists():
        users = {"admin": {"password": generate_password_hash("admin123"), "name": "系统管理员", "role": "管理员"},
                 "analyst": {"password": generate_password_hash("analyst123"), "name": "数据分析员", "role": "分析员"}}
        config.USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(config.USERS_FILE.read_text(encoding="utf-8"))


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "未登录或登录已过期"}), 401
            return redirect("/login")
        return fn(*a, **kw)
    return wrapper


def ok(data=None, **kw):
    out = {"ok": True, "data": data}
    out.update(kw)
    return jsonify(out)


@app.errorhandler(Exception)
def handle_error(e):
    import traceback
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"ok": False, "error": e.description}), e.code
    traceback.print_exc()
    return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/login")
def login_page():
    return send_from_directory(STATIC, "login.html")


@app.route("/")
@login_required
def index():
    return send_from_directory(STATIC, "index.html")


@app.post("/api/login")
def api_login():
    body = request.get_json(force=True)
    users = load_users()
    u = users.get(body.get("username", ""))
    if not u or not check_password_hash(u["password"], body.get("password", "")):
        return jsonify({"ok": False, "error": "用户名或密码错误"}), 401
    session["user"] = body["username"]
    session["login_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return ok({"username": body["username"], "name": u["name"]})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return ok()


@app.get("/api/me")
@login_required
def api_me():
    u = load_users()[session["user"]]
    return ok({"username": session["user"], "name": u["name"], "role": u["role"],
               "login_time": session.get("login_time"), "software": SOFTWARE_NAME, "version": VERSION})


@app.post("/api/password")
@login_required
def api_password():
    body = request.get_json(force=True)
    users = load_users()
    u = users[session["user"]]
    if not check_password_hash(u["password"], body.get("old", "")):
        raise ValueError("原密码错误")
    if len(body.get("new", "")) < 6:
        raise ValueError("新密码长度至少 6 位")
    u["password"] = generate_password_hash(body["new"])
    config.USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    return ok()


@app.get("/files/<path:sub>")
@login_required
def files(sub):
    return send_from_directory(WORKSPACE, sub, max_age=0)


# ----------------------------------------------------------------------------
# 首页 / 系统
# ----------------------------------------------------------------------------
@app.get("/api/dashboard")
@login_required
def api_dashboard():
    exps = ex.list_experiments()
    lib = library.list_entries()
    models = modeling.list_models()
    tasks = manager.list(10)
    steps = {k: sum(1 for e in exps if e["status"].get(k)) for k, _ in ex.STEPS}
    best = max(models, key=lambda m: m["metrics"]["accuracy"]) if models else None
    return ok({"experiments": len(exps), "fish": sum(len(e["fish"]) for e in exps), "library": len(lib),
               "library_groups": {"0": sum(1 for e in lib if e["label"] == 0),
                                  "1": sum(1 for e in lib if e["label"] == 1)},
               "datasets": len(datasets.list_datasets()), "models": len(models),
               "best_model": best, "tasks": tasks, "steps": steps, "step_names": dict(ex.STEPS),
               "recent_experiments": exps[:5],
               "running": sum(1 for t in manager.list() if t["status"] == "running")})


@app.get("/api/system")
@login_required
def api_system():
    import numpy, scipy, sklearn, cv2, ants, flask  # noqa
    stats = legacy.source_statistics()
    by_lang = {}
    for s in stats:
        by_lang[s["lang"]] = by_lang.get(s["lang"], 0) + s["lines"]
    du = sum(f.stat().st_size for f in WORKSPACE.rglob("*") if f.is_file()) / 1024 ** 3
    return ok({"software": SOFTWARE_NAME, "version": VERSION, "python": platform.python_version(),
               "platform": platform.platform(), "cpu": os.cpu_count(),
               "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                            "scikit-learn": sklearn.__version__, "opencv": cv2.__version__,
                            "antspyx": getattr(ants, "__version__", "-"), "flask": flask.__version__},
               "device": modeling.device_info(), "workspace": str(WORKSPACE),
               "workspace_gb": round(du, 2), "source": stats, "source_by_lang": by_lang,
               "source_total": sum(s["lines"] for s in stats), "settings": load_settings()})


@app.post("/api/settings")
@login_required
def api_settings():
    s = load_settings()
    for k, v in request.get_json(force=True).items():
        if k in s and isinstance(v, dict):
            s[k].update(v)
    save_settings(s)
    return ok(s)


@app.get("/api/browse")
@login_required
def api_browse():
    path = Path(request.args.get("path") or config.DATA_DIR)
    if not path.exists():
        raise FileNotFoundError(f"路径不存在：{path}")
    items = []
    for p in sorted(path.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            n = sum(1 for f in p.iterdir() if f.suffix.lower() in (".tif", ".tiff", ".npy"))
            items.append({"name": p.name, "path": str(p), "dir": True, "files": n})
    return ok({"path": str(path), "parent": str(path.parent), "items": items})


# ----------------------------------------------------------------------------
# 实验数据管理
# ----------------------------------------------------------------------------
@app.get("/api/experiments")
@login_required
def api_experiments():
    return ok(ex.list_experiments())


@app.post("/api/experiments/scan")
@login_required
def api_scan():
    return ok(ex.scan_raw(request.get_json(force=True)["path"]))


@app.post("/api/experiments")
@login_required
def api_create_experiment():
    b = request.get_json(force=True)
    meta = ex.create_experiment(b["name"], b["seq_dir"], b.get("layout", "double"),
                                tuple(int(g) for g in b.get("groups", [0, 1])), b.get("description", ""),
                                session["user"], b.get("species", "斑马鱼幼鱼 (7 dpf)"),
                                b.get("indicator", "GCaMP6s"), float(b.get("frame_rate", 2.0)))
    return ok(meta)


@app.route("/api/experiments/<exp_id>", methods=["GET", "PUT", "DELETE"])
@login_required
def api_experiment(exp_id):
    if request.method == "DELETE":
        ex.delete_experiment(exp_id)
        return ok()
    if request.method == "PUT":
        return ok(ex.update_experiment(exp_id, request.get_json(force=True)))
    meta = ex.read_meta(exp_id)
    meta["tasks"] = [t for t in manager.list() if t.get("target") == exp_id][:10]
    return ok(meta)


@app.get("/api/experiments/<exp_id>/frame")
@login_required
def api_frame(exp_id):
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    return ok(ex.frame_preview(exp_id, request.args.get("idx", 0, type=int), request.args.get("cmap", "gray"),
                               vmin, vmax))


@app.get("/api/experiments/<exp_id>/profile")
@login_required
def api_profile(exp_id):
    return ok(ex.intensity_profile(exp_id))


@app.post("/api/experiments/<exp_id>/preprocess/preview")
@login_required
def api_pre_preview(exp_id):
    b = request.get_json(force=True)
    return ok(pipeline.preprocess_preview(exp_id, b.get("frame"), b.get("params")))


@app.post("/api/experiments/<exp_id>/preprocess/apply")
@login_required
def api_pre_apply(exp_id):
    return ok(pipeline.apply_preprocess(exp_id, request.get_json(force=True).get("params", {})))


def _submit(ttype, title, func, target, **kw):
    if target and manager.running_for(target):
        raise RuntimeError("该对象已有正在运行的任务，请等待完成")
    return ok({"task_id": manager.submit(ttype, title, func, target=target, **kw)})


@app.route("/api/experiments/<exp_id>/register", methods=["GET", "POST"])
@login_required
def api_register(exp_id):
    if request.method == "POST":
        b = request.get_json(force=True)
        return _submit("register", f"{exp_id} 运动校正配准", pipeline.run_register, exp_id, exp_id=exp_id,
                       transform=b.get("transform"), workers=b.get("workers"))
    return ok(pipeline.register_result(exp_id, request.args.get("frame", type=int)))


@app.route("/api/experiments/<exp_id>/atlas", methods=["GET", "POST"])
@login_required
def api_atlas(exp_id):
    if request.method == "POST":
        b = request.get_json(force=True)
        return _submit("atlas", f"{exp_id} 脑区模板映射", pipeline.run_atlas, exp_id, exp_id=exp_id,
                       template_id=b.get("template", "zebrafish_std_100"))
    return ok(pipeline.atlas_result(exp_id))


@app.route("/api/experiments/<exp_id>/signal", methods=["GET", "POST"])
@login_required
def api_signal(exp_id):
    if request.method == "POST":
        return _submit("signal", f"{exp_id} 钙信号提取", pipeline.run_signal, exp_id, exp_id=exp_id)
    regions = [int(x) for x in request.args.get("regions", "").split(",") if x.strip()]
    return ok(pipeline.signal_result(exp_id, request.args.get("side", "l"), regions or None))


@app.route("/api/experiments/<exp_id>/feature", methods=["GET", "POST"])
@login_required
def api_feature(exp_id):
    if request.method == "POST":
        return _submit("feature", f"{exp_id} 神经活动特征计算", pipeline.run_feature, exp_id, exp_id=exp_id)
    return ok(pipeline.feature_result(exp_id, request.args.get("side", "l")))


@app.post("/api/experiments/<exp_id>/run_all")
@login_required
def api_run_all(exp_id):
    return _submit("pipeline", f"{exp_id} 一键全流程分析", pipeline.run_all, exp_id, exp_id=exp_id)


@app.get("/api/templates")
@login_required
def api_templates():
    return ok(pipeline.list_templates())


@app.get("/api/templates/<tid>")
@login_required
def api_template(tid):
    return ok(pipeline.template_preview(tid))


# ----------------------------------------------------------------------------
# 特征库
# ----------------------------------------------------------------------------
@app.get("/api/library")
@login_required
def api_library():
    return ok(library.list_entries())


@app.get("/api/library/stats")
@login_required
def api_library_stats():
    return ok(library.group_statistics())


@app.get("/api/library/<fid>")
@login_required
def api_library_entry(fid):
    regions = [int(x) for x in request.args.get("regions", "").split(",") if x.strip()]
    return ok(library.entry_detail(fid, regions or None))


@app.put("/api/library/<fid>")
@login_required
def api_library_update(fid):
    e = library.get_entry(fid)
    e["label"] = int(request.get_json(force=True)["label"])
    e["group"] = library.GROUP_NAME[e["label"]]
    library.write_entry(e)
    return ok(e)


@app.post("/api/library/from_experiment")
@login_required
def api_library_from_exp():
    return ok(library.add_from_experiment(request.get_json(force=True)["exp_id"]))


@app.post("/api/library/scan")
@login_required
def api_library_scan():
    return ok(library.scan_signal_folder(request.get_json(force=True)["folder"]))


@app.post("/api/library/import")
@login_required
def api_library_import():
    b = request.get_json(force=True)
    return _submit("library", f"信号矩阵批量分析 {Path(b['folder']).name}", library.run_import, "library",
                   folder=b["folder"], workers=int(b.get("workers", 48)), overwrite=bool(b.get("overwrite")))


@app.post("/api/library/delete")
@login_required
def api_library_delete():
    library.delete_entries(request.get_json(force=True)["ids"])
    return ok()


# ----------------------------------------------------------------------------
# 高斯核平滑 / 数据集
# ----------------------------------------------------------------------------
@app.get("/api/kernel/preview")
@login_required
def api_kernel_preview():
    return ok(datasets.kernel_preview(request.args.get("template", "zebrafish_std_100"),
                                      request.args.get("H", 0.5, type=float), request.args.get("region", 25, type=int)))


@app.get("/api/kernel/demo")
@login_required
def api_kernel_demo():
    return ok(datasets.smoothing_demo(request.args["fish"], request.args.get("feature", "num_peaks"),
                                      request.args.get("template", "zebrafish_std_100"),
                                      request.args.get("H", 0.5, type=float)))


@app.route("/api/datasets", methods=["GET", "POST"])
@login_required
def api_datasets():
    if request.method == "POST":
        b = request.get_json(force=True)
        ids = b.get("ids") or [e["fish_id"] for e in library.list_entries()]
        return _submit("dataset", f"构建数据集 {b['name']}", datasets.run_build, None, name=b["name"], ids=ids,
                       feature_keys=b.get("feature_keys") or [k for k, _ in SCALAR_FEATURES],
                       channels=b.get("channels") or ["minmax", "zscore", "kernel_minmax", "kernel_zscore"],
                       band_width=float(b.get("band_width", 0.5)), description=b.get("description", ""))
    return ok(datasets.list_datasets())


@app.route("/api/datasets/<did>", methods=["GET", "DELETE"])
@login_required
def api_dataset(did):
    if request.method == "DELETE":
        datasets.delete_dataset(did)
        return ok()
    return ok(datasets.get_dataset(did))


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
@app.get("/api/models")
@login_required
def api_models():
    return ok(modeling.list_models())


@app.post("/api/models/train")
@login_required
def api_train():
    b = request.get_json(force=True)
    return _submit("train", f"训练模型 {b['name']}", modeling.run_train, "train", dataset_id=b["dataset"],
                   name=b["name"], k_folds=int(b.get("k_folds", 5)), epochs=int(b.get("epochs", 100)),
                   lr=float(b.get("lr", 0.01)), momentum=float(b.get("momentum", 0.9)),
                   weight_decay=float(b.get("weight_decay", 1e-5)), optimizer=b.get("optimizer", "SGD"),
                   device=b.get("device", "cuda"), seed=int(b.get("seed", 42)), description=b.get("description", ""))


@app.route("/api/models/<mid>", methods=["GET", "DELETE"])
@login_required
def api_model(mid):
    if request.method == "DELETE":
        modeling.delete_model(mid)
        return ok()
    m = modeling.get_model(mid)
    m["importance_map"] = modeling.importance_map(mid)
    m["feature_names"] = modeling.feature_names(m["feature_keys"])
    return ok(m)


@app.post("/api/models/<mid>/predict")
@login_required
def api_predict(mid):
    return ok(modeling.predict(mid, request.get_json(force=True)["ids"]))


@app.get("/api/models/<mid>/predictions")
@login_required
def api_predictions(mid):
    p = config.MODEL_DIR / mid / "predictions.json"
    return ok(json.loads(p.read_text(encoding="utf-8")) if p.exists() else [])


# ----------------------------------------------------------------------------
# 任务
# ----------------------------------------------------------------------------
@app.get("/api/tasks")
@login_required
def api_tasks():
    return ok(manager.list())


@app.route("/api/tasks/<tid>", methods=["GET", "DELETE"])
@login_required
def api_task(tid):
    if request.method == "DELETE":
        return ok(manager.delete(tid))
    return ok(manager.get(tid))


@app.post("/api/demo/generate")
@login_required
def api_demo_generate():
    from services import demo_data

    def job(ctx):
        tdir = demo_data.write_template_files(config.DATA_DIR / "demo" / "template")
        demo_data.write_template_files(config.TEMPLATE_DIR / "zebrafish_std_100")
        ctx.log("脑模板与 100 脑区图谱已生成")
        demo_data.generate_experiment(config.DATA_DIR / "demo" / "raw", "EXP20260315_A", (0, 1), 100, template_dir=tdir,
                                      progress=lambda v: ctx.progress(v * 0.8))
        demo_data.generate_signal_library(config.DATA_DIR / "demo" / "signal_library", tdir, 80)
        return {"path": str(config.DATA_DIR / "demo")}
    return _submit("demo", "生成示例数据", job, "demo")


# ----------------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------------
def _csv(rows, header, filename):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/export/experiment/<exp_id>/<side>.csv")
@login_required
def export_exp_features(exp_id, side):
    res = pipeline.feature_result(exp_id, side)
    keys = [k for k, _ in SCALAR_FEATURES]
    rows = [[r + 1] + [res["table"][k][r] for k in keys] for r in range(100)]
    return _csv(rows, ["region"] + [n for _, n in SCALAR_FEATURES], f"{exp_id}_{side}_features.csv")


@app.get("/api/export/library.csv")
@login_required
def export_library():
    rows = []
    for e in library.list_entries():
        s = e["summary"]
        rows.append([e["fish_id"], e["label"], e["group"], e["source"]] +
                    [s[k]["mean"] for k, _ in SCALAR_FEATURES] + [s["sync_index"], s["total_peaks"]])
    return _csv(rows, ["fish_id", "label", "group", "source"] + [n for _, n in SCALAR_FEATURES] +
                ["同步指数", "钙峰总数"], "feature_library.csv")


@app.get("/api/export/model/<mid>/predictions.csv")
@login_required
def export_predictions(mid):
    p = config.MODEL_DIR / mid / "predictions.json"
    hist = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    rows = [[h["time"], r["fish_id"], r["true_name"], r["pred_name"], r["prob"], r["correct"]]
            for h in hist[:1] for r in h["results"]]
    return _csv(rows, ["time", "fish_id", "true", "pred", "prob_drug", "correct"], f"{mid}_predictions.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=SOFTWARE_NAME)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    load_users()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
