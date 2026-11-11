#!/usr/bin/env python
# coding=utf-8
"""操作说明书截图采集：通过浏览器真实操作软件全部功能并截图。

用法：python capture.py --url http://localhost:8765 --out ../screenshots
"""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "demo" / "raw"
LIB = ROOT / "data" / "demo" / "signal_library"

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8765")
ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--start", default="experiments", help="从指定阶段开始（用于中断后续拍）")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
U = args.url
LOG = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2, locale="zh-CN")
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))

    def api(method, url, body=None):
        return pg.evaluate("""async ([m, u, b]) => { const r = await fetch(u, {method: m, headers: {'Content-Type': 'application/json'},
            body: b === null ? undefined : JSON.stringify(b)}); return await r.json(); }""", [method, url, body])

    def shot(name, scroll=None, wait=1.2, offset=70):
        if scroll is not None:
            pg.evaluate("""([sel, off]) => { const c = document.querySelector('#content'); const el = document.querySelector(sel);
                if (el) c.scrollTop = el.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop - off; }""", [scroll, offset])
        time.sleep(wait)
        pg.screenshot(path=str(OUT / f"{name}.png"))
        LOG.append(name)
        log("shot", name)

    ORDER = ['experiments', 'templates', 'preprocess', 'register', 'atlas', 'signal', 'feature', 'library', 'kernel', 'dataset', 'train', 'evaluate', 'predict', 'final']
    prev = json.loads((OUT / "capture_log.json").read_text()) if (OUT / "capture_log.json").exists() else {}
    worst, fly_frame = prev.get("worst_frame"), prev.get("fly_frame")
    LOG.extend(prev.get("shots", []) if args.start != "experiments" else [])

    def on(stage):
        return ORDER.index(stage) >= ORDER.index(args.start)

    def go(hash_, wait=2.5):
        pg.goto(f"{U}/#{hash_}")
        time.sleep(wait)

    def wait_task(tid, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            t = api("GET", f"/api/tasks/{tid}")["data"]
            if t["status"] != "running":
                log("task", tid, t["status"], t["elapsed"])
                return t
            time.sleep(2)
        raise TimeoutError(tid)

    def wait_running(tp):
        while True:
            ts = [t for t in api("GET", "/api/tasks")["data"] if t["status"] == "running" and t["type"] == tp]
            if not ts:
                return
            time.sleep(2)

    def latest_task(tp):
        return [t for t in api("GET", "/api/tasks")["data"] if t["type"] == tp][0]["id"]

    # ------------------------------------------------------------------ 登录
    pg.goto(f"{U}/login")
    time.sleep(1)
    pg.fill("#p", "")
    shot("01_login")
    pg.fill("#p", "admin123")
    pg.click("button[type=submit]")
    pg.wait_for_url(f"{U}/")
    time.sleep(1)

    # 清理工作空间（通过软件接口）
    if args.start == "experiments":
        for e in api("GET", "/api/experiments")["data"]:
            api("DELETE", f"/api/experiments/{e['id']}")
        ids = [e["fish_id"] for e in api("GET", "/api/library")["data"]]
        if ids:
            api("POST", "/api/library/delete", {"ids": ids})
        for d in api("GET", "/api/datasets")["data"]:
            api("DELETE", f"/api/datasets/{d['id']}")
        for m in api("GET", "/api/models")["data"]:
            api("DELETE", f"/api/models/{m['id']}")
        for t in api("GET", "/api/tasks")["data"]:
            api("DELETE", f"/api/tasks/{t['id']}")

    # ------------------------------------------------------------------ 实验数据管理
    if on("experiments"):
        go("experiments")
        shot("02_exp_empty")
        pg.click("#btnNew")
        time.sleep(1)
        pg.fill("#wPath", str(RAW))
        pg.click("#wBrowse")
        time.sleep(1.2)
        shot("03_dir_picker")
        pg.click(".mask:last-child .modal-foot .btn.c")
        time.sleep(0.5)
        pg.click("#wScan")
        time.sleep(1.5)
        pg.select_option("#wG0", "0")
        pg.select_option("#wG1", "1")
        pg.fill("#wDesc", "成瘾药物急性暴露 30 min 后全脑钙成像（第 1 批）")
        shot("04_import_wizard")
        pg.click(".modal-foot .btn.primary")
        time.sleep(3)
        pg.select_option("#cmap", "inferno")
        time.sleep(1.5)
        shot("05_exp_detail")
        fly_frame = json.loads((RAW / "EXP20260315_A" / "acquisition.json").read_text())["fish"][0]["fly_frames"][0]
        pg.evaluate(f"document.querySelector('#fr').value = {fly_frame}; document.querySelector('#fr').dispatchEvent(new Event('input'))")
        time.sleep(2)
        shot("06_exp_profile", scroll="#frameBox", offset=120)
        # 第二个实验通过接口创建并在后台完成全流程（用于后续预测新样本）
        api("POST", "/api/experiments", {"name": "EXP20260315_B", "seq_dir": str(RAW / "EXP20260315_B" / "Pos0"),
                                         "description": "成瘾药物急性暴露 30 min 后全脑钙成像（第 2 批）"})
        tid_b = api("POST", "/api/experiments/EXP20260315_B/run_all", {})["data"]["task_id"]
        wait_task(tid_b)

    # ------------------------------------------------------------------ 模板库
    if on("templates"):
        go("templates", 3)
        shot("07_templates")
        shot("08_templates_table", scroll=".tbl")

    # ------------------------------------------------------------------ 预处理
    if on("preprocess"):
        api("PUT", "/api/experiments/EXP20260315_A", {})
        pg.evaluate("localStorage.setItem('bc_exp', JSON.stringify('EXP20260315_A'))")
        go("preprocess", 3.5)
        shot("09_preprocess")
        pg.click("#btnApply")
        time.sleep(3)
        shot("10_preprocess_apply", scroll=".grid")

    # ------------------------------------------------------------------ 配准
    if on("register"):
        go("register", 2.5)
        pg.click("#rRun")
        time.sleep(9)
        shot("11_register_running")
        wait_task(latest_task("register"))
        time.sleep(2)
        go("register", 4)
        shot("12_register_result")
        metrics = json.loads((ROOT / "workspace/experiments/EXP20260315_A/aligned/metrics.json").read_text())
        acq = json.loads((RAW / "EXP20260315_A" / "acquisition.json").read_text())
        fly = set(acq["fish"][0]["fly_frames"])
        cand = [(m["before"], i) for i, m in enumerate(metrics["l"]) if m and i not in fly and 50 < i < 950]
        worst = min(cand)[1]
        pg.fill("#cf", str(worst))
        pg.click("#cfGo")
        time.sleep(3)
        shot("13_register_compare", scroll="#cf", offset=110)
        shot("14_register_projection", scroll="#rRes .card:last-child", offset=80)

    # ------------------------------------------------------------------ 模板映射
    if on("atlas"):
        go("atlas", 2.5)
        pg.click("#aRun")
        time.sleep(1.2)
        shot("15_atlas_running")
        wait_task(latest_task("atlas"))
        go("atlas", 3.5)
        shot("16_atlas_result")
        shot("17_atlas_result_r", scroll="#aRes .card:nth-child(2)")

    # ------------------------------------------------------------------ 信号提取
    if on("signal"):
        go("signal", 2.5)
        pg.click("#sRun")
        time.sleep(4)
        shot("18_signal_running")
        wait_task(latest_task("signal"))
        go("signal", 4)
        shot("19_signal_result")
        shot("20_signal_traces", scroll="#tr2", offset=130)
        pg.select_option("#stage", "dff")
        time.sleep(1)
        shot("21_signal_dff", scroll="#tr2", offset=130)
        shot("22_signal_raster", scroll="#rs", offset=470)

    # ------------------------------------------------------------------ 特征
    if on("feature"):
        go("feature", 2.5)
        pg.click("#fRun")
        time.sleep(1.5)
        shot("23_feature_running")
        wait_task(latest_task("feature"))
        go("feature", 4)
        shot("24_feature_result")
        pg.select_option("#mapK", "areas_sum")
        time.sleep(1)
        shot("25_feature_maps", scroll="#mapK", offset=80)
        shot("26_feature_table", scroll="#ar", offset=130)
        pg.click("#fLib")
        time.sleep(1.5)
        shot("27_feature_tolib", wait=0.3)

    # ------------------------------------------------------------------ 特征库
    if on("library"):
        go("library", 3)
        pg.click("#lImp")
        time.sleep(1)
        pg.fill("#iP", str(LIB))
        pg.click("#iS")
        time.sleep(2)
        shot("28_library_import")
        pg.click(".modal-foot .btn.primary")
        time.sleep(5)
        shot("29_library_running")
        wait_task(latest_task("library"))
        go("library", 4)
        shot("30_library_stats")
        shot("31_library_list", scroll="#lTb", offset=160)
        pg.click("[data-v='DRG-003']")
        time.sleep(2.5)
        shot("32_library_detail")
        pg.keyboard.press("Escape")

    # ------------------------------------------------------------------ 高斯核
    if on("kernel"):
        go("kernel", 2)
        pg.evaluate("document.querySelector('#kH').value = 1.0; document.querySelector('#kH').dispatchEvent(new Event('input'))")
        pg.select_option("#kF", "DRG-003")
        pg.select_option("#kK", "areas_sum")
        pg.click("#kGo")
        time.sleep(4)
        shot("33_kernel")
        shot("34_kernel_compare", scroll="#kB", offset=120)

    # ------------------------------------------------------------------ 数据集
    if on("dataset"):
        go("dataset", 2.5)
        pg.fill("#dN", "成瘾表型识别数据集-V1")
        pg.fill("#dD", "160 例信号矩阵样本 + 2 例实验流水线样本，10 类特征 × 100 脑区，4 通道")
        pg.click("#dGo")
        wait_task(latest_task("dataset"))
        time.sleep(1)
        go("dataset", 4)
        shot("35_dataset")

        # 第 2 批实验样本在数据集构建后入库，作为模型未见过的新样本
        api("POST", "/api/library/from_experiment", {"exp_id": "EXP20260315_B"})

    # ------------------------------------------------------------------ 训练
    if on("train"):
        go("train", 3)
        pg.fill("#tN", "BrainNet-成瘾表型识别-V1")
        pg.fill("#tE", str(args.epochs))
        shot("36_train_config")
        pg.click("#tGo")
        time.sleep(150)
        shot("37_train_running", scroll="#lossC", offset=360)
        wait_task(latest_task("train"), timeout=7200)
        time.sleep(3)
        go("train", 5)
        shot("38_train_done", scroll="#lossC", offset=360)

    # ------------------------------------------------------------------ 评估
    if on("evaluate"):
        go("evaluate", 4)
        shot("39_evaluate")
        shot("40_evaluate_importance", scroll="#impF", offset=110)
        shot("41_evaluate_info", scroll="#hist", offset=80)

    # ------------------------------------------------------------------ 预测
    if on("predict"):
        go("predict", 3)
        pg.click("#sE")
        pg.evaluate("""() => { const ids = ['CTL-005','CTL-012','CTL-033','CTL-047','CTL-061','CTL-078','DRG-004','DRG-019','DRG-026','DRG-040','DRG-055','DRG-072'];
            document.querySelectorAll('.pS').forEach(c => { if (ids.includes(c.value)) c.checked = true; }); }""")
        pg.click("#pGo")
        pg.wait_for_selector("#pc canvas", timeout=120000)
        time.sleep(2)
        shot("42_predict")
        shot("43_predict_table", scroll="#pRes .card:last-child", offset=80)

    # ------------------------------------------------------------------ 任务中心 / 设置 / 首页
    if on("final"):
        go("tasks", 3)
        tid = latest_task("pipeline")
        pg.click(f"[data-v='{tid}']")
        time.sleep(2)
        shot("44_tasks")
        go("settings", 3)
        shot("45_settings")
        shot("46_settings_env", scroll=".grid.g3", offset=80)
        go("experiments", 3)
        shot("47_exp_list")
        go("dashboard", 4)
        shot("48_dashboard")
        shot("49_dashboard_bottom", scroll=".grid.g2", offset=420)

    if worst is None:
        metrics = json.loads((ROOT / "workspace/experiments/EXP20260315_A/aligned/metrics.json").read_text())
        fly = set(json.loads((RAW / "EXP20260315_A" / "acquisition.json").read_text())["fish"][0]["fly_frames"])
        worst = min((m["before"], i) for i, m in enumerate(metrics["l"]) if m and i not in fly and 50 < i < 950)[1]
    if fly_frame is None:
        fly_frame = json.loads((RAW / "EXP20260315_A" / "acquisition.json").read_text())["fish"][0]["fly_frames"][0]
    (OUT / "capture_log.json").write_text(json.dumps({"shots": sorted(set(LOG)), "errors": errors, "worst_frame": worst, "fly_frame": fly_frame},
                                                     ensure_ascii=False, indent=1))
    log("errors:", errors)
    browser.close()
