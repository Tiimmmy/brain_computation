#!/usr/bin/env python
# coding=utf-8
"""补拍单张截图：实验详情页甩动异常帧浏览（06_exp_profile）。"""
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "screenshots"
U = "http://localhost:8765"
fly = json.loads((ROOT / "data/demo/raw/EXP20260315_A/acquisition.json").read_text())["fish"][0]["fly_frames"][0]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2, locale="zh-CN").new_page()
    pg.goto(f"{U}/login"); pg.fill("#p", "admin123"); pg.click("button[type=submit]"); pg.wait_for_url(f"{U}/"); time.sleep(2)
    pg.goto(f"{U}/#experiments?id=EXP20260315_A"); time.sleep(3)
    pg.select_option("#cmap", "inferno"); time.sleep(1.5)
    pg.evaluate(f"document.querySelector('#fr').value = {fly}; document.querySelector('#fr').dispatchEvent(new Event('input'))")
    time.sleep(2.5)
    pg.evaluate("""() => { const c = document.querySelector('#content'); const el = document.querySelector('#frameBox');
        c.scrollTop = el.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop - 120; }""")
    time.sleep(1.5)
    pg.screenshot(path=str(OUT / "06_exp_profile.png"))
    # 高斯核空间平滑（色标精度修正后补拍）
    pg.goto(f"{U}/#kernel"); time.sleep(2.5)
    pg.evaluate("document.querySelector('#kH').value = 1.0; document.querySelector('#kH').dispatchEvent(new Event('input'))")
    pg.select_option("#kF", "DRG-003"); pg.select_option("#kK", "areas_sum"); pg.click("#kGo"); time.sleep(4)
    pg.screenshot(path=str(OUT / "33_kernel.png"))
    pg.evaluate("""() => { const c = document.querySelector('#content'); const el = document.querySelector('#kB');
        c.scrollTop = el.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop - 120; }""")
    time.sleep(1.5)
    pg.screenshot(path=str(OUT / "34_kernel_compare.png"))
    log = json.loads((OUT / "capture_log.json").read_text())
    log["fly_frame"] = fly
    (OUT / "capture_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    b.close()
print("retake ok, fly frame", fly)
