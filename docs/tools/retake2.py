#!/usr/bin/env python
# coding=utf-8
"""补拍：智能预测结果（42_predict）与任务中心（44_tasks）。"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "screenshots"
U = "http://localhost:8765"
IDS = ['CTL-005', 'CTL-012', 'CTL-033', 'CTL-047', 'CTL-061', 'CTL-078', 'DRG-004', 'DRG-019', 'DRG-026', 'DRG-040', 'DRG-055', 'DRG-072']
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2, locale="zh-CN").new_page()
    pg.goto(f"{U}/login"); pg.fill("#p", "admin123"); pg.click("button[type=submit]"); pg.wait_for_url(f"{U}/"); time.sleep(2)
    pg.goto(f"{U}/#predict"); time.sleep(3)
    pg.click("#sE")
    pg.evaluate("ids => document.querySelectorAll('.pS').forEach(c => { if (ids.includes(c.value)) c.checked = true; })", IDS)
    pg.click("#pGo")
    pg.wait_for_selector("#pc canvas", timeout=60000); time.sleep(2)
    pg.screenshot(path=str(OUT / "42_predict.png"))
    pg.evaluate("""() => { const c = document.querySelector('#content'); const el = document.querySelector('#pRes .card:last-child');
        c.scrollTop = el.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop - 80; }""")
    time.sleep(1.5)
    pg.screenshot(path=str(OUT / "43_predict_table.png"))
    pg.goto(f"{U}/#tasks"); time.sleep(3)
    tid = pg.evaluate("async () => (await (await fetch('/api/tasks')).json()).data.find(t => t.type === 'pipeline').id")
    pg.click(f"[data-v='{tid}']"); time.sleep(2.5)
    pg.screenshot(path=str(OUT / "44_tasks.png"))
    b.close()
print("retake2 ok")
