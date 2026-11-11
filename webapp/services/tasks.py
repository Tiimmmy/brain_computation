#!/usr/bin/env python
# coding=utf-8
"""后台任务管理：耗时计算（配准、信号提取、模型训练等）以线程方式运行，
记录进度、日志与状态，并持久化到 workspace/tasks.json。"""

import json
import threading
import time
import traceback
import uuid

from config import WORKSPACE

TASK_FILE = WORKSPACE / "tasks.json"

TASK_TYPES = {
    "import": "数据导入",
    "register": "运动校正配准",
    "atlas": "脑区模板映射",
    "signal": "钙信号提取",
    "feature": "神经活动特征计算",
    "library": "信号库批量分析",
    "kernel": "高斯核空间平滑",
    "dataset": "特征数据集构建",
    "train": "模型训练",
    "predict": "批量预测",
    "pipeline": "一键全流程分析",
    "demo": "示例数据生成",
}


class TaskContext:
    def __init__(self, manager, task_id):
        self.manager = manager
        self.task_id = task_id

    def progress(self, value, message=None):
        self.manager.update(self.task_id, progress=round(float(value), 4), message=message)

    def log(self, text):
        self.manager.append_log(self.task_id, text)

    def set_result(self, result):
        self.manager.update(self.task_id, result=result)

    def extra(self, **kwargs):
        with self.manager.lock:
            self.manager.tasks[self.task_id].setdefault("extra", {}).update(kwargs)


class TaskManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.tasks = {}
        self._load()

    def _load(self):
        if TASK_FILE.exists():
            try:
                self.tasks = json.loads(TASK_FILE.read_text(encoding="utf-8"))
                for t in self.tasks.values():
                    if t["status"] in ("running", "pending"):
                        t["status"] = "failed"
                        t["message"] = "服务重启，任务中断"
            except Exception:
                self.tasks = {}

    def _save(self):
        data = {k: {kk: vv for kk, vv in v.items()} for k, v in self.tasks.items()}
        TASK_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def submit(self, ttype, title, func, target=None, **kwargs):
        tid = time.strftime("T%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4]
        with self.lock:
            self.tasks[tid] = {
                "id": tid, "type": ttype, "type_name": TASK_TYPES.get(ttype, ttype), "title": title,
                "target": target, "status": "running", "progress": 0.0, "message": "任务已启动",
                "logs": [], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "started": time.time(),
                "finished": None, "result": None, "extra": {},
            }
            self._save()
        ctx = TaskContext(self, tid)

        def runner():
            try:
                ctx.log(f"开始执行：{title}")
                result = func(ctx, **kwargs)
                self.update(tid, status="success", progress=1.0, message="任务完成", result=result,
                            finished=time.time())
                ctx.log("任务完成")
            except Exception as e:  # noqa
                ctx.log("错误：" + "".join(traceback.format_exception_only(type(e), e)).strip())
                ctx.log(traceback.format_exc())
                self.update(tid, status="failed", message=f"任务失败：{e}", finished=time.time())

        threading.Thread(target=runner, daemon=True).start()
        return tid

    def update(self, tid, **kwargs):
        with self.lock:
            t = self.tasks.get(tid)
            if not t:
                return
            for k, v in kwargs.items():
                if v is not None or k in ("result",):
                    t[k] = v
            self._save()

    def append_log(self, tid, text):
        with self.lock:
            t = self.tasks.get(tid)
            if t is not None:
                t["logs"].append(f"[{time.strftime('%H:%M:%S')}] {text}")
                t["logs"] = t["logs"][-400:]
                self._save()

    def get(self, tid):
        with self.lock:
            t = self.tasks.get(tid)
            return self._view(t) if t else None

    def _view(self, t):
        v = dict(t)
        end = t["finished"] or time.time()
        v["elapsed"] = round(end - t["started"], 1)
        return v

    def list(self, limit=200):
        with self.lock:
            items = sorted(self.tasks.values(), key=lambda x: x["started"], reverse=True)[:limit]
            return [{k: v for k, v in self._view(t).items() if k not in ("logs", "extra")} for t in items]

    def delete(self, tid):
        with self.lock:
            if tid in self.tasks and self.tasks[tid]["status"] != "running":
                del self.tasks[tid]
                self._save()
                return True
        return False

    def running_for(self, target):
        with self.lock:
            return [t["id"] for t in self.tasks.values() if t["status"] == "running" and t["target"] == target]


manager = TaskManager()
