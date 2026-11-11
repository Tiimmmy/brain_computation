# 斑马鱼全脑钙成像神经计算分析系统 V1.0（BrainComp）Web 前端

为 `align/`、`feature_generation/`、`feature_modify/`、`Models/FC/` 中的分析算法提供图形化操作界面（Flask + 原生 JS + ECharts）。

## 安装

```bash
conda create -p ./envs/brainc python=3.11
./envs/brainc/bin/pip install -r webapp/requirements.txt   # torch 建议从 https://download.pytorch.org/whl/cu121 安装
```

## 生成示例数据（可选）

```bash
./envs/brainc/bin/python webapp/services/demo_data.py --experiments 2 --library 80
```

在 `data/demo/` 下生成：标准脑模板与 100 脑区图谱、两个 1000 帧双鱼 TIFF 实验、160 例脑区信号矩阵（含 labels.csv）。

## 启动

```bash
cd webapp && ../envs/brainc/bin/python app.py --port 8765
```

浏览器访问 `http://<host>:8765`，默认账户 `admin / admin123`（首次登录后请在“系统设置”中修改）。

运行结果保存在 `workspace/`（实验、特征库、数据集、模型、任务记录）。

## 目录

| 路径 | 说明 |
|---|---|
| `app.py` | Web 服务入口与 REST 接口 |
| `config.py` | 路径、软件信息、默认参数 |
| `services/legacy.py` | 加载项目原有算法模块 |
| `services/pipeline.py` | 预处理 → 配准 → 模板映射 → 信号提取 → 特征计算 |
| `services/reg_worker.py` | 逐帧配准子进程 |
| `services/analysis.py` | 信号预处理与特征计算封装 |
| `services/library.py` / `datasets.py` / `modeling.py` | 特征库、数据集、BrainNet 训练评估预测 |
| `services/tasks.py` | 后台任务管理 |
| `static/` | 前端页面 |

操作说明书（软著文档鉴别材料）由 `docs/tools/capture.py`（真实操作截图）与 `docs/tools/build_manual.py`（生成 PDF）产出。
