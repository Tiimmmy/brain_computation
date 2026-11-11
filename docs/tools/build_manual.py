#!/usr/bin/env python
# coding=utf-8
"""生成软件著作权登记用《操作说明书》（文档鉴别材料）PDF。

数值均从软件实际运行结果（workspace）读取，保证与截图一致。
用法：python build_manual.py
"""

import base64
import json
import sys
from pathlib import Path

import argparse
import re
import subprocess
import tempfile

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(description="生成操作说明书 / 文档鉴别材料 PDF")
ap.add_argument("--kind", choices=["manual", "jianbie"], default="manual",
                help="manual：操作说明书；jianbie：软著文档鉴别材料（页眉仅软件全称+版本号、含目录、版面紧凑并做合规检查）")
ARGS = ap.parse_args()
JB = ARGS.kind == "jianbie"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp"))
from config import SOFTWARE_NAME, VERSION  # noqa: E402
from services.legacy import source_statistics  # noqa: E402

DOCS = ROOT / "docs"
SHOT = DOCS / "screenshots"
WS = ROOT / "workspace"
TITLE = f"{SOFTWARE_NAME} {VERSION}"

# ----------------------------------------------------------------------------- 运行数据
expA = json.loads((WS / "experiments/EXP20260315_A/meta.json").read_text(encoding="utf-8"))
expB = json.loads((WS / "experiments/EXP20260315_B/meta.json").read_text(encoding="utf-8"))
reg = expA["results"]["register"]
atl = expA["results"]["atlas"]["summary"]
sig = expA["results"]["signal"]["summary"]
fea = expA["results"]["feature"]["summary"]
model = sorted([json.loads(p.read_text(encoding="utf-8")) for p in (WS / "models").glob("*/meta.json")],
               key=lambda m: m["created"])[-1]
dataset = json.loads((WS / "datasets" / model["dataset"] / "meta.json").read_text(encoding="utf-8"))
lib_n = len(list((WS / "library").glob("*/meta.json")))
cap = json.loads((SHOT / "capture_log.json").read_text(encoding="utf-8"))
pred = json.loads((WS / "models" / model["id"] / "predictions.json").read_text(encoding="utf-8"))[0]
pred_ok = sum(r["correct"] for r in pred["results"])
stats = source_statistics()
src_total = sum(s["lines"] for s in stats)
by_lang = {}
for s in stats:
    by_lang[s["lang"]] = by_lang.get(s["lang"], 0) + s["lines"]
orig = sum(s["lines"] for s in stats if not s["file"].startswith("webapp"))
mm = model["metrics"]
acqA = json.loads((ROOT / "data/demo/raw/EXP20260315_A/acquisition.json").read_text(encoding="utf-8"))
fly_match = sorted(sig["l"]["fly_frames"]) == sorted(acqA["fish"][0]["fly_frames"])
fly_note = "与示例数据采集记录中的甩动帧完全一致" if fly_match else "与示例数据采集记录基本一致"

# ----------------------------------------------------------------------------- 排版工具
FIG = [0]


def fig(name, caption):
    FIG[0] += 1
    import cv2
    img = cv2.imread(str(SHOT / f"{name}.png"))
    img = cv2.resize(img, (2000, int(img.shape[0] * 2000 / img.shape[1])), interpolation=cv2.INTER_AREA)
    data = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()
    return (f'<figure><img src="data:image/jpeg;base64,{data}">'
            f'<figcaption>图 {FIG[0]}　{caption}</figcaption></figure>')


def table(head, rows, widths=None):
    cols = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><colgroup>{cols}</colgroup><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def steps(*items):
    return "<ol class='steps'>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def ul(*items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


L, R = "左侧鱼", "右侧鱼"

# ----------------------------------------------------------------------------- 正文
H = []
H.append(f"""
<!--TOC-->

<h1>1　引言</h1>
<h2>1.1　编写目的</h2>
<p>本说明书面向{SOFTWARE_NAME}（以下简称“本系统”）的使用者，按照软件实际运行界面逐一介绍系统全部功能模块的界面组成与操作步骤，帮助科研人员快速掌握从原始钙荧光成像数据到神经活动特征、再到深度学习表型识别的完整分析流程。文中所有界面截图均来自本系统 {VERSION} 的实际运行过程，示例数据为系统自带的斑马鱼幼鱼双鱼并排全脑钙成像演示数据。</p>
<h2>1.2　软件概述</h2>
<p>斑马鱼幼鱼体型小、脑部透明，结合钙指示剂（如 GCaMP6s）可实现全脑神经元活动的在体光学成像，是神经药理学与成瘾机制研究的重要模型。全脑钙成像数据量大、存在鱼体漂移与甩动伪影、个体间脑形态差异显著，传统人工分析效率低、可重复性差。本系统以 Python 科学计算生态为基础，集成 ANTs 医学图像配准、信号处理与 PyTorch 深度学习技术，采用 B/S 架构提供图形化 Web 操作界面，实现钙成像数据的自动化、标准化、批量化计算分析。</p>
<h2>1.3　主要功能</h2>
{table(["功能模块", "功能说明"], [
    ["系统登录与首页", "用户身份认证；工作空间数据概况统计、分析流程导航、最近实验与任务展示"],
    ["实验数据管理", "TIFF 帧序列目录浏览与扫描、实验登记（编号、成像布局、鱼体分组、指示剂、帧率）、帧序列浏览、伪彩色显示、灰度直方图、全时程强度曲线、实验信息编辑与删除、一键全流程分析"],
    ["标准脑模板库", "标准脑模板图像、100 脑区分割图谱、模板叠加显示、脑区信息表与脑区中心空间分布"],
    ["图像预处理", "双鱼图像左右拆分、旋转摆正、阈值分割定位鱼体中心、居中裁剪，参数可视化预览与确认"],
    ["运动校正配准", "基于 ANTs 的逐帧非刚性配准（ElasticSyN/SyN/Affine/Rigid），多进程并行，生成对齐时空矩阵；逐帧配准质量曲线、叠加/棋盘格对比、时间投影"],
    ["脑区模板映射", "标准模板与个体脑配准，逆变换映射 100 脑区标签；轮廓叠加与 Dice 系数评价"],
    ["钙信号提取", "甩动异常帧识别、异常像素剔除、死鱼判定、脑区平均荧光、Butterworth 低通滤波、线性插值、IModPoly 基线估计、ΔF/F 归一化；过程曲线与全脑活动栅格图"],
    ["神经活动特征", "自适应阈值钙峰检测与拐点定位，计算 13 类神经活动特征；特征脑区分布图、同步连接矩阵、明细表与 CSV 导出"],
    ["个体特征库", "汇集单鱼特征样本；批量导入脑区信号矩阵并行分析；组间箱线图与 Welch t 检验；样本详情与分组修改"],
    ["高斯核空间平滑", "基于脑区中心距离的自适应带宽高斯核；核矩阵与权重分布可视化；平滑前后对比"],
    ["建模数据集", "样本、特征类型与输入通道选择，多通道归一化，PCA 二维投影预览"],
    ["模型训练", "BrainNet 深度网络，K 折交叉验证，超参数配置，GPU 加速，实时损失/准确率曲线"],
    ["模型评估", "准确率、精确率、召回率、F1、AUC、混淆矩阵、ROC 曲线、各折结果，梯度显著性特征与脑区重要性分析"],
    ["智能预测", "选择模型批量预测待测样本的药物处理（成瘾）表型概率，结果可视化与导出"],
    ["任务中心 / 系统设置", "后台任务进度与日志管理；默认参数配置、运行环境与源程序信息、密码修改"],
], ["22%", "78%"])}
<h2>1.4　技术特点</h2>
{ul("<b>全流程一体化：</b>覆盖数据导入、图像预处理、运动校正、脑区映射、信号提取、特征计算、空间平滑、数据集构建、模型训练评估与预测共 11 个环节，支持单步交互操作与“一键全流程分析”。",
    "<b>高精度非刚性配准：</b>采用 ANTs 弹性对称归一化（ElasticSyN）算法，在降采样空间估计形变场、全分辨率应用，兼顾精度与速度；模板与个体双向变换（Composite / InverseComposite）实现脑区标签自动映射。",
    "<b>稳健的信号预处理：</b>基于分位数围栏的甩动帧与异常像素自动识别、死鱼样本自动判定；零相位 Butterworth 滤波与 IModPoly 迭代多项式基线估计，有效抑制光漂白与运动伪影。",
    "<b>多维神经活动特征：</b>从放电频率、强度、持续时间、节律与脑区间同步性等维度定量刻画全脑神经活动。",
    "<b>空间先验与深度学习结合：</b>高斯核空间平滑引入脑区空间邻近先验，削弱配准误差；BrainNet 为每个特征单元配置独立子网络后接深层全连接分类头，并通过 BatchNorm 统计量重校准提升概率输出可靠性；输入梯度显著性实现可解释性分析。",
    "<b>高性能并行计算：</b>逐帧配准与批量信号分析采用多进程并行，模型训练支持 GPU 加速；耗时计算以后台任务运行，进度与日志实时可见。",
    "<b>B/S 架构与可视化：</b>基于 Flask + ECharts 的 Web 界面，无需安装客户端，图像、曲线、热图、脑区分布图等交互式可视化丰富。")}

<h1>2　运行环境</h1>
<h2>2.1　硬件环境</h2>
{table(["项目", "最低配置", "推荐配置"], [
    ["CPU", "4 核 64 位处理器", "32 核以上（并行配准）"],
    ["内存", "16 GB", "64 GB 及以上"],
    ["GPU", "可选（无 GPU 时使用 CPU 训练）", "NVIDIA GPU，显存 ≥ 8 GB（本说明书运行环境为 NVIDIA A100 40GB）"],
    ["硬盘", "100 GB 可用空间", "1 TB 以上（单个 1000 帧双鱼实验原始数据约 440 MB）"],
], ["16%", "34%", "50%"])}
<h2>2.2　软件环境</h2>
{table(["项目", "要求"], [
    ["操作系统", "Linux（CentOS / Ubuntu 等，本说明书运行环境为 Linux 6.8 x86_64）"],
    ["运行环境", "Python 3.11；numpy、scipy、scikit-learn、pandas、opencv、antspyx、PyTorch（CUDA 12.1）、Flask"],
    ["客户端", "Chrome / Edge / Firefox 等现代浏览器，推荐分辨率 1600 × 1000 及以上"],
], ["20%", "80%"])}
<h2>2.3　软件组成与源程序量</h2>
<p>本系统由算法层与应用层组成：算法层包括图像配准（align）、信号与特征计算（feature_generation）、高斯核空间平滑（feature_modify）和 BrainNet 深度网络（Models/FC）；应用层（webapp）包括 Web 服务接口、业务服务模块与前端页面。源程序共 {len(stats)} 个文件，合计 <b>{src_total:,}</b> 行，其中 {"、".join(f"{k} {v:,} 行" for k, v in by_lang.items())}。</p>
{table(["层次", "目录 / 模块", "说明"], [
    ["算法层", "align/", "双鱼图像拆分旋转、ANTs 逐帧配准、模板配准、对齐矩阵生成"],
    ["算法层", "feature_generation/", "异常帧识别、钙信号预处理、IModPoly 基线、钙峰检测与特征计算"],
    ["算法层", "feature_modify/gaussian_kernel/", "脑区距离高斯核计算与特征空间平滑"],
    ["算法层", "Models/FC/", "BrainNet 网络结构、数据加载、训练测试与混淆矩阵"],
    ["应用层", "webapp/app.py、config.py", "Web 服务入口、REST 接口、用户认证、全局配置"],
    ["应用层", "webapp/services/", "实验管理、流水线、特征库、数据集、建模、任务管理、图像渲染、示例数据"],
    ["应用层", "webapp/static/", "登录页、主框架、16 个功能页面脚本与样式"],
], ["12%", "33%", "55%"])}

<h1>3　安装与启动</h1>
<p>本系统以源程序目录方式部署，部署与启动步骤如下：</p>
{steps("创建运行环境：<code>conda create -p ./envs/brainc python=3.11</code>，并安装依赖：<code>pip install numpy scipy scikit-learn pandas opencv-python-headless imutils antspyx flask torch</code>。",
       "（可选）生成示例数据：<code>python webapp/services/demo_data.py</code>，将在 data/demo 下生成标准脑模板、两个 1000 帧双鱼实验和 160 例脑区信号矩阵。",
       "启动服务：<code>cd webapp &amp;&amp; python app.py --host 0.0.0.0 --port 8765</code>，控制台显示 “Running on http://0.0.0.0:8765” 即启动成功。",
       "在浏览器地址栏输入 <code>http://服务器地址:8765</code>，进入系统登录页面。系统默认账户为 admin（管理员）与 analyst（分析员），首次登录后请在“系统设置”中修改密码。")}
<p>系统运行时产生的实验结果、特征库、数据集、模型和任务记录均保存在工作空间目录 workspace 下，原始数据只读访问，不会被修改。</p>

<h1>4　系统登录与界面布局</h1>
<h2>4.1　用户登录</h2>
<p>打开系统地址后显示登录页面（图 1）。页面左侧为系统名称及四大核心能力介绍，右侧为登录表单。输入用户名和密码后点击“登 录”按钮，系统校验通过后进入主界面；若用户名或密码错误，表单下方以红色文字提示“用户名或密码错误”。用户密码以加盐哈希方式存储，未登录用户访问任何功能页面或数据接口都会被重定向至登录页。</p>
{fig("01_login", "系统登录界面")}
<h2>4.2　主界面布局</h2>
<p>登录成功后进入系统主界面（图 2）。主界面由三部分组成：</p>
{ul("<b>左侧导航栏：</b>按“概览、数据管理、图像处理、信号分析、智能建模、系统管理”六个分组列出全部 16 个功能入口，当前所在功能以蓝色高亮显示。",
    "<b>顶部状态栏：</b>左侧显示当前位置（功能分组 / 功能名称），右侧依次显示后台任务运行状态（有任务运行时显示橙色闪烁指示灯及运行任务数，点击可进入任务中心）、系统时间、当前登录用户及“退出”按钮。",
    "<b>右侧工作区：</b>显示当前功能页面的内容，包括页面标题与说明、操作工具栏、参数表单、图表与数据表格等。")}

<h1>5　系统首页</h1>
<p>点击左侧导航栏“系统首页”进入首页（图 2、图 3），首页对工作空间内的数据与分析结果进行汇总展示：</p>
{ul("<b>统计卡片：</b>实验数据集数量与鱼体样本数、个体特征库样本数（对照组 / 药物处理组）、建模数据集与已训练模型数量、最佳模型准确率。",
    "<b>分析流程导航：</b>以流程图形式展示系统 11 个分析环节的先后关系，点击任一环节节点即可跳转到对应功能页面，便于按流程开展分析。",
    "<b>各分析环节完成情况：</b>柱状图统计完成数据导入、图像预处理、运动校正配准、脑区模板映射、钙信号提取、特征计算各环节的实验数量。",
    "<b>特征库样本构成：</b>环形图显示特征库中对照组与药物处理组的样本数量。",
    "<b>最近实验 / 最近任务：</b>列出最近创建的实验及其流程进度条，以及最近执行的后台任务及其状态和用时。")}
<p>首页右上角提供“导入实验”和“开始建模”快捷按钮，分别跳转至实验数据管理与模型训练页面。</p>
{fig("48_dashboard", "系统首页（数据概况与分析流程导航）")}
{fig("49_dashboard_bottom", "系统首页（环节完成情况、样本构成、最近实验与任务）")}

<h1>6　实验数据管理</h1>
<p>实验数据管理模块用于登记钙成像实验并管理原始 TIFF 帧序列。系统要求每个实验的帧序列存放于同一目录，文件名格式为 <code>img_000000xxx_Default_000.tif</code>（xxx 为帧号），每帧为 16 位灰度图像。双鱼并排成像时，单帧图像左右两半分别为两条鱼。</p>
<h2>6.1　实验列表</h2>
<p>点击导航栏“实验数据管理”进入实验列表页面（图 4）。列表显示每个实验的编号与描述、成像布局、帧数、图像尺寸、鱼体分组、钙指示剂、流程进度、操作员与创建时间；右侧“操作”列提供“详情”“一键分析”“删除”按钮。列表上方检索框可按实验编号或描述关键字实时过滤。尚未导入任何实验时，列表显示空状态提示。</p>
{fig("02_exp_empty", "实验数据管理页面（初始状态）")}
<h2>6.2　导入新实验</h2>
<p>点击页面右上角“＋ 导入新实验”按钮，弹出“导入新实验”对话框，操作步骤如下：</p>
{steps("在“原始数据目录”输入框中填写数据所在目录，或点击“浏览”按钮打开“选择目录”对话框（图 5）。对话框顶部显示当前路径，列表中列出子目录及其中包含的数据文件数，点击目录进入下一级，点击“上级目录”返回，选定后点击“确定”。",
       "点击“扫描”按钮，系统递归扫描该目录，列出所有包含帧序列的子目录及帧数；帧数大于 800 的序列标记为“完整”，否则标记为“帧数不足”。点击列表中的某一序列将其选中（浅蓝色高亮），系统自动以上级目录名填充实验编号。",
       "选择成像布局（双鱼并排 / 单鱼），设置左侧鱼与右侧鱼分组（对照组 / 药物处理组），填写钙指示剂、采样帧率和实验描述（图 6）。若数据目录中含有采集信息文件 acquisition.json，系统将自动读取其中的鱼体分组信息。",
       "点击“创建实验”按钮，系统读取首帧获取图像尺寸与位深，统计数据量，生成实验记录并自动跳转至实验详情页。")}
{fig("03_dir_picker", "选择原始数据目录")}
{fig("04_import_wizard", "导入新实验对话框（扫描结果与实验信息填写）")}
<h2>6.3　实验详情与帧序列浏览</h2>
<p>实验详情页（图 7）顶部以步骤条显示该实验 6 个分析环节的完成情况，已完成环节显示绿色对勾，点击任一步骤可直接进入对应页面。页面包含以下区域：</p>
{ul("<b>基本信息：</b>实验编号、数据目录、成像布局、帧数及帧号范围、图像尺寸与位深、数据量、实验对象、钙指示剂、采样帧率、鱼体编号与分组、操作员和创建时间。",
    "<b>帧序列浏览：</b>拖动下方帧号滑块可逐帧浏览原始图像，右上角“伪彩色”下拉框可选择灰度、Inferno、Viridis、Turbo、Hot 等显示色表；右侧同步显示当前帧文件名、尺寸类型、灰度范围、均值 ± 标准差、显示窗宽以及对数坐标灰度直方图。",
    "<b>全时程平均荧光强度：</b>以每 5 帧采样计算左、右鱼区域平均灰度曲线（图 8），曲线上的骤降点通常对应鱼体甩动等异常帧，可用于快速评估数据质量。",
    "<b>相关任务记录：</b>列出针对该实验执行过的后台任务及其状态和用时。")}
<p>点击页面右上角“编辑信息”可修改实验名称、钙指示剂、各鱼分组和实验描述；点击“一键全流程分析”则以默认参数依次自动执行图像预处理确认、运动校正配准、脑区模板映射、钙信号提取和特征计算，并跳转至任务中心查看进度。</p>
{fig("05_exp_detail", f"实验详情页（帧序列浏览，Inferno 伪彩色，第 500 帧）")}
{fig("06_exp_profile", f"帧序列浏览（第 {cap['fly_frame']} 帧，甩动异常帧）与全时程平均荧光强度曲线")}
<p>图 8 中第 {cap['fly_frame']} 帧为一个甩动异常帧：鱼体发生大幅位移与离焦，图像模糊、整体亮度下降，对应全时程曲线上的骤降点。此类帧将在后续钙信号提取中被自动识别、剔除并插值。</p>

<h1>7　标准脑模板库</h1>
<p>标准脑模板库用于管理脑区映射所需的标准脑模板与脑区分割图谱。点击导航栏“标准脑模板库”进入页面（图 9），顶部统计卡片显示模板尺寸（{450} × {250} 像素）、脑区数量（100 个，左右半脑对称各 50 个）、解剖大区（端脑、视顶盖/间脑、小脑、后脑）及平均脑区面积。中部并列显示标准脑模板图像 template.tif、100 脑区分割图谱 atlas_100.png（每个脑区以不同颜色表示、黑色细线为脑区边界）以及二者的叠加效果。</p>
{fig("07_templates", "标准脑模板库（模板图像、脑区图谱与叠加显示）")}
<p>页面下部（图 10）左侧为脑区信息表，列出每个脑区的编号、名称（L-xx 表示左半脑、R-xx 表示右半脑）、所属解剖大区、面积及中心坐标；右侧散点图显示 100 个脑区中心的空间分布，点的大小与脑区面积成正比、颜色区分解剖大区，鼠标悬停可查看脑区编号与坐标。脑区中心坐标同时用于后续高斯核空间平滑。</p>
{fig("08_templates_table", "脑区信息表与脑区中心空间分布")}

<h1>8　图像预处理</h1>
<p>图像预处理对原始帧进行左右拆分、旋转摆正与居中裁剪，得到标准化的单鱼脑图像，是后续配准的基础。点击导航栏“图像预处理”进入页面（图 11），在页面顶部工具栏“当前实验”下拉框中选择待处理实验（本例为 EXP20260315_A），工具栏右侧标签显示该实验的流程进度。</p>
<p>页面左侧为“预处理参数”面板，参数含义如下：</p>
{table(["参数", "默认值", "说明"], [
    ["预览帧号", "500", "用于预览预处理效果的帧"],
    ["参考帧号", "500", "运动校正配准的固定（参考）图像帧"],
    ["左/右侧鱼旋转角度", "26° / −26°", "将倾斜放置的鱼体旋转至头部朝上；旋转后自动扩展画布以避免裁切"],
    ["裁剪高度 × 宽度", "450 × 250", "以鱼体中心为中心裁剪的区域尺寸，与标准脑模板尺寸一致"],
    ["鱼体分割阈值", "3", "16 位图像线性映射至 8 位后大于该值的像素视为鱼体，用于计算鱼体中心"],
], ["26%", "18%", "56%"])}
<p>修改参数后点击“预览效果”，右侧依次显示：① 原始帧与左右拆分线（黄色竖线）；② 每条鱼旋转摆正后的图像，其中绿色曲线为阈值分割得到的鱼体轮廓、红色十字为鱼体中心、橙色矩形为裁剪区域；③ 裁剪结果（伪彩色显示）。每条鱼下方列出旋转后图像尺寸、鱼体中心坐标、鱼体像素数和裁剪框坐标。确认效果满意后点击“确认参数”，系统保存参数并将该实验“图像预处理”环节标记为完成，参数面板标题旁标签由“未确认”变为“已确认”（图 12）。</p>
{fig("09_preprocess", "图像预处理（参数设置与原始帧拆分预览）")}
{fig("10_preprocess_apply", "图像预处理（旋转摆正、中心定位与裁剪结果）")}

<h1>9　运动校正配准</h1>
<p>成像过程中鱼体存在缓慢漂移与形变，必须对每一帧进行配准校正，使同一脑区在全时程中位于相同像素位置。点击导航栏“运动校正配准”进入页面，左侧“配准参数”面板可选择配准变换模型：ElasticSyN（仿射 + 弹性对称归一化，默认）、SyN（仿射 + 对称微分同胚）、Affine（仿射）、Rigid（刚体）；配准降采样尺寸（100 × 80）及多分辨率平滑 σ / 收缩因子为算法内置参数；“并行进程数”用于设置同时配准的进程数量。</p>
<h2>9.1　执行配准</h2>
<p>点击“开始配准”按钮，系统以参考帧为固定图像、逐帧作为浮动图像执行 ANTs 配准，每帧每条鱼输出一个复合变换文件（Composite.h5），随后将变换应用于全分辨率图像、按参考帧鱼体中心裁剪并写入对齐时空矩阵（帧数 × 像素数）。按钮下方出现任务进度框（图 13），实时显示任务名称、状态、当前进度说明（如“已配准 n/1000 帧”）、百分比、已用时及运行日志。本例 {reg["frames"]} 帧 × 2 条鱼采用 {reg["transform"]} 算法、48 进程并行，共用时 {reg["seconds"]} 秒。</p>
{fig("11_register_running", "运动校正配准（任务运行中，实时进度与日志）")}
<h2>9.2　配准结果查看</h2>
<p>配准完成后右侧显示结果（图 14）：顶部指标卡显示左、右侧鱼全部帧与参考帧的平均相关系数（配准前 → 配准后），本例左侧鱼由 {reg["summary"]["l"]["corr_before"]} 提高至 {reg["summary"]["l"]["corr_after"]}，右侧鱼由 {reg["summary"]["r"]["corr_before"]} 提高至 {reg["summary"]["r"]["corr_after"]}。“逐帧配准质量”曲线以虚线表示配准前、实线表示配准后各帧相关系数，配准后曲线整体抬升并趋于平稳；个别向下尖峰为甩动异常帧。</p>
{fig("12_register_result", "运动校正配准结果（指标与逐帧配准质量曲线）")}
<p>点击曲线上任意位置，或在“单帧配准效果对比”卡片右上角输入帧号后点击“查看”，可查看该帧的配准效果（图 15）。对比图采用品红色表示参考帧、绿色表示当前帧，二者重合处呈灰白色，错位越大彩色边缘越明显：配准前叠加图在鱼脑边缘存在明显的品红/绿色错位，配准后错位基本消除；棋盘格对比图将参考帧与配准后图像交替拼接，边界连续说明对齐良好；时间标准差投影显示配准后全时程像素波动分布。图 15 为本例中配准前相关系数最低的第 {cap["worst_frame"]} 帧。</p>
{fig("13_register_compare", f"单帧配准效果对比（第 {cap['worst_frame']} 帧）")}
<p>页面底部“时间平均投影”（图 16）对比配准前后全时程平均图像：配准前由于鱼体缓慢漂移，平均图像边缘存在轻微拖影；配准后边界更加清晰，表明运动伪影被有效消除。</p>
{fig("14_register_projection", "配准前后时间平均投影对比")}

<h1>10　脑区模板映射</h1>
<p>脑区模板映射将标准脑模板配准到个体鱼脑，并将 100 脑区图谱映射到个体空间，实现每条鱼脑区的自动标注。点击导航栏“脑区模板映射”进入页面，在工具栏中选择标准脑模板（配准算法 ElasticSyN，标签插值采用最近邻以保持脑区编号不变），点击“开始映射”。系统以模板为固定图像、个体鱼参考帧为浮动图像进行配准，生成正向变换 Composite.h5 与逆变换 InverseComposite.h5，再利用逆变换将脑区图谱变形到个体空间（图 17 为任务运行状态）。</p>
{fig("15_atlas_running", "脑区模板映射（任务运行中）")}
<p>映射完成后，每条鱼显示一组结果（图 18、图 19）：个体参考图像；映射前轮廓（黄色为个体脑轮廓、品红色为模板轮廓）；模板变形结果；映射后轮廓；个体脑区标注（彩色脑区叠加于个体图像）。卡片标题右侧标签给出映射前后脑轮廓 Dice 系数和覆盖脑区数，本例左侧鱼 Dice 由 {atl["l"]["dice_before"]} 提高到 {atl["l"]["dice_after"]}、右侧鱼由 {atl["r"]["dice_before"]} 提高到 {atl["r"]["dice_after"]}，100 个脑区全部被覆盖。下方柱状图给出各脑区在个体空间中的映射像素数，可用于检查是否存在异常缩小或缺失的脑区。</p>
{fig("16_atlas_result", f"脑区模板映射结果（{L}）")}
{fig("17_atlas_result_r", f"脑区模板映射结果（{R}）与各脑区映射像素数")}

<h1>11　钙信号提取</h1>
<p>钙信号提取模块基于映射得到的脑区标签，从对齐时空矩阵中提取 100 个脑区的钙信号并进行预处理。页面顶部工具栏除“当前实验”外还提供“鱼体位置”下拉框，用于切换查看左侧鱼或右侧鱼的结果。处理链以标签形式显示在操作区：</p>
{steps("<b>异常帧识别：</b>对鱼体像素逐像素计算时间维 20%/80% 分位数及四分位距，低于 Q20 − 1.5·IQR 的像素数超过 1000 的帧判定为甩动异常帧并剔除；",
       "<b>异常像素剔除与死鱼判定：</b>计算每个像素在全时程中的极差，极差大于 500 的像素置零剔除；若极差大于 1000 的像素超过 10000 个，判定为死鱼样本，不再计算特征；",
       "<b>脑区平均荧光：</b>由脑区标签构建稀疏指示矩阵，与时空矩阵相乘得到各脑区平均荧光时间序列；",
       "<b>低通滤波：</b>30 阶 Butterworth 低通滤波器（归一化截止频率 0.3）零相位滤波去除高频噪声；",
       "<b>线性插值：</b>在剔除的异常帧位置插入空值并线性插值，恢复完整时间轴；",
       "<b>基线估计：</b>IModPoly 迭代修正多项式拟合（6 次）估计荧光基线，消除光漂白造成的缓慢下降；",
       "<b>ΔF/F 归一化：</b>两端各裁剪 10 帧后计算 (F − F0) / F0，得到 100 脑区 × 960 帧的钙信号矩阵。")}
<p>点击“开始提取”后显示任务进度（图 20）。</p>
{fig("18_signal_running", "钙信号提取（任务运行中）")}
<p>提取完成后显示结果（图 21）：指标卡给出异常帧数、剔除异常像素数、死鱼判定、有效分析帧数与 ΔF/F 均值/最大值。本例左侧鱼自动识别出 {len(sig["l"]["fly_frames"])} 个甩动帧（第 {"、".join(map(str, sig["l"]["fly_frames"]))} 帧），{fly_note}；剔除异常像素 {sig["l"]["outlier_pixels"]:,} 个，判定为存活。“鱼体平均荧光与异常帧”曲线以红点标出被识别的甩动帧；“像素时程极差分布”直方图中红色部分为超过 500 的异常像素。</p>
{fig("19_signal_result", "钙信号提取结果（质量指标、异常帧与像素极差分布）")}
<p>“脑区信号处理过程”卡片可在右上角输入多个脑区编号（逗号分隔）后点击“更新”，并通过下拉框切换显示处理阶段：原始平均荧光、低通滤波后、插值 + 基线、ΔF/F 归一化。图 22 为“插值 + 基线”阶段，实线为滤波插值后的荧光信号、同色虚线为 IModPoly 估计的基线；图 23 为 ΔF/F 归一化结果，基线漂移已去除，钙瞬变清晰可见。图表下方滑块和鼠标滚轮可缩放时间范围。</p>
{fig("20_signal_traces", "脑区信号处理过程（插值后信号与 IModPoly 基线）")}
{fig("21_signal_dff", "脑区信号处理过程（ΔF/F 归一化结果）")}
<p>页面下部（图 24）为全脑 ΔF/F 活动栅格图，纵轴为脑区 1～100、横轴为时间帧，亮色表示钙活动强，可直观观察跨脑区的同步放电事件；“各脑区 ΔF/F 统计”图给出每个脑区的平均与最大 ΔF/F。</p>
{fig("22_signal_raster", "全脑 ΔF/F 活动栅格图与各脑区统计")}

<h1>12　神经活动特征</h1>
<p>神经活动特征模块在 ΔF/F 信号上检测钙峰并计算特征。钙峰检测以信号四分位距自适应确定高度阈值（Q3 + 2·IQR）、最小峰宽 5 帧，并通过一阶差分符号变化定位每个峰的左右拐点（谷点）。系统计算的 13 类特征如下：</p>
{table(["特征", "含义"], [
    ["钙峰数量", "各脑区检测到的钙峰个数，反映放电频率"],
    ["峰面积总和 / 平均峰面积", "以峰顶与左右拐点连线构成的三角形近似计算单峰面积，反映放电总量"],
    ["平均持续时间", "钙峰左右拐点间的平均帧数"],
    ["占空比", "全部钙峰持续时间之和占记录时长的比例"],
    ["最大峰高 / 平均峰高", "钙峰幅值的最大值与均值，反映放电强度"],
    ["平均动作电位强度", "峰高与持续时间之比的均值，反映钙瞬变上升陡峭程度"],
    ["平均峰间间隔 / 平均峰距", "相邻钙峰前峰右拐点到后峰左拐点、以及相邻峰顶之间的平均帧数，反映放电节律"],
    ["放电同步连接矩阵", "以全脑出现过钙峰的时刻为列构建脑区放电 0/1 矩阵，计算脑区间相关系数（100 × 100）"],
    ["峰值强度相关矩阵", "以峰值强度替代 0/1 计算的脑区间相关矩阵"],
    ["逐帧活跃脑区数", "每一帧出现钙峰的脑区数量，反映全脑同步事件"],
], ["30%", "70%"])}
<p>点击“开始计算”（图 25）后，系统对每条鱼计算上述特征并以 pkl 格式保存。计算完成后（图 26），顶部指标卡显示全脑钙峰总数、平均每脑区钙峰数、平均占空比、平均峰高和全脑同步指数（同步连接矩阵上三角均值）；本例左侧鱼全脑共检测到 {fea["l"]["total_peaks"]} 个钙峰。“钙峰检测结果”图以红色三角标记某一脑区检测到的钙峰位置，可通过右上角下拉框切换脑区。</p>
{fig("23_feature_running", "神经活动特征计算（任务运行中）")}
{fig("24_feature_result", "神经活动特征结果（特征概览与钙峰检测）")}
<p>“特征脑区分布图”（图 27 左）将所选特征值以 Turbo 色表绘制在个体脑区标注图上，蓝色表示低、红色表示高，可通过下拉框在 10 类标量特征间切换（图中为“峰面积总和”）；“脑区间放电同步连接矩阵”（图 27 右）以热图显示 100 × 100 相关系数，红色为正相关、蓝色为负相关，可切换为“峰值强度相关”矩阵，鼠标悬停显示脑区对与数值。</p>
{fig("25_feature_maps", "特征脑区分布图与脑区间同步连接矩阵")}
<p>页面下部（图 28）为“逐帧活跃脑区数”柱状图与“脑区特征明细表”（100 脑区 × 10 类标量特征）。点击操作区“导出特征 CSV”可下载当前鱼的脑区特征表；点击“加入个体特征库”可将该实验各鱼的特征样本加入个体特征库，成功后右上角提示“已加入特征库”（图 29）。</p>
{fig("26_feature_table", "逐帧活跃脑区数与脑区特征明细表")}
{fig("27_feature_tolib", "将实验样本加入个体特征库")}

<h1>13　个体特征库</h1>
<p>个体特征库汇集每条鱼的特征作为建模样本。样本有两种来源：一是“实验流水线”，即在神经活动特征页面加入的实验样本；二是“信号矩阵导入”，即直接导入已提取的脑区荧光信号矩阵（脑区 × 时间的 .npy 文件），由系统批量完成异常帧识别、滤波插值、基线校正、ΔF/F 与特征计算。</p>
<h2>13.1　批量导入信号矩阵</h2>
{steps("点击页面右上角“＋ 批量导入信号矩阵”，弹出导入对话框；",
       "填写或浏览选择信号矩阵所在目录，点击“扫描”。系统列出发现的矩阵文件、样本编号、矩阵形状与分组；若目录中存在 labels.csv（含 fish_id、label 列），则自动读取分组标注（图 30）；",
       "根据需要勾选“覆盖已存在样本”，设置并行进程数，点击“开始分析”。页面顶部显示批量分析任务进度与日志（图 31），本例 160 个样本使用 48 进程并行，约 20 秒完成。")}
{fig("28_library_import", "批量导入脑区信号矩阵（扫描结果）")}
{fig("29_library_running", "批量分析任务运行中")}
<h2>13.2　组间统计与样本管理</h2>
<p>导入完成后（图 32），顶部指标卡显示样本总数（本例 {lib_n - 2} 例：160 例信号矩阵导入样本 + 2 例实验流水线样本）、对照组/药物处理组数量、样本来源构成及组间显著差异特征数。“组间特征对比”以箱线图显示所选特征（脑区平均值）在两组间的分布；“组间差异检验”表对 10 类特征进行 Welch t 检验，列出两组均值、相对变化、p 值和显著性标记（*** p &lt; 0.001，** p &lt; 0.01，* p &lt; 0.05，n.s. 不显著）。</p>
{fig("30_library_stats", "个体特征库（样本统计与组间差异检验）")}
<p>“样本列表”（图 33）显示每个样本的编号、分组、来源、异常帧数、钙峰总数、平均峰面积、平均占空比、平均峰高、同步指数和入库时间，可按分组筛选或按编号检索，勾选后点击“删除选中”可批量删除样本；点击页面右上角“导出特征库 CSV”可导出全部样本的特征汇总表。</p>
{fig("31_library_list", "个体特征库样本列表")}
<p>点击样本行的“查看”按钮弹出样本详情（图 34），显示该样本的异常帧列表、若干脑区的 ΔF/F 曲线、各脑区钙峰数量柱状图和放电同步连接矩阵；如分组标注有误，可在右上角下拉框中修改分组并点击“修改分组”。</p>
{fig("32_library_detail", "样本详情")}

<h1>14　高斯核空间平滑</h1>
<p>由于个体间配准误差，相邻脑区的特征存在边界噪声。本模块依据脑区中心坐标计算两两欧氏距离 d<sub>ij</sub>，以各脑区最近邻距离乘带宽系数 H 作为核宽度 σ，构造高斯核 K<sub>ij</sub> = exp(−d<sub>ij</sub>² / 2σ<sub>j</sub>²) 并按列归一化，平滑后的特征为 X' = K · X。</p>
<p>点击导航栏“高斯核空间平滑”进入页面（图 35），拖动“带宽系数 H”滑块（0.1～2.0）、输入中心脑区编号、选择示例样本与特征后点击“计算”。指标卡显示带宽系数、平均自身权重和平均有效参与脑区数；左侧图显示中心脑区对全脑各脑区的核权重分布（颜色越深权重越大），右侧热图为 100 × 100 高斯核矩阵。H 越大，核越宽，平滑程度越高。</p>
{fig("33_kernel", "高斯核空间平滑（H = 1.0 的核权重分布与核矩阵）")}
<p>页面下部（图 36）对比示例样本某一特征在平滑前、后的脑区分布图（相同色标），并以折线图给出 100 个脑区平滑前后的特征值，平滑后曲线更加连续，局部异常值被抑制。</p>
{fig("34_kernel_compare", "平滑前后特征脑区分布对比")}

<h1>15　建模数据集</h1>
<p>建模数据集模块将特征库样本组织为深度学习输入张量。每个样本的每类特征按 100 个脑区展开为特征单元，多种归一化与空间平滑方式构成输入通道，最终张量形状为“样本数 × 特征单元数 × 通道数”。</p>
{steps("在左侧“新建数据集”面板填写数据集名称；",
       "选择样本范围（特征库全部样本 / 仅信号矩阵导入样本）；",
       "勾选特征类型（默认全选 10 类标量特征）；",
       "勾选输入通道：原始特征值、Min-Max 归一化、Z-Score 标准化、高斯核平滑以及高斯核平滑后的 Min-Max / Z-Score，默认选用 4 个归一化通道；",
       "设置高斯核带宽系数 H（默认 0.5）和描述，点击“构建数据集”。")}
<p>构建完成后数据集出现在右侧列表中，显示编号、名称、样本数（对照/药物）、张量形状、通道数、H 与创建时间。点击“预览”在下方显示数据集详情（图 37）：样本数、特征单元、通道数、数据量，PCA 二维投影散点图（绿色为对照组、橙色为药物处理组，标题给出前两个主成分的方差解释率）、特征类型与输入通道标签以及类别构成饼图。本例数据集 {dataset["id"]} 共 {dataset["samples"]} 个样本，张量形状 {" × ".join(map(str, dataset["shape"]))}。</p>
{fig("35_dataset", "建模数据集（新建、列表与 PCA 预览）")}

<h1>16　模型训练</h1>
<p>本系统采用 BrainNet 深度网络进行对照 / 药物处理（成瘾）表型二分类。网络结构在“模型训练”页面右侧以示意图展示（图 38）：输入张量中的每个特征单元由独立的 Cell 子网络（通道数 → 32 → 16 → 1，含 BatchNorm、ReLU、Dropout）编码为一个标量，全部标量拼接后依次经过 4096、2048、1024、512 维全连接层（BatchNorm + ReLU，前两层含 Dropout）输出 2 类结果。</p>
<p>左侧“训练配置”面板可设置模型名称、训练数据集、交叉验证折数、训练轮数、优化器（SGD / Adam）、学习率、动量、权重衰减、计算设备（GPU / CPU）和随机种子，损失函数为交叉熵。点击“开始训练”后，系统按 K 折交叉验证训练：每折重置网络参数，每轮记录训练与验证的损失和准确率，并保存该折验证准确率最高的轮次结果。</p>
{fig("36_train_config", "模型训练（训练配置与网络结构）")}
<p>训练过程中“训练监控”卡片实时显示任务进度、当前折与轮次、训练/验证准确率和日志，下方两图实时绘制各折训练（虚线）与验证（实线）的损失曲线和准确率曲线（图 39）。训练结束后，系统对最佳模型进行 BatchNorm 统计量重校准（关闭 Dropout 后以全量样本重估均值与方差），保存模型参数并计算梯度显著性，任务提示框出现“查看模型评估”链接（图 40）。本例 {model["hyper"]["k_folds"]} 折 × {model["hyper"]["epochs"]} 轮训练在 GPU 上用时 {round(model["seconds"] / 60, 1)} 分钟，模型参数量 {model["params"]:,}。</p>
{fig("37_train_running", "模型训练过程实时监控")}
{fig("38_train_done", "模型训练完成")}

<h1>17　模型评估</h1>
<p>点击导航栏“模型评估”，页面上方列出全部已训练模型（编号、名称、数据集、参数量、折数/轮数、准确率、精确率、召回率、AUC、训练用时和创建时间），点击“评估详情”在下方显示该模型的完整评估结果（图 41）：</p>
{ul(f"<b>性能指标：</b>交叉验证平均准确率 {mm['accuracy']}%、精确率 {mm['precision']}%、召回率 {mm['recall']}%、F1 分数 {mm['f1']}%、AUC {mm['auc']}；",
    f"<b>混淆矩阵：</b>汇总各折最佳轮次的验证结果，本例真阴性 {mm['confusion']['tn']}、假阳性 {mm['confusion']['fp']}、假阴性 {mm['confusion']['fn']}、真阳性 {mm['confusion']['tp']}；",
    "<b>ROC 曲线：</b>以各折验证样本的药物处理概率绘制，曲线下面积即 AUC；",
    "<b>各折验证准确率：</b>柱状图给出每折最佳验证准确率，橙色虚线为平均值。")}
{fig("39_evaluate", "模型评估（模型列表、性能指标、混淆矩阵与 ROC 曲线）")}
<p>可解释性分析部分（图 42）基于输入梯度显著性：计算药物处理概率对输入张量的梯度绝对值，在样本与通道维平均，得到“特征类型 × 脑区”重要性矩阵。“特征类型重要性”横向柱状图对 10 类特征排序；“脑区重要性分布图”将各脑区重要性绘制在标准脑图谱上（颜色越深越重要）；“特征类型 × 脑区显著性热图”展示二者的联合分布，可辅助定位与药物成瘾表型相关的脑区和神经活动模式。</p>
{fig("40_evaluate_importance", "模型可解释性分析（特征类型与脑区重要性）")}
<p>页面底部（图 43）给出各折验证准确率随训练轮次的变化曲线，以及模型信息：模型编号、网络结构、参数量、训练数据集、输入通道、优化器参数、交叉验证设置、计算设备、训练用时和各折明细。在模型列表中点击“删除”可删除不再需要的模型。</p>
{fig("41_evaluate_info", "训练过程曲线与模型信息")}

<h1>18　智能预测</h1>
<p>智能预测模块调用已训练模型对特征库中的样本进行表型判别。预测时系统按照模型训练数据集相同的方式（同一归一化参考样本、同一高斯核带宽）为待测样本构建输入张量，输出属于药物处理表型的概率，概率 ≥ 0.5 判定为药物处理表型。</p>
{steps("在“预测模型”下拉框中选择模型（显示名称与准确率）；",
       "在“待测样本”列表中勾选样本，可使用“全选”“仅实验流水线样本”“随机 20 例”“清空”快捷按钮；",
       "点击“开始预测”。")}
<p>预测结果（图 44）包括：预测样本数、判定为药物处理/对照表型的数量和与已知分组一致率；“预测概率分布”柱状图（柱色表示已知分组，红色虚线为 0.5 判定阈值）；“预测结果明细”表列出样本编号、来源（标注“训练集”或“新样本”）、已知分组、药物处理概率条、预测结果与是否一致（图 45）。本例共预测 {pred["count"]} 个样本，其中第 2 批实验 EXP20260315_B 的两条鱼在数据集构建之后才加入特征库，属于模型未见过的新样本；预测结果与已知分组一致 {pred_ok} 例。点击“导出 CSV”可下载预测结果。</p>
{fig("42_predict", "智能预测（预测设置与概率分布）")}
{fig("43_predict_table", "预测结果明细")}

<h1>19　任务中心</h1>
<p>系统中所有耗时计算（配准、模板映射、信号提取、特征计算、批量分析、数据集构建、模型训练、一键全流程分析等）均作为后台任务运行，用户可在任务运行期间继续浏览其他页面。点击导航栏“任务中心”（图 46），顶部统计任务总数及运行中、已完成、失败数量；左侧任务列表显示任务编号、类型、名称、状态、进度条、开始时间和用时，可按类型筛选；点击“查看”在右侧显示任务详情，包括实时进度与完整运行日志（图中为第 2 批实验“一键全流程分析”任务的日志，依次记录了配准、模板映射、信号提取和特征计算 4 个步骤的执行情况与关键结果）。已结束的任务可点击“删除”清除记录。</p>
{fig("44_tasks", "任务中心（任务列表与运行日志）")}

<h1>20　系统设置</h1>
<p>点击导航栏“系统设置”进入页面（图 47）。上部为各模块默认参数设置：图像预处理（旋转角、参考帧、裁剪尺寸、鱼体阈值）、配准（变换模型、降采样尺寸、并行进程数）、模型训练（折数、轮数、学习率、动量、权重衰减、优化器）及高斯核带宽系数，修改后点击右上角“保存设置”生效，新导入的实验将采用新的默认参数；“信号处理参数”卡片说明算法内置的滤波、基线估计、边界裁剪和异常判据参数。</p>
{fig("45_settings", "系统设置（默认参数配置）")}
<p>页面下部（图 48）显示软件信息（名称、版本号、源程序总量及各语言行数、源文件数）、运行环境（操作系统、Python 版本、CPU 核数、PyTorch 与 GPU、各依赖库版本、工作空间路径与占用空间）以及“修改密码”表单：输入原密码和两次新密码后点击“修改密码”即可。最下方“源程序文件清单”列出全部源文件及行数。</p>
{fig("46_settings_env", "软件信息、运行环境与修改密码")}

<h1>21　典型分析流程</h1>
<p>完成上述各功能的操作后，实验列表中各实验的流程进度均为 6/6（图 49）。综合以上各章，使用本系统完成一次完整分析的典型流程如下：</p>
{steps("在“实验数据管理”中导入实验 TIFF 帧序列并登记鱼体分组；",
       "在“图像预处理”中预览并确认拆分、旋转与裁剪参数；",
       "依次执行“运动校正配准”“脑区模板映射”“钙信号提取”“神经活动特征”（或在实验列表中点击“一键分析”自动完成），检查各步骤质量指标；",
       "在“神经活动特征”页将样本加入个体特征库，或在“个体特征库”中批量导入已提取的脑区信号矩阵；",
       "在“高斯核空间平滑”中确定合适的带宽系数，在“建模数据集”中构建数据集；",
       "在“模型训练”中训练 BrainNet 模型，在“模型评估”中查看性能与重要脑区；",
       "在“智能预测”中对新样本进行表型判别并导出结果。")}
{fig("47_exp_list", "实验列表（全部分析流程完成）")}
<h2>21.1　常见问题</h2>
{table(["问题", "处理方法"], [
    ["扫描目录时提示“未发现帧序列”", "检查文件名是否符合 img_帧号_*.tif 格式，且文件直接位于所选目录或其子目录中"],
    ["预处理裁剪框偏离鱼体", "调整鱼体分割阈值或旋转角度后重新预览；确保参考帧不是甩动帧"],
    ["配准后相关系数提升不明显", "尝试 SyN 或 ElasticSyN 变换模型；检查参考帧图像质量"],
    ["信号提取判定为死鱼", "检查原始数据全时程强度曲线是否存在大面积剧烈变化；死鱼样本不参与特征计算"],
    ["训练提示 GPU 不可用", "在计算设备中选择 CPU，或检查 CUDA 驱动与 PyTorch 版本"],
    ["页面提示“该对象已有正在运行的任务”", "同一实验同时只能运行一个任务，请在任务中心等待当前任务完成"],
], ["36%", "64%"])}
""")

CSS = """
@page { size: A4; margin: 24mm 18mm 20mm 18mm; }
body { font-family: 'Noto Serif CJK SC', 'Source Han Serif SC', 'SimSun', serif; font-size: 10.5pt; line-height: 1.75; color: #111; }
h1, h2, h3, .cover-title, .cover-sub, th, figcaption, .toc { font-family: 'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei', sans-serif; }
h1 { font-size: 15pt; margin: 18pt 0 8pt; page-break-after: avoid; border-bottom: 1.2pt solid #1f4e9a; padding-bottom: 3pt; color: #13325f; }
h2 { font-size: 12pt; margin: 12pt 0 5pt; page-break-after: avoid; color: #1d3f75; }
p { margin: 4pt 0; text-indent: 2em; text-align: justify; }
p.center { text-indent: 0; text-align: center; }
.small { font-size: 9.5pt; color: #444; }
ul, ol { margin: 3pt 0 5pt; padding-left: 2.4em; }
li { margin: 2pt 0; text-align: justify; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f2f4f7; padding: 0 3pt; border-radius: 2pt; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 8pt; font-size: 9.5pt; line-height: 1.5; }
th, td { border: 0.6pt solid #8a96a8; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #e6edf7; font-weight: 600; }
figure { margin: 6pt 0 8pt; text-align: center; page-break-inside: avoid; }
figure img { width: FIGW; border: 0.6pt solid #b8c2d0; }
figcaption { font-size: 9.5pt; margin-top: 3pt; color: #222; }
.cover-title { text-align: center; font-size: 20pt; font-weight: 700; color: #13325f; margin-top: 4pt; line-height: 1.5; }
.cover-title span { font-size: 15pt; }
.cover-sub { text-align: center; font-size: 16pt; letter-spacing: 6pt; margin: 4pt 0 10pt; }
.toc { font-size: 10.5pt; line-height: 1.62; page-break-after: always; }
.toc-head { text-align: center; font-size: 14pt; font-weight: 700; letter-spacing: 8pt; margin: 6pt 0 6pt; }
.toc-item { display: flex; align-items: baseline; }
.toc-item.l1 { font-weight: 600; margin-top: 1pt; }
.toc-item.l2 { padding-left: 2em; }
.toc-item .dots { flex: 1; border-bottom: 1pt dotted #777; margin: 0 5pt; transform: translateY(-3pt); }
""".replace("FIGW", "90%" if JB else "100%")
if JB:
    CSS += ".toc { page-break-after: auto; margin-bottom: 10pt; }"   # 目录后正文接续排版，避免出现行数不足的页

DOC_NAME = "用户操作手册" if JB else "操作说明书"
if JB:
    HEADER = f"""<div style="width:100%;font-size:9pt;font-family:'Noto Sans CJK SC',sans-serif;color:#222;margin:0 18mm;
  border-bottom:0.5pt solid #444;padding-bottom:3pt;display:flex;justify-content:space-between;">
  <span>{TITLE}</span><span>第 <span class="pageNumber"></span> 页　共 <span class="totalPages"></span> 页</span></div>"""
else:
    HEADER = f"""<div style="width:100%;font-size:8.5pt;font-family:'Noto Sans CJK SC',sans-serif;color:#333;margin:0 18mm;
  border-bottom:0.5pt solid #555;padding-bottom:3pt;display:flex;justify-content:space-between;">
  <span>{TITLE}　　操作说明书</span><span>第 <span class="pageNumber"></span> 页 / 共 <span class="totalPages"></span> 页</span></div>"""
FOOTER = "<div></div>"

BODY = "".join(H)
if JB:
    BODY = BODY.replace("本说明书", "本手册")
HEADS = [(int(m.group(1)), re.sub(r"<[^>]+>", "", m.group(2))) for m in re.finditer(r"<h([12])>(.*?)</h\1>", BODY)]


def toc_html(pages):
    items = "".join(f'<div class="toc-item l{lv}"><span>{t}</span><span class="dots"></span><span>{pages[k]}</span></div>'
                    for k, (lv, t) in enumerate(HEADS))
    return (f'<div class="cover-title">{SOFTWARE_NAME}<br><span>{VERSION}</span></div>'
            f'<div class="cover-sub">{" ".join(DOC_NAME)}</div>'
            f'<div class="toc"><div class="toc-head">目录</div>{items}</div>')


def render(pages, out_pdf):
    html = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{TITLE} {DOC_NAME}</title>"
            f"<style>{CSS}</style></head><body>{BODY.replace('<!--TOC-->', toc_html(pages))}</body></html>")
    out_html = Path(tempfile.mkdtemp()) / "doc.html"
    out_html.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = b.new_page()
        pg.goto(out_html.as_uri())
        pg.wait_for_timeout(1500)
        pg.pdf(path=str(out_pdf), format="A4", display_header_footer=True, header_template=HEADER, footer_template=FOOTER,
               margin={"top": "24mm", "bottom": "18mm", "left": "18mm", "right": "18mm"}, print_background=True)
        b.close()


def page_texts(pdf):
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    return out.split("\f")


def locate_headings(pdf):
    """按顺序查找各级标题所在页码：正文标题行仅含标题文字，目录行末尾带页码，据此区分。"""
    norm = lambda x: "".join(x.split())  # noqa: E731
    pages_lines = [{norm(ln) for ln in t.splitlines() if ln.strip()} for t in page_texts(pdf)]
    pages, cur = [], 0
    for _, t in HEADS:
        k = cur
        while k < len(pages_lines) and norm(t) not in pages_lines[k]:
            k += 1
        if k == len(pages_lines):
            raise RuntimeError(f"未在正文中找到标题：{t}")
        pages.append(k + 1)
        cur = k
    return pages


out_pdf = DOCS / f"{SOFTWARE_NAME}{VERSION}_{'文档鉴别材料' if JB else '操作说明书'}.pdf"
render(["00"] * len(HEADS), out_pdf)            # 第一遍：占位页码（版面与终稿一致）
pages = locate_headings(out_pdf)
render(pages, out_pdf)                          # 第二遍：填入真实页码
assert locate_headings(out_pdf) == pages, "目录页码与正文不一致"
print("figures:", FIG[0], "pdf:", out_pdf)
print("source lines:", src_total, by_lang, "original:", orig)

# ----------------------------------------------------------------------------- 合规检查
texts = page_texts(out_pdf)
texts = [t for t in texts if t.strip()]
n = len(texts)
img_pages = set()
lst = subprocess.run(["pdfimages", "-list", str(out_pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
for ln in lst:
    img_pages.add(int(ln.split()[0]))
report = {"pages": n, "short_text_pages": [], "header_ok": True, "name_variants": []}
for i, t in enumerate(texts, 1):
    lines = [x for x in t.splitlines() if x.strip()]
    if TITLE.replace(" ", "") not in "".join(lines[0].split()):
        report["header_ok"] = False
    if i not in img_pages and len(lines) - 1 < 30 and i != n:
        report["short_text_pages"].append((i, len(lines) - 1))
full = "".join("".join(t.split()) for t in texts)
for v in ["BrainComp", "BRAINCOMP", "全脑钙成像神经计算V1", "操作说明书" if JB else "@@"]:
    if v in full:
        report["name_variants"].append(v)
print("compliance:", json.dumps(report, ensure_ascii=False))
if JB and n > 60:
    sel = list(range(1, 31)) + list(range(n - 29, n + 1))
    part = out_pdf.with_name(out_pdf.stem + "_前后各30页.pdf")
    subprocess.run(["qpdf", str(out_pdf), "--pages", ".", ",".join(map(str, sel)), "--", str(part)], check=True)
    print("extracted:", part)
