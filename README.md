# MY-YOLO-FD: 多任务病鱼检测与分割模型

基于 YOLO-FD 的增强版代码库。此项目在原版 `YOLO-FD`（一个集成了目标检测与语义分割的多任务网络，专门用于病鱼检测）的基础上，修复了多项环境兼容性问题，并集成了**可视化 GUI 控制台**以及前沿的**基于大模型(LLM)的自动化调参 Agent**。

## 📌 项目状态 (Archive)

本项目已在 `2026-04-27` 进入归档状态，不再继续迭代。

- 终止复盘与经验总结: `docs/5_项目终止复盘与经验沉淀.md`
- LLM 自动调参架构蓝图: `docs/6_LLM自动调参架构蓝图.md`
- 归档说明: `PROJECT_ARCHIVE.md`

---

## 🌟 核心特性 (Features)

- **多任务学习 (Multi-Task Learning)**: 同时输出病灶的边界框 (Bounding Box) 和精确的感染区域掩码 (Mask)。
- **环境自适应修复**: 
  - 修复了 PyTorch `2.6+` 中 `torch.load(weights_only=True)` 导致的自定义模型加载失败问题。
  - 修复了 Numpy `2.0+` 中移除 `np.trapz` 导致 mAP 计算报错的问题。
- **GUI 可视化训练控制台**: 运行 `gui_train.py`，摆脱繁琐的代码修改，通过直观的界面配置超参数、切换模型结构并实时监控训练进度（内置 Windows 防休眠保护）。
- **LLM-Driven 智能炼丹 Agent**: 运行 `llm_auto_tuner.py`，利用火山引擎 (Volcengine Ark API) 的豆包大模型，根据历史 Loss 和跑分数据，全自动为你动态调整训练策略和超参数，直至突破目标跑分（mAP50 >= 0.96）。

---

## 📁 目录结构 (Directory Structure)

```text
MY-YOLO-FD/
├── docs/                 # 原工作区迁移过来的学习笔记与文献分析
├── ultralytics/          # YOLO-FD 核心算法源码 (模型结构、Loss、工具类)
├── runs/                 # 训练日志、评估结果和权重输出目录 (被 gitignore 忽略)
├── train.py              # 基础训练启动脚本
├── val.py                # 基础验证评估脚本
├── gui_train.py          # 🌈 可视化 GUI 训练控制台
├── auto_tuner.py         # 阶梯式自动调参脚本
├── llm_auto_tuner.py     # 🤖 基于大模型 API 的智能调参 Agent
├── export_onnx.py        # 模型 ONNX 导出及推理测试脚本
├── yolo-fd.pt            # 作者提供的最优预训练权重
└── norcardis_disease.yaml# 诺卡氏菌病鱼数据集配置文件
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
建议使用 Python `3.9+`，推荐安装带有 CUDA 11.8+ 支持的 PyTorch 版本的虚拟环境。
```bash
pip install -r requirements.txt
```

### 2. 准备数据集
在 `ultralytics/datasets/norcardis_disease.yaml` 中配置你本地的数据集绝对路径：
```yaml
path: C:/coding/YOLO-FD/norcardia_disease_fish_250621220254816(1)
train: det/images/train
val: det/images/val
# ...
```

### 3. 开启训练
你有三种方式来启动模型训练：

**方式一：传统脚本**
直接修改 `train.py` 中的参数，然后运行：
```bash
python train.py
```

**方式二：可视化 GUI (推荐)**
通过友好的图形界面配置所有参数：
```bash
python gui_train.py
```

**方式三：LLM 智能 Agent (实验性)**
让大模型接管你的电脑，自动分析跑分并调整参数：
*(需设置环境变量 `ARK_API_KEY`，不再支持在代码中明文填写 Key)*
```bash
python llm_auto_tuner.py
```

---

## 📈 性能评估与推理
训练完成后，你可以使用 `val.py` 进行评估，或者使用 `predict.py` 进行推理测试。若需部署到生产环境，请使用 `export_onnx.py` 将 PyTorch 权重转换为 ONNX 格式，以获得数倍的 CPU/GPU 推理加速。
