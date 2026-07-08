# 🧬 Virtual Nucleus Pulposus Cell Agent

> **虚拟髓核细胞智能体 — AI驱动的 NP 退变研究平台**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## 📋 Project Overview

**Virtual NP Cell Agent** 是一个专注于**髓核细胞（Nucleus Pulposus）** 及**椎间盘退变（IVD Degeneration）** 研究的综合计算平台。系统集成了：

- 🧠 **知识库** — 完整的 NP 细胞生物学知识图谱（标志物、通路、ECM、退变基因等）
- 📊 **差异表达分析** — 火山图、显著基因标注
- 🌡️ **表达热图** — 样本/基因聚类、分组注释
- 🔎 **生物标志物筛选** — 随机森林排序、ROC 曲线、时序趋势
- 📡 **信号通路 ODE 仿真** — 7 条核心通路集成 ODE 模型
- 🧪 **ECM 动力学模型** — 合成-降解平衡 + 多因素扰动
- 🎯 **miRNA 调控网络** — miRNA-mRNA 调控数据库、扰动模拟
- 🦴 **ABM 微环境建模** — 2D Agent 网格仿真
- 🧬 **scRNA 数据集成** — 模拟单细胞 + 真实数据管道

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Virtual NP Cell Agent                       │
├──────────┬──────────┬──────────┬──────────┬──────────┬───────┤
│   Volcano │  Heatmap │Biomarker │Signaling │   ECM    │miRNA  │
│   图       │   热图   │ RF+ROC   │ ODE 仿真 │ 动力学   │ 调控网 │
├──────────┴──────────┼──────────┼──────────┼──────────┼───────┤
│  🦴 ABM 微环境建模     │  🧬 scRNA 集成 │   📖 知识问答     │
│  Agent状态机+网格     │  PCA/t-SNE     │   内置KB          │
└──────────────────────┴────────────────┴──────────────────┘
```

## 🚀 Quick Start

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx
python run_all.py
```

All analysis results output to `output/` directory.

## 📖 Documentation

See [README.md](README.md) for full documentation.

## 📄 License

MIT License
