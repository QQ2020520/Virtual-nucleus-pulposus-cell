# 🧬 Virtual Nucleus Pulposus Cell Agent — v3.0

> **智能调度虚拟髓核细胞平台** — AI驱动的 NP 退变研究全栈系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## 📋 项目总览

**Virtual NP Cell v3.0** 是一个专注于**髓核细胞（Nucleus Pulposus）** 及**椎间盘退变（IVD Degeneration）** 研究的综合计算平台。v3.0 升级了智能调度 Orchestrator，统一路由所有 v2.0 新模块。

## 🏗️ 系统架构 (v3.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Virtual NP Cell Agent v3.0                       │
│                    🎯 智能调度 Orchestrator                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────── v1.0 核心模块 ──────────────────────┐       │
│  │  火山图  │  热图   │ 标志物  │ 信号通路 │ ECM  │  miRNA  │       │
│  │  Volcano│ Heatmap │Biomarker│ ODE     │动力学│  网络    │       │
│  ├─────────┴─────────┴─────────┴─────────┴──────┴──────────┤       │
│  │   ABM 微环境  │  scRNA 集成  │  知识库     │                     │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌───────────────────── v2.0 新增模块 ──────────────────────┐       │
│  │                                                                    │
│  │  📡 11. MRTF-A 力传导模型                                       │
│  │      (Bone Research 2025: 基质刚度→MRTF-A→糖酵解)               │
│  │                                                                    │
│  │  ⚗️ 12. NP 代谢可塑性模型                                       │
│  │      (糖酵解/OXPHOS/谷氨酰胺/线粒体/AMPK 19维ODE)               │
│  │                                                                    │
│  │  ⏳ 13. NP 细胞衰老模型                                         │
│  │      (p53-p21/p16-Rb/SASP/线粒体/Nox4 17维ODE + Senolytic)      │
│  │                                                                    │
│  │  🧬 14. m6A 表观转录组模型                                      │
│  │      (Nat Commun 2022: KDM5A-WTAP-m6A-NORAD-E2F3 13维ODE)       │
│  │                                                                    │
│  │  🗺️ 15. IVD 空间转录组模拟                                      │
│  │      (Advanced Science 2024: NP/AF/CEP分区+伪时间+生态位)        │
│  │                                                                    │
│  │  🔄 16. 多尺度耦合仿真 (可选)                                   │
│  │  💊 17. 虚拟药物筛选 (可选)                                     │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 v3.0 模块清单

| # | 模块 | 类/函数 | 状态 | 描述 |
|---|------|---------|------|------|
| 1 | 系统概览 | `VirtualNPCell` | ✅ | 状态摘要、模块自检 |
| 2 | 知识库 | `query_knowledge()` | ✅ | 内置 NP 生物学知识图谱 |
| 3 | 火山图 | `plot_volcano()` | ✅ | 差异表达分析 |
| 4 | 热图 | `plot_heatmap()` | ✅ | 样本/基因聚类 |
| 5 | 标志物 | `rank_biomarkers()` | ✅ | RF 排序 + ROC + 趋势 |
| 6 | 信号通路 | `NPSignalingModel` | ✅ | 7 通路集成 ODE |
| 7 | ECM 动力学 | `ECMDegradationModel` | ✅ | 合成-降解平衡 |
| 8 | ABM 微环境 | `run_abm_simulation()` | ✅ | 2D Agent 网格 |
| 9 | miRNA 网络 | `MIRNA_TARGET_DB` | ✅ | 调控网络 + 扰动 |
| 10 | scRNA 集成 | `simulate_scrnaseq()` | ✅ | PCA/t-SNE + 通路活性 |
| **11** | **MRTF-A 力传导** | `MRTFAMechanotransductionModel` | ✅ **新** | 刚度→糖酵解 9维ODE |
| **12** | **代谢可塑性** | `NPMetabolismModel` | ✅ **新** | 糖酵解/OXPHOS/谷氨酰胺 19维ODE |
| **13** | **细胞衰老** | `NPSenescenceModel` | ✅ **新** | p53/p16/SASP/线粒体 17维ODE |
| **14** | **m6A 表观** | `M6AEpigeneticModel` | ✅ **新** | WTAP-m6A-NORAD-E2F3 13维ODE |
| **15** | **空间转录组** | `SpatialTranscriptomics` | ✅ **新** | IVD 分区 + 伪时间 + 生态位 |
| **16** | **耦合仿真** | `NPCoupledModel` | 📦 可选 | 多尺度耦合 |
| **17** | **药物筛选** | `VirtualDrugScreening` | 📦 可选 | 虚拟筛选 |
| **18** | 综合总结 | — | ✅ | 可视化报告 |

## 🚀 快速开始

```bash
# 安装依赖
pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx

# 完整演示 (含 v2.0 新模块)
python run_all.py

# 或使用智能调度 Orchestrator
python3 -c "
from orchestrator import VirtualNPCell
v = VirtualNPCell()
print(v.status_summary())
"
```

所有分析结果输出到 `output/` 目录。

## 🧪 使用 Orchestrator v3.0

### 初始化

```python
from orchestrator import VirtualNPCell
v = VirtualNPCell()
print(v.status_summary())
```

### 运行单个模块

```python
# MRTF-A 力传导 (正常刚度)
r = v.run_module('mechanotransduction', stiffness_level=1.0)

# MRTF-A 力传导 (退变刚度 + CCG-1423 抑制剂)
r = v.run_module('mechanotransduction', stiffness_level=3.0,
                  perturbation={'CCG_inhibitor': 0.6})

# 代谢模型 (低氧)
r = v.run_module('metabolism', oxygen=0.01, glucose=1.0,
                  perturbation='hypoxia')

# 衰老模型
r = v.run_module('senescence', stress_level=2.0)

# m6A 表观模型
r = v.run_module('m6a', perturbation='NORAD_OE')

# 空间转录组
r = v.run_module('spatial', grid_size=50, n_spots=2000)
```

### 管道执行

```python
# 按序执行多个模块
results = v.run_pipeline([
    'signaling',
    'mechanotransduction',
    'metabolism',
    'senescence',
    'm6a',
    'spatial',
])
```

### 快速报告

```python
report = v.quick_report('力传导')
print(report)

report = v.quick_report('衰老')
print(report)
```

### 知识查询

```python
# v2.0 新增主题
print(v.query_knowledge('力传导'))
print(v.query_knowledge('代谢'))
print(v.query_knowledge('衰老'))
print(v.query_knowledge('表观'))
print(v.query_knowledge('空间'))
print(v.query_knowledge('治疗靶点'))
```

## 📖 模块详细说明

### v2.0 新模块

| 模块 | 文件 | 变量数 | 参考文献 |
|------|------|--------|----------|
| MRTF-A 力传导 | `simulation/mechanotransduction.py` | 9 | Bone Research 2025 |
| 代谢可塑性 | `simulation/metabolism_model.py` | 19 | Nature Reviews 代谢 |
| 细胞衰老 | `simulation/senescence_model.py` | 17 | Nature Aging |
| m6A 表观 | `regulation/epigenetics_m6a_model.py` | 13 | Nat Commun 2022 |
| 空间转录组 | `analysis/spatial_transcriptomics.py` | — | Advanced Science 2024 |

### 新模块功能亮点

**MRTF-A 力传导 (第11章)**
- 基质刚度 → Integrin-FAK → F-actin → MRTF-A 核转位
- Kidins220 ↓ → AMPK ↓ → PFKFB3/PFKM ↓ → 糖酵解 ↓
- CCG-1423 (MRTF-A 抑制剂) 可逆转上述过程
- 支持刚度剂量响应、双参数扰动热图

**代谢可塑性 (第12章)**
- 19 维 ODE: 糖酵解/OXPHOS/谷氨酰胺代谢/线粒体动力学
- 支持 8 种代谢条件 (低氧/缺糖/PFKFB3激活/AMPK激活/鱼藤酮/2DG 等)
- 氧-葡萄糖双参数代谢景观图

**细胞衰老 (第13章)**
- 17 维 ODE: p53-p21-Rb/p16-Rb/SASP/NF-κB/ROS/Nox4/线粒体
- SASP 正反馈 → 衰老自催化加速
- 多药物 Senolytic 干预模拟 (达沙替尼/槲皮素/D+Q/纳维托克)

**m6A 表观转录组 (第14章)**
- 13 维 ODE: KDM5A → H3K4me3 → WTAP → m6A-NORAD → PUM → E2F3
- 6 种基因扰动 + 7 种基因敲除模拟
- 扰动/KO 柱状图比较

**空间转录组 (第15章)**
- IVD 2D 空间网格: NP/AF/CEP 三区结构
- NP 祖细胞 → 成熟 NP 伪时间分化轨迹
- 6 种空间生态位自动注释

## 📝 输出文件清单 (v3.0)

```
output/
├── 01_volcano.png               # 差异表达火山图
├── 02_heatmap.png               # 全基因热图
├── 03_key_genes_heatmap.png     # 关键基因热图
├── 04_biomarker_importance.png  # 标志物重要性
├── 05_biomarker_trends.png      # 标志物趋势
├── 06_ROC_curves.png            # ROC 曲线
├── 07_signaling_normal.png      # 信号通路正常态
├── 08_signaling_degeneration.png# 信号通路退变态
├── 09_perturbation_comparison.png# 多扰动对比
├── 10_ecm_normal.png            # ECM 正常动态
├── 11_ecm_degeneration.png      # ECM 退变动态
├── 12_abm_normal.png            # ABM 正常态
├── 13_abm_timeseries_normal.png # ABM 时序正常
├── 14_abm_degen.png             # ABM 退变态
├── 15_abm_timeseries_degen.png  # ABM 时序退变
├── 16_mirna_network.png         # miRNA 调控网络
├── 17_mirna_roc.png             # miRNA ROC
├── 18_scrnaseq_dimreduction.png # scRNA 降维
├── 19_scrnaseq_pathway_activity.png  # scRNA 通路活性
│   ──── v2.0 新增 ────
├── 20_mechano_normal.png        # MRTF-A 正常
├── 21_mechano_degenerated.png   # MRTF-A 退变
├── 22_mechano_ccg_reversal.png  # MRTF-A CCG逆转
├── 23_mechano_dose_response.png # MRTF-A 剂量响应
├── 24_mechano_perturbation_heatmap.png # MRTF-A 热图
├── 25_metabolism_normal.png     # 代谢正常
├── 26_metabolism_ATP_landscape.png # ATP景观
├── 27_metabolism_lactate_landscape.png # 乳酸景观
├── 28_senescence_healthy.png    # 衰老健康
├── 29_senescence_stress.png     # 衰老应激
├── 30_senescence_senolytic_comparison.png # senolytic比较
├── 31_m6a_baseline.png          # m6A 基线
├── 32_m6a_NORAD_OE.png          # m6A NORAD过表达
├── 33_m6a_WTAP_KO.png           # m6A WTAP敲除
├── 34_m6a_perturbation_comparison.png # m6A扰动比较
├── 35_m6a_ko_comparison.png     # m6A KO比较
├── 36_spatial_gene_expression.png    # 空间基因表达
├── 37_spatial_pseudotime.png    # 空间伪时间
├── 38_spatial_summary.png       # 空间总结
```

## 🔧 模块可用性验证

```python
# 验证所有模块加载
from orchestrator import VirtualNPCell
v = VirtualNPCell()
print(v.status_summary())

# 验证 Mechanotransduction
r = v.run_module('mechanotransduction', stiffness_level=1.0)
assert r['success'], f"Mechanotransduction failed: {r.get('error')}"
print('✅ Mechanotransduction OK')

# 验证 Metabolism
r = v.run_module('metabolism', oxygen=0.05, glucose=1.0)
assert r['success'], f"Metabolism failed: {r.get('error')}"
print('✅ Metabolism OK')

# 验证 Senescence
r = v.run_module('senescence', stress_level=1.0)
assert r['success'], f"Senescence failed: {r.get('error')}"
print('✅ Senescence OK')
```

## 📄 License

MIT License

## 📚 参考文献

1. **Bone Research 2025** — Matrix stiffness regulates NP glycolysis by MRTF-A
2. **Nature Communications 2022** — WTAP-m6A-NORAD-E2F3 regulates IVD degeneration
3. **Advanced Science 2024** — Spatially resolved transcriptomics of mouse IVD
4. **Nature Reviews** — NP cell metabolism and metabolic plasticity
5. **Nature Aging** — Cellular senescence mechanisms in IVD degeneration
