#!/usr/bin/env python3
"""
Virtual NP Cell — 虚拟髓核细胞完整演示
============================================
展示所有核心功能: 火山图、热图、标志物预测、信号通路、ECM 动力学
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.np_knowledge_base import NP_KNOWLEDGE_BASE
from analysis.de_analysis import (
    simulate_np_de_analysis, plot_volcano
)
from analysis.heatmap import (
    simulate_np_expression_matrix, plot_heatmap, plot_gene_group_heatmap
)
from analysis.biomarker import (
    simulate_biomarker_data, rank_biomarkers,
    plot_biomarker_importance, simulate_trend_data, plot_biomarker_trends,
    plot_roc_curves
)
from simulation.signaling import (
    NPSignalingModel, plot_signaling_network, plot_perturbation_comparison
)
from simulation.ecm_model import ECMDegradationModel, plot_ecm_dynamics
from simulation.abm_microenv import (
    run_abm_simulation, plot_abm_grid, plot_abm_timeseries,
    run_full_abm_pipeline as run_abm_pipeline
)
from regulation.mirna_network import (
    MIRNA_TARGET_DB, build_mirna_network,
    simulate_mirna_expression, plot_mirna_network,
    simulate_mirna_perturbation, mirna_diagnostic_analysis,
    run_full_mirna_pipeline
)
from data.scrnaintegration import (
    simulate_scrnaseq, pca_tsne_visualization,
    calculate_pathway_activity, plot_pathway_activity,
    run_full_scrnaseq_pipeline
)
from orchestrator import VirtualNPCell, ensure_output_dir, OUTPUT_DIR

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
OUTPUT = ensure_output_dir()
DPI = 150

print("=" * 60)
print("  🧬 Virtual NP Cell — 虚拟髓核细胞系统")
print("=" * 60)

# ============ 1. 系统信息 ============
print("\n" + "=" * 60)
print("  📋 第1章: 系统概览")
print("=" * 60)
np_cell = VirtualNPCell()
print(np_cell.status_summary())

# ============ 2. 知识查询示例 ============
print("\n" + "=" * 60)
print("  📖 第2章: 知识库查询测试")
print("=" * 60)
for topic in ["标志物", "通路", "退变", "ECM"]:
    result = np_cell.query_knowledge(topic)
    key = list(result.keys())[0]
    items = result[key]
    if isinstance(items, list):
        print(f"[{topic}] {len(items)} 条记录")
        for item in items[:3]:
            if isinstance(item, dict):
                print(f"  · {item.get('gene', item.get('pathway', item.get('component', '')))}")
    elif isinstance(items, dict):
        for k, v in items.items():
            print(f"  · {k}: {str(v)[:60]}")
    print()

# ============ 3. 火山图 ============
print("\n" + "=" * 60)
print("  🌋 第3章: 差异表达分析 & 火山图")
print("=" * 60)

de_df = simulate_np_de_analysis(n_genes=1200, n_de_genes=40)
print(f"  模拟数据: {len(de_df)} 个基因")
print(f"  显著上调: {de_df['up_regulated'].sum()}")
print(f"  显著下调: {de_df['down_regulated'].sum()}")

# 标注关键 NP 基因
label_genes = [
    "KRT19", "KRT18", "PAX1", "FOXF1", "CD24", "TBXT",
    "ACAN", "COL2A1", "SOX9", "NOG", "HAPLN1", "SIRT1",
    "MMP3", "MMP13", "ADAMTS4", "ADAMTS5", "IL1B", "TNF",
    "IL6", "CXCL8", "CDKN2A"
]
fig = plot_volcano(
    de_df,
    title="NP 退变 vs 正常对照 · 差异表达火山图",
    label_genes=label_genes,
    output_path=os.path.join(OUTPUT, "01_volcano.png"),
    dpi=DPI
)
print("  [✓] 火山图完成\n")

# ============ 4. 热图 ============
print("\n" + "=" * 60)
print("  🌡️ 第4章: 基因表达热图")
print("=" * 60)

expr_df, meta = simulate_np_expression_matrix(n_samples=20, n_genes=36)
print(f"  表达矩阵: {expr_df.shape[0]} 基因 × {expr_df.shape[1]} 样本")
print(f"  样本分组: {set(meta['sample_groups'].values())}")

fig = plot_heatmap(
    expr_df, meta,
    title="NP 细胞基因表达热图 (正常 vs 退变早期/晚期)",
    output_path=os.path.join(OUTPUT, "02_heatmap.png"),
    dpi=DPI
)
print("  [✓] 全基因热图完成\n")

# 2b. 关键基因子集热图
print("  生成关键基因子集热图...")
key_genes = [
    "KRT19", "KRT18", "PAX1", "FOXF1", "CD24", "TBXT",
    "ACAN", "COL2A1", "SOX9", "HAPLN1",
    "MMP3", "MMP13", "ADAMTS4", "ADAMTS5",
    "IL1B", "TNF", "IL6", "CXCL8"
]
fig2 = plot_gene_group_heatmap(
    expr_df, meta, genes=key_genes,
    title="NP 关键标志物与退变基因热图",
    figsize=(12, 8),
    output_path=os.path.join(OUTPUT, "03_key_genes_heatmap.png"),
)
print("  [✓] 关键基因热图完成\n")

# ============ 5. 生物标志物分析 ============
print("\n" + "=" * 60)
print("  🔬 第5章: 生物标志物预测与筛选")
print("=" * 60)

bio_df = simulate_biomarker_data(n_patients=120, seed=42)
print(f"  模拟患者数据: {len(bio_df)} 例")
print(f"  分组: {bio_df['退变分期'].value_counts().to_dict()}")

importance = rank_biomarkers(bio_df, top_n=15)
fig = plot_biomarker_importance(
    importance,
    output_path=os.path.join(OUTPUT, "04_biomarker_importance.png"),
    dpi=DPI
)
print("  [✓] 标志物重要性排序完成\n")

# 5b. 标志物趋势
print("  生成退变进展趋势图...")
trend_df = simulate_trend_data(n_timepoints=6)
fig = plot_biomarker_trends(
    trend_df,
    output_path=os.path.join(OUTPUT, "05_biomarker_trends.png"),
    dpi=DPI
)
print("  [✓] 趋势图完成\n")

# 5c. ROC 曲线
print("  生成多标志物 ROC 曲线...")
roc_genes = ["MMP13", "IL1B", "CXCL8", "ACAN", "KRT19", "SOX9"]
fig = plot_roc_curves(
    bio_df, biomarkers=roc_genes,
    output_path=os.path.join(OUTPUT, "06_ROC_curves.png"),
    dpi=DPI
)
print("  [✓] ROC 曲线完成\n")

# ============ 6. 信号通路仿真 ============
print("\n" + "=" * 60)
print("  📡 第6章: 信号通路 ODE 动态仿真")
print("=" * 60)

sig_model = NPSignalingModel()

# 正常稳态仿真
t, y = sig_model.simulate()
fig = plot_signaling_network(
    t, y,
    title="正常 NP 细胞 · 信号通路稳态",
    output_path=os.path.join(OUTPUT, "07_signaling_normal.png"),
    dpi=DPI
)
print("  [✓] 正常状态信号通路完成")

# 退变扰动仿真 (IL-1β + TNF-α + 力学生物学超载)
t_deg, y_deg = sig_model.simulate(
    t_span=(0, 100), n_points=500,
    perturbations={
        'IL1B': 3.0,   # IL-1β 刺激
        'TNF': 2.0,    # TNF-α 刺激
        'HYPOXIA': 0.3, # 低氧
        'INHIBITOR': 0.2,  # 部分抑制TGF-β
    }
)
fig = plot_signaling_network(
    t_deg, y_deg,
    title="退变 NP 细胞 · IL-1β/TNF-α 刺激后信号网络",
    output_path=os.path.join(OUTPUT, "08_signaling_degeneration.png"),
    dpi=DPI
)
print("  [✓] 退变状态信号通路完成")

# 多条件扰动对比
print("\n  多条件扰动筛选...")
perturbations = [
    ("正常对照", {}),
    ("炎症刺激\n(IL-1β+TNF)", {'IL1B': 3.0, 'TNF': 2.0}),
    ("Wnt 激活", {'WNT': 2.0}),
    ("低氧应激", {'HYPOXIA': 1.0}),
    ("TGF-β 补充", {'TGFB': 2.0}),
    ("GSK3β 抑制", {'GSK3B_inhibit': 0.8}),
    ("综合退变\n(炎症+Wnt+力学)", {'IL1B': 3.0, 'TNF': 2.0, 'WNT': 1.5, 'HYPOXIA': 0.3}),
]
results, base_ecm, base_mmp = sig_model.run_perturbation_screen(perturbations)
fig = plot_perturbation_comparison(
    results, base_ecm, base_mmp,
    output_path=os.path.join(OUTPUT, "09_perturbation_comparison.png"),
    dpi=DPI
)
print("  [✓] 扰动对比完成")

for name, r in results.items():
    print(f"  · {name.strip()}: ECM变化={r['ECM_change']:+.1f}%, "
          f"MMP变化={r['MMP_change']:+.1f}%, "
          f"合成代谢评分={r['Anabolic_score']:.3f}")

# ============ 7. ECM 动力学 ============
print("\n" + "=" * 60)
print("  🧪 第7章: ECM 代谢动力学仿真")
print("=" * 60)

ecm_model = ECMDegradationModel()

# 正常 ECM 稳态
t_e, y_e = ecm_model.simulate()
fig = plot_ecm_dynamics(
    t_e, y_e,
    title="正常 NP 细胞 · ECM 合成-降解平衡",
    output_path=os.path.join(OUTPUT, "10_ecm_normal.png"),
    dpi=DPI
)
print("  [✓] 正常 ECM 动态完成")

# 退变 ECM 仿真 (炎症 + 氧化应激 + 营养不足)
t_ed, y_ed = ecm_model.simulate(
    t_span=(0, 400), n_points=500,
    perturbation={
        'INFLAM': 3.0,
        'OXSTRESS': 2.0,
        'NUTRIENT': -1.0,
        'MECHANICAL': 0.5,
    },
    degen_accel=1.0
)
fig = plot_ecm_dynamics(
    t_ed, y_ed,
    title="退变 NP 细胞 · ECM 进行性降解",
    output_path=os.path.join(OUTPUT, "11_ecm_degeneration.png"),
    dpi=DPI
)
print("  [✓] 退变 ECM 动态完成")

# ============ 8. ABM 微环境建模 ============
print("\n" + "=" * 60)
print("  🦴 第8章: ABM 微环境建模")
print("=" * 60)

print("  运行正常状态 ABM 仿真 (网格20×20, 50步)...")
abm_normal = run_abm_simulation(
    grid_size=20, n_np_cells=30, n_macrophages=6,
    n_fibroblasts=8, n_steps=50, degenerative=False, seed=42
)
print(f"  初始 NP 细胞: 30 → 最终存活: {len([a for a in abm_normal['agents'] if a.cell_type=='NP_cell'])}")
print(f"  最终 ECM 均值: {abm_normal['ecm_history'][-1][1]:.3f}")

fig = plot_abm_grid(
    abm_normal['grid_history'][-1], 20,
    output_path=os.path.join(OUTPUT, "12_abm_normal.png"),
    dpi=DPI
)
print("  [✓] ABM 网格快照完成（正常）")

fig = plot_abm_timeseries(
    abm_normal,
    output_path=os.path.join(OUTPUT, "13_abm_timeseries_normal.png"),
    dpi=DPI
)
print("  [✓] ABM 时序曲线完成（正常）")

print("\n  运行退变状态 ABM 仿真...")
abm_degen = run_abm_simulation(
    grid_size=20, n_np_cells=30, n_macrophages=6,
    n_fibroblasts=8, n_steps=50, degenerative=True,
    inflam_seed=0.5, mechanical_overload=0.6, seed=43
)
print(f"  最终 NP 存活: {len([a for a in abm_degen['agents'] if a.cell_type=='NP_cell'])}")
print(f"  最终 ECM 均值: {abm_degen['ecm_history'][-1][1]:.3f}")

fig = plot_abm_grid(
    abm_degen['grid_history'][-1], 20,
    output_path=os.path.join(OUTPUT, "14_abm_degen.png"),
    dpi=DPI
)
print("  [✓] ABM 网格快照完成（退变）")

fig = plot_abm_timeseries(
    abm_degen,
    output_path=os.path.join(OUTPUT, "15_abm_timeseries_degen.png"),
    dpi=DPI
)
print("  [✓] ABM 时序曲线完成（退变）")

# ============ 9. miRNA 调控分析 ============
print("\n" + "=" * 60)
print("  🎯 第9章: miRNA 调控网络分析")
print("=" * 60)

print("  模拟 miRNA 表达 (正常 vs 退变)...")
mirna_data = simulate_mirna_expression(n_normal=25, n_degen=25, seed=42)
print(f"  样本数: {len(mirna_data['labels'])}, miRNAs: {len(mirna_data['mirna_names'])}")
for m in ['miR-155', 'miR-21', 'miR-140-5p', 'miR-146a']:
    fc = mirna_data['log2FC'][m]
    print(f"  · {m:12s} log2FC={fc:+.2f}")

print("\n  miRNA 调控网络可视化...")
try:
    G = build_mirna_network()
    fig = plot_mirna_network(
        G,
        output_path=os.path.join(OUTPUT, "16_mirna_network.png"),
        dpi=DPI
    )
    print(f"  网络节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    print("  [✓] miRNA 调控网络完成")
except Exception as e:
    print(f"  ⚠ 网络可视化跳过: {e}")

print("\n  miRNA 扰动模拟 (miR-155 过表达)...")
pert = simulate_mirna_perturbation('miR-155', 'overexpression', fold_change=2.0)
for target, change in list(pert.items())[:5]:
    direction = '↑' if change > 0 else '↓'
    print(f"  · {target:8s} {direction} {abs(change):.2f}倍")
print("  [✓] miRNA 扰动模拟完成")

print("\n  miRNA 诊断标志物 ROC 分析...")
fig = mirna_diagnostic_analysis(
    mirna_data,
    output_path=os.path.join(OUTPUT, "17_mirna_roc.png"),
    dpi=DPI
)
print("  [✓] miRNA ROC 分析完成")

# ============ 10. scRNA 数据集成 ============
print("\n" + "=" * 60)
print("  🧬 第10章: scRNA 单细胞数据集成")
print("=" * 60)

print("  生成模拟 scRNA-seq 数据...")
ad = simulate_scrnaseq(n_cells=500, n_genes=200, seed=42)
print(f"  表达矩阵: {ad['n_cells']} cells × {ad['n_genes']} genes")
print(f"  细胞类型: {ad['cell_type_names']}")

print("\n  降维可视化...")
try:
    fig = pca_tsne_visualization(
        ad,
        output_path=os.path.join(OUTPUT, "18_scrnaseq_dimreduction.png"),
        dpi=DPI
    )
    print("  [✓] PCA + t-SNE 降维完成")
except Exception as e:
    print(f"  ⚠ 降维跳过: {e}")

print("\n  通路活性评分...")
activities = calculate_pathway_activity(ad)
for pathway, scores in activities.items():
    cell_means = {}
    for ct in set(ad['cell_types']):
        mask = ad['cell_types'] == ct
        cell_means[ct] = f"{scores[mask].mean():.3f}"
    print(f"  · {pathway:20s}  {cell_means}")

fig = plot_pathway_activity(
    ad, activities,
    output_path=os.path.join(OUTPUT, "19_scrnaseq_pathway_activity.png"),
    dpi=DPI
)
print("  [✓] 通路活性评分完成")

# ============ 11. 总结报告 ============
print("\n" + "=" * 60)
print("  📊 虚拟髓核细胞系统 — 分析完成")
print("=" * 60)
print(f"\n  输出目录: {OUTPUT}")
print(f"\n  生成文件列表:")
files = sorted(os.listdir(OUTPUT))
for f in files:
    fpath = os.path.join(OUTPUT, f)
    size = os.path.getsize(fpath) / 1024
    print(f"    · {f:40s} ({size:.1f} KB)")

print("\n" + "=" * 60)
print("  ✅ Virtual NP Cell System Ready!")
print("\n  已集成模块:")
print("    · 第1章: 系统概览")
print("    · 第2章: 知识库")
print("    · 第3章: 差异表达 & 火山图")
print("    · 第4章: 热图分析")
print("    · 第5章: 生物标志物")
print("    · 第6章: 信号通路 ODE")
print("    · 第7章: ECM 代谢动力学")
print("    · 第8章: ABM 微环境建模 (新)")
print("    · 第9章: miRNA 调控网络 (新)")
print("    · 第10章: scRNA 数据集成 (新)")
print("=" * 60)
