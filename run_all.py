#!/usr/bin/env python3
"""
Virtual NP Cell — 虚拟髓核细胞完整演示 v3.0
===============================================
展示所有核心功能: 火山图、热图、标志物预测、信号通路、ECM 动力学、
ABM 微环境、miRNA网络、scRNA数据集成 + v2.0 新模块
(力传导、代谢、衰老、m6A表观、空间转录组、耦合、药物筛选)
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

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

# ============ 配置 ============
OUTPUT = ensure_output_dir()
DPI = 150

print("=" * 60)
print("  🧬 Virtual NP Cell — 虚拟髓核细胞系统 v3.0")
print("  📦 智能调度 Orchestrator + 7 个核心模块 + 可选模块")
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

# 关键基因子集热图
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
        'IL1B': 3.0,
        'TNF': 2.0,
        'HYPOXIA': 0.3,
        'INHIBITOR': 0.2,
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

# 退变 ECM 仿真
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


# =====================================================================
#                  v2.0 新章节 ————————————————————
# =====================================================================

# ============ 11. MRTF-A 力传导 → 糖酵解仿真 ============
print("\n" + "=" * 60)
print("  🔬 第11章: MRTF-A 力传导 → 糖酵解仿真")
print("  📄 Bone Research 2025: Matrix stiffness regulates NP glycolysis by MRTF-A")
print("=" * 60)

try:
    from simulation.mechanotransduction import (
        MRTFAMechanotransductionModel,
        plot_dose_response,
        plot_perturbation_heatmap,
    )
    _HAS_MECHANO = True
except ImportError:
    _HAS_MECHANO = False
    print("  ⚠ MRTF-A 力传导模块不可用，跳过")

if _HAS_MECHANO:
    try:
        mechano_model = MRTFAMechanotransductionModel()

        # 11a. 正常刚度仿真 (1.0× normal, ~2kPa)
        print("\n  11a. 正常刚度仿真 (stiffness=1.0, ~2kPa)...")
        t_m, y_m = mechano_model.simulate(stiffness_level=1.0)
        metrics_norm = mechano_model.get_steady_state_metrics(y_m[:, -1], 1.0)
        print(f"      MRTF-A核: {metrics_norm['MRTFA_nuc']:.4f}, "
              f"Kidins220: {metrics_norm['Kidins220']:.4f}, "
              f"p-AMPK: {metrics_norm['AMPK_P']:.4f}, "
              f"糖酵解: {metrics_norm['Glycolysis_output']:.4f}")

        fig = mechano_model.plot_mechanotransduction(
            t_m, y_m, stiffness_level=1.0,
            title="正常 NP 细胞 · MRTF-A 机械力传导",
            output_path=os.path.join(OUTPUT, "20_mechano_normal.png"),
            dpi=DPI,
        )
        print("      [✓] 正常力传导仿真图")

        # 11b. 退变刚度仿真 (3.0× normal, ~6-15kPa)
        print("\n  11b. 退变刚度仿真 (stiffness=3.0)...")
        t_md, y_md = mechano_model.simulate(stiffness_level=3.0)
        metrics_deg = mechano_model.get_steady_state_metrics(y_md[:, -1], 3.0)
        print(f"      正常→退变:")
        for k in metrics_norm:
            if k == 'stiffness': continue
            base = metrics_norm[k]
            deg = metrics_deg[k]
            chg = (deg - base) / base * 100 if abs(base) > 1e-6 else 0
            print(f"        {k}: {base:.4f} → {deg:.4f} ({chg:+.1f}%)")

        fig = mechano_model.plot_mechanotransduction(
            t_md, y_md, stiffness_level=3.0,
            title="退变 NP 细胞 · MRTF-A 力传导 (刚度↑糖酵解↓)",
            output_path=os.path.join(OUTPUT, "21_mechano_degenerated.png"),
            dpi=DPI,
        )
        print("      [✓] 退变力传导仿真图")

        # 11c. CCG-1423 逆转效应
        print("\n  11c. CCG-1423 逆转效应 (stiffness=3.0, CCG=0.6)...")
        t_ccg, y_ccg = mechano_model.simulate(
            stiffness_level=3.0,
            perturbation={'CCG_inhibitor': 0.6},
        )
        metrics_ccg = mechano_model.get_steady_state_metrics(y_ccg[:, -1], 3.0)
        print(f"      Kidins220: {metrics_deg['Kidins220']:.4f} → {metrics_ccg['Kidins220']:.4f}")
        print(f"      p-AMPK:    {metrics_deg['AMPK_P']:.4f} → {metrics_ccg['AMPK_P']:.4f}")
        print(f"      糖酵解:    {metrics_deg['Glycolysis_output']:.4f} → {metrics_ccg['Glycolysis_output']:.4f}")

        fig = mechano_model.plot_mechanotransduction(
            t_ccg, y_ccg, stiffness_level=3.0,
            perturbation_label="CCG-1423 0.6",
            title="CCG-1423 逆转 MRTF-A 力传导抑制",
            output_path=os.path.join(OUTPUT, "22_mechano_ccg_reversal.png"),
            dpi=DPI,
        )
        print("      [✓] CCG逆转效应图")

        # 11d. 刚度剂量响应曲线
        print("\n  11d. 刚度剂量响应曲线...")
        dose_results = mechano_model.simulate_dose_response()
        fig = plot_dose_response(
            dose_results,
            output_path=os.path.join(OUTPUT, "23_mechano_dose_response.png"),
            dpi=DPI,
        )
        print("      [✓] 剂量响应曲线")

        # 11e. 双参数扰动热图
        print("\n  11e. 刚度×CCG-1423 双参数扰动热图...")
        fig = plot_perturbation_heatmap(
            mechano_model,
            output_path=os.path.join(OUTPUT, "24_mechano_perturbation_heatmap.png"),
            dpi=DPI,
        )
        print("      [✓] 双参数扰动热图")

        print("\n  [✓] 第11章完成")
    except Exception as e:
        print(f"\n  ⚠ 第11章错误: {e}")

# ============ 12. 代谢可塑性仿真 ============
print("\n" + "=" * 60)
print("  ⚗️ 第12章: 代谢可塑性仿真")
print("=" * 60)

try:
    from simulation.metabolism_model import NPMetabolismModel, plot_metabolic_landscape
    _HAS_METABOLISM = True
except ImportError:
    _HAS_METABOLISM = False
    print("  ⚠ 代谢模块不可用，跳过")

if _HAS_METABOLISM:
    try:
        metab_model = NPMetabolismModel()

        # 12a. 正常 NP 代谢 (5% O₂, 1mM glucose)
        print("\n  12a. 正常 NP 代谢 (5% O₂, 1mM glucose)...")
        result_norm = metab_model.simulate(oxygen_level=0.05, glucose_level=1.0)
        profile_norm = metab_model.get_metabolic_profile(result_norm)
        y_final = result_norm['y'][:, -1]
        print(f"      ATP={y_final[9]:.3f}, HIF-1α={y_final[10]:.3f}, "
              f"ROS={y_final[17]:.3f}, 膜电位={y_final[18]:.3f}")
        print(f"      糖酵解主导={profile_norm['glycolysis_dominance']:.2f}, "
              f"能量电荷={profile_norm['energy_charge']:.2f}")

        fig = metab_model.plot_metabolism(
            result_norm,
            save_path=os.path.join(OUTPUT, "25_metabolism_normal.png"),
        )
        print("      [✓] 正常代谢图")

        # 12b. 多条件代谢应激矩阵
        print("\n  12b. 代谢应激矩阵...")
        stress_results = metab_model.simulate_metabolic_stress()
        comp = stress_results['comparison']
        print(f"      {'条件':<25s} {'ATP':>6s} {'Lac':>6s} {'HIF':>6s} {'ROS':>6s} {'MMP':>6s}")
        print(f"      {'-'*55}")
        for cond, vals in comp.items():
            print(f"      {cond:<25s} {vals['ATP']:6.2f} {vals['Lactate']:6.2f} "
                  f"{vals['HIF1_alpha']:6.2f} {vals['ROS_mito']:6.2f} {vals['MMPotential']:6.2f}")

        # 12c. 代谢景观图
        print("\n  12c. 代谢景观图 (ATP)...")
        fig = plot_metabolic_landscape(
            metab_model, metric='ATP',
            save_path=os.path.join(OUTPUT, "26_metabolism_ATP_landscape.png"),
        )
        print("      [✓] ATP代谢景观图")

        fig = plot_metabolic_landscape(
            metab_model, metric='Lactate',
            save_path=os.path.join(OUTPUT, "27_metabolism_lactate_landscape.png"),
        )
        print("      [✓] 乳酸代谢景观图")

        print("\n  [✓] 第12章完成")
    except Exception as e:
        print(f"\n  ⚠ 第12章错误: {e}")
        import traceback
        traceback.print_exc()

# ============ 13. 细胞衰老模型 ============
print("\n" + "=" * 60)
print("  ⏳ 第13章: 细胞衰老模型 (含Senolytic模拟)")
print("=" * 60)

try:
    from simulation.senescence_model import NPSenescenceModel, plot_senolytic_comparison
    _HAS_SENESCENCE = True
except ImportError:
    _HAS_SENESCENCE = False
    print("  ⚠ 衰老模块不可用，跳过")

if _HAS_SENESCENCE:
    try:
        sen_model = NPSenescenceModel()

        # 13a. 健康基线
        print("\n  13a. 健康基线仿真 (stress=0.5)...")
        t_s, y_s = sen_model.simulate(stress_level=0.5)
        score = sen_model.compute_senescence_score(y_s[:, -1])
        print(f"      衰老评分: {score:.4f}")
        fig = sen_model.plot_senescence(
            t_s, y_s,
            title="NP 细胞衰老 — 健康基线 (stress=0.5)",
            output_path=os.path.join(OUTPUT, "28_senescence_healthy.png"),
            dpi=DPI,
        )
        print("      [✓] 健康基线衰老图")

        # 13b. 退变应力
        print("\n  13b. 退变应力仿真 (stress=2.0)...")
        t_sd, y_sd = sen_model.simulate(stress_level=2.0)
        score_d = sen_model.compute_senescence_score(y_sd[:, -1])
        state_d = sen_model.get_state_dict(y_sd[:, -1])
        print(f"      衰老评分: {score_d:.4f}")
        print(f"      SASP: {state_d['SASP_score']:.4f}, "
              f"ROS: {state_d['ROS_cellular']:.4f}, "
              f"线粒体障碍: {state_d['Mitochondrial_dysfunction']:.4f}")

        fig = sen_model.plot_senescence(
            t_sd, y_sd,
            title="NP 细胞衰老 — 退变应力 (stress=2.0)",
            output_path=os.path.join(OUTPUT, "29_senescence_stress.png"),
            dpi=DPI,
        )
        print("      [✓] 退变应力衰老图")

        # 13c. SASP 正反馈
        print("\n  13c. SASP 正反馈加速衰老...")
        t_fb, y_fb = sen_model.simulate_sasp_feedback(stress_level=0.8)
        score_fb = sen_model.compute_senescence_score(y_fb[:, -1])
        state_fb = sen_model.get_state_dict(y_fb[:, -1])
        print(f"      衰老评分: {score_fb:.4f}")
        print(f"      SASP: {state_fb['SASP_score']:.4f}, "
              f"NF-κB: {state_fb['NFkB_activity']:.4f}")

        # 13d. Senolytic 药物干预
        print("\n  13d. Senolytic 药物干预比较...")
        drug_configs = {
            'dasatinib': ('dasatinib', 1.5),
            'quercetin': ('quercetin', 1.5),
            'D+Q': ('dasatinib+quercetin', 1.5),
            'navitoclax': ('navitoclax', 1.5),
        }
        drug_models = {}
        for label, (drug, stress) in drug_configs.items():
            try:
                t_drug, y_drug = sen_model.simulate_senolytic(
                    drug=drug, stress_level=stress,
                    senolytic_start=200.0,
                )
                drug_models[label] = (t_drug, y_drug)
                post_score = sen_model.compute_senescence_score(y_drug[:, -1])
                pre_score = sen_model.compute_senescence_score(y_drug[:, 200])
                print(f"      {label:15s}: 治疗前={pre_score:.3f} → 治疗后={post_score:.3f}")
            except Exception as e:
                print(f"      {label:15s}: 跳过 ({e})")

        if drug_models:
            fig = plot_senolytic_comparison(
                drug_models,
                list(drug_models.keys()),
                title="Senolytic 药物干预比较 (NP 细胞衰老模型)",
                output_path=os.path.join(OUTPUT, "30_senescence_senolytic_comparison.png"),
                dpi=DPI,
            )
            print("      [✓] Senolytic比较图")

        # 13e. 完整扰动筛选
        print("\n  13e. 衰老扰动筛选...")
        from simulation.senescence_model import run_senescence_perturbation_screen
        screen_results = run_senescence_perturbation_screen(
            sen_model, stress_levels=[0.5, 1.0, 2.0],
            output_dir=OUTPUT, dpi=DPI,
        )
        top_results = [(k, v) for k, v in sorted(
            screen_results.items(),
            key=lambda x: x[1]['Senescence_score'],
            reverse=True,
        )[:5]]
        print(f"      衰老评分 Top-5:")
        for k, v in top_results:
            print(f"        {k:30s} → {v['Senescence_score']:.4f}")

        print("\n  [✓] 第13章完成")
    except Exception as e:
        print(f"\n  ⚠ 第13章错误: {e}")
        import traceback
        traceback.print_exc()

# ============ 14. m6A 表观转录组 ============
print("\n" + "=" * 60)
print("  🧬 第14章: m6A 表观转录组")
print("  📄 Nat Commun 2022: WTAP-m6A-NORAD-E2F3 调控NPC衰老")
print("=" * 60)

try:
    from regulation.epigenetics_m6a_model import (
        M6AEpigeneticModel, VAR_NAMES, simulate_ko_comparison
    )
    _HAS_M6A = True
except ImportError:
    _HAS_M6A = False
    print("  ⚠ m6A 表观模块不可用，跳过")

if _HAS_M6A:
    try:
        # 14a. 基础稳态模拟
        print("\n  14a. 基础稳态模拟...")
        m6a_model = M6AEpigeneticModel()
        ctrl = m6a_model.simulate(t_span=(0, 400))
        y_ctrl = ctrl['y'][:, -1]
        for name, val in zip(VAR_NAMES, y_ctrl):
            print(f"      {name:20s} = {val:.4f}")

        fig = m6a_model.plot_epigenetics(
            ctrl,
            output_path=os.path.join(OUTPUT, "31_m6a_baseline.png"),
            dpi=DPI,
        )
        print("      [✓] m6A 基础稳态图")

        # 14b. NORAD 过表达 (模拟退变)
        print("\n  14b. NORAD 过表达 (退变模拟)...")
        pert = m6a_model.simulate(perturbation='NORAD_OE', t_span=(0, 400))
        y_pert = pert['y'][:, -1]
        for name, cv, pv in zip(VAR_NAMES, y_ctrl, y_pert):
            if cv > 1e-10:
                fc = pv / cv
                if abs(fc - 1.0) > 0.05:
                    arrow = '↑' if fc > 1.0 else '↓'
                    print(f"      {name:20s}: {cv:.4f} → {pv:.4f} ({arrow} {fc:.2f}x)")

        fig = m6a_model.plot_epigenetics(
            pert,
            output_path=os.path.join(OUTPUT, "32_m6a_NORAD_OE.png"),
            dpi=DPI,
        )
        print("      [✓] NORAD过表达图")

        # 14c. WTAP 敲除
        print("\n  14c. WTAP 敲除...")
        ko = m6a_model.simulate_ko(ko_gene='WTAP', t_span=(0, 400))
        y_ko = ko['y'][:, -1]
        for name, cv, kv in zip(VAR_NAMES, y_ctrl, y_ko):
            if cv > 1e-10:
                fc = kv / cv
                if abs(fc - 1.0) > 0.10:
                    print(f"      {name:20s}: {cv:.4f} → {kv:.4f} ({fc:.2f}x)")

        fig = m6a_model.plot_epigenetics(
            ko,
            output_path=os.path.join(OUTPUT, "33_m6a_WTAP_KO.png"),
            dpi=DPI,
        )
        print("      [✓] WTAP敲除图")

        # 14d. 扰动&敲除比较
        print("\n  14d. 多干预比较...")
        fig = m6a_model.compare_perturbations(
            output_path=os.path.join(OUTPUT, "34_m6a_perturbation_comparison.png"),
            dpi=DPI,
        )
        print("      [✓] 扰动比较图")

        fig = m6a_model.compare_ko(
            output_path=os.path.join(OUTPUT, "35_m6a_ko_comparison.png"),
            dpi=DPI,
        )
        print("      [✓] KO比较图")

        print("\n  [✓] 第14章完成")
    except Exception as e:
        print(f"\n  ⚠ 第14章错误: {e}")
        import traceback
        traceback.print_exc()

# ============ 15. 空间转录组图谱 ============
print("\n" + "=" * 60)
print("  🗺️ 第15章: 空间转录组图谱")
print("  📄 Advanced Science 2024: IVD 空间分子图谱")
print("=" * 60)

try:
    from analysis.spatial_transcriptomics import (
        SpatialTranscriptomics, plot_spatial_summary,
        ALL_GENES, SPATIAL_NICHE_DEFS,
    )
    _HAS_SPATIAL = True
except ImportError:
    _HAS_SPATIAL = False
    print("  ⚠ 空间转录组模块不可用，跳过")

if _HAS_SPATIAL:
    try:
        # 15a. 初始化与全管道
        print("\n  15a. 初始化空间转录组模拟...")
        st = SpatialTranscriptomics(grid_size=50, n_spots=2000, random_seed=42)
        st.generate_spatial_map()
        st.simulate_npc_zones(add_noise=True)
        st.simulate_pseudotime_trajectory()
        st.identify_spatial_niches()

        print(f"\n      分区统计:")
        zones, counts = np.unique(st.spot_labels, return_counts=True)
        for z, c in zip(zones, counts):
            print(f"        {z}: {c} spots")

        # 15b. 基因空间表达图
        print("\n  15b. 基因空间表达分布...")
        fig = st.plot_spatial_gene_expression(
            genes=['TBXT', 'ACAN', 'CTSK', 'COL1A1', 'MMP13', 'COL2A1'],
            save_path=os.path.join(OUTPUT, "36_spatial_gene_expression.png"),
        )
        print("      [✓] 基因空间表达图")

        # 15c. 伪时间轨迹
        print("\n  15c. 伪时间分化轨迹...")
        fig = st.plot_pseudotime(
            save_path=os.path.join(OUTPUT, "37_spatial_pseudotime.png"),
        )
        print("      [✓] 伪时间轨迹图")

        # 15d. 空间生态位
        print("\n  15d. 空间生态位...")
        unique_niches, niche_counts = np.unique(st.niches, return_counts=True)
        for n, c in zip(unique_niches, niche_counts):
            desc = SPATIAL_NICHE_DEFS.get(n, {}).get('description', '')
            print(f"      {n:20s}: {c:4d} spots — {desc}")

        # 15e. 综合总结图
        print("\n  15e. 综合总结图...")
        fig = plot_spatial_summary(
            st,
            save_path=os.path.join(OUTPUT, "38_spatial_summary.png"),
        )
        print("      [✓] 空间转录组总结图")

        print("\n  [✓] 第15章完成")
    except Exception as e:
        print(f"\n  ⚠ 第15章错误: {e}")
        import traceback
        traceback.print_exc()

# ============ 16. 多尺度耦合仿真 (可选) ============
print("\n" + "=" * 60)
print("  🔄 第16章: 多尺度耦合仿真")
print("=" * 60)

try:
    from simulation.coupled_engine import NPCoupledModel
    _HAS_COUPLED = True
except (ImportError, ModuleNotFoundError):
    _HAS_COUPLED = False

if _HAS_COUPLED:
    try:
        coupled = NPCoupledModel()
        result = coupled.simulate()
        print(f"  [✓] 耦合仿真完成")
        coupled.print_summary()
        # 绘制耦合动力学图
        try:
            fig = coupled.plot_coupled_dynamics(
                result,
                save_path=os.path.join(OUTPUT, "41_coupled_dynamics.png"),
                dpi=DPI,
            )
            print(f"      [✓] 耦合动力学图")
        except:
            pass
    except Exception as e:
        print(f"  ⚠ 第16章错误: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  📦 多尺度耦合模块 (coupled_engine.py) 未安装，跳过第16章")

# ============ 17. 虚拟药物筛选 (可选) ============
print("\n" + "=" * 60)
print("  💊 第17章: 虚拟药物筛选")
print("=" * 60)

try:
    from analysis.drug_screening import VirtualDrugScreening
    _HAS_DRUG = True
except (ImportError, ModuleNotFoundError):
    _HAS_DRUG = False

if _HAS_DRUG:
    try:
        drug_screener = VirtualDrugScreening()
        drug_screener.add_default_drugs()
        print(f"  药物库: {len(drug_screener.list_drugs())} 种药物")
        screening_results = drug_screener.screen()
        print(f"  [✓] 药物筛选完成")
        summary = drug_screener.summary()
        print(f"  结果概要: {summary[:300] if isinstance(summary, str) else str(summary)[:300]}")
        # 绘制药物排序图
        try:
            fig = drug_screener.plot_drug_ranking(
                output_path=os.path.join(OUTPUT, "39_drug_screening_ranking.png"),
                dpi=DPI,
            )
            print(f"      [✓] 药物排序图")
            fig = drug_screener.plot_drug_heatmap(
                output_path=os.path.join(OUTPUT, "40_drug_screening_heatmap.png"),
                dpi=DPI,
            )
            print(f"      [✓] 药物热图")
        except:
            pass
    except Exception as e:
        print(f"  ⚠ 第17章错误: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  📦 虚拟药物筛选模块 (drug_screening.py) 未安装，跳过第17章")

# ============ 19. 多尺度整合器 ===========
print("\n" + "=" * 60)
print("  🧩 第19章: 多尺度整合器 (73维耦合)")
print("=" * 60)
try:
    from simulation.multiscale_integrator import MultiScaleIntegrator
    print("\n  多尺度整合器加载成功 ✅")
    ms = MultiScaleIntegrator()

    print("\n  [1] 运行正常状态下3次迭代耦合...")
    res_normal = ms.run_coupled_iteration('normal', iterations=3)
    state_normal = ms.get_integrated_state(res_normal)
    print(f"      综合状态: 8组 {len(state_normal)} 个变量")

    print("\n  [2] 三条件对比...")
    summary = ms.compare_conditions(['normal', 'early_degeneration', 'late_degeneration'])

    print("\n  [3] 干预模拟 (MitoQ)...")
    try:
        int_res = ms.simulate_intervention('MitoQ', strength=0.5, iterations=3)
        print(f"      干预完成")
    except Exception as e:
        print(f"      干预模拟跳过: {e}")

    print("\n  绘图...")
    ms.plot_integrated_heatmap({'normal': state_normal}, output_path=f'{OUTPUT}/multiscale_heatmap.png')
    ms.plot_feedback_convergence(res_normal, output_path=f'{OUTPUT}/multiscale_convergence.png')
    print(f"      多尺度热图: {OUTPUT}/multiscale_heatmap.png")
    print(f"      收敛曲线: {OUTPUT}/multiscale_convergence.png")

    _HAS_INTEGRATOR = True
except ImportError as e:
    print(f"\n  📦 多尺度整合器模块 未安装，跳过第19章 ({e})")
    _HAS_INTEGRATOR = False

# ============ 20. 综合总结 ============
print("\n" + "=" * 60)
print("  📊 第18章: 多尺度整合器\n    · 第19章: 综合总结")
print("=" * 60)

print(f"\n  输出目录: {OUTPUT}")
print(f"\n  生成文件列表:")
files = sorted(os.listdir(OUTPUT))
for f in files:
    fpath = os.path.join(OUTPUT, f)
    size = os.path.getsize(fpath) / 1024
    print(f"    · {f:40s} ({size:.1f} KB)")

print("\n" + "=" * 60)
print("  ✅ Virtual NP Cell System v3.1 Ready!")
print("\n  已集成模块:")
v1_modules = [
    "第1章: 系统概览",
    "第2章: 知识库",
    "第3章: 差异表达 & 火山图",
    "第4章: 热图分析",
    "第5章: 生物标志物",
    "第6章: 信号通路 ODE",
    "第7章: ECM 代谢动力学",
    "第8章: ABM 微环境建模",
    "第9章: miRNA 调控网络",
    "第10章: scRNA 数据集成",
]
v2_modules = [
    "第11章: MRTF-A 力传导 → 糖酵解",
    "第12章: 代谢可塑性仿真",
    "第13章: 细胞衰老 + Senolytic",
    "第14章: m6A 表观转录组",
    "第15章: 空间转录组图谱",
    ("第16章: 多尺度耦合仿真" if _HAS_COUPLED else "第16章: 多尺度耦合 [未安装]"),
    ("第17章: 虚拟药物筛选" if _HAS_DRUG else "第17章: 虚拟药物筛选 [未安装]"),
    "第18章: 多尺度整合器",
]
for m in v1_modules + v2_modules:
    if '未安装' in m:
        print(f"    📦 {m}")
    else:
        print(f"    · {m}")

if not _HAS_COUPLED and not _HAS_DRUG:
    print("\n  📦 提示: coupled_engine.py 和 drug_screening.py 为可选模块, 安装后自动激活")

print("=" * 60)
