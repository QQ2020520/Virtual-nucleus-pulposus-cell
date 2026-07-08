"""
miRNA 调控网络模块 — Virtual NP Cell
========================================
NP 退变相关的 miRNA-mRNA 调控数据库、表达模拟、网络可视化、扰动模拟、诊断标志物分析
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any

# networkx 用于构建和可视化调控网络
try:
    import networkx as nx
    NETWORKX_AVAIL = True
except ImportError:
    NETWORKX_AVAIL = False

from sklearn.metrics import roc_curve, auc
from scipy import stats

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. miRNA-mRNA 调控数据库 (NP 退变相关)
# ============================================================

MIRNA_TARGET_DB: Dict[str, List[Dict[str, Any]]] = {
    'miR-21': [
        {'target': 'MMP13',  'regulation': 'indirect_up',   'evidence': '文献支持: miR-21↑ → NF-κB↑ → MMP13↑'},
        {'target': 'ADAMTS5','regulation': 'indirect_up',   'evidence': '退变中 miR-21 上调 ADAMTS5'},
        {'target': 'PTEN',   'regulation': 'direct_down',  'evidence': 'miR-21 直接靶向抑制 PTEN'},
        {'target': 'SOX9',   'regulation': 'indirect_down','evidence': '间接抑制 SOX9 表达'},
        {'target': 'COL2A1', 'regulation': 'indirect_down','evidence': '通过抑制 SOX9 下调 COL2A1'},
    ],
    'miR-155': [
        {'target': 'MMP13',  'regulation': 'direct_up',    'evidence': 'miR-155 直接激活 MMP13 表达'},
        {'target': 'MMP3',   'regulation': 'direct_up',    'evidence': '促炎 miRNA 上调 MMP3'},
        {'target': 'IL1B',   'regulation': 'indirect_up',  'evidence': '正反馈回路: miR-155 → NF-κB → IL1B'},
        {'target': 'TNF',    'regulation': 'indirect_up',  'evidence': '促炎正反馈'},
        {'target': 'SOX9',   'regulation': 'direct_down',  'evidence': 'miR-155 直接靶向 SOX9 3\'UTR'},
    ],
    'miR-146a': [
        {'target': 'IL1B',   'regulation': 'direct_down',  'evidence': '负反馈: miR-146a 靶向 IRAK1/TRAF6 抑制 IL1B'},
        {'target': 'TNF',    'regulation': 'direct_down',  'evidence': '通过 NF-κB 负调控抑制 TNF'},
        {'target': 'MMP13',  'regulation': 'direct_down',  'evidence': '抗炎 miRNA 下调 MMP13'},
        {'target': 'NFKB1',  'regulation': 'direct_down',  'evidence': '直接靶向 NF-κB 信号通路'},
    ],
    'miR-222': [
        {'target': 'COL2A1', 'regulation': 'direct_down',  'evidence': 'miR-222 直接靶向 COL2A1 3\'UTR'},
        {'target': 'ACAN',   'regulation': 'direct_down',  'evidence': '靶向聚集蛋白聚糖 mRNA'},
        {'target': 'SOX9',   'regulation': 'direct_down',  'evidence': '抑制 SOX9 间接抑制 ECM 合成'},
        {'target': 'MMP13',  'regulation': 'indirect_up',  'evidence': '退变中上调 MMP13'},
    ],
    'miR-27a': [
        {'target': 'COL2A1', 'regulation': 'direct_down',  'evidence': 'miR-27a 直接靶向 COL2A1'},
        {'target': 'ACAN',   'regulation': 'direct_down',  'evidence': '靶向 ACAN mRNA'},
        {'target': 'SOX9',   'regulation': 'direct_down',  'evidence': '直接抑制 SOX9'},
        {'target': 'MMP13',  'regulation': 'indirect_up',  'evidence': '通过抑制 ECM 基因间接促退变'},
    ],
    'miR-140-5p': [
        {'target': 'ADAMTS5','regulation': 'direct_down',  'evidence': 'miR-140 直接靶向 ADAMTS5'},
        {'target': 'MMP13',  'regulation': 'direct_down',  'evidence': '保护性 miRNA 抑制 MMP13'},
        {'target': 'SOX9',   'regulation': 'indirect_up',  'evidence': '维持 SOX9 的 ECM 保护效应'},
        {'target': 'COL2A1', 'regulation': 'indirect_up',  'evidence': '间接保护 ECM 合成'},
    ],
    'miR-221': [
        {'target': 'COL2A1', 'regulation': 'direct_down',  'evidence': 'miR-221 靶向 ECM 基因'},
        {'target': 'ACAN',   'regulation': 'direct_down',  'evidence': '下调 ACAN 表达'},
        {'target': 'MMP13',  'regulation': 'indirect_up',  'evidence': '退变中协同上调'},
        {'target': 'TIMPs',  'regulation': 'direct_down',  'evidence': '抑制 TIMPs 打破 ECM 平衡'},
    ],
    'miR-199a': [
        {'target': 'COL2A1', 'regulation': 'direct_down',  'evidence': 'miR-199a 直接靶向 COL2A1'},
        {'target': 'ACAN',   'regulation': 'direct_down',  'evidence': '降低 ECM 合成能力'},
        {'target': 'SOX9',   'regulation': 'direct_down',  'evidence': '抑制 SOX9 介导的 ECM 转录'},
        {'target': 'MMP13',  'regulation': 'indirect_up',  'evidence': '退变 ECM 重塑中的关键节点'},
    ],
    'miR-let-7a': [
        {'target': 'IL6',    'regulation': 'direct_down',  'evidence': '靶向 IL6 3\'UTR 抑制炎症'},
        {'target': 'MMP13',  'regulation': 'indirect_down','evidence': '抗炎抗退变 miRNA'},
        {'target': 'COL2A1', 'regulation': 'indirect_up',  'evidence': '保护 ECM 稳态'},
    ],
    'miR-34a': [
        {'target': 'BCL2',   'regulation': 'direct_down',  'evidence': '促凋亡: 靶向 BCL2'},
        {'target': 'SIRT1',  'regulation': 'direct_down',  'evidence': '靶向 SIRT1 促衰老'},
        {'target': 'CDKN2A', 'regulation': 'indirect_up',  'evidence': '诱导衰老标志物'},
        {'target': 'MMP13',  'regulation': 'indirect_up',  'evidence': '退变相关'},
    ],
    'miR-15a': [
        {'target': 'BCL2',   'regulation': 'direct_down',  'evidence': '促凋亡 miRNA'},
        {'target': 'CCND1',  'regulation': 'direct_down',  'evidence': '抑制细胞周期'},
        {'target': 'MMP3',   'regulation': 'indirect_up',  'evidence': '退变上调'},
    ],
    # --- 保护性 miRNA ---
    'miR-149': [
        {'target': 'IL1B',   'regulation': 'direct_down',  'evidence': '抗炎: 靶向 IL1B'},
        {'target': 'TNF',    'regulation': 'direct_down',  'evidence': '抑制 TNF-α 表达'},
        {'target': 'MMP13',  'regulation': 'direct_down',  'evidence': '保护性下调 MMP13'},
        {'target': 'ADAMTS5','regulation': 'direct_down',  'evidence': '抑制基质降解'},
    ],
    'miR-145': [
        {'target': 'SOX9',   'regulation': 'indirect_up',  'evidence': '促进 NP 细胞表型维持'},
        {'target': 'COL2A1', 'regulation': 'indirect_up',  'evidence': '保护 ECM'},
        {'target': 'MMP13',  'regulation': 'direct_down',  'evidence': '抑制退变相关 MMP'},
    ],
    'miR-199b-5p': [
        {'target': 'COL2A1', 'regulation': 'indirect_up',  'evidence': '保护性 ECM 维持'},
        {'target': 'ACAN',   'regulation': 'indirect_up',  'evidence': '维持聚集蛋白聚糖表达'},
        {'target': 'MMP13',  'regulation': 'direct_down',  'evidence': '抑制基质金属蛋白酶'},
    ],
}

# 目标基因列表 (所有被调控的 mRNA)
TARGET_GENES = sorted(set(
    item['target']
    for mirna_targets in MIRNA_TARGET_DB.values()
    for item in mirna_targets
))


def get_mirna_target_summary() -> pd.DataFrame:
    """
    整理 miRNA 调控关系汇总表。

    Returns
    -------
    pd.DataFrame
        列: miRNA, target, regulation, evidence
    """
    rows = []
    for mirna, targets in MIRNA_TARGET_DB.items():
        for item in targets:
            rows.append({
                'miRNA': mirna,
                'target': item['target'],
                'regulation': item['regulation'],
                'evidence': item['evidence'],
            })
    return pd.DataFrame(rows)


# ============================================================
# 2. miRNA 表达模拟 (正常 vs 退变)
# ============================================================

def simulate_mirna_expression(
    n_normal: int = 30,
    n_degen: int = 30,
    seed: int = 42,
    fold_changes: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    生成正常 vs 退变的 miRNA 表达模拟数据。

    基于文献报道的 NP 退变中 miRNA 表达变化趋势:
    - 促退变 miRNA (miR-21, -155, -222, -27a, -221, -199a, -34a): 退变中↑
    - 保护性 miRNA (miR-140, -146a, -149, -145, -let-7a): 退变中↓

    Parameters
    ----------
    n_normal : int
        正常样本数
    n_degen : int
        退变样本数
    seed : int
        随机种子
    fold_changes : dict or None
        自定义 fold change 覆盖

    Returns
    -------
    dict
        {
            'expression': np.ndarray (n_samples, n_mirnas) — 表达矩阵
            'mirna_names': list
            'labels': np.ndarray (n_samples,) — 0=normal, 1=degeneration
            'group_names': ['Normal', 'Degeneration']
            'log2FC': dict — miRNA → log2FC
            'pvalues': dict — miRNA → p-value
        }
    """
    rng = np.random.RandomState(seed)
    mirna_names = sorted(MIRNA_TARGET_DB.keys())
    n_mirnas = len(mirna_names)
    n_samples = n_normal + n_degen

    # 默认 fold changes (log2 scale)
    default_fc = {
        'miR-21':     1.8,   # ↑
        'miR-155':    2.1,   # ↑↑
        'miR-146a':  -1.5,   # ↓
        'miR-222':    1.5,   # ↑
        'miR-27a':    1.3,   # ↑
        'miR-140-5p':-1.2,   # ↓
        'miR-221':    1.6,   # ↑
        'miR-199a':   1.4,   # ↑
        'miR-let-7a':-1.8,   # ↓↓
        'miR-34a':    2.5,   # ↑↑ (衰老相关)
        'miR-15a':    1.0,   # ↑
        'miR-149':   -2.0,   # ↓↓
        'miR-145':   -1.3,   # ↓
        'miR-199b-5p':-1.0, # ↓
    }
    if fold_changes:
        default_fc.update(fold_changes)

    log2fc_arr = np.array([default_fc.get(m, 0.0) for m in mirna_names])

    # 构建表达矩阵
    expr = np.zeros((n_samples, n_mirnas))
    labels = np.zeros(n_samples, dtype=int)
    labels[n_normal:] = 1  # 退变样本

    # 基础表达 (log2 scale, 均值 8, 标准差 0.5)
    base_mean = 8.0
    base_std = 0.5

    for i in range(n_samples):
        is_degen = i >= n_normal
        for j, m in enumerate(mirna_names):
            fc = log2fc_arr[j] if is_degen else 0.0
            mean = base_mean + fc
            # 添加个体差异
            individual_noise = rng.normal(0, 0.3)
            expr[i, j] = max(0, mean + individual_noise + rng.normal(0, base_std))

    # p-value 模拟 (基于效应量)
    pvals = {}
    for j, m in enumerate(mirna_names):
        normal_vals = expr[:n_normal, j]
        degen_vals = expr[n_normal:, j]
        if normal_vals.std() > 0 and degen_vals.std() > 0:
            _, p = stats.ttest_ind(degen_vals, normal_vals, equal_var=False)
        else:
            p = 0.5
        pvals[m] = max(p, 1e-10)

    return {
        'expression': expr,
        'mirna_names': mirna_names,
        'labels': labels,
        'group_names': ['Normal', 'Degeneration'],
        'log2FC': {m: log2fc_arr[i] for i, m in enumerate(mirna_names)},
        'pvalues': pvals,
        'n_normal': n_normal,
        'n_degen': n_degen,
    }


def plot_mirna_expression_heatmap(
    mirna_data: Dict[str, Any],
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制 miRNA 表达热图 (正常 vs 退变)。
    """
    expr = mirna_data['expression']
    mirna_names = mirna_data['mirna_names']
    labels = mirna_data['labels']

    # 如果指定 top_n, 按 log2FC 绝对值排序
    if top_n and top_n < len(mirna_names):
        fc_abs = np.abs([mirna_data['log2FC'][m] for m in mirna_names])
        top_idx = np.argsort(fc_abs)[::-1][:top_n]
        expr = expr[:, top_idx]
        mirna_names = [mirna_names[i] for i in top_idx]

    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=figsize)

    # 归一化每列 (z-score)
    expr_z = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-8)

    # 排序样本: 正常在前, 退变在后
    order = np.argsort(labels)
    expr_z = expr_z[order]
    sorted_labels = labels[order]

    im = ax.imshow(expr_z.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)

    # 标注分组
    n_normal = (sorted_labels == 0).sum()
    ax.axvline(n_normal - 0.5, color='black', linewidth=1.5, linestyle='--')
    ax.text(n_normal / 2, len(mirna_names) + 0.5, 'Normal',
            ha='center', fontsize=10, fontweight='bold')
    ax.text(n_normal + (len(sorted_labels) - n_normal) / 2,
            len(mirna_names) + 0.5, 'Degeneration',
            ha='center', fontsize=10, fontweight='bold', color='red')

    ax.set_yticks(range(len(mirna_names)))
    ax.set_yticklabels(mirna_names, fontsize=9)
    ax.set_xlabel('Samples', fontsize=10)
    ax.set_title('miRNA 表达热图 (Normal vs Degeneration)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] miRNA 表达热图: {output_path}")

    return fig


# ============================================================
# 3. 调控网络可视化
# ============================================================

def build_mirna_network() -> nx.DiGraph:
    """
    构建 miRNA-mRNA 调控有向图网络。

    Returns
    -------
    nx.DiGraph
        节点: miRNA + mRNA; 边: 调控关系 (激活/抑制)
    """
    G = nx.DiGraph()

    for mirna, targets in MIRNA_TARGET_DB.items():
        G.add_node(mirna, node_type='miRNA', color='#3498DB')
        for item in targets:
            target = item['target']
            reg = item['regulation']

            G.add_node(target, node_type='mRNA')
            if 'up' in reg:
                G.add_edge(mirna, target, relation='activation', color='#E74C3C')
            elif 'down' in reg:
                G.add_edge(mirna, target, relation='inhibition', color='#27AE60')

    return G


def plot_mirna_network(
    G: Optional[nx.DiGraph] = None,
    figsize: Tuple[int, int] = (14, 12),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制 miRNA-mRNA 调控网络。

    Parameters
    ----------
    G : nx.DiGraph or None
        调控网络 (None 时自动构建)
    figsize : tuple
    output_path : str or None
    dpi : int

    Returns
    -------
    plt.Figure
    """
    if not NETWORKX_AVAIL:
        raise ImportError("需要 networkx: pip install networkx")

    if G is None:
        G = build_mirna_network()

    fig, ax = plt.subplots(figsize=figsize)

    # 节点分类
    mirna_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'miRNA']
    mrna_nodes = [n for n in G.nodes(data=True) if d.get('node_type') == 'mRNA']

    # 布局: miRNA 在上方, mRNA 在下方
    pos = {}
    for i, n in enumerate(mirna_nodes):
        pos[n] = (i - len(mirna_nodes) / 2, 1.0)
    for i, n in enumerate(mrna_nodes):
        pos[n] = (i - len(mrna_nodes) / 2, -1.0)

    # 边颜色
    edge_colors = []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        edge_colors.append(d.get('color', '#95A5A6'))
        edge_widths.append(1.5 if d.get('relation') == 'activation' else 1.0)

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos, nodelist=mirna_nodes,
        node_color='#3498DB', node_size=1200, node_shape='o',
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=mrna_nodes,
        node_color='#E67E22', node_size=900, node_shape='s',
        ax=ax
    )

    # 绘制边
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=15, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1', ax=ax
    )

    # 绘制标签
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_weight='bold',
        ax=ax
    )

    # 图例
    ax.plot([], [], 'o', color='#3498DB', markersize=10, label='miRNA')
    ax.plot([], [], 's', color='#E67E22', markersize=10, label='mRNA Target')
    ax.plot([], [], '-', color='#E74C3C', linewidth=2, label='激活/上调')
    ax.plot([], [], '-', color='#27AE60', linewidth=2, label='抑制/下调')
    ax.legend(fontsize=9, loc='upper left')

    ax.set_title(
        f'NP 退变相关 miRNA-mRNA 调控网络\n'
        f'({len(mirna_nodes)} miRNAs → {len(mrna_nodes)} 靶基因)',
        fontsize=14, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] miRNA 调控网络: {output_path}")

    return fig


# ============================================================
# 4. miRNA 扰动模拟 (过表达/敲除)
# ============================================================

def simulate_mirna_perturbation(
    mirna_name: str,
    perturbation_type: str = 'overexpression',
    fold_change: float = 2.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    模拟 miRNA 过表达/敲除对下游 mRNA 的影响预测。

    基于调控数据库中的关系和简单剂量效应模型。

    Parameters
    ----------
    mirna_name : str
        要扰动的 miRNA 名称
    perturbation_type : str
        'overexpression' (过表达) 或 'knockdown' (敲除)
    fold_change : float
        扰动倍数 (overexpression: >1, knockdown: 0~1)
    seed : int

    Returns
    -------
    dict
        {靶基因: mRNA_变化倍率}
        正值表示上调, 负值表示下调
    """
    rng = np.random.RandomState(seed)

    if mirna_name not in MIRNA_TARGET_DB:
        raise ValueError(
            f"未知 miRNA: {mirna_name}. 可用: {list(MIRNA_TARGET_DB.keys())}"
        )

    targets = MIRNA_TARGET_DB[mirna_name]

    # 扰动方向
    # overexpression → 增强 miRNA 效应
    # knockdown → 减弱 miRNA 效应
    if perturbation_type == 'overexpression':
        direction = 1.0
    elif perturbation_type == 'knockdown':
        direction = -1.0
    else:
        raise ValueError("perturbation_type 必须是 'overexpression' 或 'knockdown'")

    results = {}
    for item in targets:
        target = item['target']
        reg = item['regulation']

        # 基于调控关系预测影响方向
        if 'up' in reg:
            # miRNA 上调 → 靶基因上调
            # 过表达 miRNA => 靶基因上调; 敲除 => 靶基因下调
            mrna_change = direction * fold_change
        elif 'down' in reg:
            # miRNA 下调 → 靶基因下调
            # 过表达 miRNA => 靶基因下调; 敲除 => 靶基因上调
            mrna_change = -direction * fold_change

        # 添加噪声
        noise = rng.uniform(-0.2, 0.2)
        mrna_change *= (1 + noise)

        results[target] = round(mrna_change, 3)

    return results


def plot_mirna_perturbation(
    pert_results: Dict[str, float],
    mirna_name: str,
    pert_type: str = 'overexpression',
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制 miRNA 扰动对下游 mRNA 的影响柱状图。
    """
    fig, ax = plt.subplots(figsize=figsize)

    targets = list(pert_results.keys())
    changes = list(pert_results.values())

    colors = ['#E74C3C' if c > 0 else '#27AE60' for c in changes]
    bars = ax.barh(targets, changes, color=colors, alpha=0.8, edgecolor='white')

    ax.axvline(0, color='black', linewidth=1)

    # 添加数值标签
    for bar, val in zip(bars, changes):
        label_x = val + 0.05 if val >= 0 else val - 0.3
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}', va='center', fontsize=9, fontweight='bold')

    # 图例
    ax.plot([], [], 's', color='#E74C3C', label='上调 (Up)')
    ax.plot([], [], 's', color='#27AE60', label='下调 (Down)')
    ax.legend(fontsize=9)

    pert_label = '过表达 (Overexpression)' if pert_type == 'overexpression' else '敲除 (Knockdown)'
    ax.set_xlabel('mRNA 表达变化倍数', fontsize=11)
    ax.set_title(f'miRNA 扰动模拟: {mirna_name} {pert_label}', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] miRNA 扰动模拟: {output_path}")

    return fig


# ============================================================
# 5. miRNA 诊断标志物分析 (ROC)
# ============================================================

def mirna_diagnostic_analysis(
    mirna_data: Dict[str, Any],
    target_mirnas: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    基于 miRNA 表达水平的 ROC 分析。

    对每个 miRNA 计算 ROC 曲线和 AUC，
    评估其作为 NP 退变诊断标志物的效能。

    Parameters
    ----------
    mirna_data : dict
        simulate_mirna_expression() 的输出
    target_mirnas : list or None
        要分析的 miRNA 列表 (None 时分析所有)
    figsize : tuple
    output_path : str or None
    dpi : int

    Returns
    -------
    plt.Figure
    """
    expr = mirna_data['expression']
    mirna_names = mirna_data['mirna_names']
    labels = mirna_data['labels']

    if target_mirnas is None:
        # 按 log2FC 绝对值排序选 top miRNA
        fc_abs = np.abs([mirna_data['log2FC'][m] for m in mirna_names])
        top_idx = np.argsort(fc_abs)[::-1][:6]
        target_mirnas = [mirna_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=figsize)

    auc_scores = {}
    for m in target_mirnas:
        if m not in mirna_names:
            continue
        idx = mirna_names.index(m)
        values = expr[:, idx]

        fpr, tpr, _ = roc_curve(labels, values)
        roc_auc = auc(fpr, tpr)
        auc_scores[m] = roc_auc

        ax.plot(fpr, tpr, lw=2,
                label=f'{m} (AUC = {roc_auc:.3f})')

    # 对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='随机 (AUC=0.5)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('假阳性率 (1 - Specificity)', fontsize=11)
    ax.set_ylabel('真阳性率 (Sensitivity)', fontsize=11)
    ax.set_title('miRNA 诊断标志物 ROC 曲线分析', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] miRNA ROC 分析: {output_path}")

    # 打印排序结果
    print("\n  miRNA 诊断效能排序:")
    for m, a in sorted(auc_scores.items(), key=lambda x: -x[1]):
        print(f"    {m:15s}  AUC = {a:.4f}")

    return fig


# ============================================================
# 6. 一键运行全部
# ============================================================

def run_full_mirna_pipeline(
    n_normal: int = 30,
    n_degen: int = 30,
    output_dir: str = './output',
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    运行完整 miRNA 分析管道:
    1. 表达模拟
    2. 调控网络可视化
    3. 扰动模拟
    4. ROC 分析

    Parameters
    ----------
    n_normal : int
    n_degen : int
    output_dir : str
    dpi : int

    Returns
    -------
    dict
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("🎯 miRNA 调控网络分析管道")
    print("=" * 50)

    print(f"\n[1/4] 模拟 miRNA 表达...")
    mirna_data = simulate_mirna_expression(n_normal=n_normal, n_degen=n_degen)
    print(f"  → {n_normal + n_degen} 样本, {len(mirna_data['mirna_names'])} miRNAs")

    # 表达热图
    fig1 = plot_mirna_expression_heatmap(
        mirna_data,
        output_path=os.path.join(output_dir, 'mirna_expression_hm.png'),
        dpi=dpi
    )
    print(f"  → 表达热图已保存")

    # log2FC 摘要
    print(f"\n  miRNA 差异表达摘要:")
    for m in sorted(mirna_data['log2FC'].keys(),
                    key=lambda x: -abs(mirna_data['log2FC'][x])):
        fc = mirna_data['log2FC'][m]
        p = mirna_data['pvalues'][m]
        direction = '↑' if fc > 0 else '↓'
        print(f"    {m:15s}  log2FC={fc:+.2f}  p={p:.2e}  {direction}")

    print(f"\n[2/4] 调控网络可视化...")
    if NETWORKX_AVAIL:
        fig2 = plot_mirna_network(
            output_path=os.path.join(output_dir, 'mirna_network.png'),
            dpi=dpi
        )
        print(f"  → 网络图已保存")
    else:
        print(f"  ⚠ networkx 不可用, 跳过")

    print(f"\n[3/4] miRNA 扰动模拟...")
    for mirna in ['miR-155', 'miR-140-5p', 'miR-21']:
        for ptype in ['overexpression', 'knockdown']:
            try:
                res = simulate_mirna_perturbation(mirna, ptype)
                fpath = os.path.join(
                    output_dir,
                    f'mirna_perturb_{mirna}_{ptype}.png'
                )
                plot_mirna_perturbation(res, mirna, ptype,
                                        output_path=fpath, dpi=dpi)
                n_up = sum(1 for v in res.values() if v > 0)
                n_down = sum(1 for v in res.values() if v < 0)
                print(f"  · {mirna} {ptype}: {n_up}↑ {n_down}↓")
            except Exception as e:
                print(f"  ⚠ {mirna} {ptype}: {e}")

    print(f"\n[4/4] miRNA 诊断标志物 ROC 分析...")
    fig4 = mirna_diagnostic_analysis(
        mirna_data,
        output_path=os.path.join(output_dir, 'mirna_roc.png'),
        dpi=dpi
    )
    print(f"  → ROC 分析完成")

    print(f"\n✅ miRNA 管道完成")
    return {'mirna_data': mirna_data, 'output_dir': output_dir}


if __name__ == '__main__':
    run_full_mirna_pipeline(n_normal=20, n_degen=20)
