"""
热图可视化工具 — NP 样本间的基因表达热图
支持样本聚类、基因聚类、注释条
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def simulate_np_expression_matrix(
    n_samples: int = 20,
    n_genes: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    模拟 NP 相关基因表达矩阵

    Samples: 正常 10 + 退变早期 5 + 退变晚期 5
    Genes: NP marker + 退变 + ECM + 炎症相关
    """
    np.random.seed(seed)

    # 定义感兴趣基因
    gene_groups = {
        "NP标志": ["KRT19", "KRT18", "PAX1", "FOXF1", "CD24", "TBXT", "CLEC3A", "CA12"],
        "ECM合成": ["ACAN", "COL2A1", "SOX9", "HAPLN1", "NOG", "SPON1"],
        "MMP/ADAMTS": ["MMP3", "MMP13", "MMP2", "MMP9", "ADAMTS4", "ADAMTS5"],
        "炎症因子": ["IL1B", "TNF", "IL6", "CXCL8", "CCL2", "NFKB1"],
        "衰老相关": ["CDKN2A", "TP53", "SIRT1", "SOD2", "CAT", "TERT"],
        "信号通路": ["HIF1A", "CTNNB1", "SMAD2", "SMAD3", "NOTCH1", "AKT1"],
    }

    all_genes = []
    gene_categories = []
    for cat, genes in gene_groups.items():
        for g in genes:
            all_genes.append(g)
            gene_categories.append(cat)

    # 截取到指定数量
    if n_genes <= len(all_genes):
        all_genes = all_genes[:n_genes]
        gene_categories = gene_categories[:n_genes]
    else:
        # 补充随机基因
        extra = n_genes - len(all_genes)
        for i in range(extra):
            all_genes.append(f"OTHER_{i+1}")
            gene_categories.append("其他")

    n_genes = len(all_genes)
    # 动态分配样本数
    n_norm = min(10, n_samples // 2)
    n_early = min(5, (n_samples - n_norm) // 2)
    n_late = n_samples - n_norm - n_early

    sample_labels = (
        [f"正常_{i+1}" for i in range(n_norm)]
        + [f"退变早期_{i+1}" for i in range(n_early)]
        + [f"退变晚期_{i+1}" for i in range(n_late)]
    )
    sample_groups = (
        ["正常"] * n_norm
        + ["退变早期"] * n_early
        + ["退变晚期"] * n_late
    )

    # 生成表达矩阵 (log2 标准化, centering around 10)
    expr = np.random.normal(10, 1.5, (n_genes, n_samples))

    # 注入生物学差异
    for i, gene in enumerate(all_genes):
        if gene in ["KRT19", "KRT18", "PAX1", "FOXF1", "CD24", "TBXT", "ACAN", "COL2A1", "SOX9", "HAPLN1", "NOG"]:
            # NP marker / ECM 合成：退变后下调
            expr[i, n_norm:] -= np.random.uniform(2.0, 4.0)
            expr[i, n_norm + n_early:] -= np.random.uniform(0.5, 1.5)  # 晚期更低
        elif gene in ["MMP3", "MMP13", "ADAMTS4", "ADAMTS5", "IL1B", "TNF", "IL6", "CXCL8", "CCL2", "CDKN2A"]:
            # 退变相关/炎症：上调
            expr[i, n_norm:] += np.random.uniform(2.0, 4.5)
            expr[i, n_norm + n_early:] += np.random.uniform(1.0, 2.0)
        elif gene in ["SIRT1", "SOD2", "CAT"]:
            # 保护性基因：退变下调
            expr[i, n_norm:] -= np.random.uniform(1.0, 2.5)
        elif gene in ["MMP2", "MMP9"]:
            expr[i, n_norm:] += np.random.uniform(1.0, 2.0)

    # 添加随机噪声
    expr += np.random.normal(0, 0.3, (n_genes, n_samples))

    df = pd.DataFrame(expr, index=all_genes, columns=sample_labels)

    # 存储元数据
    metadata = {
        "gene_categories": dict(zip(all_genes, gene_categories)),
        "sample_groups": dict(zip(sample_labels, sample_groups)),
    }

    return df, metadata


def plot_heatmap(
    df: pd.DataFrame,
    metadata: dict,
    title: str = "NP 细胞基因表达热图",
    z_score: bool = True,
    cmap: str = "RdBu_r",
    figsize=(14, 12),
    output_path: str = None,
    dpi: int = 150,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
):
    """
    绘制带注释的基因表达热图

    Parameters
    ----------
    df : pd.DataFrame
        表达矩阵 (genes x samples)
    metadata : dict
        包含 gene_categories 和 sample_groups
    title : str
        标题
    z_score : bool
        是否对基因行做 Z-score 标准化
    output_path : str
        保存路径
    """
    data = df.copy()

    if z_score:
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data.T).T,
            index=data.index,
            columns=data.columns
        )
    else:
        data_scaled = data

    # 准备注释
    gene_cat = metadata["gene_categories"]
    sample_grp = metadata["sample_groups"]

    # 样本注释条 (column colors)
    group_colors = {"正常": "#2ECC71", "退变早期": "#F39C12", "退变晚期": "#E74C3C"}
    col_colors = pd.DataFrame(
        {k: [group_colors[sample_grp[s]] for s in data.columns] for k in ["分组"]},
        index=data.columns
    )
    # seaborn clustermap expects Series not DataFrame with named columns for single column
    col_colors = col_colors["分组"]

    # 基因类别注释条 (row colors) — must be Series, not DataFrame with column labels as string values
    category_palette = {
        "NP标志": "#3498DB", "ECM合成": "#1ABC9C", "MMP/ADAMTS": "#E74C3C",
        "炎症因子": "#E67E22", "衰老相关": "#9B59B6", "信号通路": "#2E86C1",
        "其他": "#95A5A6"
    }
    row_color_values = [category_palette.get(gene_cat.get(g, "其他"), "#95A5A6") for g in data.index]
    row_colors = pd.Series(row_color_values, index=data.index, name="类别")

    # 构建完整的数据集
    g = sns.clustermap(
        data_scaled,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        col_colors=[col_colors],
        row_colors=[row_colors],
        cmap=cmap,
        vmin=-3, vmax=3,
        center=0,
        method="ward",
        metric="euclidean",
        figsize=figsize,
        dendrogram_ratio=(0.1, 0.05),
        cbar_pos=(0.02, 0.8, 0.03, 0.12),
        linewidths=0.3,
        linecolor='white',
        xticklabels=True,
        yticklabels=True,
        annot=False,
        fmt=".1f",
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        fontsize=7, rotation=45, ha='right'
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=7
    )

    # 标题
    g.fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    # 为样本注释添加图例
    for i, (label, color) in enumerate(group_colors.items()):
        g.ax_heatmap.bar(0, 0, color=color, label=label, alpha=0.8)
    g.ax_heatmap.legend(
        loc='upper left', bbox_to_anchor=(1.02, 1),
        fontsize=8, framealpha=0.9,
        title="分组", title_fontsize=9
    )

    g.gs.update(top=0.92)
    g.fig.set_dpi(dpi)

    if output_path:
        g.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 热图已保存: {output_path}")

    return g


def plot_gene_group_heatmap(
    df: pd.DataFrame = None,
    metadata: dict = None,
    genes: list = None,
    gene_group_name: str = "key_np_genes",
    title: str = None,
    figsize=(10, 8),
    output_path: str = None,
):
    """绘制特定基因子集的热图"""
    if df is None or metadata is None:
        df, metadata = simulate_np_expression_matrix(n_genes=40)

    if genes:
        available = [g for g in genes if g in df.index]
        if not available:
            print("[!] 指定的基因不在表达矩阵中")
            return None
        sub_df = df.loc[available]
    else:
        sub_df = df

    t = title or f"{gene_group_name} 表达热图"
    sub_metadata = {
        "gene_categories": {g: metadata["gene_categories"].get(g, "其他") for g in sub_df.index},
        "sample_groups": metadata["sample_groups"],
    }

    return plot_heatmap(sub_df, sub_metadata, title=t, figsize=figsize, output_path=output_path)
