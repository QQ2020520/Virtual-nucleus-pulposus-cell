"""
差异表达分析与火山图工具
模拟 NP vs Control 对比的差异表达数据并绘制火山图
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

# ========== 中文字体配置 ==========
plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== NP 退变已知差异基因（基准数据） ==========
NP_DEG_GROUND_TRUTH = {
    # 退变上调 (log2FC > 0)
    "MMP3": 2.8, "MMP13": 3.2, "ADAMTS4": 2.5, "ADAMTS5": 2.1,
    "IL1B": 2.9, "TNF": 2.3, "IL6": 2.6, "CXCL8": 3.0,
    "CCL2": 2.4, "NFKB1": 1.5, "CDKN2A": 1.8, "TP53": 1.2,
    "CASP3": 1.9, "BECN1": -0.8,  # 自噬相关，退变下调
    "MMP9": 2.2, "MMP2": 1.8, "ADAMTS1": 1.6,
    # 退变下调 (log2FC < 0)
    "ACAN": -2.5, "COL2A1": -2.3, "SOX9": -1.9, "KRT19": -2.7,
    "KRT18": -2.4, "PAX1": -2.1, "FOXF1": -2.0, "TBXT": -2.6,
    "CD24": -2.2, "NOG": -1.8, "HAPLN1": -2.1, "SIRT1": -1.5,
    "SOD2": -1.0, "CAT": -0.9, "TIMP1": -0.7,
    "HIF1A": -0.6, "SLC2A1": -0.5,
    # 非显著变化
    "GAPDH": 0.1, "ACTB": 0.05,
}


def simulate_np_de_analysis(
    n_genes: int = 1000,
    n_de_genes: int = 60,
    seed: int = 42,
    noise_level: float = 0.2
) -> pd.DataFrame:
    """
    模拟 NP 退变 vs 正常对照的差异表达分析结果

    Parameters
    ----------
    n_genes : int
        总基因数
    n_de_genes : int
        显著差异基因数（含已知 ground truth）
    seed : int
        随机种子
    noise_level : float
        随机噪声水平

    Returns
    -------
    pd.DataFrame
        包含 gene, log2FC, pvalue, padj, significant 等列
    """
    np.random.seed(seed)

    # 基础基因名
    base_genes = [f"GENE_{i:04d}" for i in range(n_genes)]

    # 插入已知 NP 退变相关基因
    known_genes = list(NP_DEG_GROUND_TRUTH.keys())
    for i, kg in enumerate(known_genes):
        if i < len(base_genes):
            base_genes[i] = kg

    # 模拟对数倍数变化
    log2fc = np.random.normal(0, 0.3, n_genes)

    # 注入 ground truth 并确保足够差异
    for gene, fc in NP_DEG_GROUND_TRUTH.items():
        if gene in base_genes:
            idx = base_genes.index(gene)
            log2fc[idx] = fc + np.random.normal(0, noise_level)

    # 为前 n_de_genes 增加适当差异（除已知基因外）
    known_indices = set()
    for g in NP_DEG_GROUND_TRUTH:
        if g in base_genes:
            known_indices.add(base_genes.index(g))
    extra_count = 0
    for i in range(n_genes):
        if i in known_indices:
            continue
        if extra_count >= n_de_genes:
            break
        if abs(log2fc[i]) < 0.5:  # only boost non-extreme genes
            sign = np.random.choice([-1, 1])
            log2fc[i] += sign * np.random.uniform(1.5, 3.0)
            extra_count += 1

    # 模拟 p 值
    fc_abs = np.abs(log2fc)
    raw_p = np.exp(-5.0 * fc_abs)
    # 对已知显著性基因使用极小的 p 值（确保 BH 校正后仍 < 0.05）
    for gene, fc in NP_DEG_GROUND_TRUTH.items():
        if abs(fc) > 1.0 and gene in base_genes:
            idx = base_genes.index(gene)
            raw_p[idx] = 10 ** np.random.uniform(-30, -10)
    # 其他高 FC 基因也赋予显著 p 值
    for i in range(n_genes):
        if abs(log2fc[i]) > 1.8 and raw_p[i] > 1e-6:
            raw_p[i] = 10 ** np.random.uniform(-20, -8)
    raw_p = np.clip(raw_p, 1e-50, 1.0)
    # 多重检验校正 (BH)
    ranked = np.argsort(raw_p)
    padj = np.ones(n_genes, dtype=np.float64)
    padj[ranked] = np.minimum.accumulate(
        raw_p[ranked] * n_genes / (np.arange(n_genes) + 1)
    )
    padj = np.clip(padj, 1e-50, 1.0)

    significant = (abs(log2fc) > 0.8) & (padj < 0.05)
    up_regulated = log2fc > 1.0
    down_regulated = log2fc < -1.0

    df = pd.DataFrame({
        "gene": base_genes,
        "log2FC": log2fc,
        "pvalue": raw_p.copy(),
        "padj": padj,
        "-log10_padj": -np.log10(padj),
        "significant": significant,
        "up_regulated": significant & up_regulated,
        "down_regulated": significant & down_regulated,
        "is_known_np_gene": [g in NP_DEG_GROUND_TRUTH for g in base_genes],
    })

    # 添加基因类型标注
    df["gene_group"] = "other"
    df.loc[df["gene"].isin(getattr(NP_DEG_GROUND_TRUTH, 'keys', lambda: {})()), "gene_group"] = "known_np"
    # 手动标注
    marker_set = {"KRT19","KRT18","PAX1","FOXF1","CD24","TBXT","ACAN","SOX9","COL2A1","NOG","CLEC3A","CA12","HAPLN1","SPON1"}
    deg_set = {"MMP3","MMP13","ADAMTS4","ADAMTS5","IL1B","TNF","IL6","CXCL8","CCL2","NFKB1","CDKN2A","TP53","CASP3","MMP2","MMP9"}
    inflam_set = {"IL6","CXCL8","CCL2","NFKB1"}
    ecm_cat = {"ACAN","COL2A1","HAPLN1","MMP3","MMP13","ADAMTS4","ADAMTS5","TIMP1"}

    df["category"] = "background"
    df.loc[df["gene"].isin(marker_set), "category"] = "NP_marker"
    df.loc[df["gene"].isin(deg_set), "category"] = "degeneration"
    df.loc[df["gene"].isin(inflam_set), "category"] = "inflammation"
    df.loc[df["gene"].isin(ecm_cat), "category"] = "ECM"

    return df


def plot_volcano(
    df: pd.DataFrame,
    title: str = "NP 退变 vs 正常对照 · 差异表达火山图",
    log2fc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
    show_all_labels: bool = False,
    label_genes: Optional[list] = None,
    color_up: str = "#E74C3C",
    color_down: str = "#2E86C1",
    color_ns: str = "#95A5A6",
    figsize=(12, 9),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制火山图

    Parameters
    ----------
    df : pd.DataFrame
        差异表达分析结果
    title : str
        图表标题
    log2fc_threshold : float
        log2FC 显著性阈值
    padj_threshold : float
        校正 p 值阈值
    show_all_labels : bool
        是否标注所有显著基因
    label_genes : list, optional
        指定要标注的基因列表
    output_path : str, optional
        保存路径
    dpi : int
        图片分辨率

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 分类
    up = (df["log2FC"] >= log2fc_threshold) & (df["padj"] < padj_threshold)
    down = (df["log2FC"] <= -log2fc_threshold) & (df["padj"] < padj_threshold)
    ns = ~(up | down)

    # 绘制散点
    ax.scatter(df.loc[ns, "log2FC"], df.loc[ns, "-log10_padj"],
               c=color_ns, s=8, alpha=0.4, label=f"NS ({ns.sum()})", edgecolors='none')
    ax.scatter(df.loc[down, "log2FC"], df.loc[down, "-log10_padj"],
               c=color_down, s=20, alpha=0.7, label=f"下调 ({down.sum()})", edgecolors='none')
    ax.scatter(df.loc[up, "log2FC"], df.loc[up, "-log10_padj"],
               c=color_up, s=20, alpha=0.7, label=f"上调 ({up.sum()})", edgecolors='none')

    # 阈值线
    ax.axhline(-np.log10(padj_threshold), color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(log2fc_threshold, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(-log2fc_threshold, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

    # 标注基因
    if label_genes is None:
        label_genes = []

    if show_all_labels:
        label_genes = list(set(
            list(df.loc[up, "gene"][:20]) + list(df.loc[down, "gene"][:20])
        ))

    # 合并指定标注 + 已知 NP 基因中显著的
    known_sig = df[df["is_known_np_gene"] & df["significant"]]
    auto_label = set(known_sig["gene"].tolist() + label_genes)

    for gene in auto_label:
        row = df[df["gene"] == gene]
        if row.empty:
            continue
        row = row.iloc[0]
        offset_x, offset_y = 5, 5
        if row["log2FC"] < 0:
            offset_x = -12
        else:
            offset_x = 8
        if row["-log10_padj"] > 30:
            offset_y = -8
        else:
            offset_y = 6

        ax.annotate(
            gene,
            (row["log2FC"], row["-log10_padj"]),
            fontsize=7.5,
            fontweight='bold',
            color='#2C3E50',
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=0.6, alpha=0.6),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7)
        )

    # 轴标签和标题
    ax.set_xlabel("log₂(倍数变化)", fontsize=13)
    ax.set_ylabel("-log₁₀(校正后 P 值)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # 图例
    legend = ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    for lh in legend.legend_handles:
        lh.set_sizes([40])

    # 统计信息
    ax.text(0.02, 0.98,
            f"总基因: {len(df)}\n"
            f"上调: {up.sum()}  |  下调: {down.sum()}\n"
            f"NP 标志基因标注: {len(auto_label)}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 对称 x 轴
    max_fc = max(abs(df["log2FC"].max()), abs(df["log2FC"].min())) + 0.5
    ax.set_xlim(-max_fc, max_fc)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 火山图已保存: {output_path}")

    return fig
