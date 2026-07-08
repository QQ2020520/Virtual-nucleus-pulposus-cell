"""
scRNA 单细胞数据集成桩 — Virtual NP Cell
============================================
模拟 NP 组织 scRNA-seq 数据生成、降维可视化、差异表达、通路活性评分

设计原则:
- 使用 numpy + sklearn，不强制依赖 scanpy/AnnData
- 返回 dict 结构 (类似 AnnData 的简化)
- 每个主要函数可独立调用和测试
"""

import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from typing import Optional, Dict, List, Tuple, Any

# sklearn 可选降维
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAIL = True
except ImportError:
    SKLEARN_AVAIL = False

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
import seaborn as sns
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 模拟单细胞数据生成
# ============================================================

def simulate_scrnaseq(
    n_cells: int = 2000,
    n_genes: int = 500,
    n_cell_types: int = 5,
    seed: int = 42,
    base_umi: float = 50.0,
    dispersion: float = 2.0,
) -> Dict[str, Any]:
    """
    模拟 NP 组织的 scRNA-seq UMI 计数矩阵。

    使用负二项分布 (Negative Binomial) 模拟 UMI counts，
    包含多种 NP 微环境细胞类型。

    Parameters
    ----------
    n_cells : int
        模拟细胞数量 (默认 2000)
    n_genes : int
        模拟基因数量 (默认 500)
    n_cell_types : int
        模拟细胞类型数量 (默认 5)
    seed : int
        随机种子
    base_umi : float
        基础 UMI 深度
    dispersion : float
        负二项离散参数 (越大越分散)

    Returns
    -------
    dict
        {
            'X': np.ndarray (n_cells, n_genes) — UMI 计数矩阵
            'gene_names': list — 基因名
            'cell_barcodes': list — 细胞条形码
            'cell_types': np.ndarray — 细胞类型标签 (str)
            'cell_type_probs': np.ndarray — 细胞类型频率
            'n_cells': int
            'n_genes': int
        }
    """
    rng = np.random.RandomState(seed)

    # 定义细胞类型
    cell_type_names = [
        'NP_cell',           # 髓核细胞
        'Immune_macrophage', # 免疫巨噬细胞
        'Fibroblast',        # 成纤维细胞
        'Endothelial',       # 内皮细胞
        'Stem_progenitor',   # 干/祖细胞
    ][:n_cell_types]

    # 细胞类型频率 (向 NP 细胞倾斜)
    probs = np.array([0.35, 0.20, 0.18, 0.15, 0.12][:n_cell_types])
    probs = probs / probs.sum()

    # 分配细胞类型
    cell_type_indices = rng.choice(n_cell_types, size=n_cells, p=probs)
    cell_types = np.array([cell_type_names[i] for i in cell_type_indices])

    # 基团特异性表达谱 (每个细胞类型有特征基因集)
    # 每种细胞类型有 20~50 个特征基因
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

    # 构建表达均值矩阵 (n_cell_types, n_genes)
    # 每种细胞类型在一些基因上高表达
    mean_expr = np.ones((n_cell_types, n_genes)) * 0.5  # 背景表达
    for ct_idx in range(n_cell_types):
        n_marker = min(rng.randint(20, 50), n_genes)
        marker_genes = rng.choice(n_genes, size=n_marker, replace=False)
        mean_expr[ct_idx, marker_genes] = rng.uniform(5.0, 20.0, size=n_marker)

    # 采样 UMI 计数
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        ct = cell_type_indices[i]
        # 每个细胞的捕获效率/测序深度不同
        cell_depth = rng.gamma(base_umi * 2.0, scale=0.5)
        mu = mean_expr[ct] * (cell_depth / base_umi)
        # 负二项采样
        # size = mu / (dispersion - 1) 如果 mu 固定的话
        # 这里直接用 NB: var = mu + mu^2 / r
        r = dispersion
        p_nb = r / (r + mu)
        counts = rng.negative_binomial(r, p_nb)
        X[i] = np.minimum(counts, 500).astype(np.float32)  # cap

    # 过滤低表达基因
    gene_total = X.sum(axis=0)
    keep = gene_total > 5
    X = X[:, keep]
    gene_names = [g for g, k in zip(gene_names, keep) if k]

    # 生成条形码
    cell_barcodes = [f"NP_CELL_{i:06d}" for i in range(n_cells)]

    return {
        'X': X,
        'gene_names': gene_names,
        'cell_barcodes': cell_barcodes,
        'cell_types': cell_types,
        'cell_type_names': cell_type_names,
        'cell_type_probs': probs,
        'n_cells': X.shape[0],
        'n_genes': X.shape[1],
    }


# ============================================================
# 2. 降维可视化 (PCA + t-SNE)
# ============================================================

def pca_tsne_visualization(
    adata: Dict[str, Any],
    n_components_pca: int = 30,
    n_components_tsne: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    PCA + t-SNE 降维并绘制细胞类型聚类图。

    Parameters
    ----------
    adata : dict
        simulate_scrnaseq() 返回的数据
    n_components_pca : int
        PCA 主成分数
    n_components_tsne : int
        t-SNE 分量数
    perplexity : int
        t-SNE perplexity
    random_state : int
        随机种子
    figsize : tuple
        图像尺寸
    output_path : str or None
        保存路径
    dpi : int
        图像分辨率

    Returns
    -------
    plt.Figure
    """
    if not SKLEARN_AVAIL:
        raise ImportError("需要 sklearn 进行降维: pip install scikit-learn")

    X = adata['X']
    cell_types = adata['cell_types']
    cell_type_names = adata['cell_type_names']

    # 对数标准化 (log1p)
    X_log = np.log1p(X)

    # PCA
    pca = PCA(n_components=min(n_components_pca, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X_log)
    var_ratio = pca.explained_variance_ratio_

    # t-SNE
    tsne = TSNE(
        n_components=n_components_tsne,
        perplexity=min(perplexity, X.shape[0] - 1),
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    X_tsne = tsne.fit_transform(X_pca[:, :min(30, X_pca.shape[1])])

    # 颜色映射
    unique_types = sorted(set(cell_types))
    color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: color_map[i] for i, t in enumerate(unique_types)}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左: PCA
    ax = axes[0]
    for ct in unique_types:
        mask = cell_types == ct
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[type_to_color[ct]],
            label=ct, alpha=0.6, s=5, edgecolors='none'
        )
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} 方差)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} 方差)", fontsize=10)
    ax.set_title("PCA 降维 — 细胞聚类", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='best', markerscale=2)

    # 右: t-SNE
    ax = axes[1]
    for ct in unique_types:
        mask = cell_types == ct
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=[type_to_color[ct]],
            label=ct, alpha=0.6, s=5, edgecolors='none'
        )
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("t-SNE 降维 — 细胞类型聚类", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='best', markerscale=2)

    fig.suptitle(
        f"NP 微环境 scRNA-seq 降维可视化 "
        f"({adata['n_cells']} cells, {adata['n_genes']} genes)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] scRNA 降维可视化: {output_path}")

    return fig


# ============================================================
# 3. 差异表达 (伪 bulk 差异分析)
# ============================================================

def pseudo_bulk_de_analysis(
    adata: Dict[str, Any],
    group_a: str = 'NP_cell',
    group_b: Optional[str] = None,
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    基于模拟数据的伪 bulk 差异表达分析。

    将单细胞按细胞类型聚合成伪 bulk，然后计算差异。
    使用简化方法: log2FC + 简单 t 检验。

    Parameters
    ----------
    adata : dict
        simulate_scrnaseq() 返回的数据
    group_a : str
        实验组细胞类型
    group_b : str or None
        对照组细胞类型 (None 时自动选择第二多的)
    min_cells : int
        最少细胞数

    Returns
    -------
    pd.DataFrame
        差异分析结果表
    """
    from scipy import stats

    X = adata['X']
    cell_types = adata['cell_types']
    gene_names = adata['gene_names']

    # 确定组 B
    if group_b is None:
        type_counts = pd.Series(cell_types).value_counts()
        for t in type_counts.index:
            if t != group_a:
                group_b = t
                break

    mask_a = cell_types == group_a
    mask_b = cell_types == group_b

    if mask_a.sum() < min_cells or mask_b.sum() < min_cells:
        raise ValueError(
            f"细胞数不足: {group_a}={mask_a.sum()}, {group_b}={mask_b.sum()}, "
            f"最少需要 {min_cells}"
        )

    # 伪 bulk: 取均值
    expr_a = X[mask_a].mean(axis=0)
    expr_b = X[mask_b].mean(axis=0)

    # log2 标准化 (加伪计数)
    log_a = np.log2(expr_a + 1)
    log_b = np.log2(expr_b + 1)

    log2fc = log_a - log_b

    # 简化 p-value: 用独立 t 检验
    pvals = np.ones(X.shape[1])
    for g in range(X.shape[1]):
        vals_a = X[mask_a, g]
        vals_b = X[mask_b, g]
        if vals_a.std() > 0 and vals_b.std() > 0:
            _, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            pvals[g] = max(p, 1e-300)

    # 多重检验校正 (Bonferroni)
    p_adj = np.minimum(pvals * X.shape[1], 1.0)

    # 标记方向
    direction = np.where(log2fc > 0.5, 'up',
                         np.where(log2fc < -0.5, 'down', 'ns'))

    df = pd.DataFrame({
        'gene': gene_names,
        'log2FC': log2fc,
        'pvalue': pvals,
        'padj': p_adj,
        'mean_A': expr_a,
        'mean_B': expr_b,
        'direction': direction,
        'significant': (p_adj < 0.05) & (np.abs(log2fc) > 0.5),
    })
    df = df.sort_values('pvalue')
    return df


# ============================================================
# 4. 通路活性评分 (简化版 AUCell)
# ============================================================

def calculate_pathway_activity(
    adata: Dict[str, Any],
    gene_sets: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, np.ndarray]:
    """
    简化版 AUCell 通路活性评分。

    对每个细胞、每个基因集计算基于排名的 AUC 评分。
    不使用完整的 AUCell 实现，而是使用快速排名法。

    Parameters
    ----------
    adata : dict
        simulate_scrnaseq() 返回的数据
    gene_sets : dict or None
        基因集字典 {通路名: [基因名列表]}
        None 时使用默认的 NP 相关通路

    Returns
    -------
    dict
        {通路名: np.ndarray (n_cells,)} 每个细胞的通路活性
    """
    if gene_sets is None:
        gene_sets = _default_np_gene_sets(adata['gene_names'])

    X = adata['X']
    n_cells = X.shape[0]

    activities = {}
    for pathway, genes in gene_sets.items():
        # 匹配基因索引
        gene_indices = []
        for g in genes:
            if g in adata['gene_names']:
                gene_indices.append(adata['gene_names'].index(g))

        if len(gene_indices) < 5:
            activities[pathway] = np.zeros(n_cells)
            continue

        # 基因表达子矩阵 (n_cells, n_genes_in_pathway)
        sub_X = X[:, gene_indices]

        # 对每个细胞，对基因表达进行排名 (降序)
        # 然后取前 20% 的基因平均排名分
        top_frac = 0.2
        scores = np.zeros(n_cells)

        for i in range(n_cells):
            ranks = np.argsort(np.argsort(-sub_X[i])) + 1  # 降序排名
            n_top = max(1, int(len(gene_indices) * top_frac))
            top_ranks = ranks[sub_X[i].argsort()[::-1][:n_top]]
            # 评分: 1 - (avg_rank / n_genes), 越高表示越靠前
            avg_rank = top_ranks.mean()
            scores[i] = 1.0 - (avg_rank / len(gene_indices))

        activities[pathway] = scores

    return activities


def _default_np_gene_sets(gene_names: List[str]) -> Dict[str, List[str]]:
    """
    默认的 NP 相关通路基因集。

    基于基因名中的 GENE_XXXX 格式生成模拟通路。
    """
    rng = np.random.RandomState(42)

    # 定义通路及在每个通路中的基因数量
    pathway_config = {
        'ECM_Synthesis': 30,       # ECM 合成
        'Inflammation': 25,        # 炎症
        'Apoptosis': 20,           # 凋亡
        'Hypoxia_Response': 25,    # 低氧应答
        'Matrix_Degradation': 20,  # 基质降解
    }

    gene_sets = {}
    all_indices = list(range(len(gene_names)))

    for pathway, n_genes in pathway_config.items():
        selected = sorted(rng.choice(
            all_indices, size=min(n_genes, len(gene_names)), replace=False
        ))
        gene_sets[pathway] = [gene_names[i] for i in selected]

    return gene_sets


def plot_pathway_activity(
    adata: Dict[str, Any],
    activities: Dict[str, np.ndarray],
    cell_type_order: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制通路活性评分热图/箱线图。

    Parameters
    ----------
    adata : dict
        simulate_scrnaseq() 返回的数据
    activities : dict
        calculate_pathway_activity() 返回的数据
    cell_type_order : list or None
        细胞类型排序
    figsize : tuple
        图像尺寸
    output_path : str or None
        保存路径
    dpi : int

    Returns
    -------
    plt.Figure
    """
    cell_types = adata['cell_types']
    unique_types = sorted(set(cell_types))
    if cell_type_order:
        unique_types = [t for t in cell_type_order if t in unique_types]

    # 构建 DataFrame
    records = []
    for pathway, scores in activities.items():
        for i, ct in enumerate(cell_types):
            records.append({
                'Pathway': pathway,
                'CellType': ct,
                'Activity': scores[i],
            })
    df = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左: 箱线图
    sns.boxplot(
        data=df, x='Pathway', y='Activity', hue='CellType',
        ax=ax1, palette='tab10', linewidth=0.8, fliersize=2
    )
    ax1.set_title("通路活性评分 (按细胞类型)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("")
    ax1.set_ylabel("AUCell 样评分", fontsize=10)
    ax1.tick_params(axis='x', rotation=25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=7, loc='upper right')

    # 右: 热图 (每条通路在各细胞类型的均值)
    heat_data = df.groupby(['Pathway', 'CellType'])['Activity'].mean().unstack()
    sns.heatmap(
        heat_data, ax=ax2, cmap='YlOrRd', annot=True, fmt='.3f',
        linewidths=0.5, cbar_kws={'label': '平均活性'}
    )
    ax2.set_title("平均通路活性热图", fontsize=11, fontweight='bold')
    ax2.set_xlabel("细胞类型", fontsize=10)
    ax2.set_ylabel("通路", fontsize=10)

    fig.suptitle(
        "NP 微环境细胞通路活性评分",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 通路活性分析图: {output_path}")

    return fig


# ============================================================
# 5. 一键运行全部
# ============================================================

def run_full_scrnaseq_pipeline(
    n_cells: int = 1500,
    n_genes: int = 400,
    output_dir: str = './output',
    dpi: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    运行完整单细胞分析流程:
    1. 生成模拟数据
    2. 降维可视化
    3. 差异表达分析
    4. 通路活性评分

    Parameters
    ----------
    n_cells : int
    n_genes : int
    output_dir : str
    dpi : int
    seed : int

    Returns
    -------
    dict
        包含数据和分析结果的字典
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("🧬 scRNA-seq 集成分析管道")
    print("=" * 50)

    # 1. 生成数据
    print(f"\n[1/4] 生成模拟 scRNA-seq 数据...")
    adata = simulate_scrnaseq(
        n_cells=n_cells, n_genes=n_genes, seed=seed
    )
    print(f"  → {adata['n_cells']} cells × {adata['n_genes']} genes")
    print(f"  → 细胞类型: {adata['cell_type_names']}")

    # 2. 降维可视化
    print(f"\n[2/4] PCA + t-SNE 降维...")
    try:
        fig1 = pca_tsne_visualization(
            adata,
            output_path=os.path.join(output_dir, 'scrnaseq_dimreduction.png'),
            dpi=dpi
        )
        print(f"  → 降维图已保存")
    except Exception as e:
        print(f"  ⚠ 降维失败: {e}")

    # 3. 差异表达
    print(f"\n[3/4] 伪 bulk 差异表达分析...")
    try:
        de_df = pseudo_bulk_de_analysis(adata)
        n_sig = de_df['significant'].sum()
        print(f"  → 显著差异基因: {n_sig}")
        print(f"  → Top5:")
        for _, row in de_df.head(5).iterrows():
            print(f"    {row['gene']:12s}  log2FC={row['log2FC']:+.3f}  padj={row['padj']:.2e}")
    except Exception as e:
        print(f"  ⚠ 差异表达分析失败: {e}")
        de_df = None

    # 4. 通路活性评分
    print(f"\n[4/4] 通路活性评分...")
    try:
        activities = calculate_pathway_activity(adata)
        fig2 = plot_pathway_activity(
            adata, activities,
            output_path=os.path.join(output_dir, 'scrnaseq_pathway_activity.png'),
            dpi=dpi
        )
        for pathway, scores in activities.items():
            print(f"  · {pathway}: mean={scores.mean():.3f} ± {scores.std():.3f}")
    except Exception as e:
        print(f"  ⚠ 通路活性评分失败: {e}")

    print(f"\n✅ scRNA-seq 管道完成")
    return {
        'adata': adata,
        'de_results': de_df,
        'pathway_activities': activities if 'activities' in dir() else None,
        'output_dir': output_dir,
    }


# 独立运行
if __name__ == '__main__':
    result = run_full_scrnaseq_pipeline(n_cells=800, n_genes=300)
