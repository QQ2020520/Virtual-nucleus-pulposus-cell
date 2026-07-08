"""
spatial_transcriptomics.py — IVD 空间转录组模拟与分析模块
========================================================

基于 Advanced Science 2024 (厦门大学许韧团队) 的空间转录组研究，
模拟小鼠椎间盘三个组分 (NP, AF, CEP) 的空间分子特征、
NP 祖细胞 (NPPCs) 的空间分布与分化轨迹。

References:
    - Xu R. et al., Advanced Science, 2024. Spatially resolved transcriptomics
      of the mouse intervertebral disc.
    - 核心发现: Ctsk在NP外周表达缺失, Tie2在NP亚群中表达缺失,
      TBXT+FOXA2+ 祖细胞 → ACAN+COL2A1+ 成熟NP细胞分化轨迹.

Classes:
    SpatialTranscriptomics : 核心空间转录组模拟类
Functions:
    plot_spatial_summary  : 一键式空间转录组总结可视化

Author: Virtual NP Cell Team
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

# ── matplotlib 后端与字体 ──────────────────────────────────────────────
matplotlib.use("Agg")
plt.rcParams["font.family"] = ["HarmonyHeiTi", "Droid Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ═══════════════════════════════════════════════════════════════════════
# 基因集定义 (Gene Sets)
# ═══════════════════════════════════════════════════════════════════════

# NP 标志基因 — 髓核 (Nucleus Pulposus)
NP_MARKERS: List[str] = ["TBXT", "KRT19", "CD24", "PAX1", "FOXF1"]

# AF 标志基因 — 纤维环 (Annulus Fibrosus)
AF_MARKERS: List[str] = ["COL1A1", "THY1", "FBLN1"]

# ECM 基因 — 细胞外基质 (Extracellular Matrix)
ECM_GENES: List[str] = ["ACAN", "COL2A1", "COL9A3"]

# 退变相关基因 (Degeneration-associated)
DEGENERATION_GENES: List[str] = ["MMP13", "ADAMTS5", "IL1B"]

# 祖细胞标志基因 (Progenitor markers)
PROGENITOR_GENES: List[str] = ["CTSK", "TIE2"]

# 完整基因列表
ALL_GENES: List[str] = (
    NP_MARKERS + AF_MARKERS + ECM_GENES + DEGENERATION_GENES + PROGENITOR_GENES
)

# ── 空间生态位定义 (marker-gene-based annotation) ──────────────────────
SPATIAL_NICHE_DEFS: Dict[str, Dict[str, List[str]]] = {
    "NP_core": {
        "high": ["TBXT", "KRT19", "ACAN", "COL2A1"],
        "low": ["COL1A1", "MMP13"],
        "description": "髓核核心区 — 脊索来源成熟NP细胞",
    },
    "NP_periphery": {
        "high": ["CD24", "PAX1", "CTSK"],
        "low": ["TBXT", "COL1A1"],
        "description": "髓核外周区 — 活跃的基质重塑区域, Ctsk+ 细胞参与NP形态构建",
    },
    "AF_inner": {
        "high": ["COL1A1", "THY1", "FBLN1"],
        "low": ["ACAN"],
        "description": "内层纤维环 — 高COL1A1表达的纤维软骨细胞",
    },
    "AF_outer": {
        "high": ["COL1A1", "FBLN1"],
        "low": ["ACAN", "COL2A1"],
        "description": "外层纤维环 — 成纤维细胞样细胞",
    },
    "CEP": {
        "high": ["COL2A1", "COL9A3", "ACAN"],
        "low": ["TBXT", "KRT19"],
        "description": "软骨终板 — 透明软骨特征",
    },
    "NPPC_zone": {
        "high": ["TBXT", "FOXF1", "KRT19", "CTSK"],
        "low": ["COL1A1", "MMP13"],
        "description": "NP祖细胞区 — TBXT+FOXA2+ 祖细胞富集, 具备分化潜能",
    },
    "degenerating": {
        "high": ["MMP13", "ADAMTS5", "IL1B"],
        "low": ["ACAN", "COL2A1"],
        "description": "退变区域 — 基质降解酶高表达, ECM丢失",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数 (Helper Functions)
# ═══════════════════════════════════════════════════════════════════════

def _gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """生成2D高斯核 (Gaussian kernel) 用于空间平滑"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _make_gene_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """将稀疏点表达值插值到规则网格 (Interpolate sparse expression to grid)"""
    xi = np.column_stack([x.ravel(), y.ravel()])
    zi = griddata(xi, values.ravel(), (grid_x, grid_y), method=method)
    # 填补网格插值导致的边界NaN
    mask = np.isnan(zi)
    if mask.any():
        zi[mask] = griddata(xi, values.ravel(), (grid_x[mask], grid_y[mask]), method="nearest")
    return zi


# ═══════════════════════════════════════════════════════════════════════
# SpatialTranscriptomics 类
# ═══════════════════════════════════════════════════════════════════════

class SpatialTranscriptomics:
    """IVD 空间转录组模拟与分析

    模拟小鼠椎间盘 (IVD) 的2D空间网格，包含:
    - NP (髓核) — 圆形中心区
    - AF (纤维环) — 环状外区
    - CEP (软骨终板) — 上下边界

    支持基因表达分布模拟、空间细胞类型注释和伪时间分化轨迹分析。

    Parameters
    ----------
    grid_size : int, default=50
        空间网格边长 (Grid size for spatial map)
    n_spots : int, default=2000
        模拟的spot数量 (Number of simulated spatial spots)
    random_seed : int, default=42
        随机种子，保证可重复性
    """

    def __init__(
        self,
        grid_size: int = 50,
        n_spots: int = 2000,
        random_seed: int = 42,
    ) -> None:
        self.grid_size = grid_size
        self.n_spots = n_spots
        self.rng = np.random.default_rng(random_seed)

        # 初始化容器
        self.coordinates: Optional[np.ndarray] = None          # (n_spots, 2)
        self.spot_labels: Optional[np.ndarray] = None          # (n_spots,)  NP/AF/CEP
        self.gene_expression: Optional[pd.DataFrame] = None     # (n_spots, n_genes)
        self.pseudotime: Optional[np.ndarray] = None            # (n_spots,)
        self.niches: Optional[np.ndarray] = None                # (n_spots,) 空间生态位标签

        # 内部网格 (用于可视化插值)
        self._grid_x: Optional[np.ndarray] = None
        self._grid_y: Optional[np.ndarray] = None

    # ── 公共方法 ─────────────────────────────────────────────────────

    def generate_spatial_map(self, grid_size: Optional[int] = None) -> None:
        """生成IVD的2D空间网格

        在 [-1, 1]^2 的正方形区域内随机采样spot坐标,
        构成椎间盘切面形状 (圆形NP + 环状AF + 上下CEP)。

        Parameters
        ----------
        grid_size : int, optional
            重设网格大小
        """
        if grid_size is not None:
            self.grid_size = grid_size

        # 规则网格 (用于插值和绘图)
        x_lin = np.linspace(-1.2, 1.2, self.grid_size)
        y_lin = np.linspace(-1.2, 1.2, self.grid_size)
        self._grid_x, self._grid_y = np.meshgrid(x_lin, y_lin)

        # 随机采样spot (带一定密度偏好: 中心NP区域密度略高)
        spots = []
        while len(spots) < self.n_spots:
            x = self.rng.uniform(-1.2, 1.2)
            y = self.rng.uniform(-1.2, 1.2)
            r = np.sqrt(x**2 + y**2)
            # 只保留在IVD范围内的点 (半径 < 1.1); NP区域密度提升1.3倍
            if r < 1.1:
                accept = True
                if r < 0.35:  # NP区域 - 更高密度
                    accept = self.rng.random() < 0.90
                elif r < 0.7:  # AF区域
                    accept = self.rng.random() < 0.75
                else:  # CEP/边缘
                    accept = self.rng.random() < 0.60
                if accept:
                    spots.append([x, y])

        self.coordinates = np.array(spots[: self.n_spots])
        print(f"  [generate_spatial_map] 生成了 {len(self.coordinates)} 个spot坐标")
        return self

    def simulate_npc_zones(self, add_noise: bool = True) -> None:
        """模拟 NP / AF / CEP 三个组分的空间分区与基因表达

        基于空间位置分配 zone 标签, 并模拟各zone特异性的基因表达谱。
        - NP (半径 ≤ 0.35) : NP标志基因高表达, AF标志低表达
        - AF (0.35 < 半径 ≤ 0.70) : AF标志高表达
        - CEP (半径 > 0.70 且靠近上下边界) : ECM基因高表达

        核心发现建模:
        - Ctsk在NP外周区(0.25 < r < 0.40)表达, 中心区缺失
        - Tie2在NP亚群部分表达缺失
        - TBXT+FOXA2+祖细胞集中在NP核心区

        Parameters
        ----------
        add_noise : bool, default=True
            是否添加表达噪声以模拟生物学变异性
        """
        if self.coordinates is None:
            raise RuntimeError("请先调用 generate_spatial_map() 生成空间坐标")

        coords = self.coordinates
        n = len(coords)
        radii = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        y_abs = np.abs(coords[:, 1])

        # ── Step 1: 分配 zone 标签 ──────────────────────────────────
        self.spot_labels = np.full(n, "unknown", dtype=object)
        # NP : 半径 ≤ 0.35
        mask_np = radii <= 0.35
        self.spot_labels[mask_np] = "NP"
        # AF : 0.35 < 半径 ≤ 0.70
        mask_af = (radii > 0.35) & (radii <= 0.70) & (y_abs < 0.65)
        self.spot_labels[mask_af] = "AF"
        # CEP : (半径 > 0.70 且 y 靠近边界) 或 (y_abs > 0.65 且半径 < 0.70)
        mask_cep = (~mask_np) & (~mask_af)
        self.spot_labels[mask_cep] = "CEP"
        # 对CEP中靠近NP区域的点重新分类
        mask_cep_np = mask_cep & (radii <= 0.70)
        self.spot_labels[mask_cep_np] = "NP"

        print(f"  [simulate_npc_zones] Zone分布: NP={np.sum(self.spot_labels=='NP')}, "
              f"AF={np.sum(self.spot_labels=='AF')}, CEP={np.sum(self.spot_labels=='CEP')}")

        # ── Step 2: 模拟基因表达 ────────────────────────────────────
        expr = np.zeros((n, len(ALL_GENES)), dtype=np.float64)
        expr_df = pd.DataFrame(expr, columns=ALL_GENES)

        # NP区域基因表达
        np_idx = self.spot_labels == "NP"
        # NP核心 (r ≤ 0.20) vs NP外周 (0.20 < r ≤ 0.35)
        np_inner = np_idx & (radii <= 0.20)
        np_outer = np_idx & (radii > 0.20)

        # -- NP核心: TBXT, KRT19, CD24, PAX1, FOXF1 高表达 --
        for g in NP_MARKERS:
            expr_df.loc[np_inner, g] = self.rng.normal(3.5, 0.5, size=np_inner.sum()).clip(0)
        expr_df.loc[np_inner, "ACAN"] = self.rng.normal(4.0, 0.6, size=np_inner.sum()).clip(0)
        expr_df.loc[np_inner, "COL2A1"] = self.rng.normal(3.8, 0.5, size=np_inner.sum()).clip(0)

        # -- NP外周: CD24, PAX1, CTSK 高, TBXT 降低 --
        for g in ["CD24", "PAX1"]:
            expr_df.loc[np_outer, g] = self.rng.normal(3.2, 0.5, size=np_outer.sum()).clip(0)
        expr_df.loc[np_outer, "TBXT"] = self.rng.normal(1.0, 0.4, size=np_outer.sum()).clip(0)
        expr_df.loc[np_outer, "KRT19"] = self.rng.normal(2.5, 0.5, size=np_outer.sum()).clip(0)
        # Ctsk 在 NP 外周表达, 但核心区缺失 (建模核心发现2)
        expr_df.loc[np_outer, "CTSK"] = self.rng.normal(3.0, 0.6, size=np_outer.sum()).clip(0)
        expr_df.loc[np_inner, "CTSK"] = self.rng.normal(0.1, 0.1, size=np_inner.sum()).clip(0)
        # Tie2 在 NP 部分亚群表达缺失 (建模核心发现3)
        # 先构建完整长度的 TIE2 列，再按分区填充
        tie2_inner_vals = np.where(
            self.rng.random(np_inner.sum()) < 0.35,
            self.rng.normal(2.5, 0.5, size=np_inner.sum()),
            self.rng.normal(0.1, 0.1, size=np_inner.sum()),
        ).clip(0)
        tie2_outer_vals = np.where(
            self.rng.random(np_outer.sum()) < 0.20,
            self.rng.normal(2.0, 0.5, size=np_outer.sum()),
            self.rng.normal(0.1, 0.1, size=np_outer.sum()),
        ).clip(0)
        # 一次性分配: 用原始布尔掩码直接索引全长度列
        tie2_full = np.zeros(n, dtype=np.float64)
        tie2_full[np_inner] = tie2_inner_vals
        tie2_full[np_outer] = tie2_outer_vals
        expr_df["TIE2"] = tie2_full

        # AF区域: COL1A1, THY1, FBLN1 高表达
        af_idx = self.spot_labels == "AF"
        for g in AF_MARKERS:
            expr_df.loc[af_idx, g] = self.rng.normal(4.0, 0.5, size=af_idx.sum()).clip(0)
        expr_df.loc[af_idx, "COL9A3"] = self.rng.normal(2.0, 0.4, size=af_idx.sum()).clip(0)
        expr_df.loc[af_idx, "ACAN"] = self.rng.normal(1.5, 0.4, size=af_idx.sum()).clip(0)
        expr_df.loc[af_idx, "COL2A1"] = self.rng.normal(2.0, 0.5, size=af_idx.sum()).clip(0)

        # CEP区域: COL2A1, COL9A3, ACAN 高表达
        cep_idx = self.spot_labels == "CEP"
        expr_df.loc[cep_idx, "ACAN"] = self.rng.normal(3.5, 0.5, size=cep_idx.sum()).clip(0)
        expr_df.loc[cep_idx, "COL2A1"] = self.rng.normal(3.8, 0.5, size=cep_idx.sum()).clip(0)
        expr_df.loc[cep_idx, "COL9A3"] = self.rng.normal(4.0, 0.6, size=cep_idx.sum()).clip(0)
        expr_df.loc[cep_idx, "COL1A1"] = self.rng.normal(1.5, 0.4, size=cep_idx.sum()).clip(0)

        # 退变基因: MMP13, ADAMTS5, IL1B — 主要在AF外侧和CEP交界处表达
        deg_mask = af_idx | (cep_idx & (radii > 0.85))
        expr_df.loc[deg_mask, "MMP13"] = self.rng.normal(
            1.5, 0.5, size=deg_mask.sum()
        ).clip(0)
        expr_df.loc[deg_mask, "ADAMTS5"] = self.rng.normal(
            1.2, 0.4, size=deg_mask.sum()
        ).clip(0)
        expr_df.loc[deg_mask, "IL1B"] = self.rng.normal(
            0.8, 0.3, size=deg_mask.sum()
        ).clip(0)

        # 噪声添加
        if add_noise:
            noise = self.rng.normal(0, 0.15, expr_df.shape)
            expr_df += noise.clip(-0.5, 0.5)
            expr_df.clip(0, inplace=True)

        self.gene_expression = expr_df.round(4)
        print(f"  [simulate_npc_zones] 模拟了 {len(ALL_GENES)} 个基因的表达分布")
        return self

    def simulate_pseudotime_trajectory(self) -> None:
        """模拟分化轨迹伪时间 (Pseudotime trajectory)

        基于核心发现5的脊索→NP分化路径:
            TBXT+FOXA2+ 祖细胞 → ACAN+COL2A1+ 成熟NP细胞

        伪时间从空间中心(NP核心)向外递增, 代表从祖细胞状态
        向成熟状态的分化进程。

        计算方法:
            pseudotime = f(r) + 梯度偏移 + 随机噪声
        其中 r 为距中心的归一化距离, f(r) 为sigmoid映射函数。
        """
        if self.coordinates is None or self.gene_expression is None:
            raise RuntimeError("请先调用 generate_spatial_map() 和 simulate_npc_zones()")

        coords = self.coordinates
        radii = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        max_r = radii.max() if radii.max() > 0 else 1.0

        # sigmoid映射: 中心区伪时间≈0, 外围≈1
        r_norm = radii / max_r
        pt_raw = 1.0 / (1.0 + np.exp(-6.0 * (r_norm - 0.5)))

        # NP区域内: 以距离中心远近模拟分化梯度
        np_mask = self.spot_labels == "NP"
        r_np_norm = np.zeros_like(pt_raw)
        if np_mask.any():
            np_radii = radii[np_mask]
            r_np_norm[np_mask] = np_radii / (np_radii.max() + 1e-8)
            pt_raw[np_mask] = 0.1 + 0.7 * r_np_norm[np_mask]

        # AF区域伪时间>NP
        af_mask = self.spot_labels == "AF"
        pt_raw[af_mask] = 0.85 + 0.10 * self.rng.random(af_mask.sum())

        # CEP区域伪时间>AF
        cep_mask = self.spot_labels == "CEP"
        pt_raw[cep_mask] = 0.90 + 0.10 * self.rng.random(cep_mask.sum())

        # 随机噪声
        noise = self.rng.normal(0, 0.05, pt_raw.shape)
        self.pseudotime = np.clip(pt_raw + noise, 0.0, 1.0)

        print(f"  [simulate_pseudotime_trajectory] 伪时间范围: "
              f"[{self.pseudotime.min():.3f}, {self.pseudotime.max():.3f}]")
        return self

    def identify_spatial_niches(self) -> np.ndarray:
        """基于标志基因组合进行空间生态位 (细胞类型) 注释

        对每个spot, 根据其基因表达谱计算与预定义生态位的匹配得分,
        分配最匹配的生态位标签。

        Returns
        -------
        np.ndarray
            每个spot的生态位标签数组
        """
        if self.gene_expression is None:
            raise RuntimeError("请先模拟基因表达 (simulate_npc_zones)")

        expr = self.gene_expression
        niches = []
        niche_names = list(SPATIAL_NICHE_DEFS.keys())

        for i in range(len(expr)):
            scores = {}
            for name, defn in SPATIAL_NICHE_DEFS.items():
                high_score = np.mean([expr.iloc[i][g] for g in defn["high"] if g in expr.columns])
                low_score = np.mean([expr.iloc[i][g] for g in defn["low"] if g in expr.columns])
                scores[name] = high_score - low_score
            best = max(scores, key=scores.get)
            niches.append(best)

        self.niches = np.array(niches)

        # 统计
        unique, counts = np.unique(self.niches, return_counts=True)
        print("  [identify_spatial_niches] 生态位分布:")
        for u, c in zip(unique, counts):
            desc = SPATIAL_NICHE_DEFS.get(u, {}).get("description", "")
            print(f"    {u:20s} : {c:4d} spots — {desc}")
        return self.niches

    # ── 可视化方法 ────────────────────────────────────────────────────

    def plot_spatial_gene_expression(
        self,
        genes: Optional[List[str]] = None,
        ncols: int = 4,
        figsize: Tuple[float, float] = (16, 12),
        cmap: str = "Reds",
        show_labels: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制基因在空间网格上的表达分布热图

        Parameters
        ----------
        genes : List[str], optional
            要绘制的基因列表, 默认使用 ALL_GENES
        ncols : int, default=4
            子图列数
        figsize : tuple, default=(16, 12)
            图形尺寸
        cmap : str, default="Reds"
            颜色映射
        show_labels : bool, default=True
            是否在图上标注NP/AF/CEP分区边界
        save_path : str, optional
            保存路径

        Returns
        -------
        plt.Figure
        """
        if self.gene_expression is None or self._grid_x is None:
            raise RuntimeError("请先运行 generate_spatial_map() 和 simulate_npc_zones()")

        genes = genes or ALL_GENES
        n_genes = len(genes)
        nrows = int(np.ceil(n_genes / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = axes.ravel() if nrows > 1 else (axes if ncols > 1 else [axes])

        x_flat = self.coordinates[:, 0]
        y_flat = self.coordinates[:, 1]

        for idx, gene in enumerate(genes):
            ax = axes_flat[idx]
            if gene not in self.gene_expression.columns:
                ax.text(0.5, 0.5, f"{gene} not found", ha="center", va="center")
                continue

            values = self.gene_expression[gene].values
            # 插值到网格
            zi = _make_gene_heatmap(x_flat, y_flat, values, self._grid_x, self._grid_y)

            im = ax.imshow(
                zi,
                extent=(-1.2, 1.2, -1.2, 1.2),
                origin="lower",
                cmap=cmap,
                aspect="equal",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if show_labels:
                # 标记NP边界
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(0.35 * np.cos(theta), 0.35 * np.sin(theta),
                        "--", color="gray", lw=0.8, alpha=0.6)
                ax.plot(0.70 * np.cos(theta), 0.70 * np.sin(theta),
                        "--", color="gray", lw=0.8, alpha=0.6)
                # 标记CEP上下边界
                ax.axhline(y=0.65, xmin=0, xmax=1, linestyle="--",
                           color="gray", lw=0.8, alpha=0.6)
                ax.axhline(y=-0.65, xmin=0, xmax=1, linestyle="--",
                           color="gray", lw=0.8, alpha=0.6)

            ax.set_title(f"{gene}", fontsize=12, fontweight="bold")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)

        # 隐藏多余子图
        for idx in range(n_genes, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            "IVD 空间基因表达分布 (Spatial Gene Expression in IVD)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  → 图片已保存: {save_path}")
        return fig

    def plot_pseudotime(
        self,
        figsize: Tuple[float, float] = (10, 8),
        cmap: str = "viridis",
        show_trajectory_arrow: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制伪时间分化轨迹的空间分布

        在空间网格上展示伪时间梯度, 并绘制分化轨迹箭头,
        可视化TBXT+FOXA2+祖细胞→成熟NP细胞的分化路径。

        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            图形尺寸
        cmap : str, default="viridis"
            颜色映射
        show_trajectory_arrow : bool, default=True
            是否绘制分化方向箭头
        save_path : str, optional
            保存路径

        Returns
        -------
        plt.Figure
        """
        if self.pseudotime is None or self._grid_x is None:
            raise RuntimeError("请先运行 simulate_pseudotime_trajectory()")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── 左图: 伪时间空间分布 ──
        x_flat = self.coordinates[:, 0]
        y_flat = self.coordinates[:, 1]
        zi = _make_gene_heatmap(x_flat, y_flat, self.pseudotime, self._grid_x, self._grid_y)

        im = ax1.imshow(
            zi,
            extent=(-1.2, 1.2, -1.2, 1.2),
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Pseudotime")

        # 分区边界
        theta = np.linspace(0, 2 * np.pi, 100)
        ax1.plot(0.35 * np.cos(theta), 0.35 * np.sin(theta),
                 "--", color="white", lw=1.0, alpha=0.7, label="NP boundary")
        ax1.plot(0.70 * np.cos(theta), 0.70 * np.sin(theta),
                 "--", color="white", lw=1.0, alpha=0.7, label="AF boundary")
        ax1.axhline(y=0.65, linestyle="--", color="white", lw=0.8, alpha=0.6)
        ax1.axhline(y=-0.65, linestyle="--", color="white", lw=0.8, alpha=0.6)

        if show_trajectory_arrow and self.coordinates is not None:
            # 从中心沿多个方向绘制分化箭头
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                dx = 0.55 * np.cos(angle)
                dy = 0.55 * np.sin(angle)
                ax1.arrow(
                    0, 0, dx, dy,
                    head_width=0.05, head_length=0.06,
                    fc="orange", ec="orange", alpha=0.6,
                    label="Differentiation" if angle == 0 else "",
                )

        ax1.set_title("Pseudotime 空间分布\n(祖细胞→成熟NP细胞)", fontsize=12)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.legend(loc="upper right", fontsize=8)

        # ── 右图: 伪时间 vs 基因表达散点图 ──
        if self.gene_expression is not None:
            pt = self.pseudotime
            tbxt = self.gene_expression["TBXT"].values
            acan = self.gene_expression["ACAN"].values
            ctsk = self.gene_expression["CTSK"].values if "CTSK" in self.gene_expression.columns else np.zeros_like(pt)

            # 排序以展示趋势
            sort_idx = np.argsort(pt)
            pt_sorted = pt[sort_idx]

            # TBXT (祖细胞标志, 随伪时间下降)
            ax2.scatter(pt, tbxt, s=8, alpha=0.4, c="#E74C3C", label="TBXT (Progenitor)", rasterized=True)
            # ACAN (成熟标志, 随伪时间上升)
            ax2.scatter(pt, acan, s=8, alpha=0.4, c="#2ECC71", label="ACAN (Mature NP)", rasterized=True)
            ax2.scatter(pt, ctsk, s=8, alpha=0.3, c="#F39C12", label="CTSK (Peripheral NP)", rasterized=True)

            # 平滑趋势线
            window = max(3, len(pt) // 50)
            tbxt_smooth = np.convolve(tbxt[sort_idx], np.ones(window) / window, mode="valid")
            acan_smooth = np.convolve(acan[sort_idx], np.ones(window) / window, mode="valid")
            ctsk_smooth = np.convolve(ctsk[sort_idx], np.ones(window) / window, mode="valid")
            x_smooth = pt_sorted[:len(tbxt_smooth)]

            ax2.plot(x_smooth, tbxt_smooth, "-", color="#C0392B", lw=2, label="TBXT trend")
            ax2.plot(x_smooth, acan_smooth, "-", color="#27AE60", lw=2, label="ACAN trend")
            ax2.plot(x_smooth, ctsk_smooth, "-", color="#D35400", lw=2, label="CTSK trend")

            ax2.set_xlabel("Pseudotime")
            ax2.set_ylabel("Normalized Expression")
            ax2.set_title("Pseudotime vs 标志基因表达\n(TBXT→ACAN 分化轨迹)")
            ax2.legend(fontsize=8, loc="upper right")
            ax2.grid(True, alpha=0.3)

        fig.suptitle(
            "IVD 伪时间分化轨迹模拟\n(TBXT+FOXA2+ Progenitor → ACAN+COL2A1+ Mature NP)",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  → 图片已保存: {save_path}")
        return fig

    def run_pipeline(
        self,
        grid_size: int = 50,
        n_spots: int = 2000,
        add_noise: bool = True,
        save_plot: bool = False,
        output_dir: str = "output",
    ) -> Dict[str, object]:
        """一键运行完整分析管道 (Full pipeline in one call)

        Parameters
        ----------
        grid_size : int, default=50
            网格大小
        n_spots : int, default=2000
            Spot数量
        add_noise : bool, default=True
            是否添加表达噪声
        save_plot : bool, default=False
            是否保存输出图片
        output_dir : str, default="output"
            保存路径

        Returns
        -------
        dict
            包含所有分析结果的字典
        """
        print("=" * 60)
        print("  IVD 空间转录组分析管道启动")
        print("=" * 60)

        self.generate_spatial_map(grid_size=grid_size)
        self.simulate_npc_zones(add_noise=add_noise)
        self.simulate_pseudotime_trajectory()
        self.identify_spatial_niches()

        if save_plot:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.plot_spatial_gene_expression(
                save_path=os.path.join(output_dir, "spatial_gene_expression.png")
            )
            self.plot_pseudotime(
                save_path=os.path.join(output_dir, "pseudotime_trajectory.png")
            )

        print("=" * 60)
        print("  分析完成!")
        print("=" * 60)

        return {
            "coordinates": self.coordinates,
            "spot_labels": self.spot_labels,
            "gene_expression": self.gene_expression,
            "pseudotime": self.pseudotime,
            "niches": self.niches,
        }

    def summarize(self) -> pd.DataFrame:
        """返回分析摘要统计 (Summary statistics)

        Returns
        -------
        pd.DataFrame
            各组分的spot数量/基因表达均值和伪时间均值
        """
        if self.spot_labels is None:
            raise RuntimeError("尚无分析结果")

        zones = np.unique(self.spot_labels)
        rows = []
        for z in zones:
            mask = self.spot_labels == z
            row = {
                "Zone": z,
                "n_spots": mask.sum(),
                "pseudotime_mean": (
                    np.mean(self.pseudotime[mask]) if self.pseudotime is not None else np.nan
                ),
            }
            if self.gene_expression is not None:
                for g in ALL_GENES:
                    row[f"{g}_mean"] = self.gene_expression.loc[mask, g].mean()
            rows.append(row)
        return pd.DataFrame(rows).set_index("Zone")


# ═══════════════════════════════════════════════════════════════════════
# 一键总结可视化 (Standalone summary function)
# ═══════════════════════════════════════════════════════════════════════

def plot_spatial_summary(
    st_obj: SpatialTranscriptomics,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """一键生成IVD空间转录组总结图 (Comprehensive spatial summary)

    包含:
    1. 空间分区图 (NP / AF / CEP)
    2. 伪时间空间分布
    3. 选定的关键基因表达 (TBXT, ACAN, CTSK, COL1A1, MMP13)
    4. 生态位注释结果

    Parameters
    ----------
    st_obj : SpatialTranscriptomics
        已运行分析的空间转录组对象
    figsize : tuple, default=(16, 12)
        图形尺寸
    save_path : str, optional
        保存路径

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    ax = axes.ravel()

    coords = st_obj.coordinates
    x, y = coords[:, 0], coords[:, 1]

    zone_colors = {"NP": "#E74C3C", "AF": "#3498DB", "CEP": "#2ECC71"}
    niche_cmap = plt.cm.tab10

    # 1. 空间分区
    ax0 = ax[0]
    for zone, color in zone_colors.items():
        mask = st_obj.spot_labels == zone
        ax0.scatter(x[mask], y[mask], c=color, s=6, label=zone, alpha=0.7, rasterized=True)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax0.plot(0.35 * np.cos(theta), 0.35 * np.sin(theta), "--k", lw=1, alpha=0.5)
    ax0.plot(0.70 * np.cos(theta), 0.70 * np.sin(theta), "--k", lw=1, alpha=0.5)
    ax0.set_title("IVD Spatial Zones\n(NP / AF / CEP)", fontsize=11)
    ax0.set_xlabel("X"); ax0.set_ylabel("Y")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.set_aspect("equal")

    # 2. 伪时间
    ax1 = ax[1]
    if st_obj.pseudotime is not None:
        sc = ax1.scatter(x, y, c=st_obj.pseudotime, s=8, cmap="viridis",
                         alpha=0.8, rasterized=True)
        plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Pseudotime")
    ax1.set_title("Pseudotime Trajectory", fontsize=11)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y")
    ax1.set_aspect("equal")

    # 3-7. 关键基因表达
    key_genes = ["TBXT", "ACAN", "CTSK", "COL1A1", "MMP13"]
    for i, gene in enumerate(key_genes):
        a = ax[2 + i]
        if gene in st_obj.gene_expression.columns:
            vals = st_obj.gene_expression[gene].values
            zi = _make_gene_heatmap(x, y, vals, st_obj._grid_x, st_obj._grid_y)
            im = a.imshow(zi, extent=(-1.2, 1.2, -1.2, 1.2),
                          origin="lower", cmap="Reds", aspect="equal")
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        a.set_title(f"{gene}", fontsize=11, fontweight="bold")
        a.set_xlabel("X"); a.set_ylabel("Y")
        a.set_aspect("equal")

    # 8. 生态位
    ax7 = ax[7]
    if st_obj.niches is not None:
        unique_niches = list(SPATIAL_NICHE_DEFS.keys())
        niche_idx = [unique_niches.index(n) if n in unique_niches else -1
                     for n in st_obj.niches]
        colors = niche_cmap(np.array(niche_idx) / max(len(unique_niches), 1))
        ax7.scatter(x, y, c=colors, s=8, alpha=0.8, rasterized=True)
        # 图例
        for j, name in enumerate(unique_niches):
            ax7.scatter([], [], c=[niche_cmap(j / max(len(unique_niches), 1))],
                        label=name, s=20)
        ax7.set_title("Spatial Niches\n(Cell-type annotation)", fontsize=11)
        ax7.set_xlabel("X"); ax7.set_ylabel("Y")
        ax7.legend(fontsize=6, loc="upper right", ncol=2)
        ax7.set_aspect("equal")

    fig.suptitle(
        "IVD 空间转录组总结 (Spatial Transcriptomics Summary)\n"
        "— Advanced Science 2024 (Xu R. et al.) 核心发现模拟",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → 总结图已保存: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# __main__ 演示 (Demo)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  IVD 空间转录组分析 - Demo")
    print("=" * 60)

    # 初始化
    st = SpatialTranscriptomics(grid_size=50, n_spots=2000, random_seed=42)

    # 完整管道
    results = st.run_pipeline(save_plot=True, output_dir="output")

    # 摘要
    print("\n分区统计摘要:")
    print(st.summarize())

    # 总结图
    plot_spatial_summary(st, save_path="output/spatial_summary.png")
    print("\n✅ Demo 完成!")
