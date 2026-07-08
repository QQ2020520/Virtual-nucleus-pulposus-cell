"""
生物标志物预测与趋势分析工具
NP 退变过程中的标志物筛选、时间趋势预测、ROC 分析
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.interpolate import make_interp_spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ========== 退变阶段定义 ==========
DEG_STAGES = ["正常", "退变早期", "退变中期", "退变晚期"]
DEG_STAGE_MAP = {s: i for i, s in enumerate(DEG_STAGES)}


def simulate_biomarker_data(
    n_patients: int = 100,
    seed: int = 42,
    noise: float = 0.15
) -> pd.DataFrame:
    """
    模拟 NP 退变患者的临床+分子生物标志物数据

    Returns
    -------
    pd.DataFrame
        包含基因表达、临床评分、退变分期的数据
    """
    np.random.seed(seed)

    stages = np.random.choice(DEG_STAGES, size=n_patients, p=[0.3, 0.3, 0.25, 0.15])
    stage_scores = np.array([DEG_STAGE_MAP[s] for s in stages])

    n = n_patients

    # 基因表达模拟 (Z-score normalized scale)
    data = {
        "样本ID": [f"NP_{i+1:03d}" for i in range(n)],
        "退变分期": stages,
        "分期编码": stage_scores,
    }

    # NP 保护性标志物 (随退变下降)
    protective_genes = {
        "ACAN": 2.5, "COL2A1": 2.3, "KRT19": 2.7, "KRT18": 2.4,
        "SOX9": 2.0, "PAX1": 2.2, "HAPLN1": 2.0, "SIRT1": 1.8,
        "SOD2": 1.5, "NOG": 1.6,
    }
    for gene, effect in protective_genes.items():
        data[gene] = np.random.normal(0, 0.3, n) - stage_scores * effect * 0.3
        # 晚期更显著
        late_mask = stage_scores >= 2
        data[gene][late_mask] -= np.random.uniform(0.3, 0.6, late_mask.sum())

    # 退变性标志物 (随退变上升)
    degenerative_genes = {
        "MMP3": 2.0, "MMP13": 2.5, "ADAMTS5": 2.0, "IL1B": 2.2,
        "TNF": 1.8, "IL6": 1.9, "CXCL8": 2.1, "CCL2": 1.7,
        "CDKN2A": 2.3,
    }
    for gene, effect in degenerative_genes.items():
        data[gene] = np.random.normal(0, 0.3, n) + stage_scores * effect * 0.35
        late_mask = stage_scores >= 2
        data[gene][late_mask] += np.random.uniform(0.2, 0.5, late_mask.sum())

    # MMP/TIMP 比值 (退变的关键指标)
    data["MMP3/TIMP1"] = np.random.normal(1.0, 0.3, n) + stage_scores * 0.8
    data["MMP13/TIMP1"] = np.random.normal(0.5, 0.2, n) + stage_scores * 1.0

    # 临床指标
    data["VAS疼痛评分"] = np.clip(np.random.normal(3, 1.5, n) + stage_scores * 1.8, 0, 10)
    data["Oswestry功能障碍指数"] = np.clip(np.random.normal(20, 10, n) + stage_scores * 15, 0, 100)
    data["椎间盘高度指数(DHI)"] = np.clip(np.random.normal(80, 10, n) - stage_scores * 12, 20, 100)
    data["Pfirrmann分级"] = np.clip(np.round(np.random.normal(2, 0.8, n) + stage_scores * 0.6), 1, 5).astype(int)

    # 添加噪声
    for col in data:
        if col not in ["样本ID", "退变分期", "分期编码", "Pfirrmann分级"]:
            data[col] += np.random.normal(0, noise, n)

    df = pd.DataFrame(data)
    return df


def rank_biomarkers(
    df: pd.DataFrame,
    target_col: str = "分期编码",
    top_n: int = 15,
    exclude_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    使用随机森林排序生物标志物重要性

    Parameters
    ----------
    df : pd.DataFrame
        标志物数据
    target_col : str
        目标变量列名
    top_n : int
        返回前 N 个标志物
    exclude_cols : list
        排除的列 (如样本ID、分期等)

    Returns
    -------
    pd.DataFrame
        按重要性排序的标志物列表
    """
    if exclude_cols is None:
        exclude_cols = ["样本ID", "退变分期", "分期编码", "Pfirrmann分级"]

    feature_cols = [c for c in df.columns if c not in exclude_cols and c != target_col]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=2
    )
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "标志物": feature_cols,
        "重要性得分": rf.feature_importances_,
        "类别": _classify_biomarker(feature_cols),
    }).sort_values("重要性得分", ascending=False).reset_index(drop=True)

    # 交叉验证 AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
        cv_auc = f"{scores.mean():.3f} ± {scores.std():.3f}"
    except Exception:
        cv_auc = "N/A"

    importance_df.index = importance_df.index + 1
    importance_df.index.name = "排名"

    print(f"[✓] 随机森林 5折CV 准确率: {cv_auc}")
    print(f"[✓] 前 {top_n} 个最重要的生物标志物:")

    return importance_df.head(top_n)


def _classify_biomarker(names: list) -> list:
    """对标志物进行分类"""
    categories = []
    for name in names:
        name_upper = name.upper()
        if any(m in name_upper for m in ["MMP", "ADAMTS"]):
            categories.append("基质降解酶")
        elif any(c in name_upper for c in ["IL", "TNF", "CCL", "CXCL"]):
            categories.append("炎症因子")
        elif any(k in name_upper for k in ["KRT", "PAX1", "FOXF1", "CD24", "TBXT", "NOG"]):
            categories.append("NP标志物")
        elif any(e in name_upper for e in ["ACAN", "COL2", "SOX9", "HAPLN1"]):
            categories.append("ECM合成")
        elif any(s in name_upper for s in ["SIRT", "CDKN2A", "TP53", "SOD", "CAT"]):
            categories.append("衰老/氧化")
        elif "TIMP" in name_upper:
            categories.append("蛋白酶抑制")
        elif any(c in name_upper for c in ["VAS", "Oswestry", "DHI", "Pfirrmann"]):
            categories.append("临床指标")
        else:
            categories.append("其他")
    return categories


def plot_biomarker_importance(
    importance_df: pd.DataFrame,
    title: str = "NP 退变生物标志物重要性排名",
    figsize=(10, 8),
    output_path: Optional[str] = None,
    dpi: int = 150,
):
    """绘制标志物重要性横向柱状图"""
    fig, ax = plt.subplots(figsize=figsize)

    df = importance_df.sort_values("重要性得分")
    colors = {
        "基质降解酶": "#E74C3C", "炎症因子": "#E67E22", "NP标志物": "#3498DB",
        "ECM合成": "#1ABC9C", "衰老/氧化": "#9B59B6", "蛋白酶抑制": "#2ECC71",
        "临床指标": "#95A5A6", "其他": "#7F8C8D"
    }
    bar_colors = [colors.get(c, "#95A5A6") for c in df["类别"]]

    bars = ax.barh(range(len(df)), df["重要性得分"], color=bar_colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["标志物"], fontsize=10)
    ax.set_xlabel("重要性得分 (基于随机森林)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # 添加数值标签
    for i, (v, cat) in enumerate(zip(df["重要性得分"], df["类别"])):
        ax.text(v + 0.002, i, f"  {v:.4f}", fontsize=8, va='center')

    # 图例
    legend_handles = []
    seen = set()
    for cat, color in colors.items():
        if cat in df["类别"].values and cat not in seen:
            seen.add(cat)
            legend_handles.append(plt.Rectangle((0,0), 1, 1, color=color, label=cat))
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8, title="类别")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 标志物重要性图: {output_path}")

    return fig


def simulate_trend_data(
    n_timepoints: int = 6,
    n_genes: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    模拟退变进展时间序列的趋势数据
    时间点: 正常 → 早期 → 中期 → 晚期 (0, 1, 2, 3, 4, 5)
    """
    np.random.seed(seed)
    timepoints = np.array([0, 1, 2, 3, 4, 5])

    trends = {
        "ACAN": {"start": 10, "end": 3, "curve": "decay"},
        "COL2A1": {"start": 10, "end": 3.5, "curve": "decay"},
        "KRT19": {"start": 10, "end": 2, "curve": "decay"},
        "SOX9": {"start": 10, "end": 4, "curve": "decay"},
        "MMP13": {"start": 3, "end": 11, "curve": "growth"},
        "IL1B": {"start": 2, "end": 10, "curve": "growth"},
        "IL6": {"start": 2, "end": 9.5, "curve": "growth"},
        "CDKN2A": {"start": 3, "end": 10, "curve": "growth"},
    }

    data = {"timepoint": timepoints}
    for gene, params in trends.items():
        if params["curve"] == "decay":
            values = params["start"] - (params["start"] - params["end"]) * (timepoints / 5) ** 1.5
        else:
            values = params["start"] + (params["end"] - params["start"]) * (timepoints / 5) ** 1.2

        values += np.random.normal(0, 0.4, len(timepoints))
        data[gene] = np.clip(values, 1, 12)

    return pd.DataFrame(data)


def plot_biomarker_trends(
    trend_df: pd.DataFrame,
    title: str = "NP 退变进展中关键标志物表达趋势",
    figsize=(12, 7),
    output_path: Optional[str] = None,
    dpi: int = 150,
):
    """绘制标志物随退变进展的时间趋势"""
    fig, ax = plt.subplots(figsize=figsize)

    timepoints = trend_df["timepoint"].values

    # 分组: 下调 vs 上调
    down_genes = ["ACAN", "COL2A1", "KRT19", "SOX9"]
    up_genes = ["MMP13", "IL1B", "IL6", "CDKN2A"]

    colors_down = ['#2E86C1', '#1ABC9C', '#3498DB', '#2874A6']
    colors_up = ['#E74C3C', '#E67E22', '#C0392B', '#8E44AD']

    fine_x = np.linspace(timepoints[0], timepoints[-1], 200)

    # 绘制下降组
    for i, gene in enumerate(down_genes):
        if gene in trend_df.columns:
            y = trend_df[gene].values
            spl = make_interp_spline(timepoints, y, k=3, bc_type='natural')
            ax.plot(fine_x, spl(fine_x), color=colors_down[i], linewidth=2.5, alpha=0.8)
            ax.scatter(timepoints, y, color=colors_down[i], s=60, zorder=5,
                       edgecolors='white', linewidth=1.5, label=gene)

    # 绘制上升组
    for i, gene in enumerate(up_genes):
        if gene in trend_df.columns:
            y = trend_df[gene].values
            spl = make_interp_spline(timepoints, y, k=3, bc_type='natural')
            ax.plot(fine_x, spl(fine_x), color=colors_up[i], linewidth=2.5,
                    alpha=0.8, linestyle='--')
            ax.scatter(timepoints, y, color=colors_up[i], s=60, zorder=5,
                       edgecolors='white', linewidth=1.5, label=gene,
                       marker='D')

    # 退变分期区域
    stage_colors = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
    stage_labels = ['正常', '早期', '中期', '晚期']
    stage_bounds = [0, 1.5, 3, 4.5, 5.5]
    for i in range(4):
        ax.axvspan(stage_bounds[i], stage_bounds[i+1],
                   alpha=0.06, color=stage_colors[i],
                   label=f"{stage_labels[i]}" if i == 0 else "")
        ax.text(
            (stage_bounds[i] + stage_bounds[i+1]) / 2,
            ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 12,
            stage_labels[i],
            ha='center', fontsize=9, color=stage_colors[i],
            fontweight='bold', alpha=0.7
        )

    ax.set_xlabel("退变进展时间", fontsize=12)
    ax.set_ylabel("相对表达量 (归一化)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(timepoints)
    ax.set_xticklabels(["正常"] + [f"T{i}" for i in range(1, len(timepoints))])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 注释
    ax.annotate("ECM 合成↓\nNP 表型丢失", xy=(3, 7), fontsize=9,
                color='#2E86C1', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.annotate("炎症↑\n基质降解↑\n衰老↑", xy=(3.5, 7.5), fontsize=9,
                color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 趋势图已保存: {output_path}")

    return fig


def plot_roc_curves(
    df: pd.DataFrame,
    biomarkers: list,
    target_col: str = "分期编码",
    threshold: float = 1.5,  # 正常 vs 退变
    figsize=(9, 7),
    title: str = "NP 退变诊断标志物 ROC 曲线",
    output_path: Optional[str] = None,
    dpi: int = 150,
):
    """绘制多标志物的 ROC 曲线"""
    fig, ax = plt.subplots(figsize=figsize)

    y_true = (df[target_col] >= threshold).astype(int)

    colors = plt.cm.Set1(np.linspace(0, 1, len(biomarkers)))
    for i, gene in enumerate(biomarkers):
        if gene not in df.columns:
            continue
        fpr, tpr, _ = roc_curve(y_true, df[gene])
        roc_auc = auc(fpr, tpr)
        # 如果 AUC < 0.5 则翻转
        if roc_auc < 0.5:
            fpr, tpr, _ = roc_curve(y_true, -df[gene])
            roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{gene} (AUC={roc_auc:.3f})", alpha=0.85)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机猜测')
    ax.set_xlabel("假阳性率 (1 - 特异性)", fontsize=12)
    ax.set_ylabel("真阳性率 (灵敏度)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # 统计
    ax.text(0.6, 0.2,
            f"样本数: {len(df)}\n"
            f"退变组 (分期≥{threshold}): {y_true.sum()}\n"
            f"正常组: {(1-y_true).sum()}",
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] ROC 曲线已保存: {output_path}")

    return fig
