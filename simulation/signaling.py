"""
NP 细胞信号通路仿真 — ODE 动力学 + 扰动模拟
核心通路: TGF-β/BMP, Wnt, HIF-1α, NF-κB, MAPK, Notch, PI3K/Akt
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NPSignalingModel:
    """
    NP 细胞信号通路集成 ODE 模型

    核心节点:
        TGFB: TGF-β 活性
        SMAD: SMAD2/3 复合物
        CTNNB: β-catenin (Wnt)
        NFKB: NF-κB 活性
        HIF1: HIF-1α 水平
        MAPK: ERK/p38 活性
        SOX9: SOX9 转录活性
        MMP: MMP 集体表达 (catabolic)
        ECM: ECM 合成指标 (anabolic)
        INFLAM: 炎症状态
    """

    def __init__(self):
        # 默认参数
        self.params = {
            # TGF-β 通路
            'k_tgfb_prod': 0.08,      # TGF-β 基础产生
            'k_tgfb_deg': 0.3,         # TGF-β 降解
            'k_smad_act': 0.25,        # SMAD 激活
            'k_smad_deact': 0.15,      # SMAD 失活
            'k_sox9_act': 0.18,        # SOX9 激活 (受 SMAD 调控)
            'k_sox9_deg': 0.08,        # SOX9 降解

            # Wnt 通路
            'k_wnt_prod': 0.06,
            'k_wnt_inh': 0.4,          # β-catenin 降解 (GSK3β 复合体)
            'k_ctnnb_prod': 0.05,      # β-catenin 基础产生

            # NF-κB 通路
            'k_il1b_prod': 0.02,
            'k_nfkb_act': 0.2,
            'k_nfkb_inh': 0.12,

            # HIF 通路
            'k_hif1_prod': 0.15,
            'k_hif1_deg': 0.5,         # 常氧下降解
            'k_hif1_stab': 0.4,        # 低氧稳定 (0-1)

            # MAPK
            'k_mapk_act': 0.15,
            'k_mapk_deact': 0.1,

            # 效应器
            'k_mmp_prod': 0.06,
            'k_mmp_deg': 0.04,
            'k_ecm_prod': 0.12,
            'k_ecm_deg': 0.03,
            'k_inflam_prod': 0.05,
            'k_inflam_cleared': 0.04,

            # 交叉调控
            'beta_cat_sox9_inh': 0.15,  # β-catenin → SOX9 抑制
            'nfkb_mmp_act': 0.3,        # NF-κB → MMP 激活
            'nfkb_ecm_inh': 0.2,        # NF-κB → ECM 抑制
            'mapk_inflam_act': 0.2,     # MAPK → 炎症正反馈
            'hif1_ecm_act': 0.15,       # HIF-1α → ECM 维持

            # 衰减常数
            'decay_base': 0.01,
        }

    def ode_system(self, t, y, perturbation=None):
        """ODE 系统"""
        (TGFB, SMAD, CTNNB, NFKB, HIF1,
         MAPK, SOX9, MMP, ECM, INFLAM) = y
        p = self.params

        # 扰动
        perturb = perturbation or {}
        p_tgfb = perturb.get('TGFB', 0)
        p_wnt = perturb.get('WNT', 0)
        p_il1b = perturb.get('IL1B', 0)
        p_tnf = perturb.get('TNF', 0)
        p_hypoxia = perturb.get('HYPOXIA', 0)   # 0~1, 低氧程度
        p_gs3k = perturb.get('GSK3B_inhibit', 0)  # GSK3β 抑制剂
        p_inhibitor = perturb.get('INHIBITOR', 0)  # 通用抑制剂

        # TGF-β 动态
        dTGFB = (p['k_tgfb_prod'] * (1 + p_tgfb)
                  - p['k_tgfb_deg'] * TGFB
                  - p['decay_base'] * TGFB)

        # SMAD 动态 (由 TGFβ 激活)
        dSMAD = (p['k_smad_act'] * TGFB / (1 + TGFB)
                 - p['k_smad_deact'] * SMAD)

        # Wnt/β-catenin 动态
        wnt_signal = p['k_wnt_prod'] * (1 + p_wnt)
        gsk3_deg = p['k_wnt_inh'] * (1 - p_gs3k)
        dCTNNB = (wnt_signal + p['k_ctnnb_prod']
                  - gsk3_deg * CTNNB
                  - p['decay_base'] * CTNNB)

        # NF-κB 动态 (由 IL-1β, TNF-α 激活)
        inflam_input = p['k_il1b_prod'] * (1 + p_il1b + p_tnf)
        dNFKB = (p['k_nfkb_act'] * inflam_input / (1 + inflam_input)
                 - p['k_nfkb_inh'] * NFKB)

        # HIF-1α 动态
        hypoxic_stab = p['k_hif1_stab'] * min(p_hypoxia, 1.0)
        dHIF1 = (p['k_hif1_prod']
                 - p['k_hif1_deg'] * HIF1 * (1 - hypoxic_stab)
                 - p['decay_base'] * HIF1)

        # MAPK 动态 (受炎症和力学刺激)
        dMAPK = (p['k_mapk_act'] * (1 + inflam_input * 0.5)
                 - p['k_mapk_deact'] * MAPK)

        # SOX9 动态 (被 SMAD 激活, 被 β-catenin 抑制)
        dSOX9 = (p['k_sox9_act'] * SMAD / (1 + SMAD)
                 - p['beta_cat_sox9_inh'] * CTNNB * SOX9
                 - p['k_sox9_deg'] * SOX9)

        # MMP 动态 (被 NF-κB, MAPK 激活; 被 SOX9 抑制)
        dMMP = (p['k_mmp_prod'] * (1 + p['nfkb_mmp_act'] * NFKB
                                    + p['mapk_inflam_act'] * MAPK)
                * (1 - 0.3 * SOX9 / (1 + SOX9))
                - p['k_mmp_deg'] * MMP)

        # ECM 动态 (被 SOX9, HIF-1 促进; 被 NF-κB, MAPK 抑制)
        ecm_synth = p['k_ecm_prod'] * (
            1 + 0.5 * SOX9 / (1 + SOX9) + p['hif1_ecm_act'] * HIF1
            - p['nfkb_ecm_inh'] * NFKB
            - 0.1 * MAPK
        )
        ecm_deg = p['k_ecm_deg'] * (1 + MMP * 0.5)
        dECM = max(0, ecm_synth) - ecm_deg * ECM

        # 炎症状态 (被 NF-κB, MAPK 维持)
        dINFLAM = (p['k_inflam_prod'] * (NFKB + MAPK)
                   - p['k_inflam_cleared'] * INFLAM
                   + inflam_input * 0.1)

        # 数值安全保护: 限制所有变量在合理范围内
        max_val = 20.0
        dTGFB = np.clip(dTGFB, -max_val, max_val)
        dSMAD = np.clip(dSMAD, -max_val, max_val)
        dCTNNB = np.clip(dCTNNB, -max_val, max_val)
        dNFKB = np.clip(dNFKB, -max_val, max_val)
        dHIF1 = np.clip(dHIF1, -max_val, max_val)
        dMAPK = np.clip(dMAPK, -max_val, max_val)
        dSOX9 = np.clip(dSOX9, -max_val, max_val)
        dMMP = np.clip(dMMP, -max_val, max_val)
        dECM = np.clip(dECM, -max_val, max_val)
        dINFLAM = np.clip(dINFLAM, -max_val, max_val)

        return [dTGFB, dSMAD, dCTNNB, dNFKB, dHIF1,
                dMAPK, dSOX9, dMMP, dECM, dINFLAM]

    def simulate(
        self,
        t_span=(0, 200),
        n_points=500,
        initial_state=None,
        perturbations=None,
    ):
        """运行仿真"""
        if initial_state is None:
            initial_state = [0.3, 0.2, 0.5, 0.05, 0.6,
                             0.1, 0.4, 0.05, 0.8, 0.05]

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # 基线: 先跑到稳态
        sol_base = solve_ivp(
            lambda t, y: self.ode_system(t, y),
            t_span, initial_state,
            method='RK45', dense_output=True,
            max_step=5.0
        )
        steady_state = sol_base.y[:, -1]

        # 带扰动的仿真
        sol_perturbed = None
        if perturbations:
            sol_perturbed = solve_ivp(
                lambda t, y: self.ode_system(t, y, perturbations),
                (0, 100), steady_state,
                method='RK45', dense_output=True,
                max_step=2.0
            )
            full_t = np.concatenate([t_eval[t_eval <= 0],  # 用基线稳态之前
                                     np.linspace(0, 100, 300)])
            # 简化: 直接拼接
            pre_t = t_eval[t_eval <= 0]
            if len(pre_t) == 0:
                pre_t = np.linspace(-50, 0, 100)
            full_y = np.column_stack([
                np.tile(steady_state, (len(pre_t), 1)).T,
                sol_perturbed.y
            ])
            full_t = np.concatenate([pre_t,
                                     sol_perturbed.t + 0])
        else:
            full_t = sol_base.t
            full_y = sol_base.y

        return full_t, full_y

    def run_perturbation_screen(self, perturbations: list) -> dict:
        """运行多个扰动条件并提取关键指标"""
        results = {}
        base_t, base_y = self.simulate(t_span=(0, 1000))
        base_ecm = base_y[8, -1]
        base_mmp = base_y[7, -1]

        for name, p_dict in perturbations:
            t, y = self.simulate(t_span=(0, 500), perturbations=p_dict)
            ecm_final = y[8, -1]
            mmp_final = y[7, -1]
            sox9_final = y[6, -1]
            nfkb_final = y[3, -1]

            results[name] = {
                "ECM_change": (ecm_final - base_ecm) / base_ecm * 100,
                "MMP_change": (mmp_final - base_mmp) / base_mmp * 100,
                "SOX9": sox9_final,
                "NFKB": nfkb_final,
                "Anabolic_score": ecm_final / (ecm_final + mmp_final + 1e-6),
            }

        return results, base_ecm, base_mmp


def plot_signaling_network(
    t: np.ndarray,
    y: np.ndarray,
    title: str = "NP 细胞信号通路动态仿真",
    highlight_var: Optional[str] = None,
    figsize=(14, 10),
    output_path: Optional[str] = None,
    dpi: int = 150,
):
    """绘制信号通路仿真结果"""
    var_names = [
        "TGF-β", "SMAD2/3", "β-catenin", "NF-κB", "HIF-1α",
        "MAPK", "SOX9", "MMPs", "ECM", "炎症"
    ]
    colors = ['#3498DB', '#2E86C1', '#E67E22', '#E74C3C', '#1ABC9C',
              '#9B59B6', '#2ECC71', '#C0392B', '#27AE60', '#F39C12']

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    for i, (name, color) in enumerate(zip(var_names, colors)):
        ax = axes[i]
        yi = np.nan_to_num(y[i], nan=0.0, posinf=5.0, neginf=0.0)
        ax.plot(t, yi, color=color, linewidth=2)
        ax.set_title(name, fontsize=11, fontweight='bold', color=color)

        final_val = yi[-1]
        ax.axhline(final_val, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)

        # 标注最终值
        ax.text(t[-1] * 0.85, final_val + 0.02,
                f"{final_val:.3f}", fontsize=7, color='grey')

        # 突出高亮变量
        if highlight_var and name == highlight_var:
            ax.set_facecolor('#FFF9C4')

        ax.set_xlim(t.min(), t.max())
        ymin, ymax = yi.min(), yi.max()
        if np.isnan(ymin) or np.isinf(ymin): ymin = 0
        if np.isnan(ymax) or np.isinf(ymax): ymax = 5
        if ymax - ymin < 0.01: ymax = ymin + 1.0
        ax.set_ylim(ymin - 0.1, ymax + 0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 信号通路仿真图: {output_path}")

    return fig


def plot_perturbation_comparison(
    results: dict,
    base_ecm: float,
    base_mmp: float,
    title: str = "不同扰动条件下 NP 细胞表型变化",
    figsize=(12, 6),
    output_path: Optional[str] = None,
    dpi: int = 150,
):
    """绘制扰动对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    names = list(results.keys())
    ecm_changes = [results[n]["ECM_change"] for n in names]
    mmp_changes = [results[n]["MMP_change"] for n in names]
    scores = [results[n]["Anabolic_score"] for n in names]

    x = np.arange(len(names))
    w = 0.35

    # ECM 和 MMP 变化
    bars1 = ax1.bar(x - w/2, ecm_changes, w, label='ECM 变化%', color='#27AE60', alpha=0.8)
    bars2 = ax1.bar(x + w/2, mmp_changes, w, label='MMP 变化%', color='#E74C3C', alpha=0.8)

    ax1.axhline(0, color='grey', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    ax1.set_ylabel("变化百分比 (%)", fontsize=11)
    ax1.set_title("ECM 与 MMP 变化", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 合成代谢评分
    colors_score = ['#27AE60' if s > 0.5 else '#E74C3C' for s in scores]
    ax2.bar(names, scores, color=colors_score, alpha=0.8, edgecolor='white', width=0.6)
    ax2.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    ax2.set_ylabel("合成代谢评分", fontsize=11)
    ax2.set_title("ECM/(ECM+MMP) 合成代谢指数", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 注释基底状态
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.5, 0.01,
             f"基底 ECM={base_ecm:.3f}, 基底 MMP={base_mmp:.3f}",
             ha='center', fontsize=9, color='grey')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 扰动对比图: {output_path}")

    return fig
