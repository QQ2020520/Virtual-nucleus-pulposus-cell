"""
NP 细胞 ECM 代谢与退变模型
模拟 ECM 合成/降解平衡、基质成分动态变化
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ECMDegradationModel:
    """
    NP 细胞 ECM 合成-降解平衡模型

    变量:
        Aggrecan: 聚集蛋白聚糖
        Col2: II 型胶原
        MMP_activity: MMP 总活性 (降解性)
        TIMP_activity: TIMP 总活性 (保护性)
        Inflammatory: 炎症水平
        Cell_density: NP 细胞密度
        Water_content: 水含量
    """

    def __init__(self):
        self.params = {
            'k_agg_synth': 0.10,
            'k_col2_synth': 0.08,
            'k_mmp_prod': 0.04,
            'k_timp_prod': 0.06,
            'k_inflam_prod': 0.02,
            'k_apoptosis': 0.005,
            'k_mmp_deg_agg': 0.15,
            'k_mmp_deg_col2': 0.08,
            'k_timp_inh_mmp': 0.3,
            'k_timp_loss': 0.03,
            'k_agg_loss': 0.02,
            'k_water_in': 0.01,
            'k_water_out': 0.005,
            # 退变加速因子
            'degen_accel': 0.0,
        }

    def ode_system(self, t, y, perturbation=None):
        (Agg, Col2, MMP, TIMP, Inflam, Cell, Water) = y
        p = self.params

        degen = p['degen_accel']

        # 退变扰动
        perturb = perturbation or {}
        p_inflam = perturb.get('INFLAM', 0)
        p_oxstress = perturb.get('OXSTRESS', 0)
        p_nutrient = perturb.get('NUTRIENT', 0)  # 负值 = 营养不足
        p_mech = perturb.get('MECHANICAL', 0)   # 力学超载

        # 有效炎症
        effective_inflam = max(0, Inflam + p_inflam + p_oxstress * 0.5)

        # MMP 动态
        mmp_synth = p['k_mmp_prod'] * (1 + effective_inflam * 3 + degen)
        mmp_inh = p['k_timp_inh_mmp'] * TIMP * MMP
        dMMP = mmp_synth - p['k_mmp_deg_agg'] * Agg * MMP - mmp_inh

        # TIMP 动态
        dTIMP = (p['k_timp_prod'] * Cell
                 - p['k_timp_loss'] * TIMP
                 - mmp_inh)

        # Aggrecan 动态
        agg_synth = p['k_agg_synth'] * Cell * max(0, 1 - effective_inflam * 0.5)
        agg_deg = p['k_mmp_deg_agg'] * Agg * MMP
        dAgg = agg_synth - agg_deg - p['k_agg_loss'] * Agg + p_mech * 0.02

        # Collagen II 动态
        col2_synth = p['k_col2_synth'] * Cell * max(0, 1 - effective_inflam * 0.3)
        col2_deg = p['k_mmp_deg_col2'] * Col2 * MMP
        dCol2 = col2_synth - col2_deg

        # 炎症动态
        dInflam = (p['k_inflam_prod'] * (1 + p_inflam + p_oxstress)
                   + p_mech * 0.01
                   - (p['k_timp_prod'] * TIMP * 0.1 + 0.02) * Inflam)

        # 细胞密度 (凋亡)
        dCell = (-p['k_apoptosis'] * Cell * (1 + effective_inflam * 2 + p_oxstress)
                 - 0.01 * Cell * max(0, -p_nutrient))

        # 水含量 (受 Aggrecan 调控)
        dWater = (p['k_water_in'] * Agg / (1 + Agg) * 2
                  - p['k_water_out'] * Water)

        return [dAgg, dCol2, dMMP, dTIMP, dInflam, dCell, dWater]

    def simulate(self, t_span=(0, 500), n_points=500, initial_state=None,
                 perturbation=None, degen_accel=0):
        if initial_state is None:
            initial_state = [1.0, 1.0, 0.05, 0.15, 0.02, 1.0, 0.8]

        self.params['degen_accel'] = degen_accel
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # 先跑到基线稳态
        sol_base = solve_ivp(
            lambda t, y: self.ode_system(t, y),
            (0, 1000), initial_state,
            method='RK45', max_step=10.0
        )
        steady = sol_base.y[:, -1]

        # 扰动
        sol = solve_ivp(
            lambda t, y: self.ode_system(t, y, perturbation),
            t_span, steady,
            method='RK45', t_eval=t_eval, max_step=5.0
        )

        return sol.t, sol.y


def plot_ecm_dynamics(
    t: np.ndarray, y: np.ndarray,
    title: str = "NP 细胞 ECM 动态仿真",
    figsize=(14, 8),
    output_path: Optional[str] = None,
    dpi: int = 150,
):
    """绘制 ECM 动态图"""
    var_names = [
        "Aggrecan (蛋白聚糖)", "Collagen II (II型胶原)",
        "MMP 活性", "TIMP 活性",
        "炎症水平", "NP 细胞密度", "水含量"
    ]
    colors = ['#2E86C1', '#1ABC9C', '#E74C3C', '#2ECC71',
              '#E67E22', '#8E44AD', '#3498DB']

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    axes[-1].set_visible(False)  # 隐藏最后一个

    for i, (name, color) in enumerate(zip(var_names, colors)):
        ax = axes[i]
        ax.plot(t, y[i], color=color, linewidth=2)
        ax.set_title(name, fontsize=11, fontweight='bold', color=color)
        ax.fill_between(t, 0, y[i], color=color, alpha=0.08)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

        # 终值标注
        ax.text(0.95, 0.95, f"终值: {y[i,-1]:.3f}",
                transform=ax.transAxes, fontsize=8,
                va='top', ha='right', color='grey')

    # MMP/TIMP 比值
    ax_mmp_timp = axes[6]
    ratio = y[2] / (y[3] + 1e-6)
    ax_mmp_timp.plot(t, ratio, color='#C0392B', linewidth=2, linestyle='--')
    ax_mmp_timp.set_title("MMP/TIMP 比值", fontsize=11, fontweight='bold', color='#C0392B')
    ax_mmp_timp.axhline(1, color='grey', linestyle=':', alpha=0.5)
    ax_mmp_timp.fill_between(t, 0, ratio, color='#C0392B', alpha=0.08)
    ax_mmp_timp.spines['top'].set_visible(False)
    ax_mmp_timp.spines['right'].set_visible(False)
    ax_mmp_timp.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] ECM 动态图: {output_path}")

    return fig
