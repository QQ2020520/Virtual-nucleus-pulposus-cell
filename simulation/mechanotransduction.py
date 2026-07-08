"""
MRTF-A 介导的基质刚度-糖酵解力传导模型
===============================================
基于 Bone Research 2025 论文:
    Matrix stiffness regulates nucleus pulposus cell glycolysis
    by MRTF-A-dependent mechanotransduction

核心信号链 (Core signaling cascade):
    Matrix stiffness ↑ → Integrin/FAK → F-actin ↑
    → MRTF-A nuclear translocation ↑ → Kidins220 ↓
    → AMPK phosphorylation ↓ → PFKFB3↓, PFKM↓ → Glycolysis ↓

变量列表 (9 state variables):
    y[0] = Stiffness_signal   — 感知到的基质刚度 (normalized)
    y[1] = F_actin            — F-actin 聚合水平
    y[2] = MRTFA_nuc          — 核内 MRTF-A (nuclear fraction)
    y[3] = Kidins220          — Kidins220 表达水平
    y[4] = AMPK_P             — 磷酸化 AMPK (p-AMPK)
    y[5] = PFKFB3             — PFKFB3 糖酵解酶水平
    y[6] = PFKM               — PFKM 糖酵解酶水平
    y[7] = Glycolysis_output  — 糖酵解通量输出
    y[8] = ECM_degradation    — ECM 退化状态

扰动支持 (Perturbation support):
    stiffness      — 基质刚度水平 (0.5-3.0 相对倍数; 1.0=~2kPa 正常)
    CCG_inhibitor  — CCG-1423, MRTF-A 抑制剂 (0~1)
    GSK3B          — GSK3β 活性调节 (负值=抑制, 正值=激活)
    mechanical_load— 额外力学负载刺激 (0~1)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MRTFAMechanotransductionModel:
    """
    MRTF-A介导的基质刚度→糖酵解力传导 ODE 模型

    建模基于 Bone Research 2025 的核心发现:
    1. NP退变时基质刚度从 ~2kPa 增至 ~15kPa (约7.5倍)
    2. 刚度↑ → Integrin-FAK → F-actin ↑ → MRTF-A核转位↑
    3. MRTF-A/SRF → Kidins220 转录抑制↓ → AMPK磷酸化↓ → 糖酵解↓
    4. CCG-1423 (MRTF-A抑制剂) 可逆转此过程
    """

    VAR_NAMES = [
        "Stiffness_signal",     # 0
        "F_actin",              # 1
        "MRTFA_nuc",            # 2
        "Kidins220",            # 3
        "AMPK_P",               # 4
        "PFKFB3",               # 5
        "PFKM",                 # 6
        "Glycolysis_output",    # 7
        "ECM_degradation",      # 8
    ]

    # 用于人类可读的中文变量名
    VAR_CN_NAMES = [
        "基质刚度",        # Stiffness_signal
        "F-肌动蛋白",      # F_actin
        "核MRTF-A",        # MRTFA_nuc
        "Kidins220",       # Kidins220
        "p-AMPK",          # AMPK_P
        "PFKFB3",          # PFKFB3
        "PFKM",            # PFKM
        "糖酵解通量",      # Glycolysis_output
        "ECM降解",         # ECM_degradation
    ]

    def __init__(self):
        """
        初始化模型参数 — 基于 Bone Research 2025 文献校准

        靶向稳态 (stiffness=1, 正常):
            Stiffness_signal ≈ 0.7  | F_actin ≈ 0.5  | MRTFA_nuc ≈ 0.4
            Kidins220 ≈ 0.75        | AMPK_P ≈ 0.6   | PFKFB3 ≈ 0.6
            PFKM ≈ 0.6              | Glycolysis ≈ 0.6 | ECM_degrad ≈ 0.12

        靶向退变 (stiffness=3, ~15kPa):
            Stiffness_signal ≈ 2.1  | F_actin ≈ 1.0  | MRTFA_nuc ≈ 0.8
            Kidins220 ≈ 0.20        | AMPK_P ≈ 0.25  | PFKFB3 ≈ 0.30
            PFKM ≈ 0.30             | Glycolysis ≈ 0.25 | ECM_degrad ≈ 0.4
        """
        self.params = {

            # ============================================================
            # 1. 刚度感知与 F-actin 动力学
            #    Stiffness (0.5-3x) → Stiffness_signal (linear transduction)
            #    → Integrin/FAK → F-actin (Hill activation)
            # ============================================================
            'k_stiff_prod': 0.25,         # 刚度信号基础产生率
            'k_stiff_decay': 0.35,        # 刚度信号衰减率

            'k_factin_act': 0.14,         # F-actin 最大聚合速率
            'k_factin_depol': 0.12,       # F-actin 解聚速率
            'km_stiff_actin': 1.5,        # 刚度→F-actin Hill EC50 (提高使正常刚度~50%激活)
            'n_stiff_actin': 1.5,         # 刚度→F-actin Hill 系数 (n=1.5温和激活)
            'integrin_gain': 1.5,         # Integrin-FAK 信号增益

            # ============================================================
            # 2. MRTF-A 核转位动力学
            #    F-actin ↑ → MRTF-A 从 G-actin 释放 → SRF 共激活 → 核转位↑
            #    Hill cooperativity: n=3 确保动态范围
            # ============================================================
            'k_mrtfa_trans': 0.22,        # MRTF-A 核转位最大速率
            'k_mrtfa_export': 0.20,       # MRTF-A 核输出速率
            'km_mrtfa_factin': 0.70,      # F-actin→MRTF-A Hill EC50 (调高确保动态范围)
            'n_mrtfa_factin': 3.0,        # F-actin→MRTF-A Hill 系数
            'ccg_inhibition': 0.85,       # CCG-1423 最大抑制效率 (0~1)

            # ============================================================
            # 3. Kidins220 表达 (MRTF-A/SRF 靶基因抑制)
            #    MRTF-A_nuc ↑ → SRF + 共抑制因子 → Kidins220 转录↓
            # ============================================================
            'k_kidins_prod': 0.40,        # Kidins220 最大转录速率
            'k_kidins_deg': 0.30,         # Kidins220 降解速率 (提高加速更新)
            'kidins_repress_max': 0.80,   # MRTF-A 最大抑制幅度 (0~1)
            'kidins_repress_ec50': 0.6,   # 抑制 EC50 (MRTF-A 半抑制浓度)
            'kidins_repress_hill': 3.0,   # 抑制 Hill 系数 (n=3更陡峭)
            'kidins_basal': 0.02,         # Kidins220 基础渗漏表达

            # ============================================================
            # 4. AMPK 磷酸化 (由 Kidins220 通过 CaMKKβ 促进)
            #    Kidins220 支架 CaMKKβ → AMPK Thr172 磷酸化↑
            #    使用 Hill 激活函数 (n=2) 模拟阈值效应
            # ============================================================
            'k_ampk_act': 0.15,           # AMPK 磷酸化最大速率
            'k_ampk_dephos': 0.22,        # AMPK 去磷酸化速率
            'km_ampk': 0.80,              # AMPK 激活 Hill EC50
            'n_ampk': 3.0,                # AMPK 激活 Hill 系数 (n=3 更陡的阈值)
            'ampk_basal': 0.01,           # AMPK 基础磷酸化 (无Kidins220时低)

            # ============================================================
            # 5. PFKFB3 & PFKM 糖酵解酶 (受 AMPK 转录调控)
            #    p-AMPK ↑ → PGC-1α → PFKFB3/PFKM 转录↑
            #    线性激活: dE/dt = k_prod + k_ampk*A - k_deg*E
            # ============================================================
            'k_pfkfb3_prod': 0.03,        # PFKFB3 基础转录速率
            'k_pfkfb3_ampk': 0.15,        # AMPK→PFKFB3 转录激活速率
            'k_pfkfb3_deg': 0.20,         # PFKFB3 降解速率

            'k_pfkm_prod': 0.03,          # PFKM 基础转录速率
            'k_pfkm_ampk': 0.15,          # AMPK→PFKM 转录激活速率
            'k_pfkm_deg': 0.20,           # PFKM 降解速率

            # ============================================================
            # 6. 糖酵解通量 (Glycolysis flux)
            #    反映 PFKFB3 + PFKM 的联合功能输出
            # ============================================================
            'k_glyc_base': 0.02,          # 糖酵解基础通量
            'k_glyc_enzyme': 0.30,        # 酶活性→通量增益
            'k_glyc_cons': 0.25,          # 通量消耗/产物转化

            # ============================================================
            # 7. ECM 降解与反馈
            #    高刚度 + 低糖酵解 → ECM 降解↑ (退变正反馈)
            # ============================================================
            'k_ecmdeg_prod': 0.08,        # ECM 降解基础速率
            'k_ecmdeg_stiff': 0.40,       # 刚度对 ECM 降解的促进
            'k_ecmdeg_glyc_protect': 0.50,  # 糖酵解对 ECM 的保护
            'k_ecmdeg_repair': 0.06,      # ECM 修复速率
            'ecm_stiff_feedback': 0.30,   # ECM 降解→刚度正反馈

            # ============================================================
            # 8. 力学负载 (Mechanical load)
            # ============================================================
            'mechload_gain': 0.25,        # 额外力学负载增益

            # 通用数值安全
            'decay_base': 0.01,
        }

    def ode_system(self, t: float, y: np.ndarray,
                   perturbation: Optional[Dict] = None) -> List[float]:
        """
        ODE 系统 — 机械力传导微分方程组

        Args:
            t: 时间点
            y: 9维状态向量
            perturbation: 扰动参数字典
                - stiffness: float (0.5~3.0) 刚度倍数
                - CCG_inhibitor: float (0~1) CCG-1423 MRTF-A抑制剂
                - GSK3B: float, GSK3β活性调节 (-1~1)
                - mechanical_load: float (0~1) 额外力学负载

        Returns:
            9维导数向量
        """
        (Stiffness_signal, F_actin, MRTFA_nuc, Kidins220, AMPK_P,
         PFKFB3, PFKM, Glycolysis_output, ECM_degradation) = y

        p = self.params
        perturb = perturbation or {}

        # ---- 解析扰动参数 ----
        stiff_level = perturb.get('stiffness', 1.0)
        ccg_dose = perturb.get('CCG_inhibitor', 0.0)
        gsk3b = perturb.get('GSK3B', 0.0)
        mech_load = perturb.get('mechanical_load', 0.0)

        # 数值安全裁剪
        stiff_level = np.clip(stiff_level, 0.1, 5.0)
        ccg_dose = np.clip(ccg_dose, 0.0, 1.0)
        mech_load = np.clip(mech_load, 0.0, 1.0)

        # ============================================================
        # 1. 刚度信号 (Stiffness_signal)
        #    外部刚度输入 + ECM 降解正反馈
        # ============================================================
        external_stiff = stiff_level * (1 + p['mechload_gain'] * mech_load)
        ecm_feedback = 1 + p['ecm_stiff_feedback'] * ECM_degradation

        dStiffness = (
            p['k_stiff_prod'] * external_stiff * ecm_feedback
            - p['k_stiff_decay'] * Stiffness_signal
            - p['decay_base'] * Stiffness_signal
        )

        # ============================================================
        # 2. F-actin 动力学 (Integrin-FAK 介导)
        #    刚度↑ → Integrin/FAK → F-actin 聚合↑
        #    GSK3β 通过 cofilin/actin 动力学调节
        # ============================================================
        gsk3b_effect = 1 + 0.2 * gsk3b
        # Hill 型刚度→F-actin 激活
        stiff_norm = p['integrin_gain'] * Stiffness_signal
        factin_activation = (
            stiff_norm ** p['n_stiff_actin']
            / (stiff_norm ** p['n_stiff_actin']
               + p['km_stiff_actin'] ** p['n_stiff_actin'])
        )
        dF_actin = (
            p['k_factin_act'] * factin_activation * gsk3b_effect
            - p['k_factin_depol'] * F_actin
            - p['decay_base'] * F_actin
        )

        # ============================================================
        # 3. MRTF-A 核转位
        #    F-actin ↑ → G-actin ↓ → MRTF-A 释放 → 核转位↑
        #    CCG-1423 抑制 MRTF-A 核转位/转录活性
        # ============================================================
        ccg_effect = 1 - p['ccg_inhibition'] * ccg_dose
        # Hill 型 F-actin→MRTF-A 核转位 (n=3 确保动态范围)
        mrtfa_translocation = (
            p['k_mrtfa_trans']
            * (F_actin ** p['n_mrtfa_factin']
               / (F_actin ** p['n_mrtfa_factin']
                  + p['km_mrtfa_factin'] ** p['n_mrtfa_factin']))
        )
        dMRTFA_nuc = (
            mrtfa_translocation * ccg_effect
            - p['k_mrtfa_export'] * MRTFA_nuc
            - p['decay_base'] * MRTFA_nuc
        )

        # ============================================================
        # 4. Kidins220 表达 (MRTF-A/SRF 靶基因抑制)
        #    MRTF-A 核内↑ → SRF 结合 → Kidins220 转录↓
        # ============================================================
        # Hill 型抑制函数: MRTF-A 浓度越高, Kidins220 转录越低
        repression = (
            MRTFA_nuc ** p['kidins_repress_hill']
            / (MRTFA_nuc ** p['kidins_repress_hill']
               + p['kidins_repress_ec50'] ** p['kidins_repress_hill'])
        )
        kidins_production = (
            p['kidins_basal']
            + p['k_kidins_prod'] * (1 - p['kidins_repress_max'] * repression)
        )
        kidins_production = max(kidins_production, 0.01)

        dKidins220 = (
            kidins_production
            - p['k_kidins_deg'] * Kidins220
            - p['decay_base'] * Kidins220
        )

        # ============================================================
        # 5. AMPK 磷酸化 (由 Kidins220 通过 CaMKKβ 促进)
        #    Kidins220 ↑ → CaMKKβ 活性↑ → AMPK Thr172 磷酸化↑
        #    Hill 型 (n=2) 模拟阈值效应: 需要足够的 Kidins220
        #    来招募 CaMKKβ 到 AMPK
        # ============================================================
        # Hill 型 Kidins220→AMPK 激活
        kidins_norm = Kidins220
        ampk_activation = (
            p['k_ampk_act']
            * (kidins_norm ** p['n_ampk'])
            / (kidins_norm ** p['n_ampk'] + p['km_ampk'] ** p['n_ampk'])
        )
        dAMPK_P = (
            p['ampk_basal']
            + ampk_activation
            - p['k_ampk_dephos'] * AMPK_P
            - p['decay_base'] * AMPK_P
        )

        # ============================================================
        # 6-7. PFKFB3 & PFKM 糖酵解酶 (受 AMPK 转录调控)
        #     p-AMPK ↑ → PGC-1α → PFKFB3/PFKM 转录↑ → 糖酵解↑
        #     使用线性激活: dE/dt = prod_base + ampk_coeff*A - deg*E
        # ============================================================
        pfkfb3_synthesis = (
            p['k_pfkfb3_prod']
            + p['k_pfkfb3_ampk'] * AMPK_P
        )
        dPFKFB3 = (
            pfkfb3_synthesis
            - p['k_pfkfb3_deg'] * PFKFB3
            - p['decay_base'] * PFKFB3
        )

        pfkm_synthesis = (
            p['k_pfkm_prod']
            + p['k_pfkm_ampk'] * AMPK_P
        )
        dPFKM = (
            pfkm_synthesis
            - p['k_pfkm_deg'] * PFKM
            - p['decay_base'] * PFKM
        )

        # ============================================================
        # 8. 糖酵解通量 (Glycolysis flux output)
        #    反映 PFKFB3 + PFKM 的联合功能输出
        # ============================================================
        enzyme_activity = (PFKFB3 + PFKM) / 2.0
        glycolysis_rate = (
            p['k_glyc_base']
            + p['k_glyc_enzyme'] * enzyme_activity
        )
        dGlycolysis = (
            glycolysis_rate
            - p['k_glyc_cons'] * Glycolysis_output
        )

        # ============================================================
        # 9. ECM 降解 (ECM degradation)
        #    高刚度 + 低糖酵解 → ECM 降解↑ (退变正反馈)
        # ============================================================
        stiff_deg = p['k_ecmdeg_stiff'] * Stiffness_signal
        glyc_protection = (
            p['k_ecmdeg_glyc_protect'] * Glycolysis_output
            / (0.5 + Glycolysis_output)
        )
        ecm_deg_rate = p['k_ecmdeg_prod'] * (
            1 + stiff_deg - glyc_protection
        )
        ecm_deg_rate = max(ecm_deg_rate, 0.0)

        dECM_degradation = (
            ecm_deg_rate
            - p['k_ecmdeg_repair'] * ECM_degradation
        )

        # ============================================================
        # 数值安全保护
        # ============================================================
        max_deriv = 10.0
        dStiffness = np.clip(dStiffness, -max_deriv, max_deriv)
        dF_actin = np.clip(dF_actin, -max_deriv, max_deriv)
        dMRTFA_nuc = np.clip(dMRTFA_nuc, -max_deriv, max_deriv)
        dKidins220 = np.clip(dKidins220, -max_deriv, max_deriv)
        dAMPK_P = np.clip(dAMPK_P, -max_deriv, max_deriv)
        dPFKFB3 = np.clip(dPFKFB3, -max_deriv, max_deriv)
        dPFKM = np.clip(dPFKM, -max_deriv, max_deriv)
        dGlycolysis = np.clip(dGlycolysis, -max_deriv, max_deriv)
        dECM_degradation = np.clip(dECM_degradation, -max_deriv, max_deriv)

        return [
            dStiffness, dF_actin, dMRTFA_nuc, dKidins220, dAMPK_P,
            dPFKFB3, dPFKM, dGlycolysis, dECM_degradation,
        ]

    def simulate(
        self,
        stiffness_level: float = 1.0,
        perturbation: Optional[Dict] = None,
        t_span: Tuple[float, float] = (0, 500),
        n_points: int = 500,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行机械力传导仿真至稳态

        Args:
            stiffness_level: 基质刚度倍数 (1.0 = 正常 NP ~2kPa)
            perturbation: 额外扰动参数字典
            t_span: 仿真时间范围 (默认 0~500 确保稳态)
            n_points: 输出点数
            initial_state: 初始状态 (9维向量, 默认使用生理稳态)

        Returns:
            (t_array, y_matrix) — 时间序列和9个变量的时间历程
        """
        if initial_state is None:
            # 默认生理初始状态: 正常刚度下的稳态值
            initial_state = np.array([
                0.7,    # Stiffness_signal: 归一化刚度信号
                0.5,    # F_actin: 基础F-肌动蛋白聚合
                0.4,    # MRTFA_nuc: 基础核MRTF-A水平
                0.75,   # Kidins220: 正常表达 (受MRTF-A轻度抑制)
                0.6,    # AMPK_P: 基础磷酸化AMPK
                0.6,    # PFKFB3: 基础糖酵解酶
                0.6,    # PFKM: 基础糖酵解酶
                0.6,    # Glycolysis_output: 基础糖酵解通量
                0.12,   # ECM_degradation: 低ECM降解
            ])

        # 合并刚度水平到扰动字典
        combined_perturb = dict(perturbation or {})
        combined_perturb['stiffness'] = stiffness_level

        # 主仿真
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            lambda t, y: self.ode_system(t, y, combined_perturb),
            t_span, initial_state,
            method='RK45',
            t_eval=t_eval,
            max_step=5.0,
            rtol=1e-6,
            atol=1e-8,
        )

        return sol.t, sol.y

    def simulate_dose_response(
        self,
        stiffness_range: Optional[List[float]] = None,
        perturbation: Optional[Dict] = None,
        t_span: Tuple[float, float] = (0, 500),
        n_points: int = 300,
    ) -> Dict[str, np.ndarray]:
        """
        刚度剂量响应仿真 — 每个刚度水平都跑到稳态

        Args:
            stiffness_range: 刚度倍数列表 (默认 0.5~3.0)
            perturbation: 额外扰动
            t_span: 每个条件的仿真时间
            n_points: 每个条件的采样点数

        Returns:
            dict: {
                'stiffness_levels': np.ndarray,
                'steady_states': dict[var_name → array],
                'trajectories': dict[stiffness → (t, y)],
            }
        """
        if stiffness_range is None:
            stiffness_range = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

        n_conditions = len(stiffness_range)
        steady_states = {name: np.zeros(n_conditions)
                         for name in self.VAR_NAMES}
        trajectories = {}

        # 从正常刚度 (1.0) 开始, 逐步向两边扫描
        normal_idx = 0
        for i, s in enumerate(stiffness_range):
            if abs(s - 1.0) < 1e-6:
                normal_idx = i
                break

        # 先计算正常刚度稳态
        _, y_normal = self.simulate(
            stiffness_level=1.0,
            perturbation=perturbation,
            t_span=(0, 800),
            n_points=n_points,
        )
        steady_state = y_normal[:, -1].copy()

        # 正向: normal → high stiffness
        for i in range(normal_idx, n_conditions):
            stiff = stiffness_range[i]
            if i == normal_idx:
                pass  # 已计算
            else:
                # 逐步增加刚度, 避免跳跃
                n_steps = max(1, int(abs(stiff - 1.0) / 0.3))
                current_state = steady_state.copy()
                for step in range(n_steps):
                    step_stiff = (1.0 + (stiff - 1.0) * (step + 1) / n_steps)
                    _, y_step = self.simulate(
                        stiffness_level=step_stiff,
                        perturbation=perturbation,
                        t_span=t_span,
                        n_points=n_points // n_steps + 50,
                        initial_state=current_state,
                    )
                    current_state = y_step[:, -1].copy()
                steady_state = current_state

            for j, name in enumerate(self.VAR_NAMES):
                steady_states[name][i] = steady_state[j]

            t, y = self.simulate(
                stiffness_level=stiff,
                perturbation=perturbation,
                t_span=(0, 600),
                n_points=n_points,
                initial_state=steady_state,
            )
            trajectories[stiff] = (t, y)

        # 负向: normal → low stiffness
        _, y_normal2 = self.simulate(
            stiffness_level=1.0,
            perturbation=perturbation,
            t_span=(0, 800),
            n_points=n_points,
        )
        steady_state2 = y_normal2[:, -1].copy()

        for i in range(normal_idx - 1, -1, -1):
            stiff = stiffness_range[i]
            n_steps = max(1, int(abs(1.0 - stiff) / 0.3))
            current_state = steady_state2.copy()
            for step in range(n_steps):
                step_stiff = (1.0 - (1.0 - stiff) * (step + 1) / n_steps)
                _, y_step = self.simulate(
                    stiffness_level=step_stiff,
                    perturbation=perturbation,
                    t_span=t_span,
                    n_points=n_points // n_steps + 50,
                    initial_state=current_state,
                )
                current_state = y_step[:, -1].copy()
            steady_state2 = current_state

            for j, name in enumerate(self.VAR_NAMES):
                steady_states[name][i] = steady_state2[j]

            t, y = self.simulate(
                stiffness_level=stiff,
                perturbation=perturbation,
                t_span=(0, 600),
                n_points=n_points,
                initial_state=steady_state2,
            )
            trajectories[stiff] = (t, y)

        return {
            'stiffness_levels': np.array(stiffness_range),
            'steady_states': steady_states,
            'trajectories': trajectories,
        }

    def get_steady_state_metrics(
        self, y_steady: np.ndarray, stiffness_level: float
    ) -> Dict[str, float]:
        """
        从稳态向量提取关键指标

        Args:
            y_steady: 稳态状态向量 (9维)
            stiffness_level: 对应的刚度水平

        Returns:
            关键指标字典
        """
        return {
            'stiffness': stiffness_level,
            'Stiffness_signal': y_steady[0],
            'F_actin': y_steady[1],
            'MRTFA_nuc': y_steady[2],
            'Kidins220': y_steady[3],
            'AMPK_P': y_steady[4],
            'PFKFB3': y_steady[5],
            'PFKM': y_steady[6],
            'Glycolysis_output': y_steady[7],
            'ECM_degradation': y_steady[8],
            # 衍生指标
            'Glycolysis_vs_normal': y_steady[7] / 0.5,
            'MRTFA_activation_ratio': y_steady[2] / 0.4,
        }

    def plot_mechanotransduction(
        self,
        t: np.ndarray,
        y: np.ndarray,
        title: str = "MRTF-A 机械力传导通路动态仿真",
        stiffness_level: float = 1.0,
        perturbation_label: str = "",
        figsize: Tuple[int, int] = (14, 10),
        output_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        绘制 2×4 子图 — 展示全部9个变量的时间历程
        (2行4列, 最后右下角显示 ECM 降解)

        Args:
            t: 时间数组
            y: 9×n 状态矩阵
            title: 图标题
            stiffness_level: 当前刚度水平
            perturbation_label: 扰动标签
            figsize: 图尺寸
            output_path: 保存路径 (可选)
            dpi: 图像分辨率

        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()

        # 颜色方案 (与论文通路色彩匹配)
        colors = [
            '#34495E',  # Stiffness_signal: 深灰
            '#E74C3C',  # F_actin: 红色 (细胞骨架)
            '#8E44AD',  # MRTFA_nuc: 紫色 (转录因子)
            '#2ECC71',  # Kidins220: 绿色
            '#F39C12',  # AMPK_P: 橙色 (能量传感)
            '#3498DB',  # PFKFB3: 蓝色 (糖酵解)
            '#1ABC9C',  # PFKM: 青色 (糖酵解)
            '#E67E22',  # Glycolysis_output: 橙红 (代谢输出)
        ]

        # 绘制前8个变量
        for i in range(8):
            ax = axes[i]
            yi = np.nan_to_num(y[i], nan=0.0, posinf=10.0, neginf=0.0)
            name_en = self.VAR_NAMES[i]
            name_cn = self.VAR_CN_NAMES[i]

            ax.plot(t, yi, color=colors[i], linewidth=2)
            ax.set_title(f"{name_cn}\n({name_en})", fontsize=10,
                         fontweight='bold', color=colors[i])

            # 标注最终稳态值
            final_val = yi[-1]
            ax.axhline(final_val, color='grey', linestyle='--',
                       linewidth=0.7, alpha=0.5)
            y_range = max(yi.max() - yi.min(), 0.1)
            ax.text(t[-1] * 0.75, final_val + 0.05 * y_range,
                    f"稳态={final_val:.3f}", fontsize=7, color='grey',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7))

            ax.set_xlim(t.min(), t.max())
            ymin, ymax = yi.min(), yi.max()
            if np.isnan(ymin) or np.isinf(ymin):
                ymin = 0
            if np.isnan(ymax) or np.isinf(ymax):
                ymax = 1
            y_range = max(ymax - ymin, 0.1)
            ax.set_ylim(ymin - 0.1 * y_range, ymax + 0.3 * y_range)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('时间 (a.u.)', fontsize=7)

        # 第9个变量 (ECM_degradation) 替换最后一个子图
        ax_extra = axes[7]
        ax_extra.clear()
        yi9 = np.nan_to_num(y[8], nan=0.0, posinf=10.0, neginf=0.0)
        ax_extra.plot(t, yi9, color='#C0392B', linewidth=2.5,
                      label='ECM_degradation')
        ax_extra.set_title(f"ECM降解\n(ECM_degradation)", fontsize=10,
                           fontweight='bold', color='#C0392B')

        final_val9 = yi9[-1]
        ax_extra.axhline(final_val9, color='grey', linestyle='--',
                         linewidth=0.7, alpha=0.5)
        y_range9 = max(yi9.max() - yi9.min(), 0.1)
        ax_extra.text(t[-1] * 0.75, final_val9 + 0.05 * y_range9,
                      f"稳态={final_val9:.3f}", fontsize=7, color='grey',
                      bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='white', alpha=0.7))

        ax_extra.set_xlim(t.min(), t.max())
        ymin, ymax = yi9.min(), yi9.max()
        if np.isnan(ymin) or np.isinf(ymin):
            ymin = 0
        if np.isnan(ymax) or np.isinf(ymax):
            ymax = 1
        y_range = max(ymax - ymin, 0.1)
        ax_extra.set_ylim(ymin - 0.1 * y_range, ymax + 0.3 * y_range)
        ax_extra.spines['top'].set_visible(False)
        ax_extra.spines['right'].set_visible(False)
        ax_extra.tick_params(labelsize=7)
        ax_extra.set_xlabel('时间 (a.u.)', fontsize=7)

        # 图标题
        subtitle = f"刚度水平: {stiffness_level:.1f}× normal"
        if perturbation_label:
            subtitle += f" | {perturbation_label}"
        fig.suptitle(
            f"{title}\n{subtitle}",
            fontsize=14, fontweight='bold', y=1.02,
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 机械力传导仿真图保存: {output_path}")

        return fig


def plot_dose_response(
    dose_results: Dict[str, np.ndarray],
    title: str = "刚度剂量响应 — MRTF-A 力传导通路稳态变化",
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制刚度剂量响应曲线 (2×3 子图)

    展示关键变量随基质刚度变化的稳态响应曲线,
    模拟从正常 NP (~2kPa) 到退变 NP (~15kPa) 的力传导变化

    Args:
        dose_results: simulate_dose_response() 返回的结果字典
        title: 图标题
        figsize: 图尺寸
        output_path: 保存路径
        dpi: 图像分辨率

    Returns:
        matplotlib Figure 对象
    """
    stiffness_levels = dose_results['stiffness_levels']
    steady = dose_results['steady_states']

    # 选择需要绘制的变量 (6个关键变量)
    plot_vars = [
        ('F_actin', 'F-肌动蛋白', '#E74C3C'),
        ('MRTFA_nuc', '核MRTF-A', '#8E44AD'),
        ('Kidins220', 'Kidins220', '#2ECC71'),
        ('AMPK_P', 'p-AMPK', '#F39C12'),
        ('PFKFB3', 'PFKFB3', '#3498DB'),
        ('Glycolysis_output', '糖酵解通量', '#E67E22'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, (var_name, var_cn, color) in enumerate(plot_vars):
        ax = axes[i]
        values = steady[var_name]

        # 标准化为相对正常刚度水平 (1.0) 的倍数
        normal_idx = np.argmin(np.abs(stiffness_levels - 1.0))
        if normal_idx < len(values) and values[normal_idx] != 0:
            rel_values = values / values[normal_idx]
        else:
            rel_values = values

        ax.plot(stiffness_levels, rel_values, 'o-', color=color,
                linewidth=2.5, markersize=7, markerfacecolor='white',
                markeredgewidth=2)

        # 标记正常刚度位置
        ax.axvline(1.0, color='green', linestyle=':', linewidth=1,
                   alpha=0.6, label='正常 NP (~2kPa)')

        # 标记退变刚度范围 (>2x, >~4-15kPa)
        ax.axvspan(2.0, stiffness_levels.max(), alpha=0.08,
                   color='red', label='退变范围')

        ax.set_title(f"{var_cn} ({var_name})", fontsize=10,
                     fontweight='bold', color=color)
        ax.set_xlabel('基质刚度 (相对倍数)', fontsize=8)
        ax.set_ylabel('相对变化 (倍数)', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')

        # 标注关键数值
        for j, (s, v) in enumerate(zip(stiffness_levels, rel_values)):
            if j % 2 == 0 or j == len(stiffness_levels) - 1:
                ax.annotate(f'{v:.2f}', (s, v),
                            textcoords="offset points",
                            xytext=(0, 10), fontsize=5.5,
                            ha='center', color=color, alpha=0.7)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 剂量响应曲线图保存: {output_path}")

    return fig


def plot_perturbation_heatmap(
    model: MRTFAMechanotransductionModel,
    stiffness_range: Optional[List[float]] = None,
    ccg_range: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制刚度×CCG-1423 双参数扰动热图

    系统性地扫描不同基质刚度与 CCG-1423 抑制剂浓度组合下的
    AMPK_P 和 Glycolysis_output 稳态响应

    Args:
        model: MRTFAMechanotransductionModel 实例
        stiffness_range: 刚度扫描范围
        ccg_range: CCG-1423 浓度扫描范围
        output_path: 保存路径
        dpi: 图像分辨率

    Returns:
        matplotlib Figure 对象
    """
    if stiffness_range is None:
        stiffness_range = np.linspace(0.5, 3.0, 8)
    if ccg_range is None:
        ccg_range = np.linspace(0, 1.0, 6)

    n_stiff = len(stiffness_range)
    n_ccg = len(ccg_range)

    ampk_matrix = np.zeros((n_ccg, n_stiff))
    glyc_matrix = np.zeros((n_ccg, n_stiff))
    mrtfa_matrix = np.zeros((n_ccg, n_stiff))
    kidins_matrix = np.zeros((n_ccg, n_stiff))

    for i, ccg in enumerate(ccg_range):
        for j, stiff in enumerate(stiffness_range):
            _, y = model.simulate(
                stiffness_level=stiff,
                perturbation={'CCG_inhibitor': ccg},
                t_span=(0, 500),
                n_points=200,
            )
            steady = y[:, -1]
            ampk_matrix[i, j] = steady[4]
            glyc_matrix[i, j] = steady[7]
            mrtfa_matrix[i, j] = steady[2]
            kidins_matrix[i, j] = steady[3]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    def plot_heatmap(ax, data, title, cmap, label):
        im = ax.pcolormesh(
            stiffness_range, ccg_range, data,
            cmap=cmap, shading='auto', edgecolors='white',
            linewidth=0.5,
        )
        ax.set_xlabel('基质刚度 (相对倍数)', fontsize=9)
        ax.set_ylabel('CCG-1423 剂量', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=8)
        ax.tick_params(labelsize=7)
        for i_ccg in range(n_ccg):
            for j_stiff in range(n_stiff):
                val = data[i_ccg, j_stiff]
                ax.text(
                    stiffness_range[j_stiff], ccg_range[i_ccg],
                    f'{val:.2f}', ha='center', va='center',
                    fontsize=6,
                    color='black' if 0.2 < val < 0.8 else 'white',
                )

    plot_heatmap(axes[0], mrtfa_matrix,
                 '核 MRTF-A (CCG-1423 响应)', 'Purples', 'MRTFA_nuc')
    plot_heatmap(axes[1], kidins_matrix,
                 'Kidins220 (MRTF-A 靶基因)', 'Greens', 'Kidins220')
    plot_heatmap(axes[2], ampk_matrix,
                 'p-AMPK (能量传感)', 'Oranges', 'AMPK_P')
    plot_heatmap(axes[3], glyc_matrix,
                 '糖酵解通量 (Glycolysis)', 'Blues', 'Glycolysis')

    fig.suptitle(
        "刚度 × CCG-1423 双参数扰动热图\n"
        "(Bone Research 2025: MRTF-A 介导的 NP 细胞机械力传导)",
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] 扰动热图保存: {output_path}")

    return fig


# ============================================================
# main: 快速演示与自检
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MRTF-A 机械力传导模型 — 快速演示")
    print("Bone Research 2025: Matrix stiffness regulates NP cell glycolysis")
    print("=" * 60)

    model = MRTFAMechanotransductionModel()

    # ---- 演示1: 正常刚度 ----
    print("\n[1/4] 正常刚度仿真 (stiffness=1.0 × normal, ~2kPa)")
    t, y = model.simulate(stiffness_level=1.0)
    metrics = model.get_steady_state_metrics(y[:, -1], 1.0)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- 演示2: 退变刚度 ----
    print("\n[2/4] 退变刚度仿真 (stiffness=3.0 × normal, ~6-15kPa)")
    t_deg, y_deg = model.simulate(stiffness_level=3.0)
    metrics_deg = model.get_steady_state_metrics(y_deg[:, -1], 3.0)
    for k, v in metrics_deg.items():
        print(f"  {k}: {v:.4f}")

    print("\n  → 退变 vs 正常变化:")
    for k in metrics:
        if k not in ('stiffness',):
            base = metrics[k]
            deg_val = metrics_deg[k]
            if abs(base) > 1e-6:
                chg = (deg_val - base) / base * 100
                print(f"    {k}: {chg:+.1f}%")

    # ---- 演示3: CCG-1423 逆转效应 ----
    print("\n[3/4] CCG-1423 逆转效应 (stiffness=3.0, CCG=0.6)")
    t_ccg, y_ccg = model.simulate(
        stiffness_level=3.0,
        perturbation={'CCG_inhibitor': 0.6},
    )
    metrics_ccg = model.get_steady_state_metrics(y_ccg[:, -1], 3.0)
    print(f"  Kidins220: {metrics_deg['Kidins220']:.4f} → {metrics_ccg['Kidins220']:.4f}")
    print(f"  Glycolysis: {metrics_deg['Glycolysis_output']:.4f} → {metrics_ccg['Glycolysis_output']:.4f}")
    print(f"  MRTF-A核: {metrics_deg['MRTFA_nuc']:.4f} → {metrics_ccg['MRTFA_nuc']:.4f}")

    # ---- 演示4: 剂量响应 ----
    print("\n[4/4] 刚度剂量响应扫描...")
    dose_results = model.simulate_dose_response(
        stiffness_range=[0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
    print("  刚度 | MRTF-A核 | Kidins220 | p-AMPK | 糖酵解")
    for i, s in enumerate(dose_results['stiffness_levels']):
        mrtfa = dose_results['steady_states']['MRTFA_nuc'][i]
        kidins = dose_results['steady_states']['Kidins220'][i]
        ampk = dose_results['steady_states']['AMPK_P'][i]
        glyc = dose_results['steady_states']['Glycolysis_output'][i]
        print(f"  {s:.1f}×  | {mrtfa:.4f}  | {kidins:.4f}  | "
              f"{ampk:.4f}  | {glyc:.4f}")

    # ---- 生成演示图 ----
    print("\n  生成演示图...")
    plot_dir = "/home/sandbox/.openclaw/workspace/repo/Virtual-nucleus-pulposus-cell/output"
    import os
    os.makedirs(plot_dir, exist_ok=True)

    model.plot_mechanotransduction(
        t, y, stiffness_level=1.0,
        title="正常 NP 细胞机械力传导",
        output_path=f"{plot_dir}/mechano_normal.png",
    )
    model.plot_mechanotransduction(
        t_deg, y_deg, stiffness_level=3.0,
        title="退变 NP 细胞机械力传导 (刚度↑)",
        output_path=f"{plot_dir}/mechano_degenerated.png",
    )
    model.plot_mechanotransduction(
        t_ccg, y_ccg, stiffness_level=3.0,
        perturbation_label="CCG-1423 0.6",
        title="CCG-1423 逆转效应",
        output_path=f"{plot_dir}/mechano_ccg_reversal.png",
    )
    plot_dose_response(
        dose_results,
        output_path=f"{plot_dir}/mechano_dose_response.png",
    )
    plot_perturbation_heatmap(
        model,
        output_path=f"{plot_dir}/mechano_perturbation_heatmap.png",
    )

    print(f"\n✅ 演示完成! 图文件保存至: {plot_dir}/")
    print("=" * 60)
