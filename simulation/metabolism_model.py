"""
NP 细胞代谢可塑性模型 — ODE 动力学 + 代谢扰动模拟
================================================================================
核心生物学发现:
  1. 糖酵解是 NP 主要 ATP 来源。HIF-1α → PFKFB3↑ → F-2,6-BP↑ → PFK1↑ → 糖酵解↑
     退变 NP 中 PFKFB3 下降 ~7 倍，糖酵解受损
  2. OXPHOS: 正常 NP 中受抑制 (PDK1 → PDH 磷酸化失活)，退变随血管化氧张力升高部分恢复
  3. 谷氨酰胺代谢: 补充 TCA 回补，退变中代谢重编程
  4. AMPK-mTOR 轴: 营养感知中心。AMPK 下调(通过 Kidins220) → 自噬受损
  5. 线粒体: 膜电位↓, ROS↑, 碎片化↑, 自噬↓ (PINK1/Parkin)

变量列表 (19 ODE state variables):
  Glucose_ext   - 细胞外葡萄糖浓度
  Glucose_int   - 细胞内葡萄糖浓度
  G6P           - 葡萄糖-6-磷酸
  F6P           - 果糖-6-磷酸
  F26BP         - 果糖-2,6-二磷酸 (PFKFB3 催化产物，PFK1 最强变构激活剂)
  F16BP         - 果糖-1,6-二磷酸
  Pyruvate      - 丙酮酸
  Lactate       - 乳酸
  NADH          - 还原型烟酰胺腺嘌呤二核苷酸
  ATP           - 三磷酸腺苷
  HIF1_alpha    - 低氧诱导因子 1α
  PDK1          - 丙酮酸脱氢酶激酶 1 (抑制 PDH)
  PFKFB3        - 6-磷酸果糖-2-激酶/果糖-2,6-二磷酸酶 3
  PKM2          - M2 型丙酮酸激酶
  Glutamine     - 谷氨酰胺
  Glutamate     - 谷氨酸
  alpha_KG      - α-酮戊二酸
  ROS_mito      - 线粒体活性氧
  MMPotential   - 线粒体膜电位 (归一化 0-1)

English interface with Chinese annotations.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 常量定义 / Physical & biochemical constants
# ============================================================
# 变量索引 / State variable indices (for vectorized ODE)
IDX = {
    'Glucose_ext': 0, 'Glucose_int': 1, 'G6P': 2, 'F6P': 3,
    'F26BP': 4, 'F16BP': 5, 'Pyruvate': 6, 'Lactate': 7,
    'NADH': 8, 'ATP': 9, 'HIF1_alpha': 10, 'PDK1': 11,
    'PFKFB3': 12, 'PKM2': 13, 'Glutamine': 14, 'Glutamate': 15,
    'alpha_KG': 16, 'ROS_mito': 17, 'MMPotential': 18,
}
N_VARS = 19  # 总变量数

# 变量名称 (用于绘图显示) / Variable display names
VAR_NAMES = {
    'Glucose_ext': r'Glucose$_{ext}$',
    'Glucose_int': r'Glucose$_{int}$',
    'G6P': 'G6P',
    'F6P': 'F6P',
    'F26BP': r'F-2,6-BP',
    'F16BP': r'F-1,6-BP',
    'Pyruvate': 'Pyr',
    'Lactate': 'Lac',
    'NADH': 'NADH',
    'ATP': 'ATP',
    'HIF1_alpha': r'HIF-1$\alpha$',
    'PDK1': 'PDK1',
    'PFKFB3': 'PFKFB3',
    'PKM2': 'PKM2',
    'Glutamine': 'Gln',
    'Glutamate': 'Glu',
    'alpha_KG': r'$\alpha$-KG',
    'ROS_mito': 'ROS$_{mito}$',
    'MMPotential': r'$\Delta\Psi_m$',
}


class NPMetabolismModel:
    """
    NP 细胞代谢可塑性模型: 糖酵解/OXPHOS/谷氨酰胺/AMPK-线粒体轴

    NPMetabolismModel integrates glycolysis, oxidative phosphorylation,
    glutaminolysis, and mitochondrial dynamics into a unified ODE system
    for nucleus pulposus cell metabolic plasticity.

    Parameters
    ----------
    params : dict, optional
        Custom parameter dictionary overriding defaults.

    Attributes
    ----------
    params : dict
        Full parameter set used by the model.
    state_names : list
        Ordered list of state variable names.
    """

    def __init__(self, params: Optional[Dict] = None):
        # ========== 默认参数 / Default Parameters ==========

        # --- 葡萄糖转运 / Glucose transport (GLUT1/3) ---
        self.params = {
            'V_GLUT': 0.8,          # 最大转运速率 / Max glucose uptake rate
            'Km_GLUT': 1.0,         # GLUT Michaelis 常数 (mM)

            # --- 己糖激酶 / Hexokinase (HK) ---
            'V_HK': 2.0,            # 最大 HK 速率
            'Km_HK': 0.05,          # HK 对 Glucose_int 的 Km (mM)

            # --- 磷酸葡萄糖异构酶 / Phosphoglucose isomerase (PGI) ---
            'V_PGI': 5.0,           # 最大 PGI 速率
            'Keq_PGI': 0.5,         # G6P ↔ F6P 平衡常数

            # --- PFK1 (糖酵解限速酶) / PFK1 (rate-limiting) ---
            'V_PFK1': 1.5,          # 最大 PFK1 速率
            'Km_PFK1_F6P': 0.1,     # PFK1 对 F6P 的 Km (mM)
            'Ka_F26BP': 0.01,       # F26BP 激活 PFK1 的 Ka (mM)
            'n_F26BP': 1.5,         # F26BP 协同激活 Hill 系数

            # --- F26BP 代谢 / F26BP metabolism ---
            'k_F26BP_synth': 0.05,  # PFKFB3 催化 F26BP 合成速率常数
            'k_F26BP_deg': 0.08,    # F26BP 降解速率常数

            # --- 下游糖酵解 / Downstream glycolysis ---
            'V_glyc': 3.0,          # F16BP → 2×Pyr 最大速率
            'Km_glyc_F16BP': 0.15,  # 对 F16BP 的 Km

            # --- LDH (乳酸脱氢酶) ---
            'V_LDH': 4.0,           # 最大 LDH 速率
            'Km_LDH_Pyr': 0.2,      # LDH 对 Pyr 的 Km (mM)
            'Km_LDH_NADH': 0.01,    # LDH 对 NADH 的 Km (mM)

            # --- 乳酸外排 / Lactate efflux ---
            'k_lac_efflux': 0.3,    # 乳酸外排速率常数

            # --- PDH (丙酮酸脱氢酶) / OXPHOS entry ---
            'V_PDH': 0.25,          # 最大 PDH 速率 (正常低)
            'Ki_PDK1_PDH': 0.05,    # PDK1 抑制 PDH 的 IC50 (mM)

            # --- OXPHOS / 氧化磷酸化 ---
            'V_OXPHOS': 1.2,        # NADH 最大氧化速率
            'Km_OXPHOS_NADH': 0.05, # OXPHOS 对 NADH 的 Km
            'P_O': 2.5,             # P/O 比 (每 NADH 生成 ATP 数)
            'k_ATP_synth_base': 0.1, # 底物水平磷酸化 ATP 生成

            # --- ATP 消耗 / ATP consumption ---
            'k_ATP_usage': 1.5,     # ATP 消耗速率常数
            'ATP_setpoint': 1.0,    # ATP 稳态设定点 (mM)

            # --- HIF-1α 调控 / HIF-1α regulation ---
            'k_HIF_prod': 0.12,     # HIF-1α 基础合成速率
            'k_HIF_deg': 0.6,       # HIF-1α 常氧下降解速率
            'Km_O2_HIF': 0.03,      # O2 PHD 的 Km (O2 分数, ~3%)

            # --- PDK1 表达 / PDK1 expression ---
            'k_PDK1_base': 0.02,    # PDK1 基础表达
            'k_PDK1_HIF': 0.15,     # HIF-1α → PDK1 转录激活强度
            'Km_HIF_PDK1': 0.1,     # HIF-1α 激活 PDK1 半饱和浓度
            'k_PDK1_deg': 0.1,      # PDK1 降解速率

            # --- PFKFB3 表达 / PFKFB3 expression ---
            'k_PFKFB3_base': 0.03,  # PFKFB3 基础表达
            'k_PFKFB3_HIF': 0.18,   # HIF-1α → PFKFB3 转录激活强度
            'Km_HIF_PFKFB3': 0.08,  # HIF-1α 激活 PFKFB3 半饱和浓度
            'k_PFKFB3_deg': 0.12,   # PFKFB3 降解速率

            # --- PKM2 表达 / PKM2 expression ---
            'k_PKM2_base': 0.05,    # PKM2 基础表达
            'k_PKM2_HIF': 0.10,     # HIF-1α → PKM2 转录激活强度
            'k_PKM2_deg': 0.08,     # PKM2 降解速率

            # --- 谷氨酰胺代谢 / Glutamine metabolism ---
            'k_Gln_supply': 0.5,    # 谷氨酰胺供应速率
            'V_Glnase': 0.6,        # 谷氨酰胺酶最大速率
            'Km_Glnase': 0.3,       # 谷氨酰胺酶 Km
            'V_GLUD': 0.4,          # 谷氨酸脱氢酶最大速率
            'Km_GLUD': 0.2,         # GLUD Km
            'k_GOT': 0.25,          # GOT (谷草转氨酶) 速率
            'k_alphaKG_use': 0.35,  # α-KG 进入 TCA 消耗速率
            'k_Gln_deg': 0.05,      # 谷氨酰胺非特异性降解

            # --- 线粒体 ROS / Mitochondrial ROS ---
            'k_ROS_base': 0.02,     # 基础 ROS 产生
            'k_ROS_MMP': 0.15,      # 膜电位下降导致 ROS 增加系数
            'k_ROS_O2': 0.3,        # 氧浓度依赖的 ROS 产生系数
            'k_ROS_scavenge': 0.2,  # ROS 清除速率 (SOD/GSH 等)

            # --- 线粒体膜电位 / Mitochondrial membrane potential ---
            'k_MMP_recovery': 0.1,  # 膜电位恢复速率
            'k_MMP_damage': 0.08,   # ROS 损伤膜电位速率
            'MMP_healthy': 0.95,    # 健康膜电位 (归一化)
            'MMP_min': 0.15,        # 最小膜电位

            # --- 营养感知 / Nutrient sensing (AMPK-mTOR proxy) ---
            'AMPK_basal': 0.3,      # 基础 AMPK 活性水平
            'AMPK_ATP_sens': 0.5,   # AMPK 对 ATP/ADP 的敏感度
            'Km_AMPK_ATP': 0.8,     # AMPK 激活的 ATP 阈值

            # --- 扰动参数 / Perturbation-specific ---
            'hypoxia_boost': 2.5,   # 低氧时 HIF-1α 稳定倍数
            'rotenone_OXPHOS': 0.05, # 鱼藤酮抑制后 OXPHOS 残留
            'DG_glucose': 0.02,     # 2DG 抑制后 HK 残留活性
        }

        # 覆盖自定义参数 / Override with custom params
        if params is not None:
            self.params.update(params)

        self.state_names = list(IDX.keys())

    # ============================================================
    # ODE 系统 / ODE System
    # ============================================================

    def _ode_system(self, t: float, y: np.ndarray,
                    oxygen_level: float, glucose_level: float,
                    perturbation: Optional[str]) -> np.ndarray:
        """
        ODE 右侧函数 / Right-hand side of the ODE system.

        Parameters
        ----------
        t : float
            当前时间 / Current time.
        y : np.ndarray
            状态向量 / State vector (19 variables).
        oxygen_level : float
            氧浓度 (0-1, 分数) / Oxygen concentration (fraction).
        glucose_level : float
            细胞外葡萄糖供应水平 (mM).
        perturbation : str or None
            当前扰动模式 / Current perturbation mode.

        Returns
        -------
        np.ndarray
            各变量的时间导数 / Time derivatives.
        """
        p = self.params

        # ---- 解包状态变量 / Unpack state ----
        G_ext, G_int, G6P, F6P, F26BP, F16BP = y[0:6]
        Pyr, Lac, NADH, ATP = y[6:10]
        HIF, PDK1, PFKFB3, PKM2 = y[10:14]
        Gln, Glu, aKG, ROS, MMP = y[14:19]

        # ---- 扰动调整 / Perturbation adjustments ----
        _oxy = oxygen_level
        _glu = glucose_level
        _hypoxia_boost = p['hypoxia_boost']

        if perturbation == 'hypoxia':
            _oxy = 0.01  # 严重低氧 / Severe hypoxia
            _hypoxia_boost = 4.0
        elif perturbation == 'glucose_deprivation':
            _glu = max(0.01, p['Km_GLUT'] * 0.05)  # ~5% 正常
        elif perturbation == 'PFKFB3_act':
            pass  # 通过 PFKFB3 初始值处理
        elif perturbation == 'AMPK_act':
            pass  # 通过 ATP 感知调整
        elif perturbation == 'rotenone':
            pass  # 在 OXPHOS 项中处理
        elif perturbation == '2DG':
            pass  # 在 HK 项中处理

        # ---- 氧气对 HIF-1α 稳定的影响 / O2 effect on HIF-1α stability ----
        # 常氧时 PHD 活性高 → HIF 降解; 低氧时 PHD 受抑制 → HIF 稳定
        phd_activity = _oxy / (p['Km_O2_HIF'] + _oxy)
        hif_stabilization = 1.0 - phd_activity * 0.95  # 正常0-0.95范围
        hif_stabilization = np.clip(hif_stabilization, 0.05, 1.0)
        # 低氧增强: 严重低氧时 HIF 大幅稳定 (但总量有天花板)
        if _oxy < 0.03:
            hif_stabilization = min(1.0, hif_stabilization * min(_hypoxia_boost, 2.0))

        # ---- 1. 葡萄糖转运 / Glucose transport ----
        glut_uptake = p['V_GLUT'] * G_ext / (p['Km_GLUT'] + G_ext)
        # 外部葡萄糖供应
        glucose_supply = _glu * 0.5  # 供应速率与设定水平成正比

        # ---- 2. 己糖激酶 / Hexokinase ----
        # 2DG 扰动下 HK 受抑制
        hk_Vmax = p['V_HK']
        if perturbation == '2DG':
            hk_Vmax *= p['DG_glucose']
        hk_flux = hk_Vmax * G_int / (p['Km_HK'] + G_int)

        # ---- 3. PGI (G6P ↔ F6P) ----
        pgi_flux = p['V_PGI'] * (G6P - F6P / p['Keq_PGI'])

        # ---- 4. PFK1 (F6P → F16BP, 激活受 F26BP 调控) ----
        # F26BP 激活 PFK1: 变构激活效应
        f26bp_activation = 1.0 + (F26BP ** p['n_F26BP']) / \
                           (p['Ka_F26BP'] ** p['n_F26BP'] + F26BP ** p['n_F26BP']) * 20.0
        pfk1_flux = p['V_PFK1'] * f26bp_activation * F6P / (p['Km_PFK1_F6P'] + F6P)

        # ---- 5. F26BP 代谢 (PFKFB3 产物) ----
        f26bp_synth = p['k_F26BP_synth'] * PFKFB3
        f26bp_degr = p['k_F26BP_deg'] * F26BP

        # ---- 6. 下游糖酵解 / Downstream glycolysis ----
        glyc_flux = p['V_glyc'] * F16BP / (p['Km_glyc_F16BP'] + F16BP)

        # ---- 7. LDH (Pyr + NADH ↔ Lac + NAD+) ----
        ldh_flux = p['V_LDH'] * (Pyr / (p['Km_LDH_Pyr'] + Pyr)) * \
                   (NADH / (p['Km_LDH_NADH'] + NADH))

        # ---- 8. 乳酸外排 / Lactate efflux ----
        lac_efflux = p['k_lac_efflux'] * Lac

        # ---- 9. PDH (丙酮酸进入 TCA/OXPHOS) ----
        # PDK1 磷酸化 PDH 使其失活: 抑制函数
        pdk1_inhibition = PDK1 / (PDK1 + p['Ki_PDK1_PDH'])
        pdh_flux = p['V_PDH'] * Pyr * (1.0 - pdk1_inhibition)
        pdh_flux = max(0, pdh_flux)

        # ---- 10. OXPHOS (NADH → ATP, 依赖于氧和膜电位) ----
        # 鱼藤酮抑制复合体 I
        oxphos_Vmax = p['V_OXPHOS']
        if perturbation == 'rotenone':
            oxphos_Vmax *= p['rotenone_OXPHOS']

        # OXPHOS 速率: NADH + O2 + 膜电位 → ATP
        oxphos_flux = oxphos_Vmax * \
                      (NADH / (p['Km_OXPHOS_NADH'] + NADH)) * \
                      (_oxy / (0.05 + _oxy)) * \
                      max(0, (MMP - 0.1) / 0.9)

        # ATP 生成: 糖酵解 (底物水平) + OXPHOS
        atp_glyc = 2.0 * glyc_flux  # 糖酵解净产 2 ATP/葡萄糖
        atp_oxphos = p['P_O'] * oxphos_flux  # OXPHOS ATP 生成

        # ATP 消耗: 基础消耗 + 依赖于 ATP 水平的消耗 (负反馈)
        # 加入 AMPK-like 调控: 低 ATP 时抑制消耗
        ampk_activity = p['AMPK_basal'] + \
                        p['AMPK_ATP_sens'] * (1.0 - ATP / (p['Km_AMPK_ATP'] + ATP))
        ampk_activity = np.clip(ampk_activity, 0.1, 1.5)

        # 扰动: AMPK 激活
        if perturbation == 'AMPK_act':
            ampk_activity = min(2.0, ampk_activity * 2.5)

        atp_consum = p['k_ATP_usage'] * ATP * (1.2 - 0.4 * ampk_activity)
        atp_consum = max(0.05 * ATP, atp_consum)

        # ---- 11. 谷氨酰胺代谢 / Glutamine metabolism ----
        # 谷氨酰胺供应
        gln_supply_in = p['k_Gln_supply']
        # 谷氨酰胺酶 (GLS): Gln → Glu
        gls_flux = p['V_Glnase'] * Gln / (p['Km_Glnase'] + Gln)
        # 谷氨酸脱氢酶 (GLUD): Glu → α-KG
        glud_flux = p['V_GLUD'] * Glu / (p['Km_GLUD'] + Glu)
        # GOT/transaminase: Glu → α-KG
        got_flux = p['k_GOT'] * Glu
        # α-KG 进入 TCA 消耗
        akg_use = p['k_alphaKG_use'] * aKG

        # ---- 12. 线粒体 ROS / Mitochondrial ROS ----
        # ROS 产生: 基础 + 膜电位下降 → 电子漏↑ + 氧浓度依赖
        mmp_dysfunction = max(0, (p['MMP_healthy'] - MMP) / p['MMP_healthy'])
        ros_prod = p['k_ROS_base'] + \
                   p['k_ROS_MMP'] * mmp_dysfunction + \
                   p['k_ROS_O2'] * _oxy * mmp_dysfunction
        # ROS 清除
        ros_deg = p['k_ROS_scavenge'] * ROS

        # ---- 13. 线粒体膜电位 / Mitochondrial membrane potential ----
        # 膜电位恢复 (依赖于 OXPHOS 活性和健康自噬)
        mmp_recovery = p['k_MMP_recovery'] * (p['MMP_healthy'] - MMP) * \
                       (1.0 / (1.0 + ROS * 5.0))  # ROS 抑制恢复
        # 膜电位损伤 (ROS & 功能障碍)
        mmp_damage = p['k_MMP_damage'] * ROS * max(0, MMP - p['MMP_min'])

        # ---- 组装导数 / Assemble derivatives ----
        dydt = np.zeros(N_VARS)

        # d(Glucose_ext)/dt: 供应 - 摄取
        dydt[IDX['Glucose_ext']] = glucose_supply - glut_uptake

        # d(Glucose_int)/dt: 摄取 - HK
        dydt[IDX['Glucose_int']] = glut_uptake - hk_flux

        # d(G6P)/dt: HK - PGI
        dydt[IDX['G6P']] = hk_flux - pgi_flux

        # d(F6P)/dt: PGI + FBPase - PFK1
        dydt[IDX['F6P']] = pgi_flux - pfk1_flux

        # d(F26BP)/dt: PFKFB3 合成 - 降解
        dydt[IDX['F26BP']] = f26bp_synth - f26bp_degr

        # d(F16BP)/dt: PFK1 - 下游糖酵解
        dydt[IDX['F16BP']] = pfk1_flux - glyc_flux

        # d(Pyruvate)/dt: 糖酵解产生 ×2 - LDH - PDH
        dydt[IDX['Pyruvate']] = 2.0 * glyc_flux - ldh_flux - pdh_flux

        # d(Lactate)/dt: LDH - 外排
        dydt[IDX['Lactate']] = ldh_flux - lac_efflux

        # d(NADH)/dt: 糖酵解产 NADH + TCA NADH - OXPHOS - LDH
        # 每分子葡萄糖经糖酵解净产 2 NADH (GAPDH ×2); α-KG → TCA 也贡献
        nadh_glycolysis = 2.0 * glyc_flux  # 每 F16BP → 2×Pyr 产生 2 NADH
        nadh_tca = 3.0 * akg_use  # α-KG → ... → malate 约 3 NADH/FADH2/圈
        dydt[IDX['NADH']] = nadh_glycolysis + nadh_tca - oxphos_flux - ldh_flux * 0.8

        # d(ATP)/dt: 生成 - 消耗
        dydt[IDX['ATP']] = atp_glyc + atp_oxphos + p['k_ATP_synth_base'] * glyc_flux - atp_consum

        # d(HIF1_alpha)/dt: 合成 - 氧依赖降解
        hif_degradation = p['k_HIF_deg'] * max(0.05, 1.0 - hif_stabilization) * HIF
        dydt[IDX['HIF1_alpha']] = p['k_HIF_prod'] - hif_degradation

        # d(PDK1)/dt: 基础 + HIF 激活 - 降解
        pdK1_activation = p['k_PDK1_HIF'] * HIF / (p['Km_HIF_PDK1'] + HIF)
        dydt[IDX['PDK1']] = p['k_PDK1_base'] + pdK1_activation - p['k_PDK1_deg'] * PDK1

        # d(PFKFB3)/dt: 基础 + HIF 激活 - 降解
        pfkfb3_activation = p['k_PFKFB3_HIF'] * HIF / (p['Km_HIF_PFKFB3'] + HIF)
        # PFKFB3_act 扰动: 人为提升 PFKFB3 表达
        if perturbation == 'PFKFB3_act':
            pfkfb3_activation *= 3.0
        dydt[IDX['PFKFB3']] = p['k_PFKFB3_base'] + pfkfb3_activation - p['k_PFKFB3_deg'] * PFKFB3

        # d(PKM2)/dt: 基础 + HIF 激活 - 降解
        pkm2_activation = p['k_PKM2_HIF'] * HIF / (0.1 + HIF)
        dydt[IDX['PKM2']] = p['k_PKM2_base'] + pkm2_activation - p['k_PKM2_deg'] * PKM2

        # d(Glutamine)/dt: 供应 - GLS - 非特异性降解
        dydt[IDX['Glutamine']] = gln_supply_in - gls_flux - p['k_Gln_deg'] * Gln

        # d(Glutamate)/dt: GLS - GLUD - GOT
        dydt[IDX['Glutamate']] = gls_flux - glud_flux - got_flux

        # d(alpha_KG)/dt: GLUD + GOT - TCA 消耗
        dydt[IDX['alpha_KG']] = glud_flux + got_flux - akg_use

        # d(ROS_mito)/dt: 产生 - 清除
        dydt[IDX['ROS_mito']] = ros_prod - ros_deg

        # d(MMPotential)/dt: 恢复 - 损伤
        dydt[IDX['MMPotential']] = mmp_recovery - mmp_damage

        return dydt

    # ============================================================
    # 稳态求解 / Steady-state solver
    # ============================================================

    def _get_initial_conditions(self, oxygen_level: float = 0.05,
                                glucose_level: float = 1.0,
                                perturbation: Optional[str] = None) -> np.ndarray:
        """
        获取合理的初始条件 / Get reasonable initial conditions.

        使用基于生物知识的估算值，确保 ODE 从生理相关的起点开始集成。
        Uses biologically-informed estimates to start integration from
        physiologically relevant states.
        """
        y0 = np.zeros(N_VARS)

        # 正常 NP 代谢状态的估算稳态值
        if perturbation == 'hypoxia':
            # 低氧: HIF 升高, 糖酵解增强
            y0[IDX['Glucose_ext']] = glucose_level
            y0[IDX['Glucose_int']] = 0.3
            y0[IDX['G6P']] = 0.05
            y0[IDX['F6P']] = 0.02
            y0[IDX['F26BP']] = 0.3       # PFKFB3 高, F26BP 升高
            y0[IDX['F16BP']] = 0.08
            y0[IDX['Pyruvate']] = 0.15
            y0[IDX['Lactate']] = 2.5      # 乳酸增加 (糖酵解↑)
            y0[IDX['NADH']] = 0.3
            y0[IDX['ATP']] = 0.8
            y0[IDX['HIF1_alpha']] = 0.7   # HIF 稳定
            y0[IDX['PDK1']] = 0.6         # PDK1 被 HIF 激活
            y0[IDX['PFKFB3']] = 0.7       # PFKFB3 被 HIF 激活
            y0[IDX['PKM2']] = 0.6
            y0[IDX['Glutamine']] = 0.6
            y0[IDX['Glutamate']] = 0.4
            y0[IDX['alpha_KG']] = 0.15
            y0[IDX['ROS_mito']] = 0.15    # 低氧下 ROS 适度
            y0[IDX['MMPotential']] = 0.75 # 膜电位稍降
        elif perturbation == 'glucose_deprivation':
            # 缺糖: 代谢全面下降
            y0[IDX['Glucose_ext']] = glucose_level * 0.05
            y0[IDX['Glucose_int']] = 0.02
            y0[IDX['G6P']] = 0.01
            y0[IDX['F6P']] = 0.005
            y0[IDX['F26BP']] = 0.05
            y0[IDX['F16BP']] = 0.01
            y0[IDX['Pyruvate']] = 0.03
            y0[IDX['Lactate']] = 0.2
            y0[IDX['NADH']] = 0.05
            y0[IDX['ATP']] = 0.2          # ATP 严重下降
            y0[IDX['HIF1_alpha']] = 0.1
            y0[IDX['PDK1']] = 0.1
            y0[IDX['PFKFB3']] = 0.08
            y0[IDX['PKM2']] = 0.1
            y0[IDX['Glutamine']] = 0.3    # 转向谷氨酰胺
            y0[IDX['Glutamate']] = 0.2
            y0[IDX['alpha_KG']] = 0.1
            y0[IDX['ROS_mito']] = 0.3     # 应激 ROS 升高
            y0[IDX['MMPotential']] = 0.5  # 膜电位下降
        else:
            # 正常 NP 低氧代谢状态 (生理 1-5% O2)
            y0[IDX['Glucose_ext']] = glucose_level
            y0[IDX['Glucose_int']] = 0.2
            y0[IDX['G6P']] = 0.04
            y0[IDX['F6P']] = 0.015
            y0[IDX['F26BP']] = 0.15       # 正常 F26BP (PFKFB3 维持)
            y0[IDX['F16BP']] = 0.05
            y0[IDX['Pyruvate']] = 0.1
            y0[IDX['Lactate']] = 1.5      # 正常 NP 乳酸生产
            y0[IDX['NADH']] = 0.2
            y0[IDX['ATP']] = 1.0
            y0[IDX['HIF1_alpha']] = 0.4   # 低氧驱动基础 HIF
            y0[IDX['PDK1']] = 0.35        # HIF 驱动 PDK1
            y0[IDX['PFKFB3']] = 0.4       # 正常 PFKFB3
            y0[IDX['PKM2']] = 0.35
            y0[IDX['Glutamine']] = 0.5
            y0[IDX['Glutamate']] = 0.3
            y0[IDX['alpha_KG']] = 0.12
            y0[IDX['ROS_mito']] = 0.1
            y0[IDX['MMPotential']] = 0.85 # 健康线粒体膜电位

        return y0

    # ============================================================
    # 主模拟接口 / Main simulation interface
    # ============================================================

    def simulate(self, oxygen_level: float = 0.05,
                 glucose_level: float = 1.0,
                 perturbation: Optional[str] = None,
                 t_span: Tuple[float, float] = (0.0, 200.0),
                 t_eval: Optional[np.ndarray] = None,
                 method: str = 'BDF',
                 rtol: float = 1e-6,
                 atol: float = 1e-9) -> Dict:
        """
        运行代谢 ODE 模拟 / Run metabolic ODE simulation.

        Parameters
        ----------
        oxygen_level : float, default=0.05
            氧浓度 (体积分数, 0-1) / Oxygen concentration.
            生理 NP 环境 = 0.01-0.05 (1-5% O2)
        glucose_level : float, default=1.0
            细胞外葡萄糖浓度 (mM) / Extracellular glucose.
        perturbation : str or None
            扰动模式 / Perturbation type:
            - None: 无扰动 / No perturbation
            - 'hypoxia': 严重低氧 / Severe hypoxia
            - 'glucose_deprivation': 缺糖 / Glucose deprivation
            - 'PFKFB3_act': PFKFB3 激活 / PFKFB3 activation
            - 'AMPK_act': AMPK 激活 / AMPK activation
            - 'rotenone': 复合体 I 抑制 / Complex I inhibition
            - '2DG': 2-脱氧葡萄糖 / 2-deoxyglucose
        t_span : tuple, default=(0, 200)
            模拟时间范围 / Simulation time span.
        t_eval : np.ndarray, optional
            输出时间点 / Time points for output.
        method : str, default='BDF'
            solve_ivp 积分方法 / Integration method.
        rtol, atol : float
            相对/绝对容差 / Relative/absolute tolerance.

        Returns
        -------
        dict
            't': 时间点 / Time points
            'y': 状态矩阵 (N_vars × N_timesteps) / State matrix
            'params': 使用的参数 / Parameters used
            'perturbation': 扰动类型 / Perturbation type
            'oxygen_level': 氧浓度 / Oxygen level
            'glucose_level': 葡萄糖浓度 / Glucose level
            'success': 是否成功 / Integration success
        """
        # 初始条件 / Initial conditions
        y0 = self._get_initial_conditions(oxygen_level, glucose_level, perturbation)

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        # 求解 ODE / Solve ODE
        sol = solve_ivp(
            self._ode_system,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            args=(oxygen_level, glucose_level, perturbation),
        )

        return {
            't': sol.t,
            'y': sol.y,
            'params': self.params.copy(),
            'perturbation': perturbation,
            'oxygen_level': oxygen_level,
            'glucose_level': glucose_level,
            'success': sol.success,
            'message': sol.message,
        }

    # ============================================================
    # 代谢应激模拟 / Metabolic Stress Simulation
    # ============================================================

    def simulate_metabolic_stress(self) -> Dict:
        """
        模拟多种代谢应激条件的影响 / Simulate multiple metabolic stress conditions.

        运行 8 种不同条件的模拟，涵盖:
        1. 正常 NP 生理状态 (5% O2, 1 mM 葡萄糖)
        2. 严重低氧 (1% O2)
        3. 缺糖
        4. PFKFB3 激活 (模拟退变挽救)
        5. AMPK 激活
        6. 鱼藤酮 (复合体 I 抑制)
        7. 2-脱氧葡萄糖 (糖酵解抑制)
        8. 高糖 + 常氧 (模拟退变早期)

        Returns
        -------
        dict
            'conditions': 各条件模拟结果列表
            'comparison': 关键指标汇总矩阵 (ATP, glycolysis, OXPHOS, ROS, MMP)
        """
        conditions = [
            ('normal',      0.05, 1.0,  None),
            ('hypoxia',     0.01, 1.0,  'hypoxia'),
            ('glucose_deprivation', 0.05, 0.05, 'glucose_deprivation'),
            ('PFKFB3_act',  0.05, 1.0,  'PFKFB3_act'),
            ('AMPK_act',    0.05, 1.0,  'AMPK_act'),
            ('rotenone',    0.05, 1.0,  'rotenone'),
            ('2DG',         0.05, 1.0,  '2DG'),
            ('hyperglycemia_normoxia', 0.20, 2.0, None),
        ]

        results = {}
        comparison = {}

        for name, oxy, glu, pert in conditions:
            sol = self.simulate(
                oxygen_level=oxy,
                glucose_level=glu,
                perturbation=pert,
                t_span=(0, 300),
            )
            results[name] = sol
            # 取最后时间点的稳态值
            y_final = sol['y'][:, -1]
            comparison[name] = {
                'ATP': y_final[IDX['ATP']],
                'Lactate': y_final[IDX['Lactate']],
                'HIF1_alpha': y_final[IDX['HIF1_alpha']],
                'PFKFB3': y_final[IDX['PFKFB3']],
                'ROS_mito': y_final[IDX['ROS_mito']],
                'MMPotential': y_final[IDX['MMPotential']],
                'Glycolysis_rate': 2.0 * y_final[IDX['F16BP']],  # 近似
                'NADH': y_final[IDX['NADH']],
                'Lactate_Pyruvate': max(0.01, y_final[IDX['Lactate']] /
                                         max(0.001, y_final[IDX['Pyruvate']])),
                'Glutamine': y_final[IDX['Glutamine']],
                'alpha_KG': y_final[IDX['alpha_KG']],
            }

        return {
            'conditions': results,
            'comparison': comparison,
        }

    # ============================================================
    # 绘图 / Plotting
    # ============================================================

    def plot_metabolism(self, result: Dict,
                        variables: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (14, 10),
                        show_legend: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制代谢模拟时间历程 / Plot metabolism simulation time course.

        Parameters
        ----------
        result : dict
            simulate() 方法返回的结果字典.
        variables : list, optional
            要绘制的变量名列表. 默认为核心代谢指标.
        figsize : tuple, default=(14, 10)
            图形尺寸 / Figure size.
        show_legend : bool, default=True
            是否显示图例 / Whether to show legend.
        save_path : str, optional
            保存路径 / Path to save the figure.

        Returns
        -------
        plt.Figure
        """
        if variables is None:
            variables = [
                'Glucose_ext', 'Glucose_int',
                'F26BP', 'F16BP',
                'Pyruvate', 'Lactate',
                'ATP', 'NADH',
                'HIF1_alpha', 'PFKFB3',
                'PDK1', 'PKM2',
                'Glutamine', 'alpha_KG',
                'ROS_mito', 'MMPotential',
            ]

        t = result['t']
        y = result['y']
        pert = result.get('perturbation', 'None')
        oxy = result.get('oxygen_level', 0.05)
        glu = result.get('glucose_level', 1.0)

        n_vars = len(variables)
        n_cols = 4
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, var in enumerate(variables):
            if var not in IDX:
                axes[i].set_title(f'{var} (unknown)')
                axes[i].text(0.5, 0.5, 'N/A', ha='center', va='center',
                             transform=axes[i].transAxes)
                continue
            idx = IDX[var]
            axes[i].plot(t, y[idx, :], linewidth=1.5, color='#2c7bb6')
            axes[i].set_xlabel('Time (a.u.)', fontsize=8)
            axes[i].set_ylabel(VAR_NAMES.get(var, var), fontsize=9)
            axes[i].set_title(VAR_NAMES.get(var, var), fontsize=10, fontweight='bold')
            axes[i].grid(True, alpha=0.3)

            # 标注稳态值 / Annotate steady state
            steady_val = y[idx, -1]
            axes[i].axhline(y=steady_val, color='#d7191c', linestyle='--',
                            linewidth=0.8, alpha=0.6)
            axes[i].annotate(f'{steady_val:.3f}',
                             xy=(t[-1], steady_val),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=7, color='#d7191c')

        # 隐藏多余子图 / Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)

        # 总标题 / Suptitle
        pert_label = pert if pert else 'none'
        fig.suptitle(
            f'NP Cell Metabolism | O₂={oxy*100:.1f}%, Glucose={glu:.2f} mM, '
            f'Perturbation={pert_label}',
            fontsize=13, fontweight='bold', y=1.01
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")

        return fig

    # ============================================================
    # 分析工具 / Analysis utilities
    # ============================================================

    def get_metabolic_profile(self, result: Dict) -> Dict:
        """
        从模拟结果提取代谢特征 / Extract metabolic profile from simulation.

        Parameters
        ----------
        result : dict
            simulate() 返回的结果.

        Returns
        -------
        dict
            代谢特征参数 / Metabolic profile parameters.
        """
        y_final = result['y'][:, -1]

        # 估算代谢流 / Estimate metabolic fluxes
        glycolysis_est = 2.0 * y_final[IDX['F16BP']]  # 近似糖酵解速率
        oxphos_est = y_final[IDX['NADH']] * y_final[IDX['MMPotential']]  # 近似

        profile = {
            'glycolysis_dominance': glycolysis_est / max(0.01, glycolysis_est + oxphos_est),
            'energy_charge': y_final[IDX['ATP']] / (1.0 + y_final[IDX['ATP']]),
            'lactate_pyruvate_ratio': y_final[IDX['Lactate']] / max(0.001, y_final[IDX['Pyruvate']]),
            'redox_state': y_final[IDX['NADH']],
            'hif_activity': y_final[IDX['HIF1_alpha']] / 0.5,
            'mito_health': y_final[IDX['MMPotential']] / 0.9,
            'oxidative_stress': y_final[IDX['ROS_mito']],
            'glutaminolysis': y_final[IDX['alpha_KG']] * y_final[IDX['Glutamine']],
        }
        return profile


# ============================================================
# 代谢景观图 / Metabolic Landscape Plotting
# ============================================================

def plot_metabolic_landscape(model: NPMetabolismModel,
                             oxygen_range: Optional[np.ndarray] = None,
                             glucose_range: Optional[np.ndarray] = None,
                             metric: str = 'ATP',
                             figsize: Tuple[int, int] = (10, 8),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制不同氧/葡萄糖浓度下的代谢表型景观图.

    Plot metabolic phenotype landscape across oxygen and glucose gradients.

    Parameters
    ----------
    model : NPMetabolismModel
        已初始化的代谢模型实例.
    oxygen_range : np.ndarray, optional
        氧浓度扫描范围 / O2 levels to scan (default: 0.5% to 21%).
    glucose_range : np.ndarray, optional
        葡萄糖浓度扫描范围 / Glucose levels to scan (default: 0.01 to 5 mM).
    metric : str, default='ATP'
        要绘制的指标 / Metric to plot:
        'ATP', 'Lactate', 'HIF1_alpha', 'PFKFB3', 'ROS_mito', 'MMPotential'.
    figsize : tuple
        图形尺寸.
    save_path : str, optional
        保存路径.

    Returns
    -------
    plt.Figure
    """
    if oxygen_range is None:
        oxygen_range = np.logspace(np.log10(0.005), np.log10(0.21), 20)
    if glucose_range is None:
        glucose_range = np.logspace(np.log10(0.01), np.log10(5.0), 20)

    if metric not in IDX:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(IDX.keys())}")

    metric_idx = IDX[metric]
    landscape = np.zeros((len(oxygen_range), len(glucose_range)))

    print(f"Scanning {len(oxygen_range)}×{len(glucose_range)} conditions...")
    for i, oxy in enumerate(oxygen_range):
        for j, glu in enumerate(glucose_range):
            try:
                sol = model.simulate(
                    oxygen_level=oxy,
                    glucose_level=glu,
                    perturbation=None,
                    t_span=(0, 200),
                    t_eval=np.linspace(0, 200, 100),
                )
                landscape[i, j] = sol['y'][metric_idx, -1]
            except Exception as e:
                landscape[i, j] = np.nan

    # 绘图 / Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    O2_grid, Glu_grid = np.meshgrid(oxygen_range, glucose_range, indexing='ij')

    # 使用对数颜色映射 / Log color scale
    valid = np.isfinite(landscape)
    if valid.any():
        vmin = np.nanmin(landscape[valid])
        vmax = np.nanmax(landscape[valid])
    else:
        vmin, vmax = 0, 1

    im = ax.pcolormesh(O2_grid * 100, Glu_grid, landscape,
                       shading='auto', cmap='viridis',
                       norm=plt.matplotlib.colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
                       if vmin > 0 else None)
    plt.colorbar(im, ax=ax, label=VAR_NAMES.get(metric, metric))

    ax.set_xlabel('Oxygen concentration (%)', fontsize=12)
    ax.set_ylabel('Glucose concentration (mM)', fontsize=12)
    ax.set_title(f'Metabolic Landscape: {VAR_NAMES.get(metric, metric)}',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 标注生理区域 / Annotate physiological zones
    # 生理 NP: 1-5% O2, 0.5-1.5 mM glucose
    ax.fill_between([1, 5], 0.5, 1.5, alpha=0.15, color='green',
                    label='Physiological NP')
    # 退变: >5% O2 (血管化), 血糖可能更高
    ax.fill_between([5, 21], 1.0, 5.0, alpha=0.15, color='red',
                    label='Degenerated NP')
    ax.legend(loc='upper right', fontsize=9)

    ax.set_xlim(oxygen_range[0] * 100, oxygen_range[-1] * 100)
    ax.set_ylim(glucose_range[0], glucose_range[-1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Landscape saved: {save_path}")

    return fig


# ============================================================
# 快速测试入口 / Quick test entry point
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("NP Cell Metabolism Model — Quick Test")
    print("=" * 60)

    # 初始化模型 / Initialize model
    model = NPMetabolismModel()
    print("✓ Model initialized")

    # 1. 正常 NP 模拟 / Normal NP simulation
    print("\n--- 1. Normal NP (5% O₂, 1 mM glucose) ---")
    result_normal = model.simulate(oxygen_level=0.05, glucose_level=1.0)
    profile = model.get_metabolic_profile(result_normal)
    print(f"   ATP={result_normal['y'][IDX['ATP'], -1]:.3f}, "
          f"HIF={result_normal['y'][IDX['HIF1_alpha'], -1]:.3f}, "
          f"Lactate={result_normal['y'][IDX['Lactate'], -1]:.3f}")
    print(f"   Glycolysis dominance={profile['glycolysis_dominance']:.2f}, "
          f"Energy charge={profile['energy_charge']:.2f}")

    # 2. 缺糖模拟 / Glucose deprivation
    print("\n--- 2. Glucose Deprivation ---")
    result_stress = model.simulate(
        oxygen_level=0.05, glucose_level=0.05,
        perturbation='glucose_deprivation'
    )
    print(f"   ATP={result_stress['y'][IDX['ATP'], -1]:.3f}, "
          f"Lactate={result_stress['y'][IDX['Lactate'], -1]:.3f}")

    # 3. 严重低氧 / Severe hypoxia
    print("\n--- 3. Severe Hypoxia (1% O₂) ---")
    result_hypoxia = model.simulate(
        oxygen_level=0.01, glucose_level=1.0,
        perturbation='hypoxia'
    )
    print(f"   HIF={result_hypoxia['y'][IDX['HIF1_alpha'], -1]:.3f}, "
          f"PFKFB3={result_hypoxia['y'][IDX['PFKFB3'], -1]:.3f}, "
          f"F26BP={result_hypoxia['y'][IDX['F26BP'], -1]:.3f}")

    # 4. 鱼藤酮 (复合体 I 抑制) / Rotenone
    print("\n--- 4. Rotenone (Complex I inhibition) ---")
    result_rotenone = model.simulate(
        oxygen_level=0.05, glucose_level=1.0,
        perturbation='rotenone'
    )
    print(f"   ATP={result_rotenone['y'][IDX['ATP'], -1]:.3f}, "
          f"ROS={result_rotenone['y'][IDX['ROS_mito'], -1]:.3f}")

    # 5. 全应激模拟 / Full stress comparison
    print("\n--- 5. Metabolic Stress Comparison ---")
    stress_results = model.simulate_metabolic_stress()
    comp = stress_results['comparison']
    for cond, vals in comp.items():
        print(f"   {cond:30s} | ATP={vals['ATP']:.2f} | Lac={vals['Lactate']:.2f} | "
              f"ROS={vals['ROS_mito']:.2f} | MMP={vals['MMPotential']:.2f}")

    # 6. 绘图 / Plotting
    print("\n--- 6. Plotting ---")
    fig = model.plot_metabolism(result_normal, save_path='output/normal_metabolism.png')
    plt.close(fig)
    print("   Saved: output/normal_metabolism.png")

    # 代谢景观图 / Metabolic landscape
    fig2 = plot_metabolic_landscape(
        model, metric='ATP',
        save_path='output/ATP_landscape.png'
    )
    plt.close(fig2)
    print("   Saved: output/ATP_landscape.png")

    print("\n✓ All tests completed successfully!")
