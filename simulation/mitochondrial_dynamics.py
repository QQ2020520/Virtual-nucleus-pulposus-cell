"""
髓核细胞线粒体动力学 ODE 模型
=================================
建模线粒体核心机制：融合/分裂平衡、膜电位调控、PINK1/Parkin 线粒体自噬、
NAD+/SIRT3 抗氧化防御、mtDNA 损伤恶性循环、凋亡触发。

Mitochondrial dynamics ODE model for NP cells, covering:
  - Fission/fusion balance (Drp1 / Mfn1/Mfn2 / OPA1)
  - Membrane potential (Δψm)
  - miROS production & scavenging (SIRT3/SOD2 axis)
  - mtDNA damage & vicious cycle
  - PINK1/Parkin mitophagy
  - Cytochrome c release & apoptosis initiation
  - PGC-1α mediated mitochondrial biogenesis

All ODE equations are clipped for numerical stability.
Default parameters produce a healthy baseline:
  Δψm ≈ 0.85, miROS ≈ 0.1, Fragmentation ≈ 0.3

Author: Virtual NP Cell Team
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# 中文字体配置 / Chinese font configuration
# ============================================================
plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 变量索引 / State variable indices
# ============================================================
IDX = {
    'Mito_mass': 0,                # 线粒体总质量/生物发生
    'Mito_fission': 1,             # Drp1 介导的分裂活性
    'Mito_fusion': 2,              # Mfn1/Mfn2/OPA1 融合活性
    'Fragmentation_idx': 3,        # 碎片化指数 = fission/(fusion+ε)
    'Mito_membrane_potential': 4,  # 线粒体膜电位 Δψm
    'miROS': 5,                    # 线粒体 ROS
    'mtDNA_damage': 6,             # 线粒体 DNA 损伤
    'NAD_NADH_ratio': 7,           # NAD+/NADH 比值
    'SIRT3_activity': 8,           # SIRT3 去乙酰化酶活性
    'PINK1_level': 9,              # PINK1 积累
    'Parkin_recruit': 10,          # Parkin 招募
    'Mitophagy_flux': 11,          # 线粒体自噬通量
    'CytC_release': 12,            # 细胞色素 c 释放
    'Mito_biogenesis': 13,         # PGC-1α 介导的生物发生
}
N_VARS = 14                        # 总变量数 / Total state variables

# 变量显示名 / Variable display names for plotting
VAR_NAMES = {
    'Mito_mass': 'Mito Mass',
    'Mito_fission': 'Fission (Drp1)',
    'Mito_fusion': 'Fusion (Mfn/OPA1)',
    'Fragmentation_idx': 'Fragmentation',
    'Mito_membrane_potential': r'deltaPsi_m',
    'miROS': 'miROS',
    'mtDNA_damage': 'mtDNA Damage',
    'NAD_NADH_ratio': r'NAD+/NADH',
    'SIRT3_activity': 'SIRT3 Activity',
    'PINK1_level': 'PINK1',
    'Parkin_recruit': 'Parkin Recruit',
    'Mitophagy_flux': 'Mitophagy Flux',
    'CytC_release': 'CytC Release',
    'Mito_biogenesis': 'PGC-1alpha',
}


class MitochondrialDynamicsModel:
    """
    线粒体动力学 ODE 模型 (14 维状态空间)

    ｜ 变量 ｜ 说明 ｜ 生物学依据 ｜
    ｜------｜------｜-----------｜
    ｜ Mito_mass ｜ 线粒体总质量/生物发生 ｜ PGC-1α→NRF1/TFAM 调控 ｜
    ｜ Mito_fission ｜ Drp1 介导的分裂活性 ｜ Drp1 Ser616 磷酸化激活 ｜
    ｜ Mito_fusion ｜ Mfn1/Mfn2/OPA1 融合活性 ｜ Mfn 外膜 + OPA1 内膜融合 ｜
    ｜ Fragmentation_idx ｜ 碎片化指数 ｜ 退变中分裂↑融合↓ ｜
    ｜ Mito_membrane_potential ｜ 线粒体膜电位 Δψm ｜ ETC 质子梯度 ｜
    ｜ miROS ｜ 线粒体 ROS ｜ 复合体 I/III 泄漏 ｜
    ｜ mtDNA_damage ｜ 线粒体 DNA 损伤 ｜ ROS→8-oxo-dG→突变 ｜
    ｜ NAD_NADH_ratio ｜ NAD+/NADH 比值 ｜ SIRT3 活性依赖 NAD+ ｜
    ｜ SIRT3_activity ｜ SIRT3 去乙酰化酶活性 ｜ 靶向 SOD2/IDH2→ROS↓ ｜
    ｜ PINK1_level ｜ PINK1 积累 ｜ Δψm↓→PINK1 稳定化 ｜
    ｜ Parkin_recruit ｜ Parkin 招募 ｜ PINK1→Parkin→泛素化 ｜
    ｜ Mitophagy_flux ｜ 线粒体自噬通量 ｜ PINK1/Parkin 介导 ｜
    ｜ CytC_release ｜ 细胞色素 c 释放 (凋亡启动) ｜ MOMP ｜
    ｜ Mito_biogenesis ｜ PGC-1α 介导的生物发生 ｜ AMPK→PGC-1α→TFAM ｜

    Parameters
    ----------
    params : dict, optional
        自定义参数，覆盖默认参数。
    """

    def __init__(self, params: Optional[Dict] = None):
        # ============================================================
        # 默认参数 (生物学来源标注) / Default Parameters
        # ============================================================
        self.params = {
            # --- 线粒体质量 / Mitochondrial mass turnover ---
            'k_biogenesis': 0.10,        # PGC-1α 驱动生物发生速率
            'k_mass_deg': 0.05,          # 线粒体基础周转降解
            'k_mito_deg_mitophagy': 0.08, # 自噬介导的线粒体降解系数
            'Mito_mass_max': 2.0,        # 线粒体质量上限 (稳态归一化)

            # --- 线粒体分裂 / Fission (Drp1) ---
            'k_fission_base': 0.032,     # Drp1 基础活性 (健康基线)
            'k_ros_fission': 0.08,       # ROS→Drp1 Ser616 磷酸化激活强度
            'Km_ros_fission': 0.25,      # ROS 激活 Drp1 半饱和常数
            'n_ros_fission': 2.0,        # ROS 激活 Drp1 Hill 系数
            'k_energy_fission': 0.04,    # 低能量→Drp1 激活 (AMPK→Drp1)
            'Km_atp_fission': 0.40,      # 能量感知半饱和常数
            'k_fission_decay': 0.35,     # Drp1 活性衰减速率

            # --- 线粒体融合 / Fusion (Mfn1/Mfn2/OPA1) ---
            'k_atp_fusion': 0.45,        # ATP 依赖的融合维持
            'Km_atp_fusion': 0.40,       # 融合 ATP 感知半饱和常数
            'k_psi_fusion': 0.45,        # Δψm 依赖的融合维持
            'Km_psi_fusion': 0.25,       # 融合 Δψm 感知半饱和常数
            'k_ros_inhibit_fusion': 0.15, # ROS 抑制融合 (Mfn 降解, OPA1 切割)
            'k_fusion_decay': 0.90,      # 融合活性衰减速率

            # --- 碎片化指数 / Fragmentation index ---
            'k_frag_adapt': 0.25,        # 碎片化指数适应速率

            # --- 膜电位 Δψm / Membrane potential ---
            'k_mmp_recovery': 0.35,      # Δψm 恢复速率
            'MMP_healthy': 0.90,         # 健康 Δψm 设定点
            'MMP_min': 0.10,             # 最小 Δψm (防止负值)
            'k_mmp_damage_ros': 0.04,    # miROS 损伤 Δψm 速率 (弱化)
            'k_mmp_damage_frag': 0.015,  # 碎片化损伤 Δψm 速率
            'k_mmp_mtdna': 0.02,         # mtDNA 损伤抑制 Δψm 恢复

            # --- 线粒体 ROS / miROS ---
            'k_ros_basal': 0.035,        # ETC 基础 ROS 泄漏
            'k_ros_dysfunction': 0.08,   # Δψm↓ → 电子漏↑ → ROS↑
            'k_ros_mtdna': 0.04,         # mtDNA 损伤 → ETC 泄漏 → ROS↑
            'k_ros_scavenge': 0.28,      # ROS 基础清除速率 (SOD/GSH)
            'k_sirt3_boost': 0.80,       # SIRT3 增强 ROS 清除 (SOD2 K68 去乙酰化)

            # --- mtDNA 损伤 / mtDNA damage ---
            'k_mtdna_damage': 0.03,      # miROS→mtDNA 氧化损伤 (8-oxo-dG)
            'Km_mtdna_damage': 0.25,     # mtDNA 损伤半饱和常数
            'n_mtdna': 2.0,              # mtDNA 损伤协同系数
            'k_mtdna_repair': 0.06,      # mtDNA 基础修复速率

            # --- NAD+/NADH 比值 ---
            'k_nad_reg': 0.15,           # OXPHOS 再生 NAD+
            'k_nad_cons': 0.18,          # NADH 消耗 (糖酵解/应激)
            'k_nad_sirt3': 0.05,         # SIRT3→IDH2→NADPH→NAD+ 再生
            'NAD_ratio_min': 0.05,       # NAD+/NADH 最小值
            'NAD_ratio_max': 2.0,        # NAD+/NADH 最大值

            # --- SIRT3 活性 ---
            'k_sirt3_act': 0.25,         # SIRT3 激活最大速率
            'Km_sirt3': 0.5,             # SIRT3 激活半饱和 (NAD+/NADH)
            'n_sirt3': 2.0,              # SIRT3 激活 Hill 系数
            'k_sirt3_decay': 0.25,       # SIRT3 衰减速率

            # --- PINK1/Parkin ---
            'k_pink1_prod': 0.08,        # PINK1 最大积累速率
            'Km_pink1_psi': 0.12,        # Δψm 感知 PINK1 切割半饱和
            'n_pink1_psi': 4.0,          # PINK1 对 Δψm 的陡峭阈值响应
            'k_pink1_deg': 0.15,         # PINK1 降解 (健康时 PARL 切割)
            'k_parkin_act': 0.12,        # PINK1→Parkin 招募速率
            'Km_parkin': 0.12,           # Parkin 招募半饱和常数
            'n_parkin': 2.0,             # Parkin 招募 Hill 系数
            'k_parkin_decay': 0.10,      # Parkin 活性衰减

            # --- 线粒体自噬 / Mitophagy ---
            'k_mito_parkin': 0.15,       # PINK1/Parkin→自噬速率
            'k_mito_decay': 0.08,        # 自噬通量衰减

            # --- 凋亡 / Apoptosis ---
            'k_cytC_trigger': 0.06,      # CytC 释放触发速率
            'Km_cytC_psi': 0.18,         # Δψm 阈值半饱和 (低 Δψm → 触发)
            'Km_cytC_frag': 0.60,        # 碎片化阈值 (高碎片化 → 触发)
            'n_cytC_psi': 4.0,           # Δψm 触发陡峭系数
            'n_cytC_frag': 3.0,          # 碎片化触发陡峭系数
            'k_cytC_clear': 0.04,        # CytC 清除/凋亡体消耗
            'k_fusion_protect': 0.5,     # 融合保护系数 (Mfn 抗凋亡)

            # --- 线粒体生物发生 / Biogenesis (PGC-1α) ---
            'k_pgc1a_base': 0.03,        # PGC-1α 基础表达
            'k_ampk_pgc1a': 0.10,        # AMPK (低能量)→PGC-1α 激活
            'KM_ampk_pgc1a': 0.30,       # AMPK 激活半饱和
            'k_nad_pgc1a': 0.06,         # NAD+ (SIRT1)→PGC-1α 激活
            'Km_nad_pgc1a': 0.4,         # NAD+ 激活半饱和
            'k_ros_inhibit_pgc1a': 0.15, # ROS 抑制 PGC-1α
            'k_pgc1a_decay': 0.18,       # PGC-1α 衰减速率
        }

        # 覆盖自定义参数
        if params is not None:
            self.params.update(params)

        # 状态变量名称列表
        self.state_names = list(IDX.keys())

    # ============================================================
    # ATP 代理估算 (由线粒体状态推算)
    # ============================================================

    def _estimate_atp_proxy(self, y: np.ndarray) -> float:
        """
        从线粒体状态估算细胞内 ATP 水平 (归一化 0-1)

        使用 Mito_mass, Δψm, mtDNA_damage, Fragmentation_idx 综合估算。
        健康线粒体: 高质量 × 高膜电位 × 低损伤 × 低碎片化

        Returns
        -------
        float : 估算的 ATP 归一化水平
        """
        mass = y[IDX['Mito_mass']]
        psi = y[IDX['Mito_membrane_potential']]
        mtdna = y[IDX['mtDNA_damage']]
        frag = y[IDX['Fragmentation_idx']]

        # 有效线粒体产能 = 质量 × 膜电位 × (1 - 损伤) × 碎片化效率惩罚
        efficiency = 1.0 / (1.0 + 2.0 * frag)  # 碎片化降低产能效率
        atp_proxy = mass * psi * (1.0 - 0.6 * mtdna) * efficiency
        return np.clip(atp_proxy, 0.01, 1.2)

    # ============================================================
    # ODE 系统 / ODE System
    # ============================================================

    def ode_system(self, t: float, y: np.ndarray,
                   perturbation: Optional[str] = None) -> List[float]:
        """
        ODE 右侧函数 — 14 维线粒体动力学

        Parameters
        ----------
        t : float
            当前时间 / Current time.
        y : np.ndarray
            状态向量 (14 变量) / State vector.
        perturbation : str or None
            扰动模式:
            - None: 无扰动 (健康基线)
            - 'fission_stress': Drp1 过表达 (退变)
            - 'fusion_enhance': Mfn2 过表达 (保护)
            - 'nad_depletion': NAD+ 耗竭 (衰老)
            - 'mito_poison': CCCP/Rotenone (线粒体损伤)

        Returns
        -------
        List[float]
            各变量的时间导数 (14 维)
        """
        p = self.params

        # ---- 解包状态变量 ----
        mass, fission, fusion, frag, psi = y[0:5]
        miros, mtdna, nad_ratio, sirt3 = y[5:9]
        pink1, parkin, mitophagy, cytc, biogenesis = y[9:14]

        # ---- 数值保护 ----
        psi = max(p['MMP_min'], min(1.0, psi))
        miros = max(0.0, miros)
        mtdna = max(0.0, min(1.0, mtdna))
        nad_ratio = max(p['NAD_ratio_min'], min(p['NAD_ratio_max'], nad_ratio))
        frag = max(0.0, frag)
        mass = max(0.01, mass)
        mitophagy = max(0.0, min(1.0, mitophagy))
        cytc = max(0.0, min(1.0, cytc))
        biogenesis = max(0.0, biogenesis)

        # ---- 估算 ATP 代理 ----
        atp_proxy = self._estimate_atp_proxy(y)

        # ---- 扰动调整 ----
        _fission_mod = 1.0       # 分裂调制器
        _fusion_mod = 1.0        # 融合调制器
        _nad_mod = 1.0           # NAD 调制器
        _mito_poison = 0.0       # 线粒体毒物强度

        if perturbation == 'fission_stress':
            _fission_mod = 3.0   # Drp1 过表达 → 分裂↑3倍
        elif perturbation == 'fusion_enhance':
            _fusion_mod = 2.5    # Mfn2 过表达 → 融合↑2.5倍
        elif perturbation == 'nad_depletion':
            _nad_mod = 0.2       # NAD+ 耗竭 → NAD+/NADH↓
        elif perturbation == 'mito_poison':
            _mito_poison = 0.6   # CCCP 去极化 + 复合体抑制

        # ============================================================
        # 1. 线粒体质量 / Mito_mass
        #    生物发生 (PGC-1α 驱动) - 基础周转 - 自噬降解
        # ============================================================
        mass_growth = p['k_biogenesis'] * biogenesis * (1.0 - mass / p['Mito_mass_max'])
        mass_turnover = p['k_mass_deg'] * mass
        mass_mitophagy = p['k_mito_deg_mitophagy'] * mitophagy * mass
        dMito_mass = mass_growth - mass_turnover - mass_mitophagy

        # ============================================================
        # 2. 线粒体分裂 / Mito_fission (Drp1 活性)
        #    激活: ROS (Drp1 Ser616 磷酸化) + 低能量 (AMPK→Drp1)
        # ============================================================
        # ROS 激活 Drp1
        ros_fission_act = p['k_ros_fission'] * (
            miros ** p['n_ros_fission'] /
            (p['Km_ros_fission'] ** p['n_ros_fission'] + miros ** p['n_ros_fission'])
        )
        # 低能量激活 Drp1
        energy_fission_act = p['k_energy_fission'] * (
            1.0 - atp_proxy / (atp_proxy + p['Km_atp_fission'])
        )
        dMito_fission = (
            (p['k_fission_base'] + ros_fission_act + energy_fission_act) * _fission_mod
            - p['k_fission_decay'] * fission
        )

        # ============================================================
        # 3. 线粒体融合 / Mito_fusion (Mfn1/Mfn2/OPA1)
        #    维持: ATP↑, Δψm↑; 抑制: ROS
        # ============================================================
        atp_fusion_act = p['k_atp_fusion'] * atp_proxy / (atp_proxy + p['Km_atp_fusion'])
        psi_fusion_act = p['k_psi_fusion'] * psi / (psi + p['Km_psi_fusion'])
        ros_fusion_inh = p['k_ros_inhibit_fusion'] * miros * fusion
        dMito_fusion = (
            (atp_fusion_act + psi_fusion_act) * _fusion_mod
            - ros_fusion_inh
            - p['k_fusion_decay'] * fusion
        )

        # ============================================================
        # 4. 碎片化指数 / Fragmentation_idx
        #    趋近于 fission/(fusion+ε) 的慢动力学
        # ============================================================
        target_frag = fission / (fusion + 1e-8)
        dFragmentation_idx = p['k_frag_adapt'] * (target_frag - frag)

        # ============================================================
        # 5. 膜电位 Δψm / Mito_membrane_potential
        #    恢复 → 健康设定点 | 损伤 ← ROS + 碎片化 + mtDNA 损伤
        # ============================================================
        # ETC 效率受 mtDNA 损伤影响
        etc_efficiency = max(0.05, 1.0 - mtdna)
        # 恢复 (EMF-like)
        mmp_recovery = (
            p['k_mmp_recovery'] * (p['MMP_healthy'] - psi) * etc_efficiency
        )
        # 毒物导致的额外去极化
        poison_depolarization = _mito_poison * 0.3 * psi
        # miROS 损伤 Δψm (ROS → 心磷脂过氧化 → 复合体损伤)
        mmp_damage_ros = (
            p['k_mmp_damage_ros'] * miros * max(0.0, psi - p['MMP_min'])
        )
        # 碎片化损伤 Δψm (嵴重构受损)
        mmp_damage_frag = (
            p['k_mmp_damage_frag'] * frag * max(0.0, psi - p['MMP_min'])
        )
        dMito_membrane_potential = (
            mmp_recovery - mmp_damage_ros - mmp_damage_frag - poison_depolarization
        )

        # ============================================================
        # 6. 线粒体 ROS / miROS
        #    产生: 基础泄漏 + Δψm↓ 电子漏 + mtDNA 损伤泄漏
        #    清除: SIRT3→SOD2→O2-· 清除
        # ============================================================
        # 基础 ROS 泄漏
        ros_basal = p['k_ros_basal']
        # Δψm 下降 → 电子传递链反向/泄漏 ↑
        ros_dysfunction = p['k_ros_dysfunction'] * max(0.0, p['MMP_healthy'] - psi)
        # mtDNA 损伤 → ETC 亚基缺陷 → 更多泄漏 (恶性正反馈)
        ros_mtdna_leak = p['k_ros_mtdna'] * mtdna
        # 毒物增强 ROS (Rotenone 抑制复合体 I → 电子泄漏↑)
        ros_poison = _mito_poison * 0.2
        # SIRT3→SOD2 去乙酰化 → ROS 清除↑
        sirt3_boost = 1.0 + p['k_sirt3_boost'] * sirt3
        ros_scavenge = p['k_ros_scavenge'] * sirt3_boost * miros
        dmiROS = (
            ros_basal + ros_dysfunction + ros_mtdna_leak + ros_poison
            - ros_scavenge
        )

        # ============================================================
        # 7. mtDNA 损伤 / mtDNA_damage
        #    损伤: miROS→8-oxo-dG→突变 (协同 Hill)
        #    修复: 基础修复机制
        # ============================================================
        mtdna_damage_rate = (
            p['k_mtdna_damage'] *
            (miros ** p['n_mtdna']) /
            (p['Km_mtdna_damage'] ** p['n_mtdna'] + miros ** p['n_mtdna'])
        )
        mtdna_damage = mtdna_damage_rate * (1.0 - mtdna)  # 有上限
        mtdna_repair = p['k_mtdna_repair'] * mtdna
        dmtDNA_damage = mtdna_damage - mtdna_repair

        # ============================================================
        # 8. NAD+/NADH 比值 / NAD_NADH_ratio
        #    再生: OXPHOS (健康 ETC) + SIRT3→IDH2→NADPH
        #    消耗: 糖酵解 (低氧/应激)
        # ============================================================
        # OXPHOS 再生 NAD+ (需要健康 Δψm 和线粒体质量)
        nad_regeneration = p['k_nad_reg'] * psi * mass
        # SIRT3 通过 IDH2 去乙酰化促进 NADPH → NAD+ 再生
        nad_sirt3 = p['k_nad_sirt3'] * sirt3
        # NADH 消耗/积累 (应激时 NADH↑ → NAD+/NADH↓)
        nad_consumption = p['k_nad_cons'] * (1.0 + miros) * nad_ratio
        # NAD+ 耗竭扰动
        dNAD_NADH_ratio = (
            (nad_regeneration + nad_sirt3) * _nad_mod
            - nad_consumption
        )

        # ============================================================
        # 9. SIRT3 活性 / SIRT3_activity
        #    NAD+ 依赖 (Hill 函数) → SOD2 K68 去乙酰化
        # ============================================================
        sirt3_activation = p['k_sirt3_act'] * (
            (nad_ratio ** p['n_sirt3']) /
            (p['Km_sirt3'] ** p['n_sirt3'] + nad_ratio ** p['n_sirt3'])
        )
        dSIRT3_activity = sirt3_activation - p['k_sirt3_decay'] * sirt3

        # ============================================================
        # 10. PINK1 积累 / PINK1_level
        #     健康 Δψm → PINK1 导入线粒体→PARL 切割→降解
        #     受损 Δψm → PINK1 稳定在线粒体外膜
        # ============================================================
        # Δψm 感知: 低 Δψm → PINK1 稳定 (陡峭阈值)
        pink1_stabilization = 1.0 / (
            1.0 + (psi / p['Km_pink1_psi']) ** p['n_pink1_psi']
        )
        pink1_accumulation = p['k_pink1_prod'] * pink1_stabilization
        pink1_deg = p['k_pink1_deg'] * pink1
        dPINK1_level = pink1_accumulation - pink1_deg

        # ============================================================
        # 11. Parkin 招募 / Parkin_recruit
        #     PINK1→Parkin 磷酸化/泛素化激活 → 线粒体招募
        # ============================================================
        parkin_activation = p['k_parkin_act'] * (
            (pink1 ** p['n_parkin']) /
            (p['Km_parkin'] ** p['n_parkin'] + pink1 ** p['n_parkin'])
        )
        dParkin_recruit = parkin_activation - p['k_parkin_decay'] * parkin

        # ============================================================
        # 12. 线粒体自噬通量 / Mitophagy_flux
        #     PINK1/Parkin → 泛素化 OMM → p62 → 自噬受体
        #     受底物限制 (不能超过可用线粒体)
        # ============================================================
        mito_drive = pink1 + parkin
        mitophagy_activation = (
            p['k_mito_parkin'] * mito_drive * (1.0 - mitophagy)
        )
        dMitophagy_flux = mitophagy_activation - p['k_mito_decay'] * mitophagy

        # ============================================================
        # 13. CytC 释放 (凋亡启动) / CytC_release
        #     阈值: 碎片化↑ + Δψm↓ + 融合保护↓
        # ============================================================
        # Δψm 低 → 促凋亡 (陡峭 Hill)
        psi_trigger = (1.0 - psi) ** p['n_cytC_psi'] / (
            p['Km_cytC_psi'] ** p['n_cytC_psi'] + (1.0 - psi) ** p['n_cytC_psi']
        )
        # 碎片化高 → 促凋亡
        frag_trigger = (frag ** p['n_cytC_frag']) / (
            p['Km_cytC_frag'] ** p['n_cytC_frag'] + frag ** p['n_cytC_frag']
        )
        # 融合保护 (Mfn1/Mfn2 抗凋亡)
        fusion_protection = 1.0 / (1.0 + p['k_fusion_protect'] * fusion)
        cytc_trigger = (
            p['k_cytC_trigger'] * psi_trigger * frag_trigger * fusion_protection
        )
        dCytC_release = cytc_trigger * (1.0 - cytc) - p['k_cytC_clear'] * cytc

        # ============================================================
        # 14. 线粒体生物发生 / Mito_biogenesis (PGC-1α)
        #     激活: AMPK (低ATP), NAD+/SIRT1
        #     抑制: ROS
        # ============================================================
        # AMPK 激活 (低能量)
        ampk_activation = p['k_ampk_pgc1a'] * (
            1.0 - atp_proxy / (atp_proxy + p['KM_ampk_pgc1a'])
        )
        # NAD+/SIRT1 激活
        nad_activation = p['k_nad_pgc1a'] * (
            nad_ratio / (nad_ratio + p['Km_nad_pgc1a'])
        )
        # ROS 抑制
        ros_inhibition = p['k_ros_inhibit_pgc1a'] * miros * biogenesis
        dMito_biogenesis = (
            p['k_pgc1a_base'] + ampk_activation + nad_activation
            - ros_inhibition
            - p['k_pgc1a_decay'] * biogenesis
        )

        # ---- 组装导数 + 数值裁剪 ----
        dy = np.array([
            np.clip(dMito_mass, -2.0, 2.0),
            np.clip(dMito_fission, -2.0, 2.0),
            np.clip(dMito_fusion, -2.0, 2.0),
            np.clip(dFragmentation_idx, -2.0, 2.0),
            np.clip(dMito_membrane_potential, -2.0, 2.0),
            np.clip(dmiROS, -2.0, 2.0),
            np.clip(dmtDNA_damage, -2.0, 2.0),
            np.clip(dNAD_NADH_ratio, -2.0, 2.0),
            np.clip(dSIRT3_activity, -2.0, 2.0),
            np.clip(dPINK1_level, -2.0, 2.0),
            np.clip(dParkin_recruit, -2.0, 2.0),
            np.clip(dMitophagy_flux, -2.0, 2.0),
            np.clip(dCytC_release, -2.0, 2.0),
            np.clip(dMito_biogenesis, -2.0, 2.0),
        ], dtype=float)

        return dy.tolist()

    # ============================================================
    # 初始条件 / Initial Conditions
    # ============================================================

    def _get_initial_conditions(self, perturbation: Optional[str] = None,
                                healthy_level: float = 1.0) -> np.ndarray:
        """
        获取生理初始条件

        基于生物学知识设定健康 NP 细胞的线粒体基线状态。
        扰动场景使用不同初始值以加速收敛。

        Parameters
        ----------
        perturbation : str or None
            扰动模式 (影响初始条件)
        healthy_level : float
            健康程度缩放 (0.5=退变, 1.0=健康)

        Returns
        -------
        np.ndarray : 初始状态向量 (14 维)
        """
        h = healthy_level

        if perturbation == 'mito_poison':
            # CCCP/Rotenone: 从健康状态快速去极化
            y0 = np.array([
                1.0,      # Mito_mass
                0.5,      # Mito_fission (应激升高)
                0.8,      # Mito_fusion
                0.4,      # Fragmentation_idx
                0.4,      # Δψm (初始即受损)
                0.4,      # miROS (初始升高)
                0.1,      # mtDNA_damage
                0.5,      # NAD_NADH_ratio
                0.4,      # SIRT3_activity
                0.3,      # PINK1_level (受损→积累)
                0.15,     # Parkin_recruit
                0.2,      # Mitophagy_flux (应激激活)
                0.05,     # CytC_release
                0.3,      # Mito_biogenesis
            ])
        elif perturbation == 'fission_stress':
            # Drp1 过表达: 高分裂基线
            y0 = np.array([
                1.0,      # Mito_mass
                0.8,      # Mito_fission (高)
                0.7,      # Mito_fusion (稍低)
                0.6,      # Fragmentation_idx (高)
                0.7,      # Δψm (稍降)
                0.25,     # miROS (应激升高)
                0.08,     # mtDNA_damage
                0.55,     # NAD_NADH_ratio
                0.4,      # SIRT3_activity
                0.15,     # PINK1_level
                0.08,     # Parkin_recruit
                0.1,      # Mitophagy_flux
                0.01,     # CytC_release
                0.35,     # Mito_biogenesis
            ])
        elif perturbation == 'fusion_enhance':
            # Mfn2 过表达: 高融合基线 (保护状态)
            y0 = np.array([
                1.1,      # Mito_mass (稍高)
                0.25,     # Mito_fission (正常)
                1.3,      # Mito_fusion (高)
                0.15,     # Fragmentation_idx (低)
                0.9,      # Δψm (健康)
                0.08,     # miROS (低)
                0.03,     # mtDNA_damage (低)
                0.7,      # NAD_NADH_ratio
                0.55,     # SIRT3_activity
                0.03,     # PINK1_level (低—健康)
                0.01,     # Parkin_recruit (低)
                0.05,     # Mitophagy_flux (正常基线)
                0.001,    # CytC_release (极低)
                0.5,      # Mito_biogenesis
            ])
        elif perturbation == 'nad_depletion':
            # NAD+ 耗竭: NAD+/NADH↓ → SIRT3↓ → ROS↑
            y0 = np.array([
                1.0,      # Mito_mass
                0.4,      # Mito_fission (稍高)
                0.8,      # Mito_fusion (稍低)
                0.35,     # Fragmentation_idx
                0.7,      # Δψm (稍降)
                0.3,      # miROS (升高—清除不足)
                0.1,      # mtDNA_damage
                0.2,      # NAD_NADH_ratio (严重降低)
                0.2,      # SIRT3_activity (低)
                0.2,      # PINK1_level
                0.1,      # Parkin_recruit
                0.15,     # Mitophagy_flux
                0.02,     # CytC_release
                0.2,      # Mito_biogenesis (低—NAD/SIRT1↓)
            ])
        else:
            # 健康 NP 细胞: 正常线粒体基线
            y0 = np.array([
                1.0 * h,      # Mito_mass (≈1.0)
                0.30 * h,     # Mito_fission (正常分裂)
                1.0 * h,      # Mito_fusion (正常融合)
                0.25 * h,     # Fragmentation_idx
                0.85 * h,     # Δψm (健康≈0.85)
                0.08 * h,     # miROS (基础低水平)
                0.04 * h,     # mtDNA_damage (低)
                0.65 * h,     # NAD_NADH_ratio (~0.65)
                0.50 * h,     # SIRT3_activity (中等)
                0.04 * h,     # PINK1_level (低—健康时被切割)
                0.02 * h,     # Parkin_recruit (低)
                0.08 * h,     # Mitophagy_flux (基础质量控制)
                0.005 * h,    # CytC_release (近零)
                0.45 * h,     # Mito_biogenesis (中等)
            ])

        return np.clip(y0, 0.0, None)  # 确保非负

    # ============================================================
    # 主仿真接口 / Main Simulation Interface
    # ============================================================

    def simulate(self, t_span: Tuple[float, float] = (0.0, 200.0),
                 n_points: int = 500,
                 perturbation: Optional[str] = None,
                 method: str = 'BDF',
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 initial_state: Optional[np.ndarray] = None) -> Dict:
        """
        运行线粒体动力学 ODE 仿真

        Parameters
        ----------
        t_span : tuple, default=(0, 200)
            模拟时间范围
        n_points : int, default=500
            输出时间点数
        perturbation : str or None
            扰动模式 (详见 ode_system)
        method : str, default='BDF'
            solve_ivp 积分方法 (刚性系统用 BDF)
        rtol, atol : float
            相对/绝对容差
        initial_state : np.ndarray or None
            初始状态 (None=使用默认)

        Returns
        -------
        dict
            't': 时间点
            'y': 状态矩阵 (14 × n_points)
            'params': 使用的参数
            'perturbation': 扰动类型
            'success': 积分是否成功
            'message': 积分器消息
        """
        # 初始条件
        if initial_state is not None:
            y0 = np.array(initial_state, dtype=float)
        else:
            y0 = self._get_initial_conditions(perturbation)

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # 求解 ODE
        sol = solve_ivp(
            self.ode_system,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            args=(perturbation,),
        )

        # 状态裁剪确保物理合理性
        y_safe = sol.y.copy()
        y_safe[IDX['Mito_mass'], :] = np.clip(y_safe[IDX['Mito_mass'], :], 0.01, 2.0)
        y_safe[IDX['Mito_membrane_potential'], :] = np.clip(
            y_safe[IDX['Mito_membrane_potential'], :], 0.0, 1.0
        )
        y_safe[IDX['miROS'], :] = np.clip(y_safe[IDX['miROS'], :], 0.0, 2.0)
        y_safe[IDX['mtDNA_damage'], :] = np.clip(y_safe[IDX['mtDNA_damage'], :], 0.0, 1.0)
        y_safe[IDX['CytC_release'], :] = np.clip(y_safe[IDX['CytC_release'], :], 0.0, 1.0)
        y_safe[IDX['Mitophagy_flux'], :] = np.clip(y_safe[IDX['Mitophagy_flux'], :], 0.0, 1.0)

        return {
            't': sol.t,
            'y': y_safe,
            'params': self.params.copy(),
            'perturbation': perturbation,
            'success': sol.success,
            'message': sol.message,
        }

    # ============================================================
    # 状态解读 / State Interpretation
    # ============================================================

    def get_mito_state(self, y: np.ndarray) -> Dict:
        """
        从状态向量提取线粒体功能状态摘要

        Parameters
        ----------
        y : np.ndarray
            状态向量 (14 维)

        Returns
        -------
        dict
            功能分类摘要
            'fragmentation': 碎片化相关指标
            'energetics': 能量代谢状态
            'quality_control': 质量控制 (PINK1/Parkin/自噬)
            'apoptosis_risk': 凋亡风险
        """
        state = {
            'fragmentation': {
                'fission': float(y[IDX['Mito_fission']]),
                'fusion': float(y[IDX['Mito_fusion']]),
                'fragmentation_index': float(y[IDX['Fragmentation_idx']]),
                'fission_fusion_ratio': float(
                    y[IDX['Mito_fission']] / max(1e-8, y[IDX['Mito_fusion']])
                ),
            },
            'energetics': {
                'membrane_potential': float(y[IDX['Mito_membrane_potential']]),
                'mito_mass': float(y[IDX['Mito_mass']]),
                'nad_nadh_ratio': float(y[IDX['NAD_NADH_ratio']]),
                'estimated_atp': float(self._estimate_atp_proxy(y)),
            },
            'quality_control': {
                'miROS': float(y[IDX['miROS']]),
                'mtDNA_damage': float(y[IDX['mtDNA_damage']]),
                'sirt3_activity': float(y[IDX['SIRT3_activity']]),
                'pink1_level': float(y[IDX['PINK1_level']]),
                'parkin_recruitment': float(y[IDX['Parkin_recruit']]),
                'mitophagy_flux': float(y[IDX['Mitophagy_flux']]),
                'biogenesis': float(y[IDX['Mito_biogenesis']]),
            },
            'apoptosis_risk': {
                'cytochrome_c_release': float(y[IDX['CytC_release']]),
                'apoptosis_threshold_exceeded': bool(
                    y[IDX['Fragmentation_idx']] > 0.7
                    and y[IDX['Mito_membrane_potential']] < 0.3
                ),
                'apoptosis_risk_score': float(
                    y[IDX['CytC_release']]
                    * (1.0 + y[IDX['Fragmentation_idx']])
                    / (1.0 + y[IDX['Mito_membrane_potential']] + 1e-8)
                ),
            },
        }
        return state

    def get_steady_state_metrics(self, y: np.ndarray) -> Dict:
        """
        从状态向量提取稳态关键指标

        Parameters
        ----------
        y : np.ndarray
            终态状态向量

        Returns
        -------
        dict
            稳态指标评分
        """
        atp_proxy = self._estimate_atp_proxy(y)
        psi = y[IDX['Mito_membrane_potential']]
        miros = y[IDX['miROS']]
        frag = y[IDX['Fragmentation_idx']]
        mtdna = y[IDX['mtDNA_damage']]
        cytc = y[IDX['CytC_release']]

        return {
            'mito_health_score': float(
                psi * (1.0 - miros) * (1.0 - 0.5 * frag) * (1.0 - mtdna)
            ),
            'energy_status': float(atp_proxy),
            'oxidative_stress': float(miros),
            'structural_integrity': float(1.0 / (1.0 + frag)),
            'genomic_integrity': float(1.0 - mtdna),
            'apoptosis_risk': float(cytc),
            'fission_fusion_balance': float(
                y[IDX['Mito_fission']] / max(1e-8, y[IDX['Mito_fusion']])
            ),
        }

    # ============================================================
    # 扰动仿真 / Perturbation Simulations
    # ============================================================

    def simulate_perturbation(self, pert_type: str) -> Dict:
        """
        运行特定扰动仿真并返回完整结果

        支持的扰动:
        1) 'fission_stress'  — Drp1 过表达模拟退变
        2) 'fusion_enhance'  — Mfn2 过表达模拟保护
        3) 'nad_depletion'   — NAD+ 耗竭模拟衰老
        4) 'mito_poison'     — CCCP/Rotenone 模拟线粒体损伤

        Parameters
        ----------
        pert_type : str
            扰动类型

        Returns
        -------
        dict : simulate() 返回的结果字典
        """
        valid_perturbations = [
            'fission_stress', 'fusion_enhance', 'nad_depletion', 'mito_poison'
        ]
        if pert_type not in valid_perturbations:
            raise ValueError(
                f"Unknown perturbation '{pert_type}'. "
                f"Valid: {valid_perturbations}"
            )

        # 使用更长仿真时间以确保达到稳态
        result = self.simulate(
            t_span=(0.0, 500.0),
            n_points=500,
            perturbation=pert_type,
        )
        return result

    # ============================================================
    # 绘图 / Plotting
    # ============================================================

    def plot_trajectories(self, result: Dict,
                          output_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        4×4 子图展示所有变量的时间轨迹

        Parameters
        ----------
        result : dict
            simulate() 返回的结果
        output_path : str or None
            保存路径 (None=不保存)
        figsize : tuple, default=(16, 12)
            图形尺寸

        Returns
        -------
        plt.Figure
        """
        t = result['t']
        y = result['y']

        n_rows, n_cols = 4, 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9898',
        ]

        var_order = list(IDX.keys())

        pert_label = result.get('perturbation', 'none')

        for i, var_name in enumerate(var_order):
            ax = axes[i]
            idx = IDX[var_name]
            ax.plot(t, y[idx, :], color=colors[i], linewidth=1.5)
            ax.set_title(VAR_NAMES.get(var_name, var_name), fontsize=10,
                         fontweight='bold')
            ax.set_xlabel('Time', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # 标注终值
            final_val = y[idx, -1]
            ax.axhline(y=final_val, color=colors[i], linestyle='--',
                       linewidth=0.8, alpha=0.5)
            ax.annotate(f'{final_val:.3f}',
                        xy=(t[-1], final_val),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=7, color=colors[i])

        # 隐藏多余子图 (14 vars → 16 slots, 隐藏最后2个)
        for i in range(N_VARS, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f'Mitochondrial Dynamics Trajectories | Perturbation: {pert_label}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {output_path}")

        return fig

    def plot_fission_fusion(self, result: Dict,
                            output_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制分裂/融合/碎片化 + Δψm 双轴图

        Parameters
        ----------
        result : dict
            simulate() 返回的结果
        output_path : str or None
            保存路径
        figsize : tuple, default=(10, 6)
            图形尺寸

        Returns
        -------
        plt.Figure
        """
        t = result['t']
        y = result['y']
        pert_label = result.get('perturbation', 'none')

        fig, ax1 = plt.subplots(figsize=figsize)

        # 左轴: 分裂, 融合, 碎片化
        ax1.plot(t, y[IDX['Mito_fission'], :], '#d62728', linewidth=2.0,
                 label='Fission (Drp1)')
        ax1.plot(t, y[IDX['Mito_fusion'], :], '#2ca02c', linewidth=2.0,
                 label='Fusion (Mfn/OPA1)')
        ax1.plot(t, y[IDX['Fragmentation_idx'], :], '#9467bd', linewidth=2.5,
                 linestyle='--', label='Fragmentation Index')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Activity / Index', fontsize=12, color='#333')
        ax1.tick_params(axis='y', labelsize=10)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 右轴: Δψm
        ax2 = ax1.twinx()
        ax2.plot(t, y[IDX['Mito_membrane_potential'], :], '#1f77b4',
                 linewidth=2.5, linestyle=':', label=r'$\Delta\Psi_m$')
        ax2.set_ylabel(r'Membrane Potential $\Delta\Psi_m$', fontsize=12,
                       color='#1f77b4')
        ax2.tick_params(axis='y', labelsize=10, colors='#1f77b4')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_ylim(-0.05, 1.05)

        # 标注健康/退变区域
        ax1.axhline(y=0.3, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
        ax1.annotate('Healthy frag. ≈ 0.3', xy=(0, 0.3),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color='gray', alpha=0.6)

        fig.suptitle(
            f'Fission–Fusion Balance | Perturbation: {pert_label}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {output_path}")

        return fig

    def plot_mito_quality_control(self, result: Dict,
                                  output_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制 PINK1/Parkin/自噬 + NAD+/SIRT3 双轴图

        Parameters
        ----------
        result : dict
            simulate() 返回的结果
        output_path : str or None
            保存路径
        figsize : tuple, default=(10, 6)
            图形尺寸

        Returns
        -------
        plt.Figure
        """
        t = result['t']
        y = result['y']
        pert_label = result.get('perturbation', 'none')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # ---- 上子图: PINK1/Parkin/自噬 ----
        ax1.plot(t, y[IDX['PINK1_level'], :], '#e377c2', linewidth=2.0,
                 label='PINK1')
        ax1.plot(t, y[IDX['Parkin_recruit'], :], '#ff7f0e', linewidth=2.0,
                 label='Parkin Recruitment')
        ax1.plot(t, y[IDX['Mitophagy_flux'], :], '#2ca02c', linewidth=2.5,
                 linestyle='--', label='Mitophagy Flux')
        ax1.set_ylabel('Level / Flux', fontsize=11)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('PINK1/Parkin Mitophagy Pathway', fontsize=12,
                      fontweight='bold')

        # ---- 下子图: NAD+/SIRT3 ----
        ax2.plot(t, y[IDX['NAD_NADH_ratio'], :], '#17becf', linewidth=2.0,
                 label=r'NAD$^+$/NADH')
        ax2.plot(t, y[IDX['SIRT3_activity'], :], '#bcbd22', linewidth=2.0,
                 label='SIRT3 Activity')
        ax2.plot(t, y[IDX['miROS'], :], '#d62728', linewidth=2.0,
                 linestyle='--', label='miROS')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Level / Ratio', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('NAD$^+$/SIRT3 Antioxidant Defense', fontsize=12,
                      fontweight='bold')

        fig.suptitle(
            f'Mitochondrial Quality Control | Perturbation: {pert_label}',
            fontsize=14, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {output_path}")

        return fig

    def plot_perturbation_comparison(self,
                                     results: Dict[str, Dict],
                                     output_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        比较不同扰动下的关键指标

        Parameters
        ----------
        results : dict
            {pert_name: simulate() result} 字典
        output_path : str or None
            保存路径
        figsize : tuple, default=(14, 10)
            图形尺寸

        Returns
        -------
        plt.Figure
        """
        metrics = ['Mito_membrane_potential', 'miROS', 'Fragmentation_idx',
                   'CytC_release', 'NAD_NADH_ratio', 'SIRT3_activity',
                   'PINK1_level', 'Mitophagy_flux']

        pert_names = list(results.keys())
        n_pert = len(pert_names)
        n_plots = len(metrics)
        n_cols = 4
        n_rows = int(np.ceil(n_plots / n_cols))

        colors = plt.cm.Set1(np.linspace(0, 1, max(n_pert, 3)))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            for j, (name, res) in enumerate(results.items()):
                t = res['t']
                y = res['y']
                idx = IDX[metric]
                ax.plot(t, y[idx, :], color=colors[j], linewidth=1.5,
                        label=name)
            ax.set_title(VAR_NAMES.get(metric, metric), fontsize=10,
                         fontweight='bold')
            ax.set_xlabel('Time', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7, loc='best')

        # 隐藏多余子图
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            'Mitochondrial Dynamics — Perturbation Comparison',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {output_path}")

        return fig


# ============================================================
# 模块级辅助函数 / Module-level helper functions
# ============================================================

def run_all_perturbations(model: Optional[MitochondrialDynamicsModel] = None
                          ) -> Dict[str, Dict]:
    """
    运行所有 4 种扰动仿真并返回对比结果

    Parameters
    ----------
    model : MitochondrialDynamicsModel or None
        模型实例 (None = 新建默认)

    Returns
    -------
    dict
        'results': {pert_name: simulate result}
        'summary': 各扰动终态关键指标对比表
    """
    if model is None:
        model = MitochondrialDynamicsModel()

    pert_types = ['fission_stress', 'fusion_enhance',
                  'nad_depletion', 'mito_poison']

    results = {}
    for pert in pert_types:
        print(f"  Running '{pert}'...")
        results[pert] = model.simulate_perturbation(pert)

    # 无扰动基线
    print("  Running 'baseline'...")
    results['baseline'] = model.simulate()

    # 汇总表
    summary = {}
    for name, res in results.items():
        y_final = res['y'][:, -1]
        summary[name] = {
            'Δψm': y_final[IDX['Mito_membrane_potential']],
            'miROS': y_final[IDX['miROS']],
            'Fragmentation': y_final[IDX['Fragmentation_idx']],
            'mtDNA_damage': y_final[IDX['mtDNA_damage']],
            'NAD+/NADH': y_final[IDX['NAD_NADH_ratio']],
            'SIRT3': y_final[IDX['SIRT3_activity']],
            'PINK1': y_final[IDX['PINK1_level']],
            'Mitophagy': y_final[IDX['Mitophagy_flux']],
            'CytC': y_final[IDX['CytC_release']],
            'Biogenesis': y_final[IDX['Mito_biogenesis']],
        }

    return {'results': results, 'summary': summary}


# ============================================================
# 快速测试入口 / Quick Test Entry Point
# ============================================================

if __name__ == '__main__':
    import os

    print("=" * 60)
    print("线粒体动力学模型 — 快速测试")
    print("Mitochondrial Dynamics — Quick Test")
    print("=" * 60)

    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)

    # 1. 初始化模型
    print("\n[1] 初始化模型...")
    model = MitochondrialDynamicsModel()
    print("    ✓ 14 state variables initialized")

    # 2. 健康基线仿真
    print("\n[2] 健康基线仿真 (200 time units)...")
    res_base = model.simulate()
    y_final = res_base['y'][:, -1]
    print(f"    Δψm = {y_final[IDX['Mito_membrane_potential']]:.3f} "
          f"(target: ~0.85)")
    print(f"    miROS = {y_final[IDX['miROS']]:.3f} "
          f"(target: ~0.10)")
    print(f"    Fragmentation = {y_final[IDX['Fragmentation_idx']]:.3f} "
          f"(target: ~0.30)")
    print(f"    Integration success: {res_base['success']}")

    # 3. 扰动仿真
    print("\n[3] 扰动仿真...")
    for pert in ['fission_stress', 'fusion_enhance',
                 'nad_depletion', 'mito_poison']:
        res = model.simulate_perturbation(pert)
        state = model.get_mito_state(res['y'][:, -1])
        e = state['energetics']
        q = state['quality_control']
        print(f"    {pert:20s} | Δψm={e['membrane_potential']:.3f} "
              f"| miROS={q['miROS']:.3f} "
              f"| frag={state['fragmentation']['fragmentation_index']:.3f}")

    # 4. 状态解读
    print("\n[4] 基线状态解读...")
    mito_state = model.get_mito_state(y_final)
    for category, metrics in mito_state.items():
        print(f"    {category}:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"      {key:25s} = {val:.4f}")
            elif isinstance(val, bool):
                print(f"      {key:25s} = {val}")

    # 5. 稳态指标
    print("\n[5] 稳态指标...")
    metrics = model.get_steady_state_metrics(y_final)
    for key, val in metrics.items():
        print(f"    {key:25s} = {val:.4f}")

    # 6. 绘图
    print("\n[6] 绘图...")
    fig1 = model.plot_trajectories(
        res_base, output_path='output/mito_trajectories.png'
    )
    plt.close(fig1)
    print("    Saved: output/mito_trajectories.png")

    fig2 = model.plot_fission_fusion(
        res_base, output_path='output/mito_fission_fusion.png'
    )
    plt.close(fig2)
    print("    Saved: output/mito_fission_fusion.png")

    fig3 = model.plot_mito_quality_control(
        res_base, output_path='output/mito_quality_control.png'
    )
    plt.close(fig3)
    print("    Saved: output/mito_quality_control.png")

    # 7. 全面扰动对比
    print("\n[7] 全面扰动对比...")
    all_results = run_all_perturbations(model)
    print("\n    扰动终态对比表:")
    print(f"    {'Condition':20s} | {'Δψm':8s} | {'miROS':8s} | "
          f"{'Frag':8s} | {'CytC':8s} | {'SIRT3':8s}")
    print(f"    {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, s in all_results['summary'].items():
        print(f"    {name:20s} | {s['Δψm']:.4f}  | {s['miROS']:.4f}  | "
              f"{s['Fragmentation']:.4f}  | {s['CytC']:.4f}  | "
              f"{s['SIRT3']:.4f}")

    # 扰动对比图
    fig4 = model.plot_perturbation_comparison(
        all_results['results'],
        output_path='output/mito_perturbation_comparison.png'
    )
    plt.close(fig4)
    print("\n    Saved: output/mito_perturbation_comparison.png")

    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
