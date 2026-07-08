"""
mRNA 动力学与相分离调控模块 — Virtual NP Cell
=============================================
NP 细胞 mRNA 生命周期动力学模型，整合 RBP 调控网络、
应激颗粒 (SG) / P-body 组装、液-液相分离 (LLPS) 及 circRNA/miRNA 缓冲机制。

核心耦合机制:
  1. 应激颗粒↔mRNA存贮↔ECM合成: eIF2α-P→SG形成→ECM_mRNA捕获→翻译停止
  2. HuR/TTP 跷跷板: 健康态 HuR↑TTP↓ 促ECM；退变态 HuR↓+失活TTP→ECM↓SASP↑
  3. m6A↔IGF2BP↔mRNA稳定性: m6A→IGF2BP→3'UTR稳定 mRNA
  4. NEAT1 lncRNA↔miRNA海绵: NEAT1↑→miRNA海绵→ADAMTS5↑→ECM降解
  5. LLPS相分离↔衰老: 异常相分离→核仁应力→p53→衰老程序

Author: Virtual NP Cell Team
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any, Union

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 默认参数集 (Default Parameters)
# ============================================================
# 命名规则: k_ = 合成/激活速率, d_ = 降解/失活速率, K_ = Michaelis常数
#            _act = 激活, _inh = 抑制, _trap = 捕获, _rel = 释放

PARAMS_DEFAULT = {
    # --- Total_mRNA_pool ---
    'k_pool_syn': 0.025,           # mRNA池基础合成 (转录活性)
    'd_pool': 0.010,               # mRNA池基础降解速率
    'K_pool_mir': 0.3,             # miRNA机器对mRNA池降解的半饱和常数

    # --- ECM_mRNA (Aggrecan/Col2) ---
    'k_ecm_syn': 0.030,            # ECM mRNA 基础转录 (SOX9驱动)
    'd_ecm': 0.015,                # ECM mRNA 基础降解
    'K_hur_ecm': 0.4,              # HuR稳定ECM mRNA的半饱和常数
    'K_ttp_ecm': 0.5,              # TTP促ECM mRNA降解的半饱和常数

    # --- SASP_mRNA (IL1B/IL6/MMP) ---
    'k_sasp_syn': 0.012,           # SASP mRNA 基础转录 (NF-κB驱动)
    'd_sasp': 0.020,               # SASP mRNA 基础降解
    'K_ttp_sasp': 0.4,             # TTP促SASP mRNA降解的半饱和常数
    'sasp_nfkb_boost': 1.5,        # NF-κB信号对SASP转录增强倍数

    # --- miRNA_machinery (Drosha/Dicer/AGO2) ---
    'k_mir_syn': 0.020,            # miRNA机器基础合成
    'd_mir': 0.010,                # miRNA机器降解
    'd_mir_degen_factor': 1.8,     # 退变中miRNA机器加速降解因子
    'mir_global_repress': 0.6,     # miRNA对靶mRNA全局抑制强度 [0,1]

    # --- lncRNA_NEAT1 ---
    'k_neat1_syn': 0.020,          # NEAT1 基础转录
    'd_neat1': 0.020,              # NEAT1 降解 (更快, 使基线更低)
    'k_neat1_degen': 0.060,        # 退变中NEAT1转录增强
    'neat1_sponge_eff': 0.6,       # NEAT1 miRNA海绵效率 [0,1]
    'K_neat1_sponge': 0.4,         # NEAT1海绵半饱和常数 (更低=更敏感)

    # --- P_body_assembly ---
    'k_pbody_syn': 0.025,          # P-body 组装速率
    'd_pbody': 0.012,              # P-body 拆解/降解
    'K_pbody_mir': 0.4,            # miRNA机器驱动P-body的半饱和常数
    'k_pbody_stress_disrupt': 0.03, # 应激诱导P-body拆解速率

    # --- Stress_granule ---
    'k_sg_form': 0.120,            # SG 组装最大速率
    'K_sg_eif2a': 0.45,           # eIF2α-P 驱动SG的半饱和常数 (更高阈值)
    'k_sg_dis': 0.040,             # SG 拆解速率 (更快拆解)
    'k_sg_persist': 0.003,         # 持续应激下SG不可逆转化速率

    # --- SG_mRNA_trapping ---
    'k_trap_ecm': 0.060,           # SG捕获ECM mRNA速率
    'k_trap_sasp': 0.030,          # SG捕获SASP mRNA速率 (ECM偏向捕获)
    'K_trap': 0.5,                 # 捕获半饱和常数
    'k_release': 0.015,            # SG释放mRNA速率

    # --- RBP_HuR (ELAVL1) ---
    'k_hur_syn': 0.025,            # HuR 基础合成
    'd_hur': 0.012,                # HuR 降解
    'k_hur_degen_down': 1.2,       # 退变中HuR下调因子 (更强抑制)

    # --- RBP_TTP (ZFP36) ---
    'k_ttp_syn': 0.015,            # TTP 基础合成 (NF-κB驱动)
    'd_ttp': 0.015,                # TTP 降解
    'k_ttp_nfkb_boost': 2.0,       # NF-κB对TTP转录增强
    'k_ttp_p38_inact': 0.035,      # p38 MAPK磷酸化失活TTP速率
    'd_ttp_inact': 0.010,          # 失活TTP降解

    # --- RBP_IGF2BP (IGF2BP1-3) ---
    'k_igf2bp_syn': 0.020,         # IGF2BP 基础合成
    'd_igf2bp': 0.010,             # IGF2BP 降解
    'K_igf2bp_m6a': 0.4,          # m6A信号驱动IGF2BP的半饱和常数

    # --- LLPS_condensates ---
    'k_llps_syn': 0.030,           # LLPS 凝聚物基础形成
    'd_llps': 0.008,               # LLPS 拆解
    'K_llps_rbp': 0.4,            # RBP多价互作驱动LLPS的半饱和常数
    'k_llps_abnormal': 0.005,      # 异常相分离倾向 (退变中增加)

    # --- Nuclear_export ---
    'k_export_syn': 0.020,         # mRNA核输出效率基础值
    'd_export': 0.008,             # 核输出效率衰减
    'k_export_nuclear_damage': 0.025, # 核膜损伤对输出影响

    # --- Translation_efficiency ---
    'k_transl_base': 0.030,        # 基础翻译效率
    'd_transl': 0.010,             # 翻译效率衰减
    'K_transl_eif2a': 0.3,        # eIF2α-P抑制翻译的半饱和常数
    'k_transl_sg_block': 0.040,    # SG介导的翻译阻滞

    # --- circRNA_buffer ---
    'k_circ_syn': 0.020,           # circRNA 基础生成
    'd_circ': 0.006,               # circRNA 降解
    'k_circ_degen_down': 0.6,      # 退变中circRNA减少因子

    # --- eIF2a_phos (eIF2α磷酸化, 驱动SG和翻译抑制) ---
    'k_eif2a_phos': 0.030,         # eIF2α 磷酸化 (应激驱动)
    'K_eif2a_stress': 0.5,         # 应激驱动磷酸化的半饱和常数
    'k_eif2a_dephos': 0.050,       # eIF2α 去磷酸化 (更快去磷酸化)
    'k_eif2a_ers': 0.050,          # ER应激额外磷酸化速率

    # --- 外部信号耦合 ---
    'nfkb_signal': 0.3,            # NF-κB 基础信号水平 [0,1]
    'p38_activity': 0.2,           # p38 MAPK 基础活性 [0,1]
    'm6a_signal': 0.5,             # m6A修饰整体信号 [0,1]
    'sox9_activity': 0.7,          # SOX9转录活性 [0,1]
    'stress_input': 0.03,          # 外部应激输入 [0,1] (健康态基线很低)
    'k_eif2a_ers_baseline': 0.003, # 基线ER应激 eIF2α 磷酸化 (极低)
}

# 变量名 (英文, 用于内部索引)
VAR_NAMES = [
    'Total_mRNA_pool',
    'ECM_mRNA',
    'SASP_mRNA',
    'miRNA_machinery',
    'lncRNA_NEAT1',
    'P_body_assembly',
    'Stress_granule',
    'SG_mRNA_trapping',
    'RBP_HuR',
    'RBP_TTP',
    'RBP_IGF2BP',
    'LLPS_condensates',
    'Nuclear_export',
    'Translation_efficiency',
    'circRNA_buffer',
    'eIF2a_phos',
]

NUM_VARS = len(VAR_NAMES)

# 变量中文标签 (用于绘图)
VAR_LABELS_CN = [
    '总 mRNA 池',
    'ECM mRNA',
    'SASP mRNA',
    'miRNA 加工机器',
    'NEAT1 lncRNA',
    'P-body 组装',
    '应激颗粒 (SG)',
    'SG mRNA 存储',
    'HuR/ELAVL1',
    'TTP/ZFP36',
    'IGF2BP1-3',
    'LLPS 凝聚物',
    'mRNA 核输出效率',
    '翻译效率',
    'circRNA 缓冲',
    'eIF2α 磷酸化',
]

# 变量单位/范围提示
VAR_RANGES = {
    'Total_mRNA_pool':     (0, 5),
    'ECM_mRNA':            (0, 5),
    'SASP_mRNA':           (0, 5),
    'miRNA_machinery':     (0, 3),
    'lncRNA_NEAT1':        (0, 4),
    'P_body_assembly':     (0, 3),
    'Stress_granule':      (0, 1),
    'SG_mRNA_trapping':    (0, 5),
    'RBP_HuR':             (0, 4),
    'RBP_TTP':             (0, 4),
    'RBP_IGF2BP':          (0, 4),
    'LLPS_condensates':    (0, 3),
    'Nuclear_export':      (0, 1),
    'Translation_efficiency': (0, 2),
    'circRNA_buffer':      (0, 4),
    'eIF2a_phos':          (0, 1),
}


# ============================================================
# 2. ODE 系统定义 (ODE System)
# ============================================================

def rna_ode_system(
    t: float,
    y: np.ndarray,
    p: Dict[str, float],
    perturbation: Optional[str] = None,
    pert_severity: float = 0.5,
) -> List[float]:
    """
    mRNA动力学与相分离 ODE 右端函数.

    Parameters
    ----------
    t : float
        当前时间点
    y : np.ndarray (16,)
        状态变量向量
    p : dict
        参数集
    perturbation : str or None
        扰动类型
    pert_severity : float
        扰动强度 [0, 1], 默认 0.5

    Returns
    -------
    list : 16 个变量的导数
    """
    # 解包变量
    (Total_mRNA, ECM_mRNA, SASP_mRNA, miRNA_mach, NEAT1,
     P_body, SG, SG_trap, HuR, TTP, IGF2BP,
     LLPS, Nuclear_export, Translation_eff, circRNA, eIF2a_P) = y

    # --- 解包参数 (方便引用) ---
    k_pool_syn          = p['k_pool_syn']
    d_pool              = p['d_pool']
    K_pool_mir          = p['K_pool_mir']

    k_ecm_syn           = p['k_ecm_syn']
    d_ecm               = p['d_ecm']
    K_hur_ecm           = p['K_hur_ecm']
    K_ttp_ecm           = p['K_ttp_ecm']

    k_sasp_syn          = p['k_sasp_syn']
    d_sasp              = p['d_sasp']
    K_ttp_sasp          = p['K_ttp_sasp']
    sasp_nfkb_boost     = p['sasp_nfkb_boost']

    k_mir_syn           = p['k_mir_syn']
    d_mir               = p['d_mir']
    d_mir_degen_factor  = p['d_mir_degen_factor']
    mir_global_repress  = p['mir_global_repress']

    k_neat1_syn         = p['k_neat1_syn']
    d_neat1             = p['d_neat1']
    k_neat1_degen       = p['k_neat1_degen']
    neat1_sponge_eff    = p['neat1_sponge_eff']
    K_neat1_sponge      = p['K_neat1_sponge']

    k_pbody_syn         = p['k_pbody_syn']
    d_pbody             = p['d_pbody']
    K_pbody_mir         = p['K_pbody_mir']
    k_pbody_stress_disrupt = p['k_pbody_stress_disrupt']

    k_sg_form           = p['k_sg_form']
    K_sg_eif2a          = p['K_sg_eif2a']
    k_sg_dis            = p['k_sg_dis']
    k_sg_persist        = p['k_sg_persist']

    k_trap_ecm          = p['k_trap_ecm']
    k_trap_sasp         = p['k_trap_sasp']
    K_trap              = p['K_trap']
    k_release           = p['k_release']

    k_hur_syn           = p['k_hur_syn']
    d_hur               = p['d_hur']
    k_hur_degen_down    = p['k_hur_degen_down']

    k_ttp_syn           = p['k_ttp_syn']
    d_ttp               = p['d_ttp']
    k_ttp_nfkb_boost    = p['k_ttp_nfkb_boost']
    k_ttp_p38_inact     = p['k_ttp_p38_inact']
    d_ttp_inact         = p['d_ttp_inact']

    k_igf2bp_syn        = p['k_igf2bp_syn']
    d_igf2bp            = p['d_igf2bp']
    K_igf2bp_m6a        = p['K_igf2bp_m6a']

    k_llps_syn          = p['k_llps_syn']
    d_llps              = p['d_llps']
    K_llps_rbp          = p['K_llps_rbp']
    k_llps_abnormal     = p['k_llps_abnormal']

    k_export_syn        = p['k_export_syn']
    d_export            = p['d_export']
    k_export_nuclear_damage = p['k_export_nuclear_damage']

    k_transl_base       = p['k_transl_base']
    d_transl            = p['d_transl']
    K_transl_eif2a      = p['K_transl_eif2a']
    k_transl_sg_block   = p['k_transl_sg_block']

    k_circ_syn          = p['k_circ_syn']
    d_circ              = p['d_circ']
    k_circ_degen_down   = p['k_circ_degen_down']

    k_eif2a_phos        = p['k_eif2a_phos']
    K_eif2a_stress      = p['K_eif2a_stress']
    k_eif2a_dephos      = p['k_eif2a_dephos']
    k_eif2a_ers         = p['k_eif2a_ers']
    k_eif2a_ers_base   = p.get('k_eif2a_ers_baseline', 0.005)

    # --- 外部信号 (可被扰动修改) ---
    nfkb_signal         = p['nfkb_signal']
    p38_activity        = p['p38_activity']
    m6a_signal          = p['m6a_signal']
    sox9_activity       = p['sox9_activity']
    stress_input        = p['stress_input']

    # ============================================================
    # 应用扰动 (Perturbation)
    # ============================================================
    if perturbation == 'oxidative_stress':
        # 氧化应激 → eIF2α-P → SG; 同时增强NF-κB信号
        stress_input = min(1.0, stress_input + pert_severity * 0.7)
        nfkb_signal = min(1.0, nfkb_signal + pert_severity * 0.4)

    elif perturbation == 'm6a_dysreg':
        # m6A 失调: 模拟 WTAP OE (m6A↑) 或 KD (m6A↓)
        if pert_severity > 0:
            # OE: m6A↑ → IGF2BP↑ → mRNA稳定性改变 (双刃剑)
            m6a_signal = min(1.0, m6a_signal + pert_severity * 0.4)
        else:
            # KD (负值严重度): m6A↓ → IGF2BP↓
            m6a_signal = max(0.05, m6a_signal + pert_severity * 0.5)

    elif perturbation == 'huR_loss':
        # HuR 缺失: ECM mRNA 稳定性↓
        k_hur_syn *= max(0.1, 1.0 - pert_severity * 0.9)

    elif perturbation == 'neat1_oe':
        # NEAT1 过表达: NEAT1 海绵过量
        k_neat1_syn *= (1.0 + pert_severity * 3.0)

    elif perturbation == 'phase_sep_abnormal':
        # LLPS 异常: 异常相分离倾向增强
        k_llps_abnormal *= (1.0 + pert_severity * 4.0)
        # 同时减弱正常LLPS动力学
        d_llps *= max(0.3, 1.0 - pert_severity * 0.3)

    elif perturbation == 'er_stress':
        # 内质网应激 → eIF2α-P↑ + SG↑ + 翻译↓
        stress_input = min(1.0, stress_input + pert_severity * 0.5)
        k_eif2a_ers_boost = k_eif2a_ers * (1.0 + pert_severity * 3.0)
        # 使用临时变量

    elif perturbation == 'nfkb_activation':
        # NF-κB 激活 → SASP↑ + TTP↑
        nfkb_signal = min(1.0, nfkb_signal + pert_severity * 0.5)

    elif perturbation == 'p38_inhibition':
        # p38 抑制 → TTP保持活性 → SASP mRNA降解↑
        p38_activity = max(0.05, p38_activity - pert_severity * 0.5)

    # ============================================================
    # 核心生物物理计算
    # ============================================================

    # --- eIF2α-P 动力学 ---
    # 基线低水平 + 应激驱动磷酸化
    ers_base = k_eif2a_ers_base
    if perturbation == 'er_stress':
        ers_base += k_eif2a_ers * (1.0 + pert_severity * 3.0)

    eif2a_phos_drive = k_eif2a_phos * stress_input / (K_eif2a_stress + stress_input)
    de2if2a = (eif2a_phos_drive + ers_base) * (1.0 - eIF2a_P) - k_eif2a_dephos * eIF2a_P

    # --- SG 动力学 (Mechanism 1) ---
    # eIF2α-P → SG 组装; 持续应激 → 不可逆转化
    sg_formation = k_sg_form * (eIF2a_P ** 2) / (K_sg_eif2a ** 2 + eIF2a_P ** 2) * (1.0 - SG)
    sg_persist_effect = k_sg_persist * SG * eIF2a_P  # 持续应激下SG更稳定
    dsg = sg_formation - (k_sg_dis - sg_persist_effect) * SG
    dsg = max(dsg, -0.5)  # 防止数值振荡

    # --- SG mRNA trapping (ECM_bias: ECM被优先捕获) ---
    total_free_mrna = ECM_mRNA + SASP_mRNA + 0.01  # 避免除零
    trap_ecm_flux = k_trap_ecm * SG * ECM_mRNA / (K_trap + total_free_mrna)
    trap_sasp_flux = k_trap_sasp * SG * SASP_mRNA / (K_trap + total_free_mrna)
    release_flux = k_release * SG_trap
    dsg_trap = trap_ecm_flux + trap_sasp_flux - release_flux

    # --- P-body 动力学 ---
    # miRNA机器驱动P-body组装; 应激诱导P-body拆解
    mir_pbody_drive = k_pbody_syn * miRNA_mach / (K_pbody_mir + miRNA_mach)
    stress_pbody_disrupt = k_pbody_stress_disrupt * SG * P_body  # SG↑ → P-body↓
    dpbody = mir_pbody_drive - d_pbody * P_body - stress_pbody_disrupt

    # --- 翻译效率 ---
    # eIF2α-P → 翻译抑制; SG → 翻译阻滞 (trapped mRNA不可翻译)
    eif2a_transl_inh = k_transl_sg_block * (eIF2a_P ** 2) / (K_transl_eif2a ** 2 + eIF2a_P ** 2)
    sg_transl_block = k_transl_sg_block * SG_trap / (K_trap + SG_trap)
    dtrans = k_transl_base - (d_transl + eif2a_transl_inh + sg_transl_block) * Translation_eff
    dtrans = max(dtrans, -0.3)  # 防止负值振荡

    # --- HuR 动力学 (Mechanism 2) ---
    # 健康: HuR高; 退变: HuR↓
    # NF-κB轻度激活HuR, 但退变中整体下调占主导
    # SG 反映退变压力, 下调HuR; 早期轻度应激代偿性上调
    # HuR: 退变中SG触发HuR下调; 轻度应力代偿上调
    hur_stress_early = 0.12 * min(max(SG - 0.05, 0.0), 0.2)  # 轻度应激代偿
    hur_degen_effect = k_hur_degen_down * max(0.0, SG - 0.10)  # SG>0.1触发显著降解
    dHuR = k_hur_syn + hur_stress_early - (d_hur + hur_degen_effect) * HuR
    dHuR = max(dHuR, -HuR * 0.5)

    # --- TTP 动力学 (Mechanism 2) ---
    # NF-κB → TTP转录↑; p38 → TTP磷酸化失活
    ttp_nfkb_act = k_ttp_syn * (1.0 + k_ttp_nfkb_boost * nfkb_signal)
    ttp_inactivation = k_ttp_p38_inact * p38_activity * TTP
    dTTP = ttp_nfkb_act - d_ttp * TTP - ttp_inactivation

    # --- IGF2BP 动力学 (Mechanism 3) ---
    # m6A信号驱动IGF2BP; IGF2BP稳定mRNA
    igf2bp_m6a_drive = k_igf2bp_syn * m6a_signal / (K_igf2bp_m6a + m6a_signal)
    dIGF2BP = igf2bp_m6a_drive - d_igf2bp * IGF2BP

    # --- NEAT1 动力学 (Mechanism 4) ---
    # 退变中NEAT1转录增强 (NF-κB驱动)
    neat1_trans = k_neat1_syn + k_neat1_degen * nfkb_signal
    dNEAT1 = neat1_trans - d_neat1 * NEAT1
    # NEAT1海绵效应: NEAT1↑ → miRNA被捕获 → miRNA可用活性↓
    neat1_sponge_burden = neat1_sponge_eff * NEAT1 / (K_neat1_sponge + NEAT1)

    # --- miRNA_machinery ---
    # 退变: miRNA机器下调 + NEAT1海绵效应, 加剧去抑制
    mir_sponge_degen = 1.0 + neat1_sponge_burden  # NEAT1海绵消耗miRNA活性
    mir_degen = d_mir * (1.0 + d_mir_degen_factor * SG) * mir_sponge_degen
    dmiRNA = k_mir_syn - mir_degen * miRNA_mach

    # --- Nuclear_export ---
    # 核膜应力 (由SG/LLPS/Nuclear_damage反映) → 输出紊乱
    # 异常LLPS直接损害核孔复合物 (只在LLPS显著异常时)
    llps_nuclear_damage = max(0.0, LLPS - 4.0) / (2.0 + max(0.0, LLPS - 4.0))
    nuclear_stress = k_export_nuclear_damage * (LLPS + SG) / (1.0 + LLPS + SG)
    dExport = k_export_syn - (d_export + nuclear_stress) * Nuclear_export

    # --- LLPS 动力学 (Mechanism 5) ---
    # RBP多价互作驱动LLPS (HuR + NEAT1 + IGF2BP)
    rbp_multivalency = (HuR + IGF2BP + NEAT1) / 3.0  # 归一化
    llps_drive = k_llps_syn * (rbp_multivalency ** 2) / (K_llps_rbp ** 2 + rbp_multivalency ** 2)
    # 异常相分离 (退变中增强)
    llps_abnormal_effect = k_llps_abnormal * SG * (1.0 + LLPS)
    dLLPS = llps_drive + llps_abnormal_effect - d_llps * LLPS

    # --- circRNA_buffer ---
    # 退变中circRNA↓ → miRNA解放 → 靶mRNA下调
    circ_degen = d_circ * (1.0 + (1.0 - k_circ_degen_down) * SG)
    dcirc = k_circ_syn - circ_degen * circRNA

    # --- ECM_mRNA (Mechanisms 1, 2, 3) ---
    # 合成: SOX9驱动 + HuR稳定; 降解: TTP促降解 + SG捕获
    # HuR稳定效应: 降低有效降解
    hur_stab = HuR / (K_hur_ecm + HuR)
    # TTP促降解效应: TTP_eff = active TTP / (1 + p38_inact)
    ttp_active = TTP / (1.0 + k_ttp_p38_inact * p38_activity * TTP / (d_ttp + 0.01))
    ttp_deg_ecm = ttp_active / (K_ttp_ecm + ttp_active)
    # NET: 有效降解 = 基础降解 * (1 - HuR稳定) * (1 + TTP促降解)
    ecm_stab_factor = 1.0 + 2.0 * hur_stab  # HuR稳定性因子 [1, 3]
    ecm_deg_factor = 1.0 + 2.0 * ttp_deg_ecm  # TTP降解因子 [1, 3]

    # NEAT1海绵→miRNA-140↓→ADAMTS5↑→ECM降解 (Mechanism 4)
    # miRNA_mach下降 + NEAT1海绵联合效应
    mir_available = miRNA_mach * (1.0 - neat1_sponge_burden)  # NEAT1降低miRNA可用性
    neat1_ecm_degen = 0.8 * neat1_sponge_burden * (1.0 + mir_global_repress * (1.0 - mir_available / (K_pool_mir + mir_available)))
    ecm_synthesis = k_ecm_syn * sox9_activity * Nuclear_export * ecm_stab_factor
    ecm_degradation = d_ecm * ECM_mRNA * ecm_deg_factor * (1.0 + neat1_ecm_degen)
    dECM = ecm_synthesis - ecm_degradation - trap_ecm_flux + release_flux * (ECM_mRNA / total_free_mrna if total_free_mrna > 0.01 else 0.5)

    # --- SASP_mRNA (Mechanisms 1, 2) ---
    # 合成: NF-κB驱动; 降解: 受TTP调控 (但p38使TTP失活 → SASP稳定)
    sasp_syn_factor = k_sasp_syn * (1.0 + sasp_nfkb_boost * nfkb_signal)
    # TTP对SASP的降解 (受p38失活保护)
    ttp_deg_sasp = ttp_active / (K_ttp_sasp + ttp_active)
    # 在退变中, TTP↑但失活 → SASP mRNA稳定 (核心矛盾)
    sasp_deg_factor = 1.0 + ttp_deg_sasp * (1.0 - p38_activity)  # p38高 → TTP失活 → 降解少
    dSASP = sasp_syn_factor * Nuclear_export - d_sasp * SASP_mRNA * sasp_deg_factor \
            - trap_sasp_flux + release_flux * (SASP_mRNA / total_free_mrna if total_free_mrna > 0.01 else 0.5)

    # --- Total_mRNA_pool ---
    # miRNA机器介导全局降解调控
    mir_regulation = 1.0 - mir_global_repress * miRNA_mach / (K_pool_mir + miRNA_mach)
    dTotal = k_pool_syn * Nuclear_export - d_pool * Total_mRNA * mir_regulation

    # 组装导数向量 (必须与变量顺序一致)
    dy = [
        dTotal,
        dECM,
        dSASP,
        dmiRNA,
        dNEAT1,
        dpbody,
        dsg,
        dsg_trap,
        dHuR,
        dTTP,
        dIGF2BP,
        dLLPS,
        dExport,
        dtrans,
        dcirc,
        de2if2a,
    ]

    return dy


# ============================================================
# 3. RNADynamicsModel 类
# ============================================================

class RNADynamicsModel:
    """
    NP 细胞 mRNA 动力学与相分离调控模型.

    16个变量的 ODE 系统, 囊括:
    - mRNA生命周期 (转录、核输出、翻译、降解)
    - RBP调控网络 (HuR/TTP/IGF2BP 跷跷板)
    - 相分离事件 (SG/P-body/LLPS)
    - NEAT1-circRNA-miRNA 调控轴

    Parameters
    ----------
    params : dict or None
        参数集 (None 时使用默认参数)
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        self.params = PARAMS_DEFAULT.copy()
        if params is not None:
            self.params.update(params)
        self.y0 = self._compute_steady_state()

    def _compute_steady_state(self) -> np.ndarray:
        """
        基于稳态解析近似计算初始条件.

        对每个变量近似解析求解稳态, 作为ODE积分的初始值.
        """
        p = self.params

        # --- miRNA_machinery ---
        miRNA_ss = p['k_mir_syn'] / p['d_mir']

        # --- eIF2α-P ---
        eif2a_drive = p['k_eif2a_phos'] * p['stress_input'] / (p['K_eif2a_stress'] + p['stress_input'])
        eif2a_ers = p['k_eif2a_ers'] * p['stress_input']
        eIF2a_ss = (eif2a_drive + eif2a_ers) / (eif2a_drive + eif2a_ers + p['k_eif2a_dephos'])
        eIF2a_ss = min(eIF2a_ss, 0.3)  # 健康稳态下低磷酸化

        # --- SG ---
        sg_formation = p['k_sg_form'] * eIF2a_ss ** 2 / (p['K_sg_eif2a'] ** 2 + eIF2a_ss ** 2)
        SG_ss = sg_formation / (sg_formation + p['k_sg_dis'])
        SG_ss = min(SG_ss, 0.15)  # 健康稳态下低SG

        # --- HuR ---
        HuR_ss = p['k_hur_syn'] / p['d_hur']

        # --- TTP ---
        ttp_nfkb = p['k_ttp_syn'] * (1.0 + p['k_ttp_nfkb_boost'] * p['nfkb_signal'])
        ttp_inact = p['k_ttp_p38_inact'] * p['p38_activity']
        TTP_ss = ttp_nfkb / (p['d_ttp'] + ttp_inact)

        # --- IGF2BP ---
        igf2bp_drive = p['k_igf2bp_syn'] * p['m6a_signal'] / (p['K_igf2bp_m6a'] + p['m6a_signal'])
        IGF2BP_ss = igf2bp_drive / p['d_igf2bp']

        # --- NEAT1 ---
        neat1_trans = p['k_neat1_syn'] + p['k_neat1_degen'] * p['nfkb_signal']
        NEAT1_ss = neat1_trans / p['d_neat1']

        # --- Nuclear_export ---
        Nuclear_export_ss = p['k_export_syn'] / p['d_export']
        Nuclear_export_ss = min(Nuclear_export_ss, 1.0)

        # --- Translation_efficiency ---
        eif2a_inh = p['k_transl_sg_block'] * eIF2a_ss ** 2 / (p['K_transl_eif2a'] ** 2 + eIF2a_ss ** 2)
        Translation_ss = p['k_transl_base'] / (p['d_transl'] + eif2a_inh)
        Translation_ss = min(Translation_ss, 1.5)

        # --- circRNA ---
        circRNA_ss = p['k_circ_syn'] / p['d_circ']

        # --- LLPS ---
        rbp_val = (HuR_ss + IGF2BP_ss + NEAT1_ss) / 3.0
        llps_drive = p['k_llps_syn'] * rbp_val ** 2 / (p['K_llps_rbp'] ** 2 + rbp_val ** 2)
        LLPS_ss = llps_drive / p['d_llps']

        # --- ECM_mRNA (with HuR/TTP coupling) ---
        hur_stab = HuR_ss / (p['K_hur_ecm'] + HuR_ss)
        ttp_act = TTP_ss / (1.0 + p['k_ttp_p38_inact'] * p['p38_activity'] * TTP_ss / (p['d_ttp'] + 0.01))
        ttp_deg = ttp_act / (p['K_ttp_ecm'] + ttp_act)
        ecm_stab = 1.0 + 2.0 * hur_stab
        ecm_deg_f = 1.0 + 2.0 * ttp_deg
        ECM_ss = (p['k_ecm_syn'] * p['sox9_activity'] * Nuclear_export_ss * ecm_stab) \
                 / (p['d_ecm'] * ecm_deg_f)

        # --- SASP_mRNA ---
        sasp_syn = p['k_sasp_syn'] * (1.0 + p['sasp_nfkb_boost'] * p['nfkb_signal'])
        ttp_deg_s = ttp_act / (p['K_ttp_sasp'] + ttp_act)
        sasp_deg_f = 1.0 + ttp_deg_s * (1.0 - p['p38_activity'])
        SASP_ss = sasp_syn * Nuclear_export_ss / (p['d_sasp'] * sasp_deg_f)

        # --- SG_trap ---
        total_free = max(ECM_ss + SASP_ss, 0.01)
        trap_total = (p['k_trap_ecm'] + p['k_trap_sasp']) * SG_ss * total_free / (p['K_trap'] + total_free)
        SG_trap_ss = trap_total / p['k_release']

        # --- P_body ---
        mir_drive = p['k_pbody_syn'] * miRNA_ss / (p['K_pbody_mir'] + miRNA_ss)
        P_body_ss = mir_drive / p['d_pbody']

        # --- Total_mRNA_pool ---
        mir_reg = 1.0 - p['mir_global_repress'] * miRNA_ss / (p['K_pool_mir'] + miRNA_ss)
        Total_ss = p['k_pool_syn'] * Nuclear_export_ss / (p['d_pool'] * mir_reg)

        return np.array([
            Total_ss,
            ECM_ss,
            SASP_ss,
            miRNA_ss,
            NEAT1_ss,
            P_body_ss,
            SG_ss,
            SG_trap_ss,
            HuR_ss,
            TTP_ss,
            IGF2BP_ss,
            LLPS_ss,
            Nuclear_export_ss,
            Translation_ss,
            circRNA_ss,
            eIF2a_ss,
        ])

    def ode_system(
        self,
        t: float,
        y: np.ndarray,
        perturbation: Optional[str] = None,
    ) -> List[float]:
        """包装 ODE 右端函数, 兼容外部调用."""
        return rna_ode_system(t, y, self.params, perturbation=perturbation)

    def simulate(
        self,
        t_span: Tuple[float, float] = (0, 200),
        n_points: int = 500,
        perturbation: Optional[str] = None,
        pert_severity: float = 0.5,
        method: str = 'RK45',
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = 5.0,
    ) -> Dict[str, Any]:
        """
        运行 ODE 积分模拟.

        Parameters
        ----------
        t_span : tuple (t_start, t_end)
            模拟时间范围 (默认 0-200)
        n_points : int
            输出时间点数
        perturbation : str or None
            扰动类型:
            - 'oxidative_stress'    : 氧化应激→eIF2α-P→SG
            - 'm6a_dysreg'          : m6A失调 (WTAP OE/KD)
            - 'huR_loss'            : HuR缺失
            - 'neat1_oe'            : NEAT1过表达
            - 'phase_sep_abnormal'  : LLPS异常
            - 'er_stress'           : 内质网应激
            - 'nfkb_activation'     : NF-κB激活
            - 'p38_inhibition'      : p38抑制
        pert_severity : float
            扰动强度 [0, 1], 默认 0.5
        method : str
            solve_ivp 积分方法
        rtol, atol : float
            积分容差
        max_step : float
            最大积分步长

        Returns
        -------
        dict
            {
                't': np.ndarray — 时间点
                'y': np.ndarray — 变量矩阵 (时间点 × 16)
                'var_names': list — 变量名
                'params': dict — 参数集
                'perturbation': str or None
                'pert_severity': float
            }
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def ode_func(t, y):
            return rna_ode_system(t, y, self.params,
                                  perturbation=perturbation,
                                  pert_severity=pert_severity)

        sol = solve_ivp(
            ode_func,
            t_span,
            self.y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        # 如果积分失败, 用更鲁棒的方法重试
        if not sol.success:
            sol = solve_ivp(
                ode_func,
                t_span,
                self.y0,
                t_eval=t_eval,
                method='LSODA',
                rtol=rtol * 10,
                atol=atol,
            )

        return {
            't': sol.t,
            'y': sol.y,
            'var_names': VAR_NAMES,
            'params': self.params.copy(),
            'perturbation': perturbation,
            'pert_severity': pert_severity,
        }

    def get_transcriptome_state(self, y: np.ndarray) -> Dict[str, float]:
        """
        从状态向量提取转录组综合状态指标.

        Parameters
        ----------
        y : np.ndarray (16,)
            状态向量 (当前时刻)

        Returns
        -------
        dict
            {
                'ecm_capacity':      ECM合成能力
                'sasp_burden':       炎性负担
                'rnai_activity':     RNAi/miRNA活性
                'phase_separation':  相分离综合状态
                'stability_landscape': mRNA稳定性景观
                'stress_status':     应激状态
            }
        """
        p = self.params

        # ECM capacity: ECM_mRNA / (ECM_mRNA + SASP_mRNA) 比例 × SOX9效应
        total_mrna = max(y[1] + y[2], 0.01)
        ecm_capacity = (y[1] / total_mrna) * min(y[13] / 0.5, 1.0)  # 翻译效率归一化

        # SASP burden: SASP_mRNA / total × NF-κB信号
        sasp_burden = (y[2] / total_mrna) * (0.5 + p['nfkb_signal'])

        # RNAi activity: miRNA_machinery × circRNA_buffer效应
        mir_activity = y[3] / (0.3 + y[3])
        circ_effect = y[14] / (0.3 + y[14])
        rnai_activity = mir_activity * (1.0 + 0.5 * circ_effect) / 1.5

        # Phase separation index: SG + P-body + LLPS 综合, 加权
        sg_idx = min(y[6] / 0.5, 1.0)
        pbody_idx = min(y[5] / 2.0, 1.0)
        llps_idx = min(y[11] / 2.0, 1.0)
        phase_separation = (0.4 * sg_idx + 0.2 * pbody_idx + 0.4 * llps_idx)

        # Stability landscape: HuR / TTP 比值 (HuR=稳定, TTP=降解)
        ttp_eff = max(y[9], 0.01)
        stability_landscape = y[8] / (y[8] + ttp_eff * 2.0)

        # Stress status: eIF2α-P + SG + 翻译抑制综合
        stress_status = (y[15] * 0.4 + y[6] * 0.4 + (1.0 - min(y[13] / 1.0, 1.0)) * 0.2)

        return {
            'ecm_capacity': float(ecm_capacity),
            'sasp_burden': float(sasp_burden),
            'rnai_activity': float(rnai_activity),
            'phase_separation': float(phase_separation),
            'stability_landscape': float(stability_landscape),
            'stress_status': float(stress_status),
        }

    def simulate_perturbation(
        self,
        pert_type: str = 'oxidative_stress',
        severity: float = 0.5,
        t_span: Tuple[float, float] = (0, 200),
        n_points: int = 500,
    ) -> Dict[str, Any]:
        """
        便利方法: 直接模拟特定扰动并返回结果.

        Parameters
        ----------
        pert_type : str
            扰动类型:
            'oxidative_stress'   : 氧化应激→eIF2α-P→SG
            'm6a_dysreg'         : m6A失调 (WTAP OE/KD)
            'huR_loss'           : HuR缺失→ECM mRNA稳定性↓
            'neat1_oe'           : NEAT1过表达→lncRNA海绵过量
            'phase_sep_abnormal' : LLPS异常
        severity : float
            扰动强度 [0, 1], 默认 0.5
        t_span : tuple
            时间范围
        n_points : int
            点数

        Returns
        -------
        dict : simulate() 返回结果
        """
        valid_types = [
            'oxidative_stress', 'm6a_dysreg', 'huR_loss',
            'neat1_oe', 'phase_sep_abnormal', 'er_stress',
            'nfkb_activation', 'p38_inhibition',
        ]
        if pert_type not in valid_types:
            raise ValueError(
                f"未知扰动类型: '{pert_type}'. "
                f"可选: {valid_types}"
            )
        return self.simulate(
            t_span=t_span,
            n_points=n_points,
            perturbation=pert_type,
            pert_severity=severity,
        )

    # ============================================================
    # 绘图方法
    # ============================================================

    def _setup_figure(self, title: str = '') -> Tuple[plt.Figure, plt.Axes]:
        """通用图形设置."""
        fig, ax = plt.subplots(figsize=(10, 6))
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (a.u.)', fontsize=11)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def plot_rna_landscape(
        self,
        result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        mRNA + SG + P-body + 相分离 综合图.

        面板1: mRNA池 (Total, ECM, SASP)
        面板2: SG + P-body + SG_trap
        面板3: 相分离 (LLPS + miRNA_machinery)
        面板4: 翻译效率 + 核输出

        Parameters
        ----------
        result : dict
            simulate() 返回的结果字典
        output_path : str or None
            保存路径 (None时显示)

        Returns
        -------
        str : 保存的文件路径
        """
        t = result['t']
        y = result['y']
        pert = result.get('perturbation', None)
        title_suffix = f' [{pert}]' if pert else ' [steady]'

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'RNA Landscape{title_suffix}', fontsize=14, fontweight='bold')

        # --- 面板1: mRNA池 ---
        ax1 = axes[0, 0]
        ax1.plot(t, y[0], 'k-', linewidth=1.5, label='Total mRNA')
        ax1.plot(t, y[1], 'g-', linewidth=2.0, label='ECM mRNA')
        ax1.plot(t, y[2], 'r-', linewidth=2.0, label='SASP mRNA')
        ax1.set_ylabel('mRNA Level (a.u.)', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- 面板2: SG + P-body + Trap ---
        ax2 = axes[0, 1]
        ax2.plot(t, y[6], 'orange', linewidth=2.0, label='Stress Granule')
        ax2.plot(t, y[5], 'purple', linewidth=1.5, label='P-body')
        ax2.plot(t, y[7], 'brown', linewidth=1.5, linestyle='--', label='SG mRNA Trap')
        ax2.set_ylabel('Level (a.u.)', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- 面板3: 相分离 + miRNA ---
        ax3 = axes[1, 0]
        ax3.plot(t, y[11], 'c-', linewidth=2.0, label='LLPS Condensates')
        ax3.plot(t, y[3], 'm-', linewidth=1.5, label='miRNA Machinery')
        ax3.plot(t, y[14], 'teal', linewidth=1.5, linestyle='--', label='circRNA Buffer')
        ax3.set_xlabel('Time (a.u.)', fontsize=10)
        ax3.set_ylabel('Level (a.u.)', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # --- 面板4: 翻译 + 输出 ---
        ax4 = axes[1, 1]
        ax4.plot(t, y[13], 'blue', linewidth=2.0, label='Translation Eff.')
        ax4.plot(t, y[12], 'green', linewidth=1.5, label='Nuclear Export')
        ax4.plot(t, y[15], 'red', linewidth=1.5, linestyle='--', label='eIF2α-P')
        ax4.set_xlabel('Time (a.u.)', fontsize=10)
        ax4.set_ylabel('Level (a.u.)', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.show()
            return ''

    def plot_rbp_balance(
        self,
        result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        HuR/TTP/IGF2BP 轴随时间变化.

        面板1: RBP水平
        面板2: HuR/TTP比值 (mRNA稳定性指标)
        面板3: RBP活性对ECM/SASP的调控效应
        面板4: 热图摘要

        Parameters
        ----------
        result : dict
            simulate() 返回结果
        output_path : str or None
            保存路径

        Returns
        -------
        str : 文件路径
        """
        t = result['t']
        y = result['y']
        p = self.params
        pert = result.get('perturbation', None)
        title_suffix = f' [{pert}]' if pert else ' [steady]'

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'RBP Balance Network{title_suffix}', fontsize=14, fontweight='bold')

        # --- 面板1: RBP水平 ---
        ax1 = axes[0, 0]
        ax1.plot(t, y[8], '#2196F3', linewidth=2.0, label='HuR (stabilizer)')
        ax1.plot(t, y[9], '#F44336', linewidth=2.0, label='TTP (degrader)')
        ax1.plot(t, y[10], '#4CAF50', linewidth=2.0, label='IGF2BP (m6A reader)')
        ax1.set_ylabel('RBP Level (a.u.)', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

        # --- 面板2: HuR/TTP 比值 ---
        ax2 = axes[0, 1]
        # 避免除零
        hur_ttp_ratio = np.where(y[9] > 1e-6, y[8] / (y[9] + 1e-6), 10.0)
        ax2.plot(t, hur_ttp_ratio, '#FF9800', linewidth=2.0, label='HuR/TTP Ratio')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Balance')
        ax2.set_ylabel('HuR / TTP', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- 面板3: RBP对ECM/SASP的调控效应 ---
        ax3 = axes[1, 0]
        # HuR稳定效应
        hur_ecm_eff = p['K_hur_ecm'] / (p['K_hur_ecm'] + y[8])
        # TTP降解效应 (ECM)
        ttp_active = y[9] / (1.0 + p['k_ttp_p38_inact'] * p['p38_activity'] * y[9] / (p['d_ttp'] + 0.01))
        ttp_ecm_eff = ttp_active / (p['K_ttp_ecm'] + ttp_active)
        # IGF2BP稳定效应
        igf_stab = y[10] / (p['K_igf2bp_m6a'] + y[10])

        ax3.plot(t, hur_ecm_eff, '#2196F3', linewidth=1.5, label='HuR → ECM stability')
        ax3.plot(t, ttp_ecm_eff, '#F44336', linewidth=1.5, label='TTP → ECM deg.')
        ax3.plot(t, igf_stab, '#4CAF50', linewidth=1.5, label='IGF2BP → mRNA stab.')
        ax3.set_xlabel('Time (a.u.)', fontsize=10)
        ax3.set_ylabel('Regulatory Effect [0,1]', fontsize=10)
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # --- 面板4: 活性TTP (p38失活效应) ---
        ax4 = axes[1, 1]
        ttp_active_frac = ttp_active / (y[9] + 1e-6)
        ax4.plot(t, ttp_active_frac, '#E91E63', linewidth=2.0, label='Active TTP fraction')
        ax4.plot(t, y[9], '#9C27B0', linewidth=1.5, linestyle='--', label='Total TTP')
        ax4.set_xlabel('Time (a.u.)', fontsize=10)
        ax4.set_ylabel('Level', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, None)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.show()
        return ''

    def plot_phase_separation(
        self,
        result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        SG / P-body / LLPS 动力学 + 关联分析.

        面板1: 三种相分离事件时序
        面板2: mRNA捕获与翻译抑制
        面板3: LLPS-RBP 关联
        面板4: 综合相分离指数

        Parameters
        ----------
        result : dict
            simulate() 返回结果
        output_path : str or None
            保存路径

        Returns
        -------
        str : 文件路径
        """
        t = result['t']
        y = result['y']
        pert = result.get('perturbation', None)
        title_suffix = f' [{pert}]' if pert else ' [steady]'

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Phase Separation Dynamics{title_suffix}', fontsize=14, fontweight='bold')

        # --- 面板1: 三种相分离事件 ---
        ax1 = axes[0, 0]
        ax1.plot(t, y[6], 'orange', linewidth=2.0, label='Stress Granule (SG)')
        ax1.plot(t, y[5], 'purple', linewidth=2.0, label='P-body')
        ax1.plot(t, y[11], 'cyan', linewidth=2.0, label='LLPS Condensates')
        ax1.set_ylabel('Level (a.u.)', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- 面板2: mRNA捕获与翻译 ---
        ax2 = axes[0, 1]
        # ECM有效翻译 = 翻译效率 × ECM_mRNA_free (减去被捕获部分)
        ecm_translatable = y[1]  # free ECM_mRNA
        sasp_translatable = y[2]  # free SASP_mRNA
        transl_eff = y[13]
        ax2.plot(t, ecm_translatable * transl_eff, 'g-', linewidth=2.0,
                 label='ECM translatable')
        ax2.plot(t, sasp_translatable * transl_eff, 'r-', linewidth=2.0,
                 label='SASP translatable')
        ax2.plot(t, y[7], 'brown', linewidth=1.5, linestyle='--',
                 label='SG trapped mRNA')
        ax2.set_ylabel('mRNA Level (a.u.)', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- 面板3: LLPS-RBP三元关联 ---
        ax3 = axes[1, 0]
        rbp_sum = y[8] + y[9] + y[10]
        # RBP 多价性 → LLPS
        ax3.plot(t, y[11], 'cyan', linewidth=2.0, label='LLPS')
        ax3.plot(t, y[8], '#2196F3', linewidth=1.5, linestyle='--', label='HuR')
        ax3.plot(t, y[10], '#4CAF50', linewidth=1.5, linestyle='--', label='IGF2BP')
        ax3.plot(t, y[4], '#FF5722', linewidth=1.5, linestyle=':', label='NEAT1')
        ax3.set_xlabel('Time (a.u.)', fontsize=10)
        ax3.set_ylabel('Level (a.u.)', fontsize=10)
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # --- 面板4: 综合相分离指数 ---
        ax4 = axes[1, 1]
        # 综合指数 = 加权和
        sg_norm = np.clip(y[6] / 0.5, 0, 1)
        pbody_norm = np.clip(y[5] / 2.0, 0, 1)
        llps_norm = np.clip(y[11] / 2.0, 0, 1)
        psi = 0.4 * sg_norm + 0.2 * pbody_norm + 0.4 * llps_norm
        ax4.plot(t, psi, 'darkviolet', linewidth=2.5, label='Phase Separation Index')
        ax4.fill_between(t, 0, psi, alpha=0.2, color='darkviolet')
        ax4.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5,
                    label='Normal threshold')
        ax4.set_xlabel('Time (a.u.)', fontsize=10)
        ax4.set_ylabel('Index [0,1]', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.show()
        return ''

    def plot_comparison_heatmap(
        self,
        perturbations: List[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        多种扰动对比热图.

        Parameters
        ----------
        perturbations : list of str
            要比较的扰动列表
        output_path : str or None
            保存路径

        Returns
        -------
        str : 文件路径
        """
        if perturbations is None:
            perturbations = [
                'oxidative_stress', 'm6a_dysreg', 'huR_loss',
                'neat1_oe', 'phase_sep_abnormal',
            ]

        # 对比指标
        metrics = ['ecm_capacity', 'sasp_burden', 'rnai_activity',
                   'phase_separation', 'stability_landscape', 'stress_status']
        metric_labels = ['ECM Capacity', 'SASP Burden', 'RNAi Activity',
                         'Phase Sep.', 'Stability', 'Stress']

        # 基线
        baseline = self.simulate(t_span=(0, 300), n_points=100)
        base_state = self.get_transcriptome_state(baseline['y'][:, -1])

        # 各扰动末态
        data = np.zeros((len(perturbations), len(metrics)))
        for i, pert in enumerate(perturbations):
            res = self.simulate_perturbation(pert, severity=0.5,
                                            t_span=(0, 300), n_points=100)
            state = self.get_transcriptome_state(res['y'][:, -1])
            for j, met in enumerate(metrics):
                data[i, j] = state[met] - base_state[met]  # 相对基线变化

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, fontsize=9, rotation=30, ha='right')
        ax.set_yticks(range(len(perturbations)))
        ax.set_yticklabels(perturbations, fontsize=9)

        # 添加数值标注
        for i in range(len(perturbations)):
            for j in range(len(metrics)):
                val = data[i, j]
                color = 'white' if abs(val) > 0.25 else 'black'
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label='Δ from baseline', shrink=0.7)
        ax.set_title('Perturbation Impact on Transcriptome State',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.show()
        return ''


# ============================================================
# 4. 工具箱函数
# ============================================================

def run_perturbation_scan(
    pert_type: str = 'oxidative_stress',
    severities: List[float] = None,
    t_span: Tuple[float, float] = (0, 200),
) -> Dict[str, Any]:
    """
    对指定扰动进行强度扫描.

    Parameters
    ----------
    pert_type : str
        扰动类型
    severities : list of float
        强度列表 (默认 [0, 0.25, 0.5, 0.75, 1.0])
    t_span : tuple
        时间范围

    Returns
    -------
    dict
        {
            'severities': list of float,
            'end_states': list of dict (get_transcriptome_state at end)
            'results': list of simulate() dict
        }
    """
    if severities is None:
        severities = [0, 0.25, 0.5, 0.75, 1.0]

    model = RNADynamicsModel()
    end_states = []
    results = []

    for sev in severities:
        res = model.simulate_perturbation(pert_type, severity=sev, t_span=t_span)
        state = model.get_transcriptome_state(res['y'][:, -1])
        end_states.append(state)
        results.append(res)

    return {
        'severities': severities,
        'end_states': end_states,
        'results': results,
    }


def quick_demo(output_dir: str = './output', dpi: int = 150) -> None:
    """
    快速演示: 基线模拟 + 三种关键扰动对比.

    Parameters
    ----------
    output_dir : str
        输出目录
    dpi : int
        图片分辨率
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model = RNADynamicsModel()

    perturb_names = [
        ('oxidative_stress', 'Oxidative_Stress'),
        ('huR_loss', 'HuR_Loss'),
        ('neat1_oe', 'NEAT1_OE'),
    ]

    for pert, label in perturb_names:
        print(f"  模拟: {label} ...")
        res = model.simulate_perturbation(pert, severity=0.5)

        # RNA landscape
        path = os.path.join(output_dir, f'rna_landscape_{label}.png')
        model.plot_rna_landscape(res, output_path=path)
        print(f"     → {path}")

        # RBP balance
        path = os.path.join(output_dir, f'rbp_balance_{label}.png')
        model.plot_rbp_balance(res, output_path=path)
        print(f"     → {path}")

        # Phase separation
        path = os.path.join(output_dir, f'phase_sep_{label}.png')
        model.plot_phase_separation(res, output_path=path)
        print(f"     → {path}")

        # 转录组状态
        final_state = model.get_transcriptome_state(res['y'][:, -1])
        print(f"     末态: { {k: round(float(v), 3) for k, v in final_state.items()} }")

    # Baseline
    print(f"  基线模拟 ...")
    res = model.simulate()
    path = os.path.join(output_dir, 'rna_landscape_baseline.png')
    model.plot_rna_landscape(res, output_path=path)
    state = model.get_transcriptome_state(res['y'][:, -1])
    print(f"     基线末态: {state}")

    # 对比热图
    print(f"  扰动对比热图 ...")
    path = os.path.join(output_dir, 'perturbation_heatmap.png')
    model.plot_comparison_heatmap(output_path=path)
    print(f"     → {path}")

    print(f"✓ RNA动力学模块演示完成, 输出至: {output_dir}/")


# ============================================================
# 5. 命令行入口
# ============================================================

if __name__ == '__main__':
    quick_demo()
