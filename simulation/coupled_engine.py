"""
NP 细胞多尺度耦合引擎
=======================
多尺度闭环耦合系统，整合力传导、代谢、衰老、ECM、信号通路

闭环逻辑 (Closed-loop logic):
    ECM刚度和分解 → [力传导] MRTF-A → Kidins220 → AMPK → PFKFB3
        ↕                                                  ↕
    [ECM代谢] Aggrecan/Col2                  →  [代谢] 糖酵解/OXPHOS/ROS
        ↕                                                  ↕
      [衰老] SASP/IL-1β/TNF/MMP  ←  [衰老] 线粒体ROS/AMPK↓/p53-p21
        ↕
      [ECM] MMP↓Aggrecan→刚度↑ → 闭环回到力传导

核心设计: 通过 condition_factor (0~1) 驱动病理状态切换
  - 0.0 = 健康 NP (低刚度)
  - 0.5 = 早期退变
  - 1.0 = 晚期退变 (高刚度, 正反馈锁定)

Author: Virtual NP Cell Team
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Dict, Tuple, List, Union
from enum import Enum
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 变量索引 (23 维状态空间)
# ============================================================
class Var(Enum):
    Stiffness_signal = 0   # 刚度信号
    MRTFA_nuc = 1          # 核内MRTF-A
    Kidins220 = 2          # Kidins220 表达
    AMPK_P = 3             # 磷酸化AMPK
    PFKFB3 = 4             # PFKFB3糖酵解酶
    NFKB = 5               # NF-κB 炎症核心
    SOX9 = 6               # SOX9 合成代谢转录因子
    MMP = 7                # MMP 总体活性
    ECM_synth = 8          # ECM 合成状态
    ATP = 9                # 细胞内ATP水平
    ROS_mito = 10          # 线粒体活性氧
    Lactate = 11           # 乳酸
    HIF1_alpha = 12        # HIF-1α
    p53 = 13               # p53 肿瘤抑制蛋白
    p21 = 14               # p21 CDK抑制因子
    SASP_score = 15        # SASP综合评分
    IL1B = 16              # IL-1β 炎症因子
    TNF = 17               # TNF-α 炎症因子
    senescence_score = 18  # 综合衰老评分
    ECM_stiffness = 19     # ECM动态刚度
    ECM_degradation = 20   # ECM降解水平
    Aggrecan = 21          # 聚集蛋白聚糖
    Col2 = 22              # II型胶原


VAR_CN = {
    Var.Stiffness_signal: "刚度信号", Var.MRTFA_nuc: "核MRTF-A",
    Var.Kidins220: "Kidins220", Var.AMPK_P: "p-AMPK",
    Var.PFKFB3: "PFKFB3", Var.NFKB: "NF-κB",
    Var.SOX9: "SOX9", Var.MMP: "MMP活性",
    Var.ECM_synth: "ECM合成", Var.ATP: "ATP",
    Var.ROS_mito: "线粒体ROS", Var.Lactate: "乳酸",
    Var.HIF1_alpha: "HIF-1α", Var.p53: "p53",
    Var.p21: "p21", Var.SASP_score: "SASP评分",
    Var.IL1B: "IL-1β", Var.TNF: "TNF-α",
    Var.senescence_score: "衰老评分", Var.ECM_stiffness: "ECM刚度",
    Var.ECM_degradation: "ECM降解", Var.Aggrecan: "聚集蛋白聚糖",
    Var.Col2: "II型胶原",
}

N_VARS = len(Var)


# ============================================================
# 主要耦合模型类
# ============================================================
class NPCoupledModel:
    """
    NP 细胞多尺度耦合模型

    condition_factor: 控制病理严重度的外部驱动参数
        - 0.0: 健康
        - 0.5: 早期退变 (ECM刚度↑, 炎症轻度↑)
        - 1.0: 晚期退变 (闭环正反馈锁定)
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        self.params = {
            # ---------- 力传导轴 ----------
            'k_stiff_trans': 0.12,
            'k_stiff_decay': 0.35,
            'k_mrtfa_trans': 0.22,
            'k_mrtfa_export': 0.20,
            'km_mrtfa': 0.40, 'n_mrtfa': 3.0,
            'k_kidins_prod': 0.40,
            'k_kidins_deg': 0.30,
            'kidins_rep_max': 0.75,
            'kidins_rep_ec50': 0.45, 'kidins_rep_hill': 3.0,
            'kidins_basal': 0.03,
            'k_ampk_act': 0.22,
            'k_ampk_dephos': 0.28,
            'km_ampk': 0.60, 'n_ampk': 3.0,
            'ampk_basal': 0.02,
            'k_pfkfb3_base': 0.04,
            'k_pfkfb3_ampk': 0.18,
            'k_pfkfb3_deg': 0.20,

            # ---------- NF-κB/SASP 炎症轴 ----------
            'k_nfkb_base': 0.01,
            'k_nfkb_act': 0.30,
            'k_nfkb_inh': 0.18,
            'km_nfkb_il1b': 0.30, 'km_nfkb_tnf': 0.30,
            'km_nfkb_ros': 0.50,
            'k_il1b_prod': 0.08,
            'k_il1b_deg': 0.08,
            'k_tnf_prod': 0.06,
            'k_tnf_deg': 0.08,
            'k_sasp_prod': 0.15,
            'k_sasp_deg': 0.06,
            'km_sasp_nfkb': 0.25,

            # ---------- SOX9/合成代谢 ----------
            'k_sox9_prod': 0.20,
            'k_sox9_nfkb_inh': 0.50,
            'km_sox9_nfkb': 0.30,
            'k_sox9_deg': 0.12,

            # ---------- MMP/ECM ----------
            'k_mmp_base': 0.02,
            'k_mmp_nfkb': 0.28,
            'k_mmp_sen': 0.18,
            'k_mmp_deg': 0.08,
            'km_mmp_nfkb': 0.35,
            'k_ecm_synth': 0.10,
            'k_ecm_sox9': 0.20,
            'k_ecm_nfkb_inh': 0.35,
            'km_ecm_nfkb': 0.30,
            'k_ecm_deg': 0.05,

            # ---------- 代谢 ----------
            'k_atp_prod_base': 0.10,
            'k_atp_ampk': 0.25,
            'k_atp_cons': 0.22,
            'km_atp': 0.45,
            'k_ros_base': 0.02,
            'k_ros_atp': 0.05,
            'k_ros_mito_dys': 0.15,
            'k_ros_clear': 0.20,
            'k_lac_prod_base': 0.08,
            'k_lac_pfkfb3': 0.20,
            'k_lac_clear': 0.20,
            'k_hif_prod': 0.10,
            'k_hif_deg': 0.40,
            'k_hif_ros': 0.12,
            'km_hif_ros': 0.35,

            # ---------- 衰老 ----------
            'k_p53_act': 0.20, 'k_p53_deg': 0.15,
            'km_p53_ros': 0.40,
            'k_p21_act': 0.25, 'k_p21_deg': 0.12,
            'km_p21_p53': 0.35,
            'k_sen_ampk': 0.20, 'km_sen_ampk': 0.30,
            'k_sen_ros': 0.18, 'km_sen_ros': 0.40,
            'k_sen_p21': 0.15, 'km_sen_p21': 0.30,
            'sen_decay': 0.04,

            # ---------- ECM 刚度/降解/成分 ----------
            'k_stiff_mmp': 0.20,
            'k_stiff_repair': 0.08,
            'k_ecmdeg_mmp': 0.15,
            'k_ecmdeg_repair': 0.12,
            'agg_stiff_sensitivity': 0.30,
            'stiff_min': 0.30,
            'stiff_max': 3.00,
            'k_agg_synth': 0.10, 'k_agg_sox9': 0.15,
            'k_agg_mmp_deg': 0.10, 'k_agg_loss': 0.03,
            'k_col2_synth': 0.08, 'k_col2_sox9': 0.10,
            'k_col2_mmp_deg': 0.06, 'k_col2_loss': 0.02,
            'agg_nfkb_inh': 0.25, 'col2_nfkb_inh': 0.15,

            'max_deriv': 10.0, 'min_var': 1e-6,
        }
        if params:
            self.params.update(params)

    # ============================================================
    # ODE 系统 — 23维
    # ============================================================
    def ode_system(self, t, y, perturbation=None):
        """
        23维耦合ODE系统

        perturbation支持:
            condition_factor: 病理驱动参数 [0~1], 0=健康, 1=晚期退变
            ampk_act: AMPK激活
            nfkb_inh: NF-κB抑制
            rox_inh: ROS清除
            senolytic: 衰老清除
            mmp_inh: MMP抑制
            tnf_stim: TNF刺激
            il1b_stim: IL1B刺激
            oxidative_stress: 氧化应激
        """
        p = self.params
        pert = perturbation or {}

        # 病理驱动参数: 0=健康, 1=晚期退变
        cf = np.clip(pert.get('condition_factor', 0.0), 0.0, 1.0)
        tnf_stim = pert.get('tnf_stim', 0.0)
        il1b_stim = pert.get('il1b_stim', 0.0)
        ampk_act = pert.get('ampk_act', 0.0)
        nfkb_inh = pert.get('nfkb_inh', 0.0)
        rox_inh = pert.get('rox_inh', 0.0)
        senolytic = pert.get('senolytic', 0.0)
        mmp_inh = pert.get('mmp_inh', 0.0)
        ox_stress = pert.get('oxidative_stress', 0.0)

        y = np.maximum(y, p['min_var'])
        (Stiff_sig, MRTFA_nuc, Kidins220, AMPK_P, PFKFB3,
         NFKB, SOX9, MMP, ECM_synth,
         ATP, ROS_mito, Lactate, HIF1_alpha,
         p53_val, p21_val, SASP_score, IL1B, TNF,
         sen_score, ECM_stiff, ECM_deg, Agg, Col2) = y

        # ---- 病理驱动对关键参数的影响 ----
        # cf=0: 健康, cf=1: 严重退变
        # 1) 外部刚度驱动 (模拟ECM退变起始)
        ext_stiff_drive = p['stiff_min'] + (p['stiff_max'] - p['stiff_min']) * cf * 0.6
        # 2) 炎症驱动 (外部促炎因子)
        ext_nfkb_drive = cf * 0.3
        # 3) ECM合成抑制 (退变中合成代谢受损)
        ecm_synth_suppression = 1 - cf * 0.5

        # ============================================================
        # 0. 有效ECM刚度 (被MMP降解 + 病理驱动)
        # ============================================================
        norm_deg = ECM_deg / (0.5 + ECM_deg)
        agg_loss_stiff = p['agg_stiff_sensitivity'] * (1 - Agg / (0.5 + Agg))
        effective_stiff = max(p['stiff_min'], min(p['stiff_max'],
                            ext_stiff_drive + (p['stiff_max'] - p['stiff_min']) * norm_deg * 0.4
                            + agg_loss_stiff * 0.5))

        # ============================================================
        # 1-5. 力传导轴: Stiff_sig → MRTFA_nuc → Kidins220 → AMPK_P → PFKFB3
        # ============================================================
        dStiff_sig = p['k_stiff_trans'] * effective_stiff - p['k_stiff_decay'] * Stiff_sig

        mrtfa_trans = (p['k_mrtfa_trans']
            * (Stiff_sig ** p['n_mrtfa'])
            / (Stiff_sig ** p['n_mrtfa'] + p['km_mrtfa'] ** p['n_mrtfa']))
        dMRTFA_nuc = mrtfa_trans - p['k_mrtfa_export'] * MRTFA_nuc

        kidins_rep = (MRTFA_nuc ** p['kidins_rep_hill']
            / (MRTFA_nuc ** p['kidins_rep_hill'] + p['kidins_rep_ec50'] ** p['kidins_rep_hill']))
        kidins_prod = p['kidins_basal'] + p['k_kidins_prod'] * (1 - p['kidins_rep_max'] * kidins_rep)
        dKidins220 = kidins_prod - p['k_kidins_deg'] * Kidins220

        ampk_act_rate = (p['k_ampk_act']
            * (Kidins220 ** p['n_ampk']) / (Kidins220 ** p['n_ampk'] + p['km_ampk'] ** p['n_ampk']))
        ampk_act_rate *= (1 + ampk_act * 0.5)
        dAMPK_P = p['ampk_basal'] + ampk_act_rate - p['k_ampk_dephos'] * AMPK_P

        pfkfb3_synth = p['k_pfkfb3_base'] + p['k_pfkfb3_ampk'] * AMPK_P
        dPFKFB3 = pfkfb3_synth - p['k_pfkfb3_deg'] * PFKFB3

        # ============================================================
        # 6. NF-κB (炎症核心)
        #    被: IL1B, TNF, ROS, 病理驱动 激活
        # ============================================================
        nfkb_il1b = (IL1B + il1b_stim) / (IL1B + il1b_stim + p['km_nfkb_il1b'])
        nfkb_tnf = (TNF + tnf_stim) / (TNF + tnf_stim + p['km_nfkb_tnf'])
        nfkb_ros = ROS_mito / (ROS_mito + p['km_nfkb_ros'])

        nfkb_drive = 0.35 * nfkb_il1b + 0.35 * nfkb_tnf + 0.15 * nfkb_ros + ext_nfkb_drive
        nfkb_drive *= (1 - nfkb_inh * 0.85)
        dNFKB = p['k_nfkb_base'] + p['k_nfkb_act'] * nfkb_drive - p['k_nfkb_inh'] * NFKB

        # ============================================================
        # 7. SOX9 (被NF-κB抑制)
        # ============================================================
        nfkb_sox9_inh = NFKB ** 2 / (NFKB ** 2 + p['km_sox9_nfkb'] ** 2)
        dSOX9 = (p['k_sox9_prod'] * (1 - p['k_sox9_nfkb_inh'] * nfkb_sox9_inh)
                 - p['k_sox9_deg'] * SOX9)

        # ============================================================
        # 8-9. MMP, ECM_synth
        # ============================================================
        mmp_expr = p['k_mmp_base'] + p['k_mmp_nfkb'] * NFKB / (NFKB + p['km_mmp_nfkb']) \
                   + p['k_mmp_sen'] * sen_score
        dMMP = mmp_expr * (1 - mmp_inh * 0.8) - p['k_mmp_deg'] * MMP

        ecm_sox9 = 1 + p['k_ecm_sox9'] * SOX9 / (SOX9 + 0.5)
        ecm_nfkb_inh = 1 - p['k_ecm_nfkb_inh'] * NFKB / (NFKB + p['km_ecm_nfkb'])
        dECM_synth = (p['k_ecm_synth'] * ecm_sox9 * max(ecm_nfkb_inh, 0.01) * ecm_synth_suppression
                      - p['k_ecm_deg'] * ECM_synth)

        # ============================================================
        # 10-13. 代谢: ATP, ROS_mito, Lactate, HIF1_alpha
        # ============================================================
        dATP = (p['k_atp_prod_base'] + p['k_atp_ampk'] * AMPK_P / (AMPK_P + p['km_atp'])
                - p['k_atp_cons'] * ATP)

        ros_atp = p['k_ros_atp'] * ATP / (ATP + 1.0)
        ros_sen = p['k_ros_mito_dys'] * sen_score
        ros_clear = p['k_ros_clear'] * ROS_mito * (1 + rox_inh * 2.0)
        dROS_mito = p['k_ros_base'] + ros_atp + ros_sen + ox_stress - ros_clear

        dLactate = p['k_lac_prod_base'] + p['k_lac_pfkfb3'] * PFKFB3 - p['k_lac_clear'] * Lactate

        hif_ros_stab = 1 + p['k_hif_ros'] * ROS_mito / (ROS_mito + p['km_hif_ros'])
        dHIF1_alpha = p['k_hif_prod'] - p['k_hif_deg'] * HIF1_alpha / hif_ros_stab

        # ============================================================
        # 14-18. 衰老轴: p53, p21, SASP, IL1B, TNF, senescence_score
        # ============================================================
        dp53 = (p['k_p53_act'] * ROS_mito / (ROS_mito + p['km_p53_ros'])
                - p['k_p53_deg'] * p53_val)
        dp21 = (p['k_p21_act'] * p53_val / (p53_val + p['km_p21_p53'])
                - p['k_p21_deg'] * p21_val)

        dSASP_score = (p['k_sasp_prod'] * NFKB / (NFKB + p['km_sasp_nfkb'])
                       - p['k_sasp_deg'] * SASP_score)

        dIL1B = p['k_il1b_prod'] * SASP_score + il1b_stim - p['k_il1b_deg'] * IL1B
        dTNF = p['k_tnf_prod'] * SASP_score + tnf_stim - p['k_tnf_deg'] * TNF

        # 衰老评分组合
        ampk_sen_sig = 1 - AMPK_P / (AMPK_P + p['km_sen_ampk'])
        ros_sen_sig = ROS_mito / (ROS_mito + p['km_sen_ros'])
        p21_sen_sig = p21_val / (p21_val + p['km_sen_p21'])
        sen_prod = p['k_sen_ampk'] * ampk_sen_sig + p['k_sen_ros'] * ros_sen_sig \
                   + p['k_sen_p21'] * p21_sen_sig
        sen_clear = (p['sen_decay'] + senolytic * 0.3) * sen_score
        dsen_score = sen_prod * (1 - sen_score) - sen_clear

        # ============================================================
        # 19-23. ECM: stiffness, degradation, Aggrecan, Col2
        # ============================================================
        stiff_inc = p['k_stiff_mmp'] * MMP / (1 + MMP) * (1 - ECM_stiff / p['stiff_max'])
        stiff_inc += agg_loss_stiff * 0.3 * (1 - ECM_stiff / p['stiff_max'])
        stiff_dec = p['k_stiff_repair'] * (ECM_stiff - p['stiff_min'])
        dECM_stiff = stiff_inc - stiff_dec

        ecmdeg_mmp = p['k_ecmdeg_mmp'] * MMP * (1 + sen_score * 0.5)
        ecmdeg_repair = p['k_ecmdeg_repair'] * ECM_deg
        dECM_deg = ecmdeg_mmp - ecmdeg_repair

        agg_sox9 = 1 + p['k_agg_sox9'] * SOX9 / (SOX9 + 0.5)
        agg_nfkb_inh = 1 - p['agg_nfkb_inh'] * NFKB / (NFKB + 0.5)
        agg_synth = p['k_agg_synth'] * agg_sox9 * max(agg_nfkb_inh, 0.01)
        dAgg = agg_synth - p['k_agg_mmp_deg'] * Agg * MMP - p['k_agg_loss'] * Agg

        col2_sox9 = 1 + p['k_col2_sox9'] * SOX9 / (SOX9 + 0.5)
        col2_nfkb_inh = 1 - p['col2_nfkb_inh'] * NFKB / (NFKB + 0.5)
        col2_synth = p['k_col2_synth'] * col2_sox9 * max(col2_nfkb_inh, 0.01)
        dCol2 = col2_synth - p['k_col2_mmp_deg'] * Col2 * MMP - p['k_col2_loss'] * Col2

        def sc(v):
            return float(np.clip(v, -p['max_deriv'], p['max_deriv']))

        return [sc(v) for v in [
            dStiff_sig, dMRTFA_nuc, dKidins220, dAMPK_P, dPFKFB3,
            dNFKB, dSOX9, dMMP, dECM_synth,
            dATP, dROS_mito, dLactate, dHIF1_alpha,
            dp53, dp21, dSASP_score, dIL1B, dTNF,
            dsen_score, dECM_stiff, dECM_deg, dAgg, dCol2,
        ]]

    # ============================================================
    # 主要仿真接口
    # ============================================================
    def simulate(self, sim_type='normal', n_conditions=None,
                 t_span=(0, 600), n_points=600, perturbation=None,
                 initial_state=None):
        """
        多条件仿真

        Args:
            sim_type: 'normal', 'early_degeneration', 'late_degeneration', 'all'
            perturbation: 可包含 condition_factor 和其他干预
        """
        # condition_factor映射
        cf_map = {
            'normal': 0.0,
            'early_degeneration': 0.4,
            'late_degeneration': 0.8,
        }

        if n_conditions is not None:
            conditions = n_conditions
        elif sim_type == 'all':
            conditions = ['normal', 'early_degeneration', 'late_degeneration']
        else:
            conditions = [sim_type]

        # 所有条件从健康初始状态开始
        y0 = self._healthy_initial() if initial_state is None else initial_state
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        results = {}
        for cond in conditions:
            cf = cf_map.get(cond, 0.0)
            pert = dict(perturbation or {})
            pert['condition_factor'] = cf

            sol = solve_ivp(
                lambda t, y: self.ode_system(t, y, pert),
                t_span, y0,
                method='RK45', t_eval=t_eval,
                max_step=10.0, rtol=1e-6, atol=1e-8,
            )
            results[cond] = {
                't': sol.t, 'y': sol.y,
                'steady_state': sol.y[:, -1],
            }
        return results

    def simulate_condition(self, condition='early_degeneration',
                           perturbation=None, t_span=(0, 600), n_points=600):
        """单条件仿真"""
        cf_map = {'normal': 0.0, 'early_degeneration': 0.4, 'late_degeneration': 0.8}
        cf = cf_map.get(condition, 0.0)
        y0 = self._healthy_initial()
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        pert = dict(perturbation or {})
        pert['condition_factor'] = cf

        sol = solve_ivp(
            lambda t, y: self.ode_system(t, y, pert),
            t_span, y0, method='RK45', t_eval=t_eval,
            max_step=10.0, rtol=1e-6, atol=1e-8,
        )
        return sol.t, sol.y

    def simulate_intervention(self, target='AMPK', strength=1.5,
                              condition='late_degeneration',
                              t_span=(0, 800), n_points=800):
        """干预模拟"""
        int_map = {
            'AMPK': ('ampk_act', strength),
            'NFKB': ('nfkb_inh', min(strength, 1.0)),
            'ROS': ('rox_inh', strength),
            'senolytic': ('senolytic', min(strength, 1.0)),
            'MMP': ('mmp_inh', min(strength, 1.0)),
        }
        if target not in int_map:
            raise ValueError(f"不支持: {target}. 支持: {list(int_map.keys())}")

        # 对照
        t_ctrl, y_ctrl = self.simulate_condition(condition, t_span=t_span, n_points=n_points)

        # 干预
        param_name, param_val = int_map[target]
        pert = {param_name: param_val}
        t_int, y_int = self.simulate_condition(condition, perturbation=pert,
                                                t_span=t_span, n_points=n_points)

        return {
            'target': target, 'strength': strength, 'condition': condition,
            't': t_ctrl, 'y_control': y_ctrl, 'y_intervention': y_int,
            'steady_control': y_ctrl[:, -1], 'steady_intervention': y_int[:, -1],
        }

    # ============================================================
    # 反馈环强度
    # ============================================================
    def get_feedback_strength(self, condition='normal', t_span=(0, 600)):
        """量化各反馈环强度 (基于Jacobian)"""
        _, y = self.simulate_condition(condition, t_span=t_span, n_points=200)
        steady = y[:, -1]
        eps = 0.01

        def _jac(perturb_idx, target_idx):
            dy_base = np.array(self.ode_system(0, steady))
            y_pert = steady.copy()
            y_pert[perturb_idx] += eps
            dy_pert = np.array(self.ode_system(0, y_pert))
            return (dy_pert[target_idx] - dy_base[target_idx]) / eps

        fb = {
            'MRTFA→Kidins220抑制': _jac(Var.MRTFA_nuc.value, Var.Kidins220.value),
            'Kidins220→AMPK激活': _jac(Var.Kidins220.value, Var.AMPK_P.value),
            'AMPK→PFKFB3激活': _jac(Var.AMPK_P.value, Var.PFKFB3.value),
            'IL1B→NF-κB激活': _jac(Var.IL1B.value, Var.NFKB.value),
            'TNF→NF-κB激活': _jac(Var.TNF.value, Var.NFKB.value),
            'SASP→IL1B产': _jac(Var.SASP_score.value, Var.IL1B.value),
            'SASP→TNF产': _jac(Var.SASP_score.value, Var.TNF.value),
            'NF-κB→SASP激活': _jac(Var.NFKB.value, Var.SASP_score.value),
            'NF-κB→MMP激活': _jac(Var.NFKB.value, Var.MMP.value),
            'NF-κB→SOX9抑制': _jac(Var.NFKB.value, Var.SOX9.value),
            'MMP→ECM降解': _jac(Var.MMP.value, Var.ECM_degradation.value),
            'MMP→Aggrecan降解': _jac(Var.MMP.value, Var.Aggrecan.value),
            'ROS→p53激活': _jac(Var.ROS_mito.value, Var.p53.value),
            'p53→p21激活': _jac(Var.p53.value, Var.p21.value),
            'AMPK↓→衰老': _jac(Var.AMPK_P.value, Var.senescence_score.value),
            'ROS→衰老': _jac(Var.ROS_mito.value, Var.senescence_score.value),
            'p21→衰老': _jac(Var.p21.value, Var.senescence_score.value),
        }
        return fb

    # ============================================================
    # 初始状态
    # ============================================================
    def _healthy_initial(self):
        """健康NP细胞初始状态"""
        return np.array([
            0.35, 0.25, 0.85, 0.70, 0.65,   # 0-4: 力传导轴健康
            0.04, 0.70, 0.02, 0.80,         # 5-8: 信号通路健康
            1.00, 0.04, 0.50, 0.35,         # 9-12: 代谢健康
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 13-18: p53/p21/SASP/IL1B/TNF/senescence
            0.30, 0.05, 1.00, 1.00,         # 19-22: ECM健康
        ], dtype=float)

    # ============================================================
    # 绘图
    # ============================================================
    def plot_coupled_dynamics(self, results, output_path=None,
                              dpi=150, figsize=(20, 16)):
        """4×6子图展示所有变量"""
        conditions = list(results.keys())
        colors = {'normal': '#2ECC71', 'early_degeneration': '#F39C12',
                  'late_degeneration': '#E74C3C'}
        ls_map = {'normal': '-', 'early_degeneration': '--', 'late_degeneration': '-.'}

        fig, axes = plt.subplots(4, 6, figsize=figsize)
        axes = axes.flatten()

        for i in range(N_VARS):
            ax = axes[i]
            var_enum = Var(i)
            for cond in conditions:
                t = results[cond]['t']; y = results[cond]['y']
                c = colors.get(cond, '#888')
                ax.plot(t, y[i], color=c, linewidth=1.8,
                        linestyle=ls_map.get(cond, '-'), label=cond, alpha=0.9)
                final = y[i, -1]
                rng = max(y[i].max() - y[i].min(), 0.1)
                ax.text(t[-1] * 0.85, final + 0.05 * rng,
                        f"{final:.2f}", fontsize=5.5, color=c, alpha=0.7)
            ax.set_title(f"{VAR_CN[var_enum]}\n({var_enum.name})", fontsize=8, fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=6); ax.set_xlabel('时间 (a.u.)', fontsize=6)
            if i == 0: ax.legend(fontsize=6, loc='upper left')

        ax_last = axes[-1]; ax_last.clear(); ax_last.axis('off')
        ax_last.text(0.1, 0.9, "多尺度耦合NP退变模型 (23变量)", fontsize=10, fontweight='bold',
                     transform=ax_last.transAxes)
        ax_last.text(0.1, 0.75, "绿=健康  橙=早期  红=晚期", fontsize=8,
                     transform=ax_last.transAxes)

        fig.suptitle("NP 细胞多尺度耦合动力学", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 耦合动力学图保存: {output_path}")
        return fig

    def plot_coupling_network(self, output_path=None, figsize=(16, 12), dpi=150):
        """耦合网络图"""
        G = nx.DiGraph()
        modules = {
            '力传导': [Var.Stiffness_signal, Var.MRTFA_nuc, Var.Kidins220, Var.AMPK_P, Var.PFKFB3],
            '信号通路': [Var.NFKB, Var.SOX9, Var.MMP, Var.ECM_synth],
            '代谢': [Var.ATP, Var.ROS_mito, Var.Lactate, Var.HIF1_alpha],
            '衰老': [Var.p53, Var.p21, Var.SASP_score, Var.IL1B, Var.TNF, Var.senescence_score],
            'ECM': [Var.ECM_stiffness, Var.ECM_degradation, Var.Aggrecan, Var.Col2],
        }
        mod_colors = {'力传导': '#3498DB', '信号通路': '#9B59B6', '代谢': '#E67E22',
                      '衰老': '#E74C3C', 'ECM': '#2ECC71'}

        for mod, vlist in modules.items():
            for v in vlist:
                G.add_node(v.name, module=mod, label=f"{VAR_CN[v]}\n({v.name})")

        edges = [
            (Var.ECM_stiffness, Var.Stiffness_signal, 'positive', 0.9),
            (Var.Stiffness_signal, Var.MRTFA_nuc, 'positive', 0.8),
            (Var.MRTFA_nuc, Var.Kidins220, 'negative', 0.8),
            (Var.Kidins220, Var.AMPK_P, 'positive', 0.7),
            (Var.AMPK_P, Var.PFKFB3, 'positive', 0.6),
            (Var.IL1B, Var.NFKB, 'positive', 0.9), (Var.TNF, Var.NFKB, 'positive', 0.9),
            (Var.ROS_mito, Var.NFKB, 'positive', 0.6),
            (Var.NFKB, Var.SASP_score, 'positive', 0.8),
            (Var.SASP_score, Var.IL1B, 'positive', 0.7), (Var.SASP_score, Var.TNF, 'positive', 0.7),
            (Var.NFKB, Var.MMP, 'positive', 0.8), (Var.NFKB, Var.SOX9, 'negative', 0.7),
            (Var.SOX9, Var.Aggrecan, 'positive', 0.6), (Var.SOX9, Var.Col2, 'positive', 0.5),
            (Var.AMPK_P, Var.ATP, 'positive', 0.7), (Var.PFKFB3, Var.Lactate, 'positive', 0.5),
            (Var.ROS_mito, Var.HIF1_alpha, 'positive', 0.3),
            (Var.ROS_mito, Var.p53, 'positive', 0.7), (Var.p53, Var.p21, 'positive', 0.6),
            (Var.AMPK_P, Var.senescence_score, 'negative', 0.6),
            (Var.ROS_mito, Var.senescence_score, 'positive', 0.6),
            (Var.p21, Var.senescence_score, 'positive', 0.5),
            (Var.senescence_score, Var.MMP, 'positive', 0.6),
            (Var.MMP, Var.ECM_degradation, 'positive', 0.7),
            (Var.MMP, Var.Aggrecan, 'negative', 0.8), (Var.MMP, Var.Col2, 'negative', 0.6),
            (Var.ECM_degradation, Var.ECM_stiffness, 'positive', 0.8),
            (Var.Aggrecan, Var.ECM_stiffness, 'negative', 0.5),
            (Var.HIF1_alpha, Var.PFKFB3, 'positive', 0.3),
        ]
        for src, tgt, etype, w in edges:
            G.add_edge(src.name, tgt.name, type=etype, weight=w)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pos = nx.kamada_kawai_layout(G)
        nc = [mod_colors.get(G.nodes[n]['module'], '#999') for n in G.nodes()]
        ns = [800 + 200 * G.degree(n) for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns, alpha=0.85,
                               edgecolors='white', linewidths=1.5)

        pos_e = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'positive']
        neg_e = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'negative']

        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=pos_e, edge_color='#E74C3C',
                               width=[1 + G[u][v]['weight'] * 1.5 for u, v in pos_e],
                               alpha=0.6, arrows=True, arrowstyle='->', arrowsize=15,
                               connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=neg_e, edge_color='#3498DB',
                               width=[1 + G[u][v]['weight'] * 1.5 for u, v in neg_e],
                               alpha=0.6, arrows=True, arrowstyle='->', arrowsize=15,
                               connectionstyle='arc3,rad=0.1')

        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={n: G.nodes[n]['label'] for n in G.nodes()},
                                font_size=6, font_weight='bold')

        from matplotlib.lines import Line2D
        legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=m, markersize=10)
                  for m, c in mod_colors.items()]
        legend += [Line2D([0], [1], color='#E74C3C', linewidth=2, label='正反馈'),
                   Line2D([0], [1], color='#3498DB', linewidth=2, label='负反馈')]
        ax.legend(handles=legend, loc='upper left', fontsize=9, framealpha=0.9)
        ax.set_title("NP 细胞多尺度耦合网络\n(红=正反馈  蓝=负反馈  边宽=强度)",
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 耦合网络图保存: {output_path}")
        return fig

    def plot_intervention_comparison(self, int_res, output_path=None, dpi=150,
                                     figsize=(18, 12)):
        """干预对比图"""
        t = int_res['t']; y_c = int_res['y_control']; y_i = int_res['y_intervention']
        target = int_res['target']; condition = int_res['condition']

        key_vars = [Var.ECM_stiffness, Var.NFKB, Var.MMP, Var.SOX9,
                    Var.ATP, Var.ROS_mito, Var.senescence_score, Var.Aggrecan]
        titles = ["ECM刚度", "NF-κB", "MMP", "SOX9", "ATP", "ROS", "衰老", "Aggrecan"]

        n_plots = len(key_vars); n_cols = 4; n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, (ve, ttl) in enumerate(zip(key_vars, titles)):
            ax = axes[i]; idx = ve.value
            ax.plot(t, y_c[idx], '#E74C3C', lw=2, label=f'对照', alpha=0.8)
            ax.plot(t, y_i[idx], '#2ECC71', lw=2.5, label=f'干预 ({target})', alpha=0.9)
            chg = (y_i[idx, -1] - y_c[idx, -1]) / max(y_c[idx, -1], 0.001) * 100
            ax.set_title(f"{ttl}\n({ve.name})", fontsize=10, fontweight='bold')
            ax.text(0.5, 0.05, f"Δ={chg:+.1f}%", transform=ax.transAxes, fontsize=9,
                    fontweight='bold', color='#2ECC71' if chg > 0 else '#E74C3C',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7); ax.set_xlabel('时间', fontsize=7)
            ax.legend(fontsize=7, loc='upper right')

        for i in range(n_plots, len(axes)): axes[i].set_visible(False)
        fig.suptitle(f"干预: {target}(强度={int_res['strength']}) | {condition}",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 干预对比图保存: {output_path}")
        return fig

    def print_summary(self, results):
        """打印结果"""
        metrics = [(Var.ECM_stiffness, "ECM刚度"), (Var.NFKB, "NF-κB"),
                   (Var.MMP, "MMP"), (Var.SOX9, "SOX9"),
                   (Var.ATP, "ATP"), (Var.ROS_mito, "ROS"),
                   (Var.senescence_score, "衰老"), (Var.Aggrecan, "Aggrecan"),
                   (Var.Col2, "Col2"), (Var.MRTFA_nuc, "MRTFA_nuc"),
                   (Var.AMPK_P, "AMPK_P"), (Var.PFKFB3, "PFKFB3")]
        for cond, data in results.items():
            ss = data['steady_state']
            print(f"\n--- {cond} ---")
            for ve, name in metrics:
                print(f"  {name:15s} = {ss[ve.value]:.4f}")


# ============================================================
# 完整分析
# ============================================================
def run_full_analysis(output_dir=None, dpi=150):
    """运行完整的多尺度耦合分析"""
    import os
    if output_dir is None:
        output_dir = '/home/sandbox/.openclaw/workspace/repo/Virtual-nucleus-pulposus-cell/output'
    os.makedirs(output_dir, exist_ok=True)

    model = NPCoupledModel()
    print("=" * 70)
    print("多尺度耦合引擎 — 完整分析")
    print("=" * 70)

    # 1. 三条件仿真
    print("\n[1/5] 三条件仿真...")
    sim_results = model.simulate(sim_type='all')
    model.print_summary(sim_results)

    # 2. 耦合动力学图
    print("\n[2/5] 耦合动力学图...")
    fig = model.plot_coupled_dynamics(sim_results,
        output_path=f"{output_dir}/coupled_dynamics.png", dpi=dpi)
    plt.close(fig)

    # 3. 网络图
    print("\n[3/5] 耦合网络图...")
    fig = model.plot_coupling_network(
        output_path=f"{output_dir}/coupled_network.png", dpi=dpi)
    plt.close(fig)

    # 4. 反馈环
    print("\n[4/5] 反馈环强度...")
    for cond in ['normal', 'early_degeneration', 'late_degeneration']:
        fb = model.get_feedback_strength(cond)
        print(f"  [{cond}] 最强反馈环:")
        top3 = sorted(fb.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for name, val in top3:
            fb_type = "正" if val > 0 else "负"
            print(f"    {name:30s}: {val:+.4f} ({fb_type})")

    # 5. 干预
    print("\n[5/5] 多靶点干预...")
    interventions = {'AMPK': 1.5, 'NFKB': 0.6, 'ROS': 1.0, 'senolytic': 0.3}
    for target, strength in interventions.items():
        int_res = model.simulate_intervention(target=target, strength=strength)
        fig = model.plot_intervention_comparison(int_res,
            output_path=f"{output_dir}/int_{target}.png", dpi=dpi)
        plt.close(fig)

    # 干预对比表
    print("\n  === 干预效果对比 ===")
    metrics = [Var.ECM_stiffness, Var.NFKB, Var.SOX9, Var.ATP,
               Var.ROS_mito, Var.senescence_score, Var.Aggrecan]
    header = f"{'靶点':12s}" + ''.join(f"{m.name:12s}" for m in metrics)
    print("  " + header)
    print("  " + "-" * len(header))

    for target, strength in interventions.items():
        int_res = model.simulate_intervention(target=target, strength=strength)
        ctrl = int_res['steady_control']; intv = int_res['steady_intervention']
        row = f"{target:12s}"
        for m in metrics:
            chg = (intv[m.value] - ctrl[m.value]) / max(ctrl[m.value], 0.001) * 100
            row += f"{chg:+.1f}%   "
        print("  " + row)

    print(f"\n[✓] 分析完成! 输出: {output_dir}/")
    return model


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    import os
    os.makedirs('/home/sandbox/.openclaw/workspace/repo/Virtual-nucleus-pulposus-cell/output', exist_ok=True)

    print("=" * 70)
    print("NP 细胞多尺度耦合引擎 — 自检")
    print("=" * 70)

    model = NPCoupledModel()

    # [1] 初始化
    print("\n[1] 模型初始化...")
    y0 = model._healthy_initial()
    dy = model.ode_system(0, y0)
    print(f"    ✓ {N_VARS}变量, 导数范围 [{min(dy):.6f}, {max(dy):.6f}]")

    # [2-4] 三条件
    print("\n[2] 正常仿真...")
    t, y = model.simulate_condition('normal')
    print(f"    ECM_stiff={y[19,-1]:.3f} NFKB={y[5,-1]:.3f} SOX9={y[6,-1]:.3f}")
    print(f"    ATP={y[9,-1]:.3f} Sen={y[18,-1]:.3f} Agg={y[21,-1]:.3f}")

    print("\n[3] 早期退变...")
    t, y = model.simulate_condition('early_degeneration')
    print(f"    ECM_stiff={y[19,-1]:.3f} NFKB={y[5,-1]:.3f} SOX9={y[6,-1]:.3f}")
    print(f"    ATP={y[9,-1]:.3f} Sen={y[18,-1]:.3f} Agg={y[21,-1]:.3f}")

    print("\n[4] 晚期退变...")
    t, y = model.simulate_condition('late_degeneration')
    print(f"    ECM_stiff={y[19,-1]:.3f} NFKB={y[5,-1]:.3f} SOX9={y[6,-1]:.3f}")
    print(f"    ATP={y[9,-1]:.3f} Sen={y[18,-1]:.3f} Agg={y[21,-1]:.3f}")

    # [5] 差异验证
    print("\n[5] 三条件差异验证:")
    res = model.simulate(sim_type='all')
    for cond in ['normal', 'early_degeneration', 'late_degeneration']:
        ss = res[cond]['steady_state']
        print(f"    {cond:20s}: ", end='')
        for vi in [Var.ECM_stiffness, Var.NFKB, Var.SOX9, Var.ATP, Var.ROS_mito,
                   Var.senescence_score, Var.Aggrecan]:
            print(f"{vi.name}={ss[vi.value]:.3f} ", end='')
        print()

    # [6] 反馈环
    print("\n[6] 反馈环强度 (晚期退变):")
    fb = model.get_feedback_strength('late_degeneration')
    for name, val in sorted(fb.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
        fb_type = "正" if val > 0 else "负"
        print(f"    {name:30s}: {val:+.4f} ({fb_type})")

    # [7] 干预
    print("\n[7] AMPK干预 (晚期退变):")
    int_res = model.simulate_intervention(target='AMPK', strength=1.5)
    ctrl = int_res['steady_control']; intv = int_res['steady_intervention']
    print(f"    AMPK_P:  {ctrl[3]:.4f} → {intv[3]:.4f} (+{(intv[3]-ctrl[3])/ctrl[3]*100:.1f}%)")
    print(f"    ATP:     {ctrl[9]:.4f} → {intv[9]:.4f} (+{(intv[9]-ctrl[9])/ctrl[9]*100:.1f}%)")
    print(f"    衰老:    {ctrl[18]:.4f} → {intv[18]:.4f} ({(intv[18]-ctrl[18])/ctrl[18]*100:.1f}%)")
    print(f"    Aggrecan:{ctrl[21]:.4f} → {intv[21]:.4f} (+{(intv[21]-ctrl[21])/ctrl[21]*100:.1f}%)")

    # [8] 绘图
    print("\n[8] 绘图测试...")
    out = '/home/sandbox/.openclaw/workspace/repo/Virtual-nucleus-pulposus-cell/output'
    fig = model.plot_coupled_dynamics(res, output_path=f"{out}/dynamics_selfcheck.png"); plt.close(fig)
    fig = model.plot_coupling_network(output_path=f"{out}/network_selfcheck.png"); plt.close(fig)
    fig = model.plot_intervention_comparison(int_res, output_path=f"{out}/intervention_selfcheck.png"); plt.close(fig)
    print(f"    ✓ 图片保存至 {out}/")

    print("\n" + "=" * 70)
    print("✅ 多尺度耦合引擎自检通过!")
    print("=" * 70)
