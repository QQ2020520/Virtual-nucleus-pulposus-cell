"""
NP 细胞衰老多维度 ODE 模型
=================================
建模 NP 细胞衰老的核心分子机制，包括：

1. p53-p21-Rb 和 p16-Rb 两条平行通路驱动细胞周期停滞
2. SASP (Senescence-Associated Secretory Phenotype): NF-κB/CEBPβ 驱动的分泌表型
3. 线粒体功能障碍: Δψm↓, mtROS↑, ATP↓, 碎片化, 自噬受损
4. SASP 正反馈: SASP 因子 → NF-κB → 更多 SASP → 加速衰老
5. 氧化应激: Nox4↑ → ROS↑ → DNA 损伤 (DDR) → p53-p21
6. TonEBP 调控促炎基因

Author: Virtual NP Cell Team
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Union

# 中文字体配置
plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NPSenescenceModel:
    """
    NP 细胞衰老多维度 ODE 模型

    核心变量 (17 维状态空间):
        DNA_damage:          DNA 损伤水平 (DDR 信号强度)
        p53:                 p53 肿瘤抑制蛋白水平
        p21:                 p21^WAF1/CIP1 CDK 抑制因子
        p16:                 p16^INK4a CDK 抑制因子
        Rb_P:                磷酸化 Rb 水平 (高 → E2F 释放 → 细胞周期推进)
        E2F_active:          游离活性 E2F 转录因子
        Cell_cycle_arrest:   细胞周期停滞状态 [0,1]
        SASP_score:          SASP 综合分泌评分
        IL1B:                IL-1β 炎症因子
        IL6:                 IL-6 炎症因子
        TNF:                 TNF-α 炎症因子
        MMP_senescence:      衰老相关基质降解酶 (MMP3/MMP13)
        NFkB_activity:       NF-κB 转录活性
        ROS_cellular:        细胞内 ROS 水平
        Mitochondrial_dysfunction: 线粒体功能障碍程度 [0,1]
        Nox4:                NADPH 氧化酶 4 水平
        Apoptosis:           细胞凋亡水平
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        初始化衰老模型

        Args:
            params: 自定义参数字典 (覆盖默认参数)
        """
        # ============ 默认参数 ============
        self.params = {
            # --- DNA 损伤与修复 ---
            'k_dna_damage': 0.05,        # ROS→DNA 损伤速率
            'k_dna_repair': 0.15,        # DNA 修复速率
            'dna_damage_threshold': 0.5, # p53 激活半饱和常数

            # --- p53-p21 通路 ---
            'k_p53_act': 0.20,           # p53 最大激活速率 (由 DDR 驱动)
            'k_p53_deg': 0.10,           # p53 降解 (MDM2 介导)
            'p53_threshold': 0.3,        # p53→p21 半饱和常数
            'k_p21_act': 0.25,           # p21 最大转录速率 (由 p53 驱动)
            'k_p21_deg': 0.08,           # p21 降解

            # --- p16 通路 ---
            'k_p16_base': 0.002,         # p16 基础表达 (年龄相关)
            'k_p16_stress': 0.15,        # p16 应激诱导 (ROS/损伤)
            'k_p16_deg': 0.05,           # p16 降解
            'p16_stress_threshold': 0.4, # ROS→p16 半饱和常数

            # --- Rb 磷酸化 ---
            'k_rb_phos': 0.30,           # Rb 磷酸化速率 (CDK4/6-CyclinD, CDK2-CyclinE)
            'k_rb_dephos': 0.08,         # Rb 去磷酸化速率 (PP1/PP2A)
            'p21_inh_rbphos': 0.6,       # p21 对 Rb 磷酸化的抑制强度
            'p16_inh_rbphos': 0.7,       # p16 对 Rb 磷酸化的抑制强度

            # --- E2F ---
            'k_e2f_release': 0.25,       # Rb_P→E2F 释放速率
            'k_e2f_inhibit': 0.12,       # E2F 失活/结合速率
            'e2f_threshold': 0.3,        # Rb_P→E2F 半饱和常数

            # --- 细胞周期停滞 ---
            'k_arrest_activate': 0.10,   # 细胞周期停滞激活
            'k_arrest_release': 0.02,    # 细胞周期停滞释放
            'arrest_e2f_threshold': 0.3, # E2F 低→停滞半饱和常数

            # --- SASP ---
            'k_sasp_prod': 0.15,         # SASP 最大产生速率 (NF-κB 驱动)
            'k_sasp_deg': 0.04,          # SASP 衰减
            'sasp_nfkb_threshold': 0.3,  # NF-κB→SASP 半饱和常数

            # --- SASP 组分 ---
            'k_il1b_prod': 0.08,         # IL-1β 产生 (SASP 驱动)
            'k_il1b_deg': 0.06,          # IL-1β 降解
            'k_il6_prod': 0.10,          # IL-6 产生
            'k_il6_deg': 0.05,           # IL-6 降解
            'k_tnf_prod': 0.07,          # TNF-α 产生
            'k_tnf_deg': 0.06,           # TNF-α 降解

            # --- MMPs ---
            'k_mmp_sen_prod': 0.12,      # 衰老相关 MMP 产生 (NF-κB + SASP)
            'k_mmp_sen_deg': 0.04,       # MMP 降解

            # --- NF-κB ---
            'k_nfkb_act': 0.18,          # NF-κB 激活 (IL-1β, TNF-α, ROS, 应力)
            'k_nfkb_inh': 0.10,          # NF-κB 抑制 (IκBα)
            'nfkb_il1b_threshold': 0.2,  # IL-1β→NF-κB 半饱和常数
            'nfkb_tnf_threshold': 0.2,   # TNF-α→NF-κB 半饱和常数
            'nfkb_ros_threshold': 0.4,   # ROS→NF-κB 半饱和常数

            # --- ROS 与 Nox4 ---
            'k_ros_prod_nox4': 0.08,     # Nox4→ROS 产生速率
            'k_ros_prod_mito': 0.15,     # 线粒体→ROS 泄漏
            'k_ros_clear': 0.06,         # ROS 清除 (SOD, 谷胱甘肽)
            'k_nox4_base': 0.005,        # Nox4 基础表达
            'k_nox4_stress': 0.12,       # ROS 诱导 Nox4 正反馈
            'k_nox4_tnf': 0.08,          # TNF-α 诱导 Nox4
            'k_nox4_deg': 0.04,          # Nox4 降解
            'nox4_stress_threshold': 0.3,# ROS→Nox4 半饱和常数
            'nox4_tnf_threshold': 0.3,   # TNF→Nox4 半饱和常数
            'nox4_prod_threshold': 0.3,  # Nox4→ROS 半饱和常数

            # --- 线粒体功能障碍 ---
            'k_mito_dys': 0.08,          # ROS→线粒体损伤速率
            'k_mito_repair': 0.03,       # 线粒体自噬修复 (PINK1/Parkin)
            'k_mito_ros_inh': 0.3,       # ROS 对线粒体自噬的抑制
            'mito_ros_threshold': 0.4,   # ROS→线粒体损伤半饱和常数

            # --- 凋亡 ---
            'k_apop_mito': 0.04,         # 线粒体功能障碍→凋亡
            'k_apop_dna': 0.03,          # DNA 损伤→凋亡 (不可修复)
            'k_apop_clear': 0.01,        # 凋亡清除
            'apop_threshold': 0.6,       # 凋亡激活阈值
            'p53_apop_threshold': 0.8,   # p53→凋亡切换阈值 (高水平)

            # --- TonEBP 调控 (退变相关) ---
            'k_tonebp_ccL2': 0.05,       # TonEBP→CCL2/促炎调控
            'tonebp_stress_threshold': 0.3,

            # --- 数值安全 ---
            'max_var': 15.0,             # 变量上限
        }

        # 用户自定义参数覆盖
        if params is not None:
            self.params.update(params)

        # 变量名称与索引映射
        self.var_names = [
            'DNA_damage',
            'p53',
            'p21',
            'p16',
            'Rb_P',
            'E2F_active',
            'Cell_cycle_arrest',
            'SASP_score',
            'IL1B',
            'IL6',
            'TNF',
            'MMP_senescence',
            'NFkB_activity',
            'ROS_cellular',
            'Mitochondrial_dysfunction',
            'Nox4',
            'Apoptosis',
        ]

        self.var_indices = {name: i for i, name in enumerate(self.var_names)}
        self.n_vars = len(self.var_names)

    # ====================================================================
    # ODE 系统
    # ====================================================================

    def _ode_system(
        self,
        t: float,
        y: np.ndarray,
        stress_level: float = 1.0,
        perturbation: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """
        ODE 系统: 17 维衰老动力学

        Args:
            t: 当前时间
            y: 状态向量 [17,]
            stress_level: 基线应力倍率 (默认 1.0)
            perturbation: 扰动字典, 支持:
                'oxidative_stress': 氧化应激强度 [0,~]
                'TNF_stim':         TNF-α 刺激 [0,~]
                'DNA_damage':       DNA 损伤诱导 [0,~]
                'senolytic':        衰老细胞清除强度 [0,~]
                'Nox4_KD':          Nox4 敲低 [0,1] (1=完全敲低)

        Returns:
            dy/dt 列表 [17,]
        """
        p = self.params
        s = stress_level

        # 解析扰动
        perturb = perturbation or {}
        ox_stress = perturb.get('oxidative_stress', 0.0)
        tnf_stim = perturb.get('TNF_stim', 0.0)
        dna_damage_ind = perturb.get('DNA_damage', 0.0)
        senolytic = perturb.get('senolytic', 0.0)
        nox4_kd = perturb.get('Nox4_KD', 0.0)  # 0=正常, 1=完全敲低

        # 解包状态变量
        (DNA_damage, p53_val, p21_val, p16_val, Rb_P,
         E2F_active, Cell_cycle_arrest, SASP_score,
         IL1B_val, IL6_val, TNF_val, MMP_senescence,
         NFkB_activity, ROS_cellular, Mito_dys, Nox4_val,
         Apoptosis) = y

        # ==================================================================
        # 1. DNA 损伤动态 (DDR)
        #    来源: ROS 氧化损伤 + 外部 DNA 损伤诱导
        #    修复: 基础修复机制
        # ==================================================================
        ros_damage = p['k_dna_damage'] * ROS_cellular / (1 + ROS_cellular)
        dDNA_damage = (ros_damage * s + dna_damage_ind
                       - p['k_dna_repair'] * DNA_damage)

        # ==================================================================
        # 2. p53 动态 (核心衰老调控枢纽)
        #    激活: DNA 损伤 (ATM/ATR/CHK1/CHK2 级联)
        #    降解: MDM2 介导 (p53 自身诱导 MDM2 → 负反馈)
        # ==================================================================
        p53_activation = (p['k_p53_act']
                          * DNA_damage
                          / (DNA_damage + p['dna_damage_threshold']))
        # p53 可诱导 MDM2 → 自身负反馈 (简化: 用饱和项表示)
        p53_mdm2_deg = p['k_p53_deg'] * p53_val * (1 + 0.5 * p53_val / (1 + p53_val))
        dp53 = p53_activation - p53_mdm2_deg

        # ==================================================================
        # 3. p21 动态 (p53 下游, CDK 通用抑制因子)
        #    p53 转录激活 p21
        # ==================================================================
        p21_activation = (p['k_p21_act']
                          * p53_val
                          / (p53_val + p['p53_threshold']))
        dp21 = p21_activation - p['k_p21_deg'] * p21_val

        # ==================================================================
        # 4. p16 动态 (INK4a 家族, 选择性抑制 CDK4/6)
        #    基础表达 (年龄相关积累) + 应激诱导 (ROS/损伤)
        # ==================================================================
        p16_base = p['k_p16_base']
        p16_stress = (p['k_p16_stress']
                      * ROS_cellular
                      / (ROS_cellular + p['p16_stress_threshold']))
        dp16 = (p16_base + p16_stress - p['k_p16_deg'] * p16_val)

        # ==================================================================
        # 5. Rb 磷酸化动态 (细胞周期门控)
        #    Rb_P = 磷酸化 Rb (高 → E2F 释放 → 细胞周期推进)
        #    CDK4/6-CyclinD 和 CDK2-CyclinE 磷酸化 Rb
        #    p16 → CDK4/6 抑制; p21 → CDK2/4/6 通用抑制
        # ==================================================================
        # CDK 活性受 p21 和 p16 抑制
        cdk_activity = (1
                        - p['p21_inh_rbphos'] * p21_val / (1 + p21_val)
                        - p['p16_inh_rbphos'] * p16_val / (1 + p16_val))
        cdk_activity = max(0.01, cdk_activity)  # 保持最小活性

        rb_phosphorylation = p['k_rb_phos'] * cdk_activity
        rb_dephosphorylation = p['k_rb_dephos'] * Rb_P
        dRb_P = (rb_phosphorylation * (1 - Rb_P)   # 非磷酸化 Rb → 磷酸化
                 - rb_dephosphorylation)             # 磷酸化 Rb → 去磷酸化

        # ==================================================================
        # 6. E2F 活性 (细胞周期推进转录因子)
        #    高 Rb_P → E2F 释放 → 活性 E2F
        #    低 Rb_P → E2F 被 Rb 结合 → 失活
        # ==================================================================
        e2f_release = (p['k_e2f_release']
                       * Rb_P
                       / (Rb_P + p['e2f_threshold']))
        e2f_inhibition = p['k_e2f_inhibit'] * E2F_active
        dE2F_active = e2f_release - e2f_inhibition

        # ==================================================================
        # 7. 细胞周期停滞
        #    由低 E2F + 高 p21 + 高 p16 驱动
        #    这是一个累积性状态 (senescence 标志)
        # ==================================================================
        # E2F 低 → 停滞信号
        e2f_arrest_signal = 1 - E2F_active / (E2F_active + p['arrest_e2f_threshold'])
        # p21/p16 协同促进停滞
        cki_arrest_signal = (p21_val / (1 + p21_val)
                             + p16_val / (1 + p16_val)) / 2
        arrest_act = (p['k_arrest_activate']
                      * (0.6 * e2f_arrest_signal + 0.4 * cki_arrest_signal)
                      * (1 - Cell_cycle_arrest))  # 饱和上限
        arrest_rel = p['k_arrest_release'] * Cell_cycle_arrest
        dCell_cycle_arrest = arrest_act - arrest_rel

        # ==================================================================
        # 8. SASP 综合评分 (NF-κB 驱动的分泌表型)
        # ==================================================================
        sasp_activation = (p['k_sasp_prod']
                           * NFkB_activity
                           / (NFkB_activity + p['sasp_nfkb_threshold']))
        dSASP_score = sasp_activation - p['k_sasp_deg'] * SASP_score

        # ==================================================================
        # 9-11. SASP 炎症因子组分 (IL-1β, IL-6, TNF-α)
        # ==================================================================
        dIL1B = (p['k_il1b_prod'] * SASP_score - p['k_il1b_deg'] * IL1B_val)
        dIL6  = (p['k_il6_prod']  * SASP_score - p['k_il6_deg']  * IL6_val)
        dTNF  = (p['k_tnf_prod']  * SASP_score - p['k_tnf_deg']  * TNF_val)

        # 外部 TNF 刺激
        dTNF += tnf_stim

        # ==================================================================
        # 12. 衰老相关 MMP (MMP3/MMP13)
        #     受 NF-κB 和 SASP 双重驱动
        # ==================================================================
        mmp_drive = (NFkB_activity + SASP_score) / 2
        dMMP_senescence = (p['k_mmp_sen_prod'] * mmp_drive
                           - p['k_mmp_sen_deg'] * MMP_senescence)

        # ==================================================================
        # 13. NF-κB 活性 (SASP 核心转录调控枢纽)
        #     激活信号: IL-1β, TNF-α, ROS, 机械应力
        #     正反馈: SASP → NF-κB → 更多 SASP
        # ==================================================================
        nfkb_il1b = IL1B_val / (IL1B_val + p['nfkb_il1b_threshold'])
        nfkb_tnf  = (TNF_val + tnf_stim) / (TNF_val + tnf_stim + p['nfkb_tnf_threshold'])
        nfkb_ros  = ROS_cellular / (ROS_cellular + p['nfkb_ros_threshold'])
        nfkb_drive = 0.4 * nfkb_il1b + 0.4 * nfkb_tnf + 0.2 * nfkb_ros

        nfkb_activation = p['k_nfkb_act'] * nfkb_drive * s
        nfkb_inhibition = p['k_nfkb_inh'] * NFkB_activity
        dNFkB_activity = nfkb_activation - nfkb_inhibition

        # ==================================================================
        # 14. 细胞内 ROS
        #     来源: Nox4 产 ROS + 线粒体泄漏
        #     清除: SOD, 谷胱甘肽, 过氧化氢酶
        # ==================================================================
        nox4_ros = (p['k_ros_prod_nox4']
                    * Nox4_val * (1 - nox4_kd)
                    / (Nox4_val * (1 - nox4_kd) + p['nox4_prod_threshold']))
        mito_ros = p['k_ros_prod_mito'] * Mito_dys
        # 外源氧化应激
        external_ros = ox_stress
        ros_clear = p['k_ros_clear'] * ROS_cellular

        dROS_cellular = (nox4_ros + mito_ros + external_ros - ros_clear)

        # ==================================================================
        # 15. 线粒体功能障碍
        #     ROS 损伤线粒体 → 功能障碍
        #     功能障碍 → 更多 ROS (恶性循环)
        #     PINK1/Parkin 介导的线粒体自噬被 ROS 抑制
        # ==================================================================
        mito_ros_damage = (p['k_mito_dys']
                           * ROS_cellular
                           / (ROS_cellular + p['mito_ros_threshold']))
        # 自噬修复受 ROS 抑制
        mito_repair = (p['k_mito_repair']
                       * (1 - p['k_mito_ros_inh'] * ROS_cellular / (1 + ROS_cellular))
                       * Mito_dys)
        dMitochondrial_dysfunction = (mito_ros_damage * (1 - Mito_dys)
                                      - mito_repair)

        # ==================================================================
        # 16. Nox4 (NADPH 氧化酶 4, 主要衰老相关 ROS 来源)
        #     被 ROS 正反馈诱导 + 被 TNF-α 诱导
        # ==================================================================
        nox4_stress_ind = (p['k_nox4_stress']
                           * ROS_cellular
                           / (ROS_cellular + p['nox4_stress_threshold']))
        nox4_tnf_ind = (p['k_nox4_tnf']
                        * (TNF_val + tnf_stim)
                        / (TNF_val + tnf_stim + p['nox4_tnf_threshold']))
        nox4_production = p['k_nox4_base'] + nox4_stress_ind + nox4_tnf_ind
        nox4_degradation = p['k_nox4_deg'] * Nox4_val
        # Nox4 敲低
        nox4_production *= (1 - nox4_kd)

        dNox4 = nox4_production - nox4_degradation

        # ==================================================================
        # 17. 凋亡
        #     严重线粒体功能障碍 + 持续高 p53 + 不可修复 DNA 损伤
        # ==================================================================
        # 线粒体途径凋亡
        mito_apop = (p['k_apop_mito']
                     * Mito_dys
                     / (Mito_dys + p['apop_threshold']))
        # p53 高表达触发凋亡切换 (超过细胞周期停滞阈值)
        p53_apop = (p['k_apop_dna']
                    * max(0, p53_val - p['p53_apop_threshold'])
                    / (1 + max(0, p53_val - p['p53_apop_threshold'])))
        dApoptosis = (mito_apop + p53_apop
                      - p['k_apop_clear'] * Apoptosis
                      - senolytic * Apoptosis)  # 衰老细胞清除

        # ==================================================================
        # 数值安全裁剪
        # ==================================================================
        max_val = p['max_var']
        dy = [
            np.clip(dDNA_damage, -max_val, max_val),
            np.clip(dp53, -max_val, max_val),
            np.clip(dp21, -max_val, max_val),
            np.clip(dp16, -max_val, max_val),
            np.clip(dRb_P, -max_val, max_val),
            np.clip(dE2F_active, -max_val, max_val),
            np.clip(dCell_cycle_arrest, -max_val, max_val),
            np.clip(dSASP_score, -max_val, max_val),
            np.clip(dIL1B, -max_val, max_val),
            np.clip(dIL6, -max_val, max_val),
            np.clip(dTNF, -max_val, max_val),
            np.clip(dMMP_senescence, -max_val, max_val),
            np.clip(dNFkB_activity, -max_val, max_val),
            np.clip(dROS_cellular, -max_val, max_val),
            np.clip(dMitochondrial_dysfunction, -max_val, max_val),
            np.clip(dNox4, -max_val, max_val),
            np.clip(dApoptosis, -max_val, max_val),
        ]

        return dy

    # ====================================================================
    # 仿真接口
    # ====================================================================

    def simulate(
        self,
        stress_level: float = 1.0,
        perturbation: Optional[Dict[str, float]] = None,
        t_span: Tuple[float, float] = (0, 500),
        n_points: int = 1000,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行衰老仿真

        先跑基线至稳态, 再施加扰动继续仿真。

        Args:
            stress_level: 应力倍率 (默认 1.0, 健康状态)
            perturbation: 扰动参数字典
            t_span:       (起始, 结束) 时间范围
            n_points:     输出点数
            initial_state: 初始状态 (None = 使用默认年轻健康状态)

        Returns:
            (时间数组, 状态矩阵 [n_vars, n_t])
        """
        # 默认初始状态: 年轻健康 NP 细胞
        if initial_state is None:
            y0 = self._default_initial_state()
        else:
            y0 = np.array(initial_state, dtype=float)

        # 先跑基线至稳态 (无扰动)
        t_baseline = (0, 2000)
        sol_base = solve_ivp(
            lambda t, y: self._ode_system(t, y, stress_level=stress_level),
            t_baseline, y0,
            method='RK45',
            max_step=20.0,
            rtol=1e-6,
            atol=1e-8,
        )
        steady_state = sol_base.y[:, -1].copy()

        # 施加扰动
        if perturbation:
            # 先记录扰动前一段时间 (-200 到 0)
            pre_t = np.linspace(-200, 0, 200)
            pre_y = np.tile(steady_state.reshape(-1, 1), (1, 200))

            # 带扰动仿真
            sol_pert = solve_ivp(
                lambda t, y: self._ode_system(
                    t, y, stress_level=stress_level, perturbation=perturbation,
                ),
                (0, t_span[1]), steady_state,
                method='RK45',
                max_step=5.0,
                rtol=1e-6,
                atol=1e-8,
                t_eval=np.linspace(0, t_span[1], n_points),
            )

            full_t = np.concatenate([pre_t, sol_pert.t])
            full_y = np.column_stack([pre_y, sol_pert.y])
        else:
            # 无扰动: 直接返回基线结果
            t_eval = np.linspace(t_baseline[0], t_baseline[1], n_points)
            sol = solve_ivp(
                lambda t, y: self._ode_system(t, y, stress_level=stress_level),
                t_baseline, y0,
                method='RK45',
                max_step=20.0,
                rtol=1e-6,
                atol=1e-8,
                t_eval=t_eval,
            )
            full_t = sol.t
            full_y = sol.y

        return full_t, full_y

    def simulate_sasp_feedback(
        self,
        stress_level: float = 0.8,
        t_span: Tuple[float, float] = (0, 800),
        n_points: int = 1500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟 SASP 正反馈驱动的衰老加速

        使用较低应力启动, 观察 SASP → NF-κB → 更多 SASP
        的自催化放大效应。

        Returns:
            (时间数组, 状态矩阵)
        """
        # 添加 TNF 初始脉冲以触发 SASP 反馈
        perturbation = {
            'TNF_stim': 0.3,    # 初始 TNF 刺激
        }

        t, y = self.simulate(
            stress_level=stress_level,
            perturbation=perturbation,
            t_span=t_span,
            n_points=n_points,
        )

        return t, y

    def simulate_senolytic(
        self,
        drug: str = 'dasatinib',
        stress_level: float = 1.5,
        t_span: Tuple[float, float] = (0, 600),
        n_points: int = 1200,
        senolytic_start: float = 200.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟衰老细胞清除药物 (Senolytics) 干预

        支持不同药物:
            - 'dasatinib':  达沙替尼 (清除衰老细胞, senolytic=0.3)
            - 'quercetin':  槲皮素 (清除衰老细胞, senolytic=0.2)
            - 'dasatinib+quercetin': D+Q 联合 (senolytic=0.5)
            - 'navitoclax': 纳维托克 (BCL-2 抑制剂, senolytic=0.4)

        Args:
            drug:           药物名称
            stress_level:   基线应力水平
            t_span:         时间范围
            n_points:       输出点数
            senolytic_start: 给药时间点

        Returns:
            (时间数组, 状态矩阵)
        """
        # 药物 → 衰老清除强度映射
        senolytic_strength = {
            'dasatinib': 0.3,
            'quercetin': 0.2,
            'dasatinib+quercetin': 0.5,
            'D+Q': 0.5,
            'navitoclax': 0.4,
        }
        senolytic_val = senolytic_strength.get(drug, 0.3)

        # 先用应力跑至衰老状态
        y0 = self._default_initial_state()
        t_pre = (0, senolytic_start * 1.5)
        sol_pre = solve_ivp(
            lambda t, y: self._ode_system(t, y, stress_level=stress_level),
            t_pre, y0,
            method='RK45',
            max_step=10.0,
            rtol=1e-6,
            atol=1e-8,
        )
        aged_state = sol_pre.y[:, -1].copy()

        # 分两段: 给药前 (基线) + 给药后 (含 senolytic)
        pre_t = np.linspace(0, senolytic_start, n_points // 3)
        # 给药前不带 senolytic
        sol_pre_eval = solve_ivp(
            lambda t, y: self._ode_system(t, y, stress_level=stress_level),
            (0, senolytic_start), aged_state if False else y0,
            method='RK45',
            max_step=10.0,
            rtol=1e-6,
            atol=1e-8,
            t_eval=pre_t,
        )

        # 给药后带 senolytic
        post_t = np.linspace(senolytic_start, t_span[1], 2 * n_points // 3)
        sol_post = solve_ivp(
            lambda t, y: self._ode_system(
                t, y,
                stress_level=stress_level,
                perturbation={'senolytic': senolytic_val},
            ),
            (senolytic_start, t_span[1]),
            sol_pre_eval.y[:, -1],
            method='RK45',
            max_step=5.0,
            rtol=1e-6,
            atol=1e-8,
            t_eval=post_t,
        )

        full_t = np.concatenate([sol_pre_eval.t, sol_post.t])
        full_y = np.column_stack([sol_pre_eval.y, sol_post.y])

        return full_t, full_y

    # ====================================================================
    # 辅助方法
    # ====================================================================

    def _default_initial_state(self) -> np.ndarray:
        """默认初始状态: 年轻健康 NP 细胞"""
        return np.array([
            0.01,    # DNA_damage: 低
            0.05,    # p53: 基础水平
            0.03,    # p21: 低
            0.01,    # p16: 低 (年轻)
            0.60,    # Rb_P: 中等磷酸化 (正常细胞周期)
            0.40,    # E2F_active: 中等
            0.01,    # Cell_cycle_arrest: 极低 (正常增殖)
            0.01,    # SASP_score: 极低
            0.001,   # IL1B: 极低
            0.001,   # IL6: 极低
            0.001,   # TNF: 极低
            0.01,    # MMP_senescence: 低
            0.05,    # NFkB_activity: 基础水平
            0.05,    # ROS_cellular: 低
            0.01,    # Mitochondrial_dysfunction: 极低
            0.05,    # Nox4: 基础水平
            0.001,   # Apoptosis: 极低
        ], dtype=float)

    def get_state_dict(self, y: np.ndarray) -> Dict[str, float]:
        """将状态向量转为命名字典"""
        return {name: y[i] for i, name in enumerate(self.var_names)}

    def compute_senescence_score(self, y: np.ndarray) -> float:
        """
        计算综合衰老评分 [0,1]

        综合考量:
            - 细胞周期停滞程度
            - SASP 分泌水平
            - 线粒体功能障碍
            - ROS 水平
            - p16 表达
        """
        idx = self.var_indices
        arrest = y[idx['Cell_cycle_arrest']]
        sasp = y[idx['SASP_score']]
        mito = y[idx['Mitochondrial_dysfunction']]
        ros = y[idx['ROS_cellular']]
        p16_val = y[idx['p16']]

        score = (0.25 * arrest / (1 + arrest)
                 + 0.20 * sasp / (1 + sasp)
                 + 0.20 * mito
                 + 0.15 * ros / (1 + ros)
                 + 0.20 * p16_val / (1 + p16_val))
        return min(1.0, score)

    # ====================================================================
    # 绘制方法
    # ====================================================================

    def plot_senescence(
        self,
        t: np.ndarray,
        y: np.ndarray,
        title: str = "NP 细胞衰老多维度动态仿真",
        highlight_vars: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 12),
        output_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        绘制衰老仿真结果 (5 行面板)

        面板布局:
            行1: DNA 损伤 + p53/p21/p16 + 细胞周期
            行2: Rb/E2F + 细胞周期停滞
            行3: SASP 组分
            行4: NF-κB + ROS + 线粒体
            行5: Nox4 + 凋亡 + 综合衰老评分

        Args:
            t:            时间数组
            y:            状态矩阵 [n_vars, n_t]
            title:        图表标题
            highlight_vars: 高亮变量名列表
            figsize:      图像尺寸
            output_path:  保存路径 (None = 不保存)
            dpi:          分辨率

        Returns:
            matplotlib Figure 对象
        """
        idx = self.var_indices
        n_t = len(t)
        var_data = {name: y[i, :] for i, name in enumerate(self.var_names)}
        highlight_vars = highlight_vars or []

        # 计算综合衰老评分
        sen_score = np.array([self.compute_senescence_score(y[:, j])
                              for j in range(n_t)])

        fig, axes = plt.subplots(5, 4, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.005)

        # Color palette
        colors = {
            'DNA_damage': '#8E44AD',
            'p53': '#2E86C1',
            'p21': '#3498DB',
            'p16': '#1ABC9C',
            'Rb_P': '#E67E22',
            'E2F_active': '#F39C12',
            'Cell_cycle_arrest': '#C0392B',
            'SASP_score': '#E74C3C',
            'IL1B': '#FF6B6B',
            'IL6': '#FF8E53',
            'TNF': '#FF4757',
            'MMP_senescence': '#A30000',
            'NFkB_activity': '#9B59B6',
            'ROS_cellular': '#FF6348',
            'Mitochondrial_dysfunction': '#57606F',
            'Nox4': '#E84393',
            'Apoptosis': '#2C3E50',
        }

        def _plot_line(ax, var_name, label=None, color=None, lw=2, ls='-'):
            """辅助: 在指定轴上绘制变量"""
            c = color or colors.get(var_name, '#333')
            lbl = label or var_name
            data = var_data[var_name]
            ax.plot(t, data, color=c, linewidth=lw, linestyle=ls, label=lbl)
            if var_name in highlight_vars:
                ax.set_facecolor('#FFF9C4')
            # 标注终值
            final_val = data[-1]
            ax.text(t[-1] * 0.92, final_val + 0.02 * (max(data) - min(data) + 0.1),
                    f"{final_val:.3f}", fontsize=6, color=c, alpha=0.8)

        # ============ 行 0: DNA 损伤 + p53/p21/p16 通路 ============
        ax = axes[0, 0]
        _plot_line(ax, 'DNA_damage')
        ax.set_title('DNA 损伤 (DDR)', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[0, 1]
        _plot_line(ax, 'p53', color='#2E86C1')
        _plot_line(ax, 'p21', color='#3498DB')
        ax.set_title('p53-p21 通路', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        self._decorate_axis(ax)

        ax = axes[0, 2]
        _plot_line(ax, 'p16')
        ax.set_title('p16^INK4a', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[0, 3]
        _plot_line(ax, 'p53', color='#2E86C1')
        _plot_line(ax, 'p21', color='#3498DB')
        _plot_line(ax, 'p16', color='#1ABC9C')
        ax.set_title('p53 / p21 / p16 对比', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        self._decorate_axis(ax)

        # ============ 行 1: Rb/E2F + 细胞周期 ============
        ax = axes[1, 0]
        _plot_line(ax, 'Rb_P')
        ax.set_title('Rb 磷酸化 (Rb_P)', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[1, 1]
        _plot_line(ax, 'E2F_active')
        ax.set_title('E2F 活性', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[1, 2]
        _plot_line(ax, 'Cell_cycle_arrest')
        ax.set_title('细胞周期停滞', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[1, 3]
        _plot_line(ax, 'Rb_P', color='#E67E22')
        _plot_line(ax, 'E2F_active', color='#F39C12')
        _plot_line(ax, 'Cell_cycle_arrest', color='#C0392B')
        ax.set_title('Rb ↔ E2F ↔ 停滞', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        self._decorate_axis(ax)

        # ============ 行 2: SASP 组分 ============
        ax = axes[2, 0]
        _plot_line(ax, 'SASP_score')
        ax.set_title('SASP 综合评分', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[2, 1]
        _plot_line(ax, 'IL1B')
        ax.set_title('IL-1β', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[2, 2]
        _plot_line(ax, 'IL6')
        ax.set_title('IL-6', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[2, 3]
        _plot_line(ax, 'TNF')
        ax.set_title('TNF-α', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        # ============ 行 3: NF-κB + ROS + 线粒体 + MMP ============
        ax = axes[3, 0]
        _plot_line(ax, 'NFkB_activity')
        ax.set_title('NF-κB 活性', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[3, 1]
        _plot_line(ax, 'ROS_cellular')
        ax.set_title('胞内 ROS', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[3, 2]
        _plot_line(ax, 'Mitochondrial_dysfunction')
        ax.set_title('线粒体功能障碍', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[3, 3]
        _plot_line(ax, 'MMP_senescence')
        ax.set_title('衰老相关 MMP', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        # ============ 行 4: Nox4 + 凋亡 + 衰老评分 + 综合 ============
        ax = axes[4, 0]
        _plot_line(ax, 'Nox4')
        ax.set_title('Nox4 (NADPH 氧化酶)', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[4, 1]
        _plot_line(ax, 'Apoptosis')
        ax.set_title('凋亡水平', fontsize=10, fontweight='bold')
        self._decorate_axis(ax)

        ax = axes[4, 2]
        ax.plot(t, sen_score, color='#2C3E50', linewidth=2.5, label='衰老评分')
        ax.set_title('综合衰老评分', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.text(t[-1] * 0.92, sen_score[-1] + 0.02,
                f"{sen_score[-1]:.3f}", fontsize=7, color='#2C3E50')
        self._decorate_axis(ax)

        # 综合: SASP ↔ NF-κB 正反馈环路
        ax = axes[4, 3]
        _plot_line(ax, 'SASP_score', color='#E74C3C')
        _plot_line(ax, 'NFkB_activity', color='#9B59B6')
        _plot_line(ax, 'ROS_cellular', color='#FF6348')
        ax.set_title('SASP ↔ NF-κB ↔ ROS 环路', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        self._decorate_axis(ax)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 衰老仿真图已保存: {output_path}")

        return fig

    @staticmethod
    def _decorate_axis(ax: plt.Axes):
        """统一轴装饰"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('时间', fontsize=7)


# ====================================================================
# 模块级辅助函数
# ====================================================================

def plot_senolytic_comparison(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    drug_names: List[str],
    title: str = "Senolytic 药物干预比较",
    figsize: Tuple[int, int] = (18, 10),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    比较不同 Senolytic 药物的干预效果

    Args:
        models: {drug_name: (time_array, state_matrix)} 字典
        drug_names: 药物名称列表 (用于图例, 顺序与 models 一致)
        title: 图表标题
        figsize: 图像尺寸
        output_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure
    """
    # 变量索引
    var_names = [
        'Cell_cycle_arrest', 'SASP_score', 'IL1B', 'IL6',
        'TNF', 'MMP_senescence', 'NFkB_activity', 'ROS_cellular',
        'Mitochondrial_dysfunction', 'Nox4', 'Apoptosis'
    ]
    titles = [
        '细胞周期停滞', 'SASP 评分', 'IL-1β', 'IL-6',
        'TNF-α', 'MMP 衰老', 'NF-κB 活性', 'ROS 水平',
        '线粒体功能障碍', 'Nox4', '凋亡'
    ]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
              '#9B59B6', '#1ABC9C', '#E67E22', '#FF6348',
              '#57606F', '#E84393', '#2C3E50']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'x']

    n_plots = len(var_names)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for idx, (var_name, plot_title, color) in enumerate(
        zip(var_names, titles, colors)
    ):
        ax = axes[idx]
        for i, drug in enumerate(drug_names):
            if drug not in models:
                continue
            t, y = models[drug]
            # 查找变量索引
            var_idx = list(NPSenescenceModel().var_names).index(var_name)
            data = y[var_idx, :]
            ax.plot(t, data, color=colors[i % len(colors)],
                    linewidth=2,
                    label=drug if i < 6 else None,  # 避免图例太多
                    alpha=0.85)

        # 添加 "无干预" 参考线 (第一个模型的基线部分)
        ax.set_title(plot_title, fontsize=10, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('时间', fontsize=7)

    # 隐藏多余子图
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # 统一图例
    if drug_names:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors[i % len(colors)],
                   linewidth=2, label=name)
            for i, name in enumerate(drug_names)
        ]
        # 将图例放在底部右侧
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=min(len(drug_names), 6), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] Senolytic 比较图已保存: {output_path}")

    return fig


def run_senescence_perturbation_screen(
    model: Optional[NPSenescenceModel] = None,
    stress_levels: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    dpi: int = 150,
) -> Dict[str, Dict[str, float]]:
    """
    运行衰老扰动筛选实验

    系统评估不同应激水平和干预条件下的衰老表型。

    Args:
        model: NPSenescenceModel 实例 (None = 新建默认)
        stress_levels: 要测试的应力水平列表
        output_dir: 输出目录 (None = 不保存图片)
        dpi: 图片分辨率

    Returns:
        {perturbation_name: {metric: value}} 结果字典
    """
    if model is None:
        model = NPSenescenceModel()
    if stress_levels is None:
        stress_levels = [0.5, 1.0, 1.5, 2.0, 3.0]

    # 扰动条件列表
    perturbations = [
        ('baseline', None),
        ('oxidative_stress_0.5', {'oxidative_stress': 0.5}),
        ('oxidative_stress_1.0', {'oxidative_stress': 1.0}),
        ('TNF_stim_0.3', {'TNF_stim': 0.3}),
        ('TNF_stim_1.0', {'TNF_stim': 1.0}),
        ('DNA_damage_0.5', {'DNA_damage': 0.5}),
        ('Nox4_KD_0.5', {'Nox4_KD': 0.5}),
        ('Nox4_KD_0.8', {'Nox4_KD': 0.8}),
        ('senolytic_0.3', {'senolytic': 0.3}),
        ('senolytic_0.5', {'senolytic': 0.5}),
        ('combined_stress', {
            'oxidative_stress': 0.5,
            'TNF_stim': 0.3,
        }),
    ]

    idx = model.var_indices
    results = {}

    for stress_lvl in stress_levels:
        for name, pert in perturbations:
            label = f"stress{stress_lvl:.1f}_{name}"
            t, y = model.simulate(
                stress_level=stress_lvl,
                perturbation=pert,
                t_span=(0, 500),
                n_points=800,
            )

            # 提取终态指标
            y_final = y[:, -1]
            sen_score = model.compute_senescence_score(y_final)

            results[label] = {
                'stress_level': stress_lvl,
                'DNA_damage': y_final[idx['DNA_damage']],
                'p53': y_final[idx['p53']],
                'p21': y_final[idx['p21']],
                'p16': y_final[idx['p16']],
                'Cell_cycle_arrest': y_final[idx['Cell_cycle_arrest']],
                'SASP_score': y_final[idx['SASP_score']],
                'IL1B': y_final[idx['IL1B']],
                'IL6': y_final[idx['IL6']],
                'TNF': y_final[idx['TNF']],
                'MMP_senescence': y_final[idx['MMP_senescence']],
                'NFkB_activity': y_final[idx['NFkB_activity']],
                'ROS_cellular': y_final[idx['ROS_cellular']],
                'Mitochondrial_dysfunction': y_final[idx['Mitochondrial_dysfunction']],
                'Nox4': y_final[idx['Nox4']],
                'Apoptosis': y_final[idx['Apoptosis']],
                'Senescence_score': sen_score,
            }

        # 每个 stress 水平的基线
        if output_dir and stress_lvl == 1.5:
            t, y = model.simulate(stress_level=stress_lvl)
            fig = model.plot_senescence(
                t, y,
                title=f"NP 细胞衰老 (应力={stress_lvl})",
                output_path=f"{output_dir}/senescence_stress{stress_lvl}.png",
                dpi=dpi,
            )
            plt.close(fig)

    return results


# ====================================================================
# 示例 / 测试入口
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NP 细胞衰老模型 — 运行示例")
    print("=" * 60)

    model = NPSenescenceModel()

    # 1. 健康基线仿真
    print("\n[1/4] 健康基线仿真 (stress=0.5)...")
    t, y = model.simulate(stress_level=0.5)
    score = model.compute_senescence_score(y[:, -1])
    print(f"    终态衰老评分: {score:.4f}")

    # 2. 退变应力仿真
    print("\n[2/4] 退变应力仿真 (stress=2.0)...")
    t, y = model.simulate(stress_level=2.0)
    score = model.compute_senescence_score(y[:, -1])
    print(f"    终态衰老评分: {score:.4f}")
    print(f"    细胞周期停滞: {y[model.var_indices['Cell_cycle_arrest'], -1]:.4f}")
    print(f"    SASP 评分:     {y[model.var_indices['SASP_score'], -1]:.4f}")

    # 3. SASP 正反馈仿真
    print("\n[3/4] SASP 正反馈仿真...")
    t_fb, y_fb = model.simulate_sasp_feedback(stress_level=0.8)
    score_fb = model.compute_senescence_score(y_fb[:, -1])
    nfkb_fb = y_fb[model.var_indices['NFkB_activity'], -1]
    print(f"    终态衰老评分: {score_fb:.4f}")
    print(f"    NF-κB 活性:   {nfkb_fb:.4f}")

    # 4. Senolytic 干预仿真
    print("\n[4/4] Senolytic 干预仿真 (D+Q)...")
    t_sen, y_sen = model.simulate_senolytic(
        drug='dasatinib+quercetin',
        stress_level=1.5,
    )
    score_sen = model.compute_senescence_score(y_sen[:, -1])
    print(f"    干预终态衰老评分: {score_sen:.4f}")

    print("\n[✓] 所有仿真完成")
