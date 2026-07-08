"""
m6A 表观转录组调控 NPC 衰老模型 — Virtual NP Cell
=====================================================
基于 Nature Communications 2022:
    "WTAP-mediated m6A modification of lncRNA NORAD promotes intervertebral disc degeneration"

核心信号轴: KDM5A → H3K4me3↑(WTAP启动子) → WTAP↑ → m6A-NORAD → PUM隔离 → E2F3释放 → 衰老

变量列表 (13个ODE):
    y[0]  = KDM5A           (组蛋白去甲基化酶, 激活WTAP启动子H3K4me3)
    y[1]  = H3K4me3_WTAP    (WTAP启动子处的H3K4me3修饰水平)
    y[2]  = WTAP_mRNA       (含m6A修饰的WTAP mRNA, 反映转录活性)
    y[3]  = WTAP             (WTAP蛋白, m6A甲基转移酶复合物组分)
    y[4]  = NORAD_pre        (未修饰的NORAD lncRNA前体, 不稳定)
    y[5]  = NORAD_m6A        (m6A修饰的稳定型NORAD)
    y[6]  = YTHDF2           (m6A读码蛋白, 识别m6A-NORAD)
    y[7]  = PUM_free         (游离PUM1/2蛋白, 可结合E2F3 mRNA)
    y[8]  = PUM_bound        (与NORAD结合的PUM, 被隔离的失活形式)
    y[9]  = E2F3_mRNA        (E2F3 mRNA, 受PUM介导的降解调控)
    y[10] = E2F3             (E2F3蛋白, 驱动细胞周期和衰老程序)
    y[11] = Senescence_score (NP细胞衰老评分)
    y[12] = SASP_score       (衰老相关分泌表型评分)

扰动支持: KDM5A_OE, WTAP_OE, NORAD_OE, METTL3_KD, YTHDF2_KD, E2F3_KD
敲除支持: 任意变量
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any, Callable

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 默认参数集 (Default Parameters)
# ============================================================

# --- 基础参数常数, 反映正常NP细胞稳态 ---
# 命名规则: k_ = 合成/激活速率, d_ = 降解速率, K_ = Michaelis常数/半饱和

PARAMS_DEFAULT = {
    # KDM5A 动力学
    'k_KDM5A_syn': 0.015,       # KDM5A 基础合成速率
    'd_KDM5A': 0.008,           # KDM5A 降解速率

    # H3K4me3_WTAP 动力学 (KDM5A → H3K4me3↑ at WTAP promoter)
    'k_H3K4_act': 0.012,        # KDM5A 催化 H3K4me3 沉积速率
    'K_H3K4': 0.8,              # KDM5A 激活 H3K4me3 的 Michaelis 常数
    'H3K4_max': 1.0,            # H3K4me3_WTAP 最大修饰水平 (归一化)
    'd_H3K4': 0.006,            # H3K4me3 去修饰/衰减速率

    # WTAP 转录与翻译
    'k_WTAP_trans': 0.025,      # H3K4me3 驱动 WTAP 转录速率
    'K_WTAP_trans': 0.3,        # H3K4me3 半饱和转录常数
    'd_WTAP_mRNA': 0.02,        # WTAP mRNA 降解速率
    'k_WTAP_transl': 0.04,      # WTAP 翻译速率
    'd_WTAP': 0.01,             # WTAP 蛋白降解速率

    # NORAD 转录与 m6A 修饰
    'k_NORAD_trans': 0.03,      # NORAD lncRNA 基础转录速率
    'd_NORAD_pre': 0.05,        # 未修饰 NORAD 快速降解速率
    'k_m6A': 0.06,              # WTAP/METTL3 复合物 m6A 修饰速率
    'K_m6A': 0.4,               # WTAP 半饱和 m6A 催化常数
    'd_NORAD_m6A': 0.005,       # m6A 修饰 NORAD 缓慢降解速率 (稳定)

    # YTHDF2 动力学
    'k_YTHDF2_syn': 0.02,       # YTHDF2 基础合成速率
    'd_YTHDF2': 0.015,          # YTHDF2 降解速率

    # PUM (PUM1/2) 动力学与 NORAD 结合
    'k_PUM_syn': 0.025,         # PUM1/2 基础合成速率
    'k_PUM_bind': 0.08,         # PUM-NORAD_m6A 结合速率
    'K_PUM_bind': 0.5,          # 结合的半饱和常数
    'k_PUM_unbind': 0.02,       # PUM-NORAD 解离速率
    'd_PUM_free': 0.01,         # 游离 PUM 降解速率
    'd_PUM_bound': 0.005,       # 结合态 PUM 降解速率 (结合后轻微保护)

    # E2F3 mRNA 调控 (PUM 介导的降解)
    'k_E2F3_trans': 0.02,       # E2F3 基础转录速率
    'k_PUM_deg': 0.04,          # PUM 介导 E2F3 mRNA 降解速率
    'K_PUM_deg': 0.5,           # PUM 降解的半饱和常数
    'd_E2F3_mRNA': 0.015,       # E2F3 mRNA 基础降解速率
    'k_E2F3_transl': 0.035,     # E2F3 翻译速率
    'd_E2F3': 0.012,            # E2F3 蛋白降解速率

    # 衰老与 SASP 动力学
    'k_sen': 0.008,             # E2F3 驱动衰老评分积累速率
    'K_sen': 0.6,               # E2F3 半饱和衰老激活常数
    'd_sen': 0.003,             # 衰老评分衰减速率
    'k_sasp_E2F3': 0.005,       # E2F3 驱动 SASP 速率
    'k_sasp_sen': 0.006,        # 衰老正反馈驱动 SASP 速率
    'K_sasp': 0.5,              # SASP 激活半饱和常数
    'd_sasp': 0.004,            # SASP 评分衰减速率
}

# 变量的描述和单位
VAR_NAMES = [
    'KDM5A',
    'H3K4me3_WTAP',
    'WTAP_mRNA',
    'WTAP',
    'NORAD_pre',
    'NORAD_m6A',
    'YTHDF2',
    'PUM_free',
    'PUM_bound',
    'E2F3_mRNA',
    'E2F3',
    'Senescence_score',
    'SASP_score',
]

VAR_LABELS_CN = [
    'KDM5A 蛋白',
    'H3K4me3 (WTAP启动子)',
    'WTAP mRNA',
    'WTAP 蛋白',
    'NORAD 前体 (未修饰)',
    'NORAD m6A (稳定型)',
    'YTHDF2 蛋白',
    '游离 PUM1/2',
    '结合 NORAD 的 PUM',
    'E2F3 mRNA',
    'E2F3 蛋白',
    '衰老评分',
    'SASP 评分',
]

NUM_VARS = len(VAR_NAMES)


# ============================================================
# 2. ODE 系统定义 (ODE System)
# ============================================================

def m6a_ode_system(
    t: float,
    y: np.ndarray,
    p: Dict[str, float],
    perturbation: Optional[str] = None,
    ko_gene: Optional[str] = None,
) -> List[float]:
    """
    m6A表观转录组调控 ODE 右端函数.

    Parameters
    ----------
    t : float
        当前时间点
    y : np.ndarray (13,)
        状态变量向量
    p : dict
        参数集
    perturbation : str or None
        扰动类型 (None = 无扰动, 稳态)
    ko_gene : str or None
        敲除的基因名

    Returns
    -------
    list : 13 个变量的导数
    """
    # 解包变量
    KDM5A, H3K4me3_WTAP, WTAP_mRNA, WTAP, NORAD_pre, \
        NORAD_m6A, YTHDF2, PUM_free, PUM_bound, \
        E2F3_mRNA, E2F3, Sen_score, SASP_score = y

    # --- 参数 (可能会被扰动修改) ---
    k_KDM5A_syn = p['k_KDM5A_syn']
    d_KDM5A = p['d_KDM5A']

    k_H3K4_act = p['k_H3K4_act']
    K_H3K4 = p['K_H3K4']
    H3K4_max = p['H3K4_max']
    d_H3K4 = p['d_H3K4']

    k_WTAP_trans = p['k_WTAP_trans']
    K_WTAP_trans = p['K_WTAP_trans']
    d_WTAP_mRNA = p['d_WTAP_mRNA']
    k_WTAP_transl = p['k_WTAP_transl']
    d_WTAP = p['d_WTAP']

    k_NORAD_trans = p['k_NORAD_trans']
    d_NORAD_pre = p['d_NORAD_pre']
    k_m6A = p['k_m6A']
    K_m6A = p['K_m6A']
    d_NORAD_m6A = p['d_NORAD_m6A']

    k_YTHDF2_syn = p['k_YTHDF2_syn']
    d_YTHDF2 = p['d_YTHDF2']

    k_PUM_syn = p['k_PUM_syn']
    k_PUM_bind = p['k_PUM_bind']
    K_PUM_bind = p['K_PUM_bind']
    k_PUM_unbind = p['k_PUM_unbind']
    d_PUM_free = p['d_PUM_free']
    d_PUM_bound = p['d_PUM_bound']

    k_E2F3_trans = p['k_E2F3_trans']
    k_PUM_deg = p['k_PUM_deg']
    K_PUM_deg = p['K_PUM_deg']
    d_E2F3_mRNA_basal = p['d_E2F3_mRNA']
    k_E2F3_transl = p['k_E2F3_transl']
    d_E2F3 = p['d_E2F3']

    k_sen = p['k_sen']
    K_sen = p['K_sen']
    d_sen = p['d_sen']
    k_sasp_E2F3 = p['k_sasp_E2F3']
    k_sasp_sen = p['k_sasp_sen']
    K_sasp = p['K_sasp']
    d_sasp = p['d_sasp']

    # --- 应用扰动 (Perturbation) ---
    if perturbation == 'KDM5A_OE':
        # KDM5A 过表达: 合成速率加倍
        k_KDM5A_syn *= 3.0
    elif perturbation == 'WTAP_OE':
        # WTAP 过表达: 增强翻译或添加外源WTAP
        k_WTAP_transl *= 3.0
    elif perturbation == 'NORAD_OE':
        # NORAD 过表达: NORAD 转录速率大幅提升
        k_NORAD_trans *= 4.0
    elif perturbation == 'METTL3_KD':
        # METTL3 敲降: m6A 修饰效率降低
        k_m6A *= 0.2
    elif perturbation == 'YTHDF2_KD':
        # YTHDF2 敲降: YTHDF2 合成减少
        k_YTHDF2_syn *= 0.2
    elif perturbation == 'E2F3_KD':
        # E2F3 敲降: E2F3 翻译减少
        k_E2F3_transl *= 0.2

    # --- 应用敲除 (KO) ---
    if ko_gene == 'KDM5A':
        k_KDM5A_syn = 0.0
        # 初始值 KDM5A=0 后会自然衰减
    elif ko_gene == 'WTAP':
        k_WTAP_transl = 0.0  # WTAP 蛋白合成归零
        # 但 WTAP mRNA 仍会转录 (mRNA level not zero until dilution)
        # 这里把翻译置零更准确地反映蛋白水平的敲除效果
    elif ko_gene == 'NORAD':
        k_NORAD_trans = 0.0
    elif ko_gene == 'YTHDF2':
        k_YTHDF2_syn = 0.0
    elif ko_gene == 'E2F3':
        k_E2F3_transl = 0.0
    elif ko_gene == 'METTL3':
        # METTL3 敲除 → m6A 修饰速率归零
        k_m6A = 0.0
    elif ko_gene == 'PUM' or ko_gene == 'PUM1' or ko_gene == 'PUM2':
        k_PUM_syn = 0.0

    # =============================================
    # ODE 方程组 (基于机制建模)
    # =============================================

    # (1) KDM5A: 基础合成 - 降解
    dKDM5A = k_KDM5A_syn - d_KDM5A * KDM5A

    # (2) H3K4me3_WTAP: KDM5A激活H3K4me3沉积 (假设KDM5A通过去除抑制性标记促进H3K4me3)
    # Michaelis-Menten 型激活, 趋近最大修饰水平
    h3k4_activation = k_H3K4_act * KDM5A / (K_H3K4 + KDM5A) * (H3K4_max - H3K4me3_WTAP)
    dH3K4me3 = h3k4_activation - d_H3K4 * H3K4me3_WTAP

    # (3) WTAP mRNA: H3K4me3 驱动转录
    wtap_transcription = k_WTAP_trans * H3K4me3_WTAP / (K_WTAP_trans + H3K4me3_WTAP)
    dWTAP_mRNA = wtap_transcription - d_WTAP_mRNA * WTAP_mRNA

    # (4) WTAP 蛋白: 翻译 - 降解
    dWTAP = k_WTAP_transl * WTAP_mRNA - d_WTAP * WTAP

    # (5) NORAD 前体 (未修饰): 转录 - 降解 - m6A修饰
    # m6A 修饰由 WTAP/METTL3 复合物催化
    m6a_rate = k_m6A * WTAP / (K_m6A + WTAP) * NORAD_pre
    dNORAD_pre = k_NORAD_trans - d_NORAD_pre * NORAD_pre - m6a_rate

    # (6) NORAD_m6A (m6A修饰稳定型): m6A修饰生成 - 缓慢降解
    dNORAD_m6A = m6a_rate - d_NORAD_m6A * NORAD_m6A

    # (7) YTHDF2: 基础合成 - 降解
    dYTHDF2 = k_YTHDF2_syn - d_YTHDF2 * YTHDF2

    # (8) 游离 PUM: 合成 - 结合NORAD + 解离 - 降解
    pum_binding = k_PUM_bind * PUM_free * NORAD_m6A / (K_PUM_bind + NORAD_m6A)
    pum_unbinding = k_PUM_unbind * PUM_bound
    dPUM_free = k_PUM_syn - pum_binding + pum_unbinding - d_PUM_free * PUM_free

    # (9) 结合态 PUM: 结合 - 解离 - 降解
    dPUM_bound = pum_binding - pum_unbinding - d_PUM_bound * PUM_bound

    # (10) E2F3 mRNA: 转录 - PUM介导降解 - 基础降解
    pum_deg_e2f3 = k_PUM_deg * PUM_free / (K_PUM_deg + PUM_free) * E2F3_mRNA
    dE2F3_mRNA = k_E2F3_trans - pum_deg_e2f3 - d_E2F3_mRNA_basal * E2F3_mRNA

    # (11) E2F3 蛋白: 翻译 - 降解
    dE2F3 = k_E2F3_transl * E2F3_mRNA - d_E2F3 * E2F3

    # (12) 衰老评分: E2F3 激活衰老程序 (Hill型激活)
    sen_activation = k_sen * E2F3 / (K_sen + E2F3)
    dSen = sen_activation - d_sen * Sen_score

    # (13) SASP评分: E2F3 + 衰老正反馈
    sasp_activation = k_sasp_E2F3 * E2F3 / (K_sasp + E2F3) \
                      + k_sasp_sen * Sen_score / (K_sasp + Sen_score)
    dSASP = sasp_activation - d_sasp * SASP_score

    return [dKDM5A, dH3K4me3, dWTAP_mRNA, dWTAP, dNORAD_pre,
            dNORAD_m6A, dYTHDF2, dPUM_free, dPUM_bound,
            dE2F3_mRNA, dE2F3, dSen, dSASP]


# ============================================================
# 3. M6AEpigeneticModel 主类
# ============================================================

class M6AEpigeneticModel:
    """
    m6A 表观转录组调控 NPC 衰老模型.

    基于 Nature Communications 2022 论文 (WTAP-m6A-NORAD-E2F3 轴),
    通过常微分方程组模拟 KDM5A → H3K4me3 → WTAP → m6A-NORAD → PUM隔离 → E2F3释放 → 衰老
    这一表观转录组调控级联反应.

    Parameters
    ----------
    params : dict or None
        参数集 (None 时使用默认参数 PARAMS_DEFAULT)
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        self.params = PARAMS_DEFAULT.copy()
        if params is not None:
            self.params.update(params)

        # 计算稳态初始条件
        self.y0 = self._compute_steady_state()

    def _compute_steady_state(self) -> np.ndarray:
        """
        基于稳态解析近似计算初始条件.

        For each variable, compute approximate steady-state analytically
        from the ODE system (used as initial condition for simulation).
        """
        p = self.params

        # 解析稳态近似 (假设各模块解耦)
        # KDM5A 稳态
        KDM5A_ss = p['k_KDM5A_syn'] / p['d_KDM5A']

        # H3K4me3_WTAP 稳态 (当 KDM5A 达到稳态时)
        h3k4_act_ss = p['k_H3K4_act'] * KDM5A_ss / (p['K_H3K4'] + KDM5A_ss)
        H3K4me3_ss = h3k4_act_ss * p['H3K4_max'] / \
                      (h3k4_act_ss + p['d_H3K4'] * p['H3K4_max'])

        # WTAP mRNA 稳态
        wtap_trans_ss = p['k_WTAP_trans'] * H3K4me3_ss / (p['K_WTAP_trans'] + H3K4me3_ss)
        WTAP_mRNA_ss = wtap_trans_ss / p['d_WTAP_mRNA']

        # WTAP 蛋白稳态
        WTAP_ss = p['k_WTAP_transl'] * WTAP_mRNA_ss / p['d_WTAP']

        # NORAD 前体稳态
        m6a_rate_ss = p['k_m6A'] * WTAP_ss / (p['K_m6A'] + WTAP_ss)
        NORAD_pre_ss = p['k_NORAD_trans'] / (p['d_NORAD_pre'] + m6a_rate_ss)
        # 修正: 使用稳态值迭代 (简化, 仅近似)
        m6a_rate_final = p['k_m6A'] * WTAP_ss / (p['K_m6A'] + WTAP_ss) * NORAD_pre_ss
        NORAD_pre_ss = p['k_NORAD_trans'] / (p['d_NORAD_pre'] + m6a_rate_final) \
            if m6a_rate_final > 1e-10 else p['k_NORAD_trans'] / p['d_NORAD_pre']

        # 重新计算带修正的m6A速率
        m6a_rate_final = p['k_m6A'] * WTAP_ss / (p['K_m6A'] + WTAP_ss) * NORAD_pre_ss
        NORAD_m6A_ss = m6a_rate_final / p['d_NORAD_m6A']

        # YTHDF2 稳态
        YTHDF2_ss = p['k_YTHDF2_syn'] / p['d_YTHDF2']

        # PUM 系统稳态 (耦合, 近似解)
        PUM_total_ss = p['k_PUM_syn'] / p['d_PUM_free']
        # 在NORAD_m6A存在下, PUM会部分结合
        bind_frac = p['k_PUM_bind'] * NORAD_m6A_ss / (p['K_PUM_bind'] + NORAD_m6A_ss) \
            if NORAD_m6A_ss > 1e-10 else 0.0
        PUM_free_ss = PUM_total_ss * p['d_PUM_bound'] / \
                      (bind_frac + p['d_PUM_bound']) \
            if (bind_frac + p['d_PUM_bound']) > 1e-10 else PUM_total_ss
        PUM_bound_ss = bind_frac * PUM_free_ss / p['d_PUM_bound']

        # E2F3 mRNA 稳态 (受PUM_free调控的降解)
        pum_deg_ss = p['k_PUM_deg'] * PUM_free_ss / (p['K_PUM_deg'] + PUM_free_ss)
        E2F3_mRNA_ss = p['k_E2F3_trans'] / (pum_deg_ss + p['d_E2F3_mRNA'])

        # E2F3 蛋白稳态
        E2F3_ss = p['k_E2F3_transl'] * E2F3_mRNA_ss / p['d_E2F3']

        # 衰老评分稳态 (通常基线较低)
        Sen_ss = p['k_sen'] * E2F3_ss / (p['K_sen'] + E2F3_ss) / p['d_sen']

        # SASP 评分稳态
        sasp_ss = (p['k_sasp_E2F3'] * E2F3_ss / (p['K_sasp'] + E2F3_ss)
                   + p['k_sasp_sen'] * Sen_ss / (p['K_sasp'] + Sen_ss)) / p['d_sasp']

        return np.array([
            KDM5A_ss, H3K4me3_ss, WTAP_mRNA_ss, WTAP_ss,
            NORAD_pre_ss, NORAD_m6A_ss, YTHDF2_ss,
            PUM_free_ss, PUM_bound_ss,
            E2F3_mRNA_ss, E2F3_ss, Sen_ss, sasp_ss,
        ])

    def simulate(
        self,
        t_span: Tuple[float, float] = (0, 500),
        perturbation: Optional[str] = None,
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
            模拟时间范围
        perturbation : str or None
            扰动类型, 可选:
            - 'KDM5A_OE'  : KDM5A过表达 (3倍合成)
            - 'WTAP_OE'    : WTAP过表达 (3倍翻译)
            - 'NORAD_OE'   : NORAD过表达 (4倍转录)
            - 'METTL3_KD'  : METTL3敲降 (m6A修饰效率降至20%)
            - 'YTHDF2_KD'  : YTHDF2敲降 (YTHDF2合成降至20%)
            - 'E2F3_KD'    : E2F3敲降 (E2F3翻译降至20%)
        method : str
            solve_ivp 积分方法 ('RK45', 'LSODA', 'BDF'等)
        rtol, atol : float
            积分容差
        max_step : float
            最大积分步长

        Returns
        -------
        dict
            {
                't': np.ndarray — 时间点
                'y': np.ndarray — 变量矩阵 (时间点 × 13变量)
                'var_names': list — 变量名列表
                'params': dict — 使用参数集
                'perturbation': str or None
            }
        """
        if perturbation is not None:
            valid = ['KDM5A_OE', 'WTAP_OE', 'NORAD_OE',
                     'METTL3_KD', 'YTHDF2_KD', 'E2F3_KD']
            if perturbation not in valid:
                raise ValueError(
                    f"未知扰动类型: '{perturbation}'. "
                    f"可选: {valid}"
                )

        def ode_func(t, y):
            return m6a_ode_system(t, y, self.params,
                                  perturbation=perturbation,
                                  ko_gene=None)

        sol = solve_ivp(
            ode_func,
            t_span,
            self.y0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )

        # 如果积分失败, 尝试更鲁棒的方法
        if not sol.success:
            sol = solve_ivp(
                ode_func,
                t_span,
                self.y0,
                method='LSODA',
                rtol=rtol * 10,
                atol=atol,
                max_step=max_step,
            )

        return {
            't': sol.t,
            'y': sol.y,
            'var_names': VAR_NAMES,
            'params': self.params.copy(),
            'perturbation': perturbation,
        }

    def simulate_ko(
        self,
        ko_gene: str = 'WTAP',
        t_span: Tuple[float, float] = (0, 500),
        **kwargs,
    ) -> Dict[str, Any]:
        """
        模拟基因敲除 (KO) 效果.

        从稳态出发, 将目标基因的合成速率置零,
        观察系统重新达到的稳态 (通常在200-500时间单位后).

        Parameters
        ----------
        ko_gene : str
            敲除基因, 可选:
            'KDM5A', 'WTAP', 'NORAD', 'YTHDF2', 'E2F3', 'METTL3', 'PUM'
        t_span : tuple
            模拟时间范围
        **kwargs
            传递给 solve_ivp 的额外参数

        Returns
        -------
        dict
            {
                't': np.ndarray
                'y': np.ndarray
                'var_names': list
                'ko_gene': str
                'params': dict
            }
        """
        valid_ko = ['KDM5A', 'WTAP', 'NORAD', 'YTHDF2',
                    'E2F3', 'METTL3', 'PUM', 'PUM1', 'PUM2']
        if ko_gene not in valid_ko:
            raise ValueError(
                f"未知敲除基因: '{ko_gene}'. "
                f"可选: {valid_ko}"
            )

        def ode_func_ko(t, y):
            return m6a_ode_system(t, y, self.params,
                                  perturbation=None,
                                  ko_gene=ko_gene)

        sol = solve_ivp(
            ode_func_ko,
            t_span,
            self.y0,
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=5.0,
            **{k: v for k, v in kwargs.items()
               if k in ('method', 'rtol', 'atol', 'max_step')},
        )

        if not sol.success:
            sol = solve_ivp(
                ode_func_ko,
                t_span,
                self.y0,
                method='LSODA',
                rtol=1e-5,
                atol=1e-9,
                max_step=5.0,
            )

        return {
            't': sol.t,
            'y': sol.y,
            'var_names': VAR_NAMES,
            'ko_gene': ko_gene,
            'params': self.params.copy(),
        }

    def get_steady_state(
        self,
        perturbation: Optional[str] = None,
        t_max: float = 800,
    ) -> np.ndarray:
        """
        获取系统在扰动或无扰动下的稳态值.

        通过长时间积分 (t_max=800) 取最后时间点的值作为稳态估计.

        Parameters
        ----------
        perturbation : str or None
        t_max : float
            最大模拟时间

        Returns
        -------
        np.ndarray (13,)
            稳态变量值
        """
        res = self.simulate(t_span=(0, t_max), perturbation=perturbation)
        return res['y'][:, -1]

    def plot_epigenetics(
        self,
        sim_result: Dict[str, Any],
        figsize: Tuple[int, int] = (16, 12),
        output_path: Optional[str] = None,
        dpi: int = 150,
        show_legend: bool = True,
    ) -> plt.Figure:
        """
        绘制 m6A 表观转录组调控模型的模拟结果.

        分4个子图:
        - 图A: 染色质+转录 (KDM5A, H3K4me3, WTAP_mRNA, WTAP)
        - 图B: m6A-NORAD 轴 (NORAD_pre, NORAD_m6A, YTHDF2)
        - 图C: PUM-E2F3 轴 (PUM_free, PUM_bound, E2F3_mRNA, E2F3)
        - 图D: 衰老输出 (Senescence_score, SASP_score)

        Parameters
        ----------
        sim_result : dict
            simulate() 或 simulate_ko() 的返回值
        figsize : tuple
        output_path : str or None
        dpi : int
        show_legend : bool

        Returns
        -------
        plt.Figure
        """
        t = sim_result['t']
        y = sim_result['y']

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            'm6A 表观转录组调控 NPC 衰老 — 模拟结果\n'
            '(KDM5A → H3K4me3 → WTAP → m6A-NORAD → PUM隔离 → E2F3 → 衰老)',
            fontsize=13, fontweight='bold', y=0.98,
        )

        # ---- 图A: 染色质状态 + WTAP 转录 ----
        ax1 = axes[0, 0]
        ax1.plot(t, y[0], label='KDM5A', color='#8E44AD', linewidth=2)
        ax1.plot(t, y[1], label='H3K4me3_WTAP', color='#E74C3C', linewidth=2,
                 linestyle='--')
        ax1.plot(t, y[2], label='WTAP_mRNA', color='#3498DB', linewidth=2,
                 linestyle=':')
        ax1.plot(t, y[3], label='WTAP', color='#2ECC71', linewidth=2)
        ax1.set_ylabel('归一化水平', fontsize=10)
        ax1.set_title('A | 染色质修饰与 WTAP 表达', fontsize=11, fontweight='bold')
        ax1.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
        if show_legend:
            ax1.legend(fontsize=8, loc='best', framealpha=0.8)
        ax1.grid(alpha=0.2)

        # ---- 图B: m6A-NORAD 轴 ----
        ax2 = axes[0, 1]
        ax2.plot(t, y[4], label='NORAD_pre (未修饰)', color='#F39C12',
                 linewidth=2, linestyle='--')
        ax2.plot(t, y[5], label='NORAD_m6A (稳定型)', color='#E67E22',
                 linewidth=2.5, linestyle='-')
        ax2.plot(t, y[6], label='YTHDF2', color='#1ABC9C',
                 linewidth=2, linestyle=':')
        ax2.set_ylabel('归一化水平', fontsize=10)
        ax2.set_title('B | m6A 修饰与 NORAD 稳定性',
                       fontsize=11, fontweight='bold')
        if show_legend:
            ax2.legend(fontsize=8, loc='best', framealpha=0.8)
        ax2.grid(alpha=0.2)

        # ---- 图C: PUM-E2F3 轴 ----
        ax3 = axes[1, 0]
        ax3.plot(t, y[7], label='PUM_free', color='#2980B9',
                 linewidth=2, linestyle='-')
        ax3.plot(t, y[8], label='PUM_bound', color='#9B59B6',
                 linewidth=2, linestyle='--')
        ax3.plot(t, y[9], label='E2F3_mRNA', color='#E74C3C',
                 linewidth=2, linestyle=':')
        ax3.plot(t, y[10], label='E2F3', color='#C0392B',
                 linewidth=2.5, linestyle='-')
        ax3.set_xlabel('时间 (任意单位)', fontsize=10)
        ax3.set_ylabel('归一化水平', fontsize=10)
        ax3.set_title('C | PUM 分子海绵与 E2F3 释放',
                       fontsize=11, fontweight='bold')
        if show_legend:
            ax3.legend(fontsize=8, loc='best', framealpha=0.8)
        ax3.grid(alpha=0.2)

        # ---- 图D: 衰老输出 ----
        ax4 = axes[1, 1]
        ax4.plot(t, y[11], label='Senescence_score', color='#C0392B',
                 linewidth=2.5, linestyle='-')
        ax4.plot(t, y[12], label='SASP_score', color='#E74C3C',
                 linewidth=2, linestyle='--')
        ax4.set_xlabel('时间 (任意单位)', fontsize=10)
        ax4.set_ylabel('评分', fontsize=10)
        ax4.set_title('D | NPC 衰老与 SASP', fontsize=11, fontweight='bold')
        if show_legend:
            ax4.legend(fontsize=8, loc='best', framealpha=0.8)
        ax4.grid(alpha=0.2)

        # 标注扰动的相关信息
        pert_info = sim_result.get('perturbation')
        ko_info = sim_result.get('ko_gene')
        subtitle_parts = []
        if pert_info:
            subtitle_parts.append(f'扰动: {pert_info}')
        if ko_info:
            subtitle_parts.append(f'敲除: {ko_gene_name_cn(ko_info)}')

        fig.text(
            0.5, 0.005,
            ' | '.join(subtitle_parts) if subtitle_parts else '基础稳态模拟',
            ha='center', fontsize=10, fontstyle='italic',
            color='gray',
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 表观遗传学模拟图: {output_path}")

        return fig

    def compare_perturbations(
        self,
        perturbations: Optional[List[str]] = None,
        t_span: Tuple[float, float] = (0, 500),
        figsize: Tuple[int, int] = (18, 10),
        output_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        比较不同扰动对关键变量的影响 (柱状图).

        展示各扰动下系统稳态时的关键变量值 (相对对照的变化倍数).

        Parameters
        ----------
        perturbations : list or None
            要比较的扰动列表 (None 时使用全部)
        t_span : tuple
        figsize : tuple
        output_path : str or None
        dpi : int

        Returns
        -------
        plt.Figure
        """
        if perturbations is None:
            perturbations = ['KDM5A_OE', 'WTAP_OE', 'NORAD_OE',
                             'METTL3_KD', 'YTHDF2_KD', 'E2F3_KD']

        # 感兴趣的输出变量
        key_vars = ['WTAP', 'NORAD_m6A', 'PUM_free', 'E2F3',
                    'Senescence_score', 'SASP_score']
        key_indices = [VAR_NAMES.index(v) for v in key_vars]

        # 获取对照稳态
        ctrl = self.get_steady_state(perturbation=None, t_max=t_span[1])

        # 获取各扰动稳态
        results = {}
        for p in perturbations:
            res = self.get_steady_state(perturbation=p, t_max=t_span[1])
            results[p] = res

        # 绘制
        n_pert = len(perturbations)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            '扰动比较: 各干预下 m6A 表观调控网络稳态变化\n'
            '(倍率 vs 对照)',
            fontsize=14, fontweight='bold',
        )

        colors_pert = {
            'KDM5A_OE': '#8E44AD',
            'WTAP_OE': '#2ECC71',
            'NORAD_OE': '#E67E22',
            'METTL3_KD': '#3498DB',
            'YTHDF2_KD': '#1ABC9C',
            'E2F3_KD': '#E74C3C',
        }

        x = np.arange(n_pert)
        width = 0.6

        for idx, (var_name, var_idx) in enumerate(zip(key_vars, key_indices)):
            ax = axes[idx // 3, idx % 3]
            ctrl_val = ctrl[var_idx]

            fold_changes = []
            colors = []
            for p in perturbations:
                val = results[p][var_idx]
                fc = val / ctrl_val if abs(ctrl_val) > 1e-10 else 1.0
                fold_changes.append(fc)
                colors.append(colors_pert.get(p, '#95A5A6'))

            bars = ax.bar(x, fold_changes, width, color=colors, alpha=0.8,
                          edgecolor='white', linewidth=0.5)

            # 基线
            ax.axhline(1.0, color='black', linewidth=1, linestyle='--',
                       alpha=0.6, label='对照 (1.0x)')

            # 标注数值
            for bar, fc in zip(bars, fold_changes):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05 * (1 if fc >= 0 else -1),
                    f'{fc:.2f}x',
                    ha='center', va='bottom' if fc >= 0 else 'top',
                    fontsize=8, fontweight='bold',
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [p.replace('_', '\n') for p in perturbations],
                fontsize=7,
            )
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel('变化倍数 (Fold Change)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] 扰动比较图: {output_path}")

        return fig

    def compare_ko(
        self,
        ko_genes: Optional[List[str]] = None,
        t_span: Tuple[float, float] = (0, 500),
        figsize: Tuple[int, int] = (18, 10),
        output_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        比较不同基因敲除 (KO) 对关键变量的影响.

        Parameters
        ----------
        ko_genes : list or None
            要比较的敲除基因 (None 时使用全部)
        t_span : tuple
        figsize : tuple
        output_path : str or None
        dpi : int

        Returns
        -------
        plt.Figure
        """
        if ko_genes is None:
            ko_genes = ['KDM5A', 'WTAP', 'NORAD', 'YTHDF2', 'E2F3', 'METTL3']

        # 感兴趣的输出变量
        key_vars = ['WTAP', 'NORAD_m6A', 'PUM_free', 'E2F3',
                    'Senescence_score', 'SASP_score']
        key_indices = [VAR_NAMES.index(v) for v in key_vars]

        # 对照稳态
        ctrl = self.get_steady_state(t_max=t_span[1])

        # 各KO稳态
        ko_results = {}
        for g in ko_genes:
            res = self.simulate_ko(ko_gene=g, t_span=t_span)
            ko_results[g] = res['y'][:, -1]

        # 绘制
        n_ko = len(ko_genes)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            '基因敲除比较: 靶向干预下 m6A 表观调控网络稳态变化\n'
            '(倍率 vs 对照)',
            fontsize=14, fontweight='bold',
        )

        x = np.arange(n_ko)
        width = 0.6

        for idx, (var_name, var_idx) in enumerate(zip(key_vars, key_indices)):
            ax = axes[idx // 3, idx % 3]
            ctrl_val = ctrl[var_idx]

            fold_changes = []
            for g in ko_genes:
                val = ko_results[g][var_idx]
                fc = val / ctrl_val if abs(ctrl_val) > 1e-10 else 1.0
                fold_changes.append(fc)

            bars = ax.bar(x, fold_changes, width, color='#7F8C8D',
                          alpha=0.75, edgecolor='white', linewidth=0.5)

            ax.axhline(1.0, color='black', linewidth=1, linestyle='--',
                       alpha=0.5, label='对照')

            # 标注
            for bar, fc in zip(bars, fold_changes):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05 * (1 if fc >= 0 else -1),
                    f'{fc:.2f}x',
                    ha='center', va='bottom' if fc >= 0 else 'top',
                    fontsize=8, fontweight='bold',
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f'{g} KO' for g in ko_genes],
                fontsize=8,
            )
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel('变化倍数 (Fold Change)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"[✓] KO 比较图: {output_path}")

        return fig


# ============================================================
# 4. 辅助函数 (Utility Functions)
# ============================================================

def ko_gene_name_cn(ko_gene: str) -> str:
    """返回基因敲除的中文名称."""
    name_map = {
        'KDM5A': 'KDM5A',
        'WTAP': 'WTAP',
        'NORAD': 'NORAD',
        'YTHDF2': 'YTHDF2',
        'E2F3': 'E2F3',
        'METTL3': 'METTL3',
        'PUM': 'PUM1/2',
        'PUM1': 'PUM1',
        'PUM2': 'PUM2',
    }
    return name_map.get(ko_gene, ko_gene)


def simulate_ko_comparison(
    model: Optional[M6AEpigeneticModel] = None,
    ko_genes: Optional[List[str]] = None,
    t_span: Tuple[float, float] = (0, 500),
    output_dir: str = './output',
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    一键运行 KO 比较分析.

    Parameters
    ----------
    model : M6AEpigeneticModel or None
        (None 时自动创建默认模型)
    ko_genes : list or None
    t_span : tuple
    output_dir : str
    dpi : int

    Returns
    -------
    dict
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if model is None:
        model = M6AEpigeneticModel()

    print("=" * 55)
    print("  m6A 表观转录组调控模型 — KO 比较分析")
    print("  论文: Nat Commun 2022 | WTAP-NORAD-E2F3 轴")
    print("=" * 55)

    print(f"\n[1/3] 基础稳态模拟...")
    ctrl_res = model.simulate(t_span=t_span)
    fig_ctrl = model.plot_epigenetics(
        ctrl_res,
        output_path=os.path.join(output_dir, 'm6a_epigenetics_baseline.png'),
        dpi=dpi,
    )
    print(f"  → 基础稳态图已保存")

    # 打印稳态值
    y_final = ctrl_res['y'][:, -1]
    print(f"\n  稳态变量值:")
    for name, val in zip(VAR_NAMES, y_final):
        print(f"    {name:20s} = {val:.4f}")

    if ko_genes is None:
        ko_genes = ['KDM5A', 'WTAP', 'NORAD', 'YTHDF2', 'E2F3', 'METTL3']

    print(f"\n[2/3] 基因敲除模拟 ({len(ko_genes)} 个KO)...")
    ko_results = {}
    for g in ko_genes:
        ko_res = model.simulate_ko(ko_gene=g, t_span=t_span)
        ko_results[g] = ko_res

        # 打印关键变量变化
        y_ko = ko_res['y'][:, -1]
        print(f"\n  {g} KO:")
        for var, ctrl_v, ko_v in zip(VAR_NAMES, y_final, y_ko):
            if ctrl_v > 1e-10:
                fc = ko_v / ctrl_v
            else:
                fc = 0.0
            print(f"    {var:20s}: {ctrl_v:.4f} → {ko_v:.4f}  ({fc:.2f}x)")

    # KO 比较图
    fig_ko = model.compare_ko(
        ko_genes=ko_genes,
        t_span=t_span,
        output_path=os.path.join(output_dir, 'm6a_ko_comparison.png'),
        dpi=dpi,
    )
    print(f"\n  → KO 比较图已保存")

    print(f"\n[3/3] 扰动模拟...")
    perturbations = ['KDM5A_OE', 'WTAP_OE', 'NORAD_OE',
                     'METTL3_KD', 'YTHDF2_KD', 'E2F3_KD']
    for p in perturbations:
        pert_res = model.simulate(perturbation=p, t_span=t_span)
        y_pert = pert_res['y'][:, -1]
        print(f"\n  {p}:")
        for var, ctrl_v, pert_v in zip(VAR_NAMES, y_final, y_pert):
            fc = pert_v / ctrl_v if ctrl_v > 1e-10 else 0.0
            if abs(fc - 1.0) > 0.05:  # 只显示变化 >5% 的变量
                arrow = '↑' if fc > 1.0 else '↓'
                print(f"    {var:20s}: {ctrl_v:.4f} → {pert_v:.4f}  ({arrow} {fc:.2f}x)")

    # 扰动比较图
    fig_pert = model.compare_perturbations(
        perturbations=perturbations,
        t_span=t_span,
        output_path=os.path.join(output_dir, 'm6a_perturbation_comparison.png'),
        dpi=dpi,
    )
    print(f"\n  → 扰动比较图已保存")

    print(f"\n{'=' * 55}")
    print(f"  ✅ m6A 表观遗传学模型分析完成")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 55}")

    plt.close('all')

    return {
        'model': model,
        'ctrl_result': ctrl_res,
        'ko_results': ko_results,
        'output_dir': output_dir,
    }


# ============================================================
# 5. 快速演示 (Quick Demo)
# ============================================================

def quick_demo(output_dir: str = './output', dpi: int = 150):
    """
    快速演示 m6A 表观转录组调控模型的基本功能.
    运行基础模拟 + 一个KO + 一个扰动, 并绘图.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("  🧬 m6A Epigenetic Model — Quick Demo")
    print("  KDM5A → H3K4me3 → WTAP → m6A-NORAD → E2F3 → Senescence")
    print("=" * 50)

    # 创建模型
    model = M6AEpigeneticModel()

    # 基础模拟
    print("\n[1] 基础稳态模拟...")
    ctrl = model.simulate(t_span=(0, 400))
    model.plot_epigenetics(
        ctrl,
        output_path=os.path.join(output_dir, 'm6a_demo_baseline.png'),
        dpi=dpi,
    )
    y_final = ctrl['y'][:, -1]
    print("  变量 | 稳态值:")
    for name, val in zip(VAR_NAMES, y_final):
        print(f"    {name:20s} = {val:.4f}")

    # WTAP KO
    print("\n[2] WTAP 敲除模拟...")
    ko = model.simulate_ko(ko_gene='WTAP', t_span=(0, 400))
    model.plot_epigenetics(
        ko,
        output_path=os.path.join(output_dir, 'm6a_demo_WTAP_KO.png'),
        dpi=dpi,
    )
    y_ko = ko['y'][:, -1]
    for var, cv, kv in zip(VAR_NAMES, y_final, y_ko):
        fc = kv / cv if cv > 1e-10 else 0
        print(f"    {var:20s}: {cv:.4f} → {kv:.4f}  ({fc:.2f}x)")

    # NORAD 过表达
    print("\n[3] NORAD 过表达 (模拟退变状态)...")
    pert = model.simulate(perturbation='NORAD_OE', t_span=(0, 400))
    model.plot_epigenetics(
        pert,
        output_path=os.path.join(output_dir, 'm6a_demo_NORAD_OE.png'),
        dpi=dpi,
    )
    y_pert = pert['y'][:, -1]
    for var, cv, pv in zip(VAR_NAMES, y_final, y_pert):
        fc = pv / cv if cv > 1e-10 else 0
        arrow = '↑' if fc > 1.05 else ('↓' if fc < 0.95 else '→')
        print(f"    {var:20s}: {cv:.4f} → {pv:.4f}  ({arrow} {fc:.2f}x)")

    print(f"\n✅ 演示完成, 输出: {output_dir}")


if __name__ == '__main__':
    quick_demo()
