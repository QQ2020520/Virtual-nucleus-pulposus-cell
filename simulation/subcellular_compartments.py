"""
髓核细胞亚细胞区室整合 ODE 模型
=====================================
整合五个亚细胞系统：内质网应激/UPR、核纤层/染色质、自噬-溶酶体、
外泌体/细胞通讯、核酸动力学与DAMP信号。

设计理念:
  - 每个子系统输出其"健康评分" (0-1)
  - 子系统间通过关键耦合机制交互 (Ca²⁺, ATP, ROS, cfDNA, 细胞因子)
  - 支持独立模拟某个通路 (simulate_pathway)
  - 与现有 MitochondrialDynamicsModel, NPMetabolismModel 兼容
  - 支持多类型扰动 (内质网应激、核膜破裂、自噬抑制、外泌体传播)

ODE 稳定性保证:
  - 每个变量 dx/dt = production * (1 - x/capacity) - decay * x
  - 稳态解 x* = production / (production/capacity + decay)
  - 天然稳定在 [0, capacity] 区间，不会发散或硬性裁剪到边界

Author: Virtual NP Cell Team
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Union
import warnings
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# 中文字体配置
# ============================================================
plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 变量索引 (20 维状态空间)
# ============================================================
IDX = {
    # === 组1: 内质网应激 / UPR ===
    'ER_fold_load': 0,
    'ER_misfolded': 1,
    'BiP_GRP78': 2,
    'PERK_act': 3,
    'ATF6_act': 4,
    'CHOP': 5,

    # === 组2: 核纤层/染色质 ===
    'Lamin_A_C': 6,
    'Heterochromatin': 7,
    'Nuclear_envelope_rupture': 8,
    'cGAS_STING_act': 9,

    # === 组3: 自噬-溶酶体 ===
    'Autophagy_flux': 10,
    'Lysosomal_function': 11,
    'p62_accumulation': 12,
    'NLRP3_act': 13,

    # === 组4: 外泌体/细胞通讯 ===
    'Exosome_secretion': 14,
    'EV_miRNA_cargo': 15,
    'cfDNA_release': 16,
    'SASP_propagation': 17,

    # === 组5: 核酸动力学 ===
    'cfRNA_level': 18,
    'DAMPs_activation': 19,
}
N_VARS = 20

VAR_NAMES = {
    'ER_fold_load': 'ER Fold Load',
    'ER_misfolded': 'Misfolded Protein',
    'BiP_GRP78': 'BiP/GRP78',
    'PERK_act': 'PERK Activity',
    'ATF6_act': 'ATF6 Activity',
    'CHOP': 'CHOP',
    'Lamin_A_C': 'Lamin A/C',
    'Heterochromatin': 'Heterochromatin',
    'Nuclear_envelope_rupture': 'NE Rupture',
    'cGAS_STING_act': 'cGAS-STING',
    'Autophagy_flux': 'Autophagy Flux',
    'Lysosomal_function': 'Lysosomal Function',
    'p62_accumulation': 'p62 Accum.',
    'NLRP3_act': 'NLRP3',
    'Exosome_secretion': 'Exosome Secretion',
    'EV_miRNA_cargo': 'EV miRNA Cargo',
    'cfDNA_release': 'cfDNA Release',
    'SASP_propagation': 'SASP Propagation',
    'cfRNA_level': 'cfRNA Level',
    'DAMPs_activation': 'DAMPs Activation',
}

SYSTEM_GROUPS = {
    'ER_stress': ['ER_fold_load', 'ER_misfolded', 'BiP_GRP78',
                  'PERK_act', 'ATF6_act', 'CHOP'],
    'Nuclear': ['Lamin_A_C', 'Heterochromatin',
                'Nuclear_envelope_rupture', 'cGAS_STING_act'],
    'Autophagy_Lysosome': ['Autophagy_flux', 'Lysosomal_function',
                           'p62_accumulation', 'NLRP3_act'],
    'Exosome_Communication': ['Exosome_secretion', 'EV_miRNA_cargo',
                              'cfDNA_release', 'SASP_propagation'],
    'Nucleic_Acid_DAMP': ['cfRNA_level', 'DAMPs_activation'],
}

HEALTH_DIRECTION = {
    'ER_fold_load': -1, 'ER_misfolded': -1, 'BiP_GRP78': 1,
    'PERK_act': -1, 'ATF6_act': -1, 'CHOP': -1,
    'Lamin_A_C': 1, 'Heterochromatin': 1, 'Nuclear_envelope_rupture': -1,
    'cGAS_STING_act': -1, 'Autophagy_flux': 1, 'Lysosomal_function': 1,
    'p62_accumulation': -1, 'NLRP3_act': -1,
    'Exosome_secretion': 1, 'EV_miRNA_cargo': -1,
    'cfDNA_release': -1, 'SASP_propagation': -1,
    'cfRNA_level': -1, 'DAMPs_activation': -1,
}


class SubcellularCompartmentsModel:
    """
    亚细胞区室整合 ODE 模型 (20 维)

    五个子系统，每个变量使用 guaranteed-stable 形式:
      dx/dt = prod*(1-x/cap) - decay*x

    Parameters
    ----------
    params : dict, optional
        自定义参数，覆盖默认参数。
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = {
            # ============================================================
            # ALL PRODUCTION & DECAY RATES
            # ============================================================
            # ER fold load
            'fold_base': 0.025,      'fold_sox9': 0.020,
            'fold_decay': 0.12,      'fold_atp_stall': 0.25,
            # ER misfolded
            'misf_base': 0.005,      'misf_stress': 0.030,
            'misf_bip': 0.20,        'misf_erad': 0.15,
            'misf_autophagy': 0.08,
            # BiP
            'bip_base': 0.06,        'bip_upr': 0.12,
            'bip_bind': 0.20,        'bip_decay': 0.08,
            # PERK
            'perk_max': 0.20,        'perk_bip_th': 0.20,
            'perk_n': 3.0,           'perk_deact': 0.22,
            # ATF6
            'atf6_max': 0.18,        'atf6_bip_th': 0.25,
            'atf6_n': 2.0,           'atf6_deact': 0.22,
            # CHOP
            'chop_perk': 0.10,       'chop_atf6': 0.06,
            'chop_decay': 0.22,      'chop_apop_th': 0.60,
            # Lamin A/C
            'lamin_base': 0.080,     'lamin_progerin': 0.008,
            'lamin_oxi': 0.006,      'lamin_decay': 0.020,
            # Heterochromatin
            'hetero_maint': 0.08,    'hetero_lamin': 0.08,
            'hetero_loss': 0.06,     'hetero_stress': 0.06,
            'hetero_sirt': 0.06,
            # Nuclear envelope rupture
            'NE_rupt_lamin': 0.08,   'NE_lamin_th': 0.40,
            'NE_rupt_mech': 0.04,    'NE_repair': 0.22,
            # cGAS-STING
            'cgas_act': 0.12,        'cgas_cfdna_th': 0.20,
            'cgas_n': 2.0,           'sting_clear': 0.18,
            'sting_auto': 0.12,
            # Autophagy
            'auto_base': 0.100,      'auto_ampk': 0.08,
            'auto_mtor': 0.04,       'auto_stress': 0.05,
            'auto_sasp': 0.06,       'auto_decay': 0.10,
            # Lysosomal function
            'lyso_base': 0.05,       'lyso_auto': 0.08,
            'lyso_ros': 0.08,        'lyso_lipo': 0.05,
            'lyso_tfeb': 0.12,       'lyso_repair': 0.10,
            # p62
            'p62_base': 0.020,       'p62_trx': 0.06,
            'p62_bind': 0.08,        'p62_auto': 0.15,
            # NLRP3
            'nlrp3_prime': 0.06,     'nlrp3_lyso': 0.12,
            'nlrp3_lyso_th': 0.30,   'nlrp3_lyso_n': 3.0,
            'nlrp3_mito': 0.03,      'nlrp3_clear': 0.30,
            'nlrp3_auto': 0.12,
            # Exosome
            'exo_base': 0.015,       'exo_stress': 0.04,
            'exo_ca': 0.04,          'exo_sasp': 0.05,
            'exo_decay': 0.16,
            # EV miRNA
            'mir_sort': 0.03,        'mir_nfkb': 0.04,
            'mir_anti': 0.04,        'mir_decay': 0.25,
            # cfDNA
            'cfdna_ne': 0.08,        'cfdna_mito': 0.06,
            'cfdna_mito_ros': 0.06,  'cfdna_clear': 0.40,
            # SASP propagation
            'sasp_exo': 0.08,        'sasp_nfkb': 0.08,
            'sasp_il1b': 0.06,       'sasp_decay': 0.30,
            # cfRNA
            'cfrna_rel': 0.04,       'cfrna_rupt': 0.06,
            'cfrna_exo': 0.03,       'cfrna_clear': 0.35,
            # DAMPs
            'damp_cfdna': 0.06,      'damp_cfrna': 0.05,
            'damp_atp': 0.05,        'damp_hmgb1': 0.05,
            'damp_nlrp3': 0.06,      'damp_sasp': 0.05,
            'damp_clear': 0.30,
            'cap_ER_fold_load': 1.5,
            'cap_ER_misfolded': 0.6,
            'cap_BiP_GRP78': 2.0,
            'cap_PERK_act': 0.6,
            'cap_ATF6_act': 0.6,
            'cap_CHOP': 0.8,
            'cap_Lamin_A_C': 1.2,
            'cap_Heterochromatin': 1.2,
            'cap_Nuclear_envelope_rupture': 1.0,
            'cap_cGAS_STING_act': 1.0,
            'cap_Autophagy_flux': 1.0,
            'cap_Lysosomal_function': 1.2,
            'cap_p62_accumulation': 1.0,
            'cap_NLRP3_act': 1.0,
            'cap_Exosome_secretion': 1.2,
            'cap_EV_miRNA_cargo': 1.0,
            'cap_cfDNA_release': 1.0,
            'cap_SASP_propagation': 1.0,
            'cap_cfRNA_level': 1.0,
            'cap_DAMPs_activation': 1.0,

            'max_derivative': 2.0,
        }

        if params is not None:
            self.params.update(params)

        self.state_names = list(IDX.keys())
        self.var_indices = IDX
        self.system_groups = SYSTEM_GROUPS
        self.health_direction = HEALTH_DIRECTION

    # ============================================================
    # 辅助方法
    # ============================================================

    def _hill(self, x: float, K: float, n: float) -> float:
        return x**n / (K**n + x**n)

    def _cap(self, x: float, var_name: str) -> float:
        """Carrying capacity saturation: production*(1 - x/cap)"""
        cap = self.params.get(f'cap_{var_name}', 1.0)
        return max(0.05, 1.0 - x / max(cap, 0.01))

    def _cap_old(self, x: float, var_name: str) -> float:
        """Legacy capacity for variables not using the new scheme"""
        cap = self.params.get(f'cap_{var_name}', 0.5)
        return 1.0 / (1.0 + (x / cap)**2)

    # ============================================================
    # 外部耦合代理
    # ============================================================

    def _estimate_atp_proxy(self) -> float:
        return 0.80

    def _estimate_ros_proxy(self) -> float:
        return 0.15

    def _estimate_ca_cytosolic(self, er_fold: float) -> float:
        return 0.1 + 0.4 * er_fold / (1.0 + er_fold)

    # ============================================================
    # ODE 系统
    # ============================================================

    def ode_system(self, t: float, y: np.ndarray,
                   perturbation: Optional[str] = None,
                   external_state: Optional[Dict] = None) -> List[float]:
        """
        ODE 右侧 — 20 维亚细胞区室动力学

        每个变量 dx/dt = production * (1 - x/cap) - decay * x
        天然稳定，不会发散。
        """
        p = self.params

        # ---- 解包 & 裁剪 ----
        yc = np.clip(y, 0.001, None)
        (ER_fold, ER_misf, BiP, PERK, ATF6, CHOP,
         Lamin, Hetero, NE_rupt, cGAS_STING,
         Auto_flux, Lyso_fn, p62, NLRP3,
         Exo_sec, EV_mir, cfDNA, SASP_prop,
         cfRNA, DAMP) = yc

        # ---- 外部状态 ----
        if external_state is not None:
            atp = np.clip(external_state.get('atp', 0.80), 0.01, 1.2)
            ros = np.clip(external_state.get('ros', 0.15), 0.0, 2.0)
            sasp_ext = np.clip(external_state.get('sasp', 0.05), 0.0, 1.0)
            nfkb = np.clip(external_state.get('nfkb', 0.10), 0.0, 1.0)
        else:
            atp, ros, sasp_ext, nfkb = 0.80, 0.15, 0.05, 0.10

        ca_cyt = self._estimate_ca_cytosolic(ER_fold)

        # ---- 扰动调制 ----
        _er_mod = 1.0
        _lamin_mod = 1.0
        _auto_mod = 1.0
        _exo_mod = 1.0

        if perturbation == 'ER_stress':
            _er_mod = 3.5
        elif perturbation == 'nuclear_damage':
            _lamin_mod = 0.3
        elif perturbation == 'autophagy_block':
            _auto_mod = 0.2
        elif perturbation == 'exosome_boost':
            _exo_mod = 2.5
        elif perturbation == 'severe_degeneration':
            _er_mod = 3.0
            _lamin_mod = 0.4
            _auto_mod = 0.3
            _exo_mod = 2.0
            atp = min(atp, 0.3)
            ros = max(ros, 0.6)
            sasp_ext = max(sasp_ext, 0.5)
            nfkb = max(nfkb, 0.6)

        # ---- ATP efficiency ----
        atp_eff = atp / (atp + 0.30)

        # ---- BiP dissociation for UPR sensors ----
        bip_dis_perk = 1.0 - BiP / (BiP + p['perk_bip_th'])
        bip_dis_atf6 = 1.0 - BiP / (BiP + p['atf6_bip_th'])

        # ============================================================
        # 组1: 内质网应激 / UPR
        # ============================================================

        # 1. ER_fold_load
        fold_prod = (p['fold_base'] + p['fold_sox9']) * _er_mod
        fold_secret = p['fold_decay'] * ER_fold
        fold_stall = p['fold_atp_stall'] * (1.0 - atp_eff) * ER_fold * \
                     0.5  # attenuated
        d0 = fold_prod - fold_secret + fold_stall * self._cap(ER_fold, 'ER_fold_load')

        # 2. ER_misfolded
        misf_rate = (p['misf_base'] + p['misf_stress'] * ros * (1.0 - atp_eff)) * \
                    (1.0 + 0.5 * ER_fold / (1.0 + ER_fold))
        bip_rem = p['misf_bip'] * BiP * ER_misf / (ER_misf + 0.25)
        erad_rem = p['misf_erad'] * ER_misf
        auto_misf = p['misf_autophagy'] * Auto_flux * ER_misf
        d1 = misf_rate - bip_rem - erad_rem - auto_misf
        d1 *= self._cap(ER_misf, 'ER_misfolded')

        # 3. BiP_GRP78
        bip_upr = p['bip_upr'] * (0.6 * PERK + 0.4 * ATF6)
        bip_prod = (p['bip_base'] + bip_upr) * self._cap(BiP, 'BiP_GRP78')
        bip_bind = p['bip_bind'] * BiP * ER_misf / (ER_misf + 0.30)
        d2 = bip_prod - bip_bind - p['bip_decay'] * BiP

        # 4. PERK_act
        perk_act = p['perk_max'] * (bip_dis_perk ** p['perk_n'])
        d3 = perk_act - p['perk_deact'] * PERK
        d3 *= self._cap(PERK, 'PERK_act')

        # 5. ATF6_act
        atf6_act = p['atf6_max'] * (bip_dis_atf6 ** p['atf6_n'])
        d4 = atf6_act - p['atf6_deact'] * ATF6
        d4 *= self._cap(ATF6, 'ATF6_act')

        # 6. CHOP
        chop_prod = p['chop_perk'] * self._hill(PERK, 0.30, 2.0) + \
                    p['chop_atf6'] * ATF6
        d5 = chop_prod * self._cap(CHOP, 'CHOP') - p['chop_decay'] * CHOP

        # ============================================================
        # 组2: 核纤层/染色质
        # ============================================================

        # 7. Lamin_A_C
        lamin_prod = p['lamin_base'] * _lamin_mod * self._cap(Lamin, 'Lamin_A_C')
        lamin_loss = (p['lamin_progerin'] + p['lamin_oxi'] * ros) * Lamin
        d6 = lamin_prod - lamin_loss - p['lamin_decay'] * Lamin

        # 8. Heterochromatin
        hetero_gain = p['hetero_maint'] * Hetero * (1.0 - Hetero / 1.2) + \
                      p['hetero_lamin'] * Lamin * (1.0 - Hetero / 1.2) + \
                      p['hetero_sirt'] * Hetero * (1.0 - Hetero / 1.2)
        hetero_loss = (p['hetero_loss'] + p['hetero_stress'] * ros) * Hetero
        d7 = hetero_gain - hetero_loss

        # 9. Nuclear_envelope_rupture
        lamin_frag = p['NE_rupt_lamin'] * (1.0 - Lamin / (Lamin + p['NE_lamin_th']))
        rupture_drive = (lamin_frag + p['NE_rupt_mech']) * (1.0 - NE_rupt)
        rupture_repair = p['NE_repair'] * NE_rupt * (1.0 + 0.5 * Auto_flux)
        d8 = rupture_drive - rupture_repair

        # 10. cGAS_STING_act
        cgas_stim = cfDNA + 0.3 * DAMP
        cgas_act = p['cgas_act'] * self._hill(cgas_stim, p['cgas_cfdna_th'], p['cgas_n']) * \
                   (1.0 + 0.5 * ER_fold / (1.0 + ER_fold))
        sting_clear = (p['sting_clear'] + p['sting_auto'] * (1.0 - Auto_flux)) * cGAS_STING
        d9 = cgas_act * self._cap(cGAS_STING, 'cGAS_STING_act') - sting_clear

        # ============================================================
        # 组3: 自噬-溶酶体
        # ============================================================

        # 11. Autophagy_flux
        ampk = 1.0 / (1.0 + atp / 0.3)
        auto_ampk = p['auto_ampk'] * ampk
        auto_mtor = p['auto_mtor'] * atp / (atp + 0.4) * Auto_flux
        auto_stress = p['auto_stress'] * (PERK + 0.3 * ATF6)
        auto_sasp = p['auto_sasp'] * sasp_ext * Auto_flux
        auto_base = (p['auto_base'] + auto_ampk + auto_stress)
        if perturbation in ('autophagy_block', 'severe_degeneration'):
            auto_base *= _auto_mod
        d10 = auto_base * self._cap(Auto_flux, 'Autophagy_flux') - \
              auto_mtor - auto_sasp - p['auto_decay'] * Auto_flux

        # 12. Lysosomal_function
        lyso_gain = p['lyso_base'] * self._cap(Lyso_fn, 'Lysosomal_function') + \
                    p['lyso_auto'] * Auto_flux * self._cap(Lyso_fn, 'Lysosomal_function') + \
                    p['lyso_tfeb'] * (1.0 - Lyso_fn) * (1.0 + 0.5 * ampk)
        lyso_damage = p['lyso_ros'] * ros * Lyso_fn + \
                      p['lyso_lipo'] * p62 * Lyso_fn + \
                      p['lyso_repair'] * Lyso_fn
        d11 = lyso_gain - lyso_damage

        # 13. p62_accumulation
        p62_prod = (p['p62_base'] + p['p62_trx'] * ros / (ros + 0.30))
        p62_cargo = p['p62_bind'] * (ER_misf + 0.3 * DAMP)
        p62_deg = p['p62_auto'] * Auto_flux * p62
        d12 = (p62_prod + p62_cargo) * self._cap(p62, 'p62_accumulation') - p62_deg

        # 14. NLRP3_act
        nlrp3_prime = p['nlrp3_prime'] * nfkb * (1.0 + 0.5 * sasp_ext)
        lyso_dys = (1.0 - Lyso_fn) / (1.0 - Lyso_fn + p['nlrp3_lyso_th'])
        nlrp3_lyso = p['nlrp3_lyso'] * (lyso_dys ** p['nlrp3_lyso_n'])
        nlrp3_mito = p['nlrp3_mito'] * (ros + 0.5 * DAMP)
        nlrp3_clear = (p['nlrp3_clear'] + p['nlrp3_auto'] * Auto_flux) * NLRP3
        d13 = (nlrp3_prime + nlrp3_lyso + nlrp3_mito) * \
              self._cap(NLRP3, 'NLRP3_act') - nlrp3_clear

        # ============================================================
        # 组4: 外泌体/细胞通讯
        # ============================================================

        # 15. Exosome_secretion
        exo_prod = p['exo_base'] * self._cap(Exo_sec, 'Exosome_secretion') + \
                   p['exo_stress'] * (ros + 0.3 * ER_fold) * self._cap(Exo_sec, 'Exosome_secretion') + \
                   p['exo_ca'] * ca_cyt / (ca_cyt + 0.2) + \
                   p['exo_sasp'] * sasp_ext * self._cap(Exo_sec, 'Exosome_secretion')
        d14 = exo_prod * _exo_mod - p['exo_decay'] * Exo_sec

        # 16. EV_miRNA_cargo
        mir_sort = p['mir_sort'] * (1.0 + 0.5 * sasp_ext) * self._cap(EV_mir, 'EV_miRNA_cargo')
        mir_pro = p['mir_nfkb'] * nfkb * self._cap(EV_mir, 'EV_miRNA_cargo')
        mir_anti = p['mir_anti'] / (1.0 + nfkb)
        d15 = mir_sort + mir_pro + mir_anti - p['mir_decay'] * EV_mir

        # 17. cfDNA_release
        cfdna_prod = p['cfdna_ne'] * NE_rupt + \
                     p['cfdna_mito'] * (1.0 - atp + 0.3 * ros) + \
                     p['cfdna_mito_ros'] * ros * (1.0 - atp)
        if perturbation == 'nuclear_damage':
            cfdna_prod *= 3.0
        d16 = cfdna_prod * self._cap(cfDNA, 'cfDNA_release') - p['cfdna_clear'] * cfDNA

        # 18. SASP_propagation
        sasp_prod = p['sasp_exo'] * Exo_sec * EV_mir + \
                    p['sasp_nfkb'] * nfkb * (1.0 + sasp_ext) + \
                    p['sasp_il1b'] * NLRP3
        d17 = sasp_prod * self._cap(SASP_prop, 'SASP_propagation') - p['sasp_decay'] * SASP_prop

        # ============================================================
        # 组5: 核酸动力学 / DAMP
        # ============================================================

        # 19. cfRNA_level
        cfrna_prod = p['cfrna_rel'] * (0.5 * ros + 0.3 * ER_fold) + \
                     p['cfrna_rupt'] * NE_rupt + \
                     p['cfrna_exo'] * Exo_sec
        d18 = cfrna_prod * self._cap(cfRNA, 'cfRNA_level') - p['cfrna_clear'] * cfRNA

        # 20. DAMPs_activation
        damp_prod = p['damp_cfdna'] * cfDNA / (cfDNA + 0.2) + \
                    p['damp_cfrna'] * cfRNA / (cfRNA + 0.2) + \
                    p['damp_atp'] * (1.0 - atp / (atp + 0.3)) + \
                    p['damp_hmgb1'] * (NE_rupt + 0.3 * PERK) + \
                    p['damp_nlrp3'] * NLRP3 + \
                    p['damp_sasp'] * SASP_prop
        d19 = damp_prod * self._cap(DAMP, 'DAMPs_activation') - p['damp_clear'] * DAMP

        # ---- 裁剪并返回 ----
        max_d = p['max_derivative']
        dy = np.clip(np.array([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9,
                               d10, d11, d12, d13, d14, d15, d16, d17, d18, d19],
                              dtype=float),
                     -max_d, max_d)

        return dy.tolist()

    # ============================================================
    # 初始条件
    # ============================================================

    def _get_initial_conditions(self, perturbation: Optional[str] = None,
                                healthy_level: float = 1.0) -> np.ndarray:
        h = healthy_level

        if perturbation in ('severe_degeneration',):
            y0 = np.array([
                0.90, 0.60, 0.30, 0.70, 0.55, 0.50,
                0.30, 0.30, 0.40, 0.50,
                0.15, 0.25, 0.85, 0.60,
                0.70, 0.65, 0.55, 0.60,
                0.40, 0.60,
            ])
        elif perturbation in ('ER_stress',):
            y0 = np.array([
                0.75, 0.35, 0.55, 0.30, 0.25, 0.10,
                0.80, 0.65, 0.02, 0.08,
                0.50, 0.65, 0.25, 0.10,
                0.35, 0.30, 0.08, 0.08,
                0.06, 0.08,
            ])
        elif perturbation in ('nuclear_damage',):
            y0 = np.array([
                0.50, 0.08, 0.75, 0.08, 0.08, 0.03,
                0.40, 0.50, 0.15, 0.15,
                0.50, 0.60, 0.30, 0.10,
                0.30, 0.30, 0.15, 0.08,
                0.10, 0.15,
            ])
        elif perturbation in ('autophagy_block',):
            y0 = np.array([
                0.50, 0.10, 0.70, 0.10, 0.08, 0.04,
                0.80, 0.70, 0.02, 0.10,
                0.50, 0.60, 0.35, 0.10,
                0.30, 0.30, 0.04, 0.05,
                0.05, 0.06,
            ])
        elif perturbation in ('exosome_boost',):
            y0 = np.array([
                0.50, 0.08, 0.70, 0.08, 0.08, 0.03,
                0.80, 0.70, 0.02, 0.06,
                0.55, 0.65, 0.25, 0.06,
                0.40, 0.40, 0.04, 0.08,
                0.05, 0.06,
            ])
        else:
            # 健康基线
            y0 = np.array([
                0.50*h, 0.08*h, 0.80*h, 0.08*h, 0.08*h, 0.05*h,
                0.80*h, 0.70*h, 0.02*h, 0.06*h,
                0.55*h, 0.65*h, 0.25*h, 0.06*h,
                0.30*h, 0.30*h, 0.04*h, 0.05*h,
                0.05*h, 0.06*h,
            ])

        return np.clip(y0, 0.001, None)

    # ============================================================
    # 主仿真接口
    # ============================================================

    def simulate(self, t_span: Tuple[float, float] = (0.0, 200.0),
                 n_points: int = 500,
                 perturbation: Optional[str] = None,
                 method: str = 'BDF',
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 initial_state: Optional[np.ndarray] = None,
                 external_state: Optional[Dict] = None) -> Dict:
        if initial_state is not None:
            y0 = np.array(initial_state, dtype=float)
        else:
            y0 = self._get_initial_conditions(perturbation)

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            self.ode_system, t_span, y0,
            method=method, t_eval=t_eval,
            rtol=rtol, atol=atol,
            args=(perturbation, external_state),
        )

        y_safe = sol.y.copy()
        for i in range(N_VARS):
            y_safe[i, :] = np.clip(y_safe[i, :], 0.0, 2.0)

        return {
            't': sol.t, 'y': y_safe,
            'params': self.params.copy(),
            'perturbation': perturbation,
            'success': sol.success, 'message': sol.message,
        }

    # ============================================================
    # 独立通路模拟
    # ============================================================

    def simulate_pathway(self, pathway: str = 'ER_stress',
                         severity: float = 0.5) -> Dict:
        pert_map = {
            'ER_stress': 'ER_stress',
            'nuclear_damage': 'nuclear_damage',
            'autophagy_block': 'autophagy_block',
            'exosome_boost': 'exosome_boost',
        }
        if pathway not in pert_map:
            raise ValueError(f"Unknown pathway '{pathway}'. "
                             f"Valid: {list(pert_map.keys())}")

        base_pert = pert_map[pathway]
        y0 = self._get_initial_conditions(perturbation=None)

        # 通过参数调整模拟严重程度
        params_mod = self.params.copy()
        if pathway == 'ER_stress':
            params_mod['fold_base'] = 0.008 * (1.0 + severity * 3.0)
        elif pathway == 'nuclear_damage':
            params_mod['lamin_base'] = 0.035 * (1.0 - severity * 0.7)
        elif pathway == 'autophagy_block':
            params_mod['auto_base'] = 0.03 * (1.0 - severity * 0.8)
        elif pathway == 'exosome_boost':
            params_mod['exo_base'] = 0.015 * (1.0 + severity * 4.0)

        temp_model = SubcellularCompartmentsModel(params_mod)
        result = temp_model.simulate(
            perturbation=base_pert if severity > 0.3 else None,
            initial_state=y0,
            t_span=(0, 200),
            n_points=500,
        )
        return result

    # ============================================================
    # 状态解读与健康评分
    # ============================================================

    def compute_health_score(self, var_name: str, value: float) -> float:
        direction = self.health_direction.get(var_name, 1)
        if direction == 1:
            return float(np.clip(value, 0.0, 1.0))
        else:
            return float(np.clip(1.0 - value, 0.0, 1.0))

    def get_subcellular_state(self, y: np.ndarray) -> Dict[str, float]:
        state = {}
        for system_name, var_list in self.system_groups.items():
            scores = [self.compute_health_score(v, float(y[self.var_indices[v]]))
                      for v in var_list]
            state[f'{system_name}_health'] = float(np.mean(scores))

        all_scores = [self.compute_health_score(v, float(y[self.var_indices[v]]))
                      for v in self.state_names]
        state['overall_subcellular_health'] = float(np.mean(all_scores))

        idx = self.var_indices
        state['UPR_activation'] = float(
            (y[idx['PERK_act']] + y[idx['ATF6_act']] + y[idx['CHOP']]) / 3.0
        )
        state['nuclear_integrity'] = float(
            (y[idx['Lamin_A_C']] + y[idx['Heterochromatin']] +
             (1.0 - y[idx['Nuclear_envelope_rupture']])) / 3.0
        )
        state['autophagy_lysosome_health'] = float(
            (y[idx['Autophagy_flux']] + y[idx['Lysosomal_function']] +
             (1.0 - y[idx['p62_accumulation']]) + (1.0 - y[idx['NLRP3_act']])) / 4.0
        )
        state['inflammatory_burden'] = float(
            (y[idx['NLRP3_act']] + y[idx['cGAS_STING_act']] +
             y[idx['DAMPs_activation']] + y[idx['SASP_propagation']]) / 4.0
        )
        state['ER_stress_severity'] = float(
            y[idx['CHOP']] / self.params['chop_apop_th']
        )
        return state

    # ============================================================
    # 扰动仿真
    # ============================================================

    def simulate_perturbation(self, pert_type: str) -> Dict:
        valid = ['ER_stress', 'nuclear_damage', 'autophagy_block',
                 'exosome_boost', 'severe_degeneration']
        if pert_type not in valid:
            raise ValueError(f"Unknown perturbation '{pert_type}'."
                             f"Valid: {valid}")
        return self.simulate(t_span=(0.0, 300.0), n_points=500,
                             perturbation=pert_type)

    # ============================================================
    # 绘图方法
    # ============================================================

    def plot_subcellular_overview(self, result: Dict,
                                  output_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (18, 14)) -> str:
        t = result['t']; y = result['y']
        pert_label = result.get('perturbation', 'none')
        n_rows, n_cols = 5, 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        system_colors = {'ER_stress': '#e41a1c', 'Nuclear': '#377eb8',
                         'Autophagy_Lysosome': '#4daf4a',
                         'Exosome_Communication': '#984ea3',
                         'Nucleic_Acid_DAMP': '#ff7f00'}
        var_color = {}
        for sys_name, var_list in self.system_groups.items():
            for var_name in var_list:
                var_color[var_name] = system_colors.get(sys_name, '#333')

        for i, var_name in enumerate(self.state_names):
            ax = axes[i]
            ax.plot(t, y[self.var_indices[var_name], :],
                    color=var_color.get(var_name, '#333'), linewidth=1.5)
            ax.set_title(VAR_NAMES.get(var_name, var_name), fontsize=9, fontweight='bold')
            ax.set_xlabel('Time', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            fv = y[self.var_indices[var_name], -1]
            ax.axhline(y=fv, color=var_color.get(var_name, '#333'),
                       linestyle='--', linewidth=0.6, alpha=0.4)
            ax.annotate(f'{fv:.3f}', xy=(t[-1], fv),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color=var_color.get(var_name, '#333'))

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=c, linewidth=3, label=s)
            for s, c in system_colors.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f'Subcellular Compartments — 20 Variables\n'
                     f'Perturbation: {pert_label}',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.close(fig)
        return None

    def plot_upr_pathway(self, result: Dict, output_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 10)) -> str:
        t = result['t']; y = result['y']
        pert_label = result.get('perturbation', 'none')
        idx = self.var_indices

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(t, y[idx['ER_fold_load'], :], '#e41a1c', lw=2, label='Fold Load')
        ax.plot(t, y[idx['ER_misfolded'], :], '#ff7f00', lw=2, label='Misfolded')
        ax.set_title('ER Protein Folding', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, y[idx['BiP_GRP78'], :], '#377eb8', lw=2, label='BiP/GRP78')
        ax.set_title('BiP/GRP78 Chaperone', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(t, y[idx['PERK_act'], :], '#e41a1c', lw=2, label='PERK')
        ax.set_title('PERK → eIF2α → ATF4', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.plot(t, y[idx['ATF6_act'], :], '#4daf4a', lw=2, label='ATF6')
        ax.set_title('ATF6 → Golgi → ERAD/XBP1', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[4]
        ax.plot(t, y[idx['CHOP'], :], '#984ea3', lw=2, label='CHOP')
        th = self.params['chop_apop_th']
        ax.axhline(y=th, color='red', ls='--', lw=1.5, alpha=0.7,
                   label=f'Apoptosis threshold ({th})')
        ax.set_title('CHOP — UPR→Apoptosis Switch', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[5]
        upr = (y[idx['PERK_act']] + y[idx['ATF6_act']] + y[idx['CHOP']]) / 3.0
        ax.plot(t, upr, '#333333', lw=2.5, label='UPR Index')
        ax.plot(t, y[idx['BiP_GRP78'], :], '#377eb8', lw=1.5, alpha=0.6, label='BiP')
        ax.set_title('UPR Index (PERK+ATF6+CHOP)/3', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle(f'ER Stress / UPR Pathway | Perturbation: {pert_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.close(fig)
        return None

    def plot_autophagy_lysosome(self, result: Dict, output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (14, 8)) -> str:
        t = result['t']; y = result['y']
        pert_label = result.get('perturbation', 'none')
        idx = self.var_indices
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0, 0]
        ax.plot(t, y[idx['Autophagy_flux'], :], '#4daf4a', lw=2, label='Autophagy')
        ax.plot(t, y[idx['Lysosomal_function'], :], '#377eb8', lw=2, ls='--', label='Lysosomal')
        ax.set_title('Autophagy-Lysosome Axis', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(t, y[idx['p62_accumulation'], :], '#ff7f00', lw=2, label='p62')
        ax.plot(t, y[idx['NLRP3_act'], :], '#e41a1c', lw=2, ls='--', label='NLRP3')
        ax.set_title('p62 Accumulation & NLRP3', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        autoph = y[idx['Autophagy_flux'], :]
        nlrp = y[idx['NLRP3_act'], :]
        ax.scatter(autoph, nlrp, c=t, cmap='viridis', s=10, alpha=0.7)
        ax.set_xlabel('Autophagy Flux', fontsize=10)
        ax.set_ylabel('NLRP3 Activity', fontsize=10)
        ax.set_title('Autophagy ↔ NLRP3 Coupling', fontsize=12, fontweight='bold')
        plt.colorbar(ax.collections[0], ax=ax).set_label('Time', fontsize=8)

        ax = axes[1, 1]
        health = ((y[idx['Autophagy_flux']] + y[idx['Lysosomal_function']] +
                   (1.0 - y[idx['p62_accumulation']]) + (1.0 - y[idx['NLRP3_act']])) / 4.0)
        ax.plot(t, health, '#4daf4a', lw=2.5, label='A-L Health Score')
        ax.axhline(y=0.5, color='red', ls='--', lw=1, alpha=0.6)
        ax.set_title('Autophagy-Lysosome Health', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle(f'Autophagy-Lysosome | Perturbation: {pert_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.close(fig)
        return None

    def plot_nuclear_damage(self, result: Dict, output_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 8)) -> str:
        t = result['t']; y = result['y']
        pert_label = result.get('perturbation', 'none')
        idx = self.var_indices
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0, 0]
        ax.plot(t, y[idx['Lamin_A_C'], :], '#377eb8', lw=2, label='Lamin A/C')
        ax.plot(t, y[idx['Heterochromatin'], :], '#4daf4a', lw=2, ls='--', label='Heterochromatin')
        ax.set_title('Nuclear Lamina & Chromatin', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(t, y[idx['Nuclear_envelope_rupture'], :], '#e41a1c', lw=2, label='NE Rupture')
        ax.plot(t, y[idx['cGAS_STING_act'], :], '#984ea3', lw=2, ls='--', label='cGAS-STING')
        ax.set_title('NE Rupture → cGAS-STING', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(t, y[idx['cfDNA_release'], :], '#ff7f00', lw=2, label='cfDNA')
        ax.plot(t, y[idx['cfRNA_level'], :], '#a65628', lw=2, ls='--', label='cfRNA')
        ax.plot(t, y[idx['DAMPs_activation'], :], '#f781bf', lw=2, ls=':', label='DAMPs')
        ax.set_title('Nucleic Acid Release & DAMPs', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        nh = (y[idx['Lamin_A_C']] + y[idx['Heterochromatin']] +
              (1.0 - y[idx['Nuclear_envelope_rupture']]) +
              (1.0 - y[idx['cGAS_STING_act']])) / 4.0
        ib = (y[idx['NLRP3_act']] + y[idx['cGAS_STING_act']] +
              y[idx['DAMPs_activation']]) / 3.0
        ax.plot(t, nh, '#377eb8', lw=2.5, label='Nuclear Health')
        ax.plot(t, ib, '#e41a1c', lw=2.5, ls='--', label='Inflammatory Burden')
        ax.set_title('Nuclear Health & Inflammation', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05)

        fig.suptitle(f'Nuclear Damage & cfDNA-STING | Perturbation: {pert_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.close(fig)
        return None

    def plot_exosome_communication(self, result: Dict, output_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (14, 8)) -> str:
        t = result['t']; y = result['y']
        pert_label = result.get('perturbation', 'none')
        idx = self.var_indices
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0, 0]
        ax.plot(t, y[idx['Exosome_secretion'], :], '#984ea3', lw=2, label='Exosome')
        ax.plot(t, y[idx['EV_miRNA_cargo'], :], '#ff7f00', lw=2, ls='--', label='miRNA (pro-inflam)')
        ax.set_title('Exosome Secretion & Cargo', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(t, y[idx['SASP_propagation'], :], '#e41a1c', lw=2, label='SASP Propagation')
        ax.set_title('SASP Propagation via Exosomes', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(t, y[idx['cfDNA_release'], :], '#a65628', lw=2, label='cfDNA Release')
        ax.set_title('cfDNA (Nuclear + mt)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(t, y[idx['DAMPs_activation'], :], '#f781bf', lw=2, label='DAMPs')
        ax.plot(t, y[idx['SASP_propagation'], :], '#e41a1c', lw=2, ls='--', label='SASP')
        ax.set_title('DAMPs & SASP: Paracrine Danger', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle(f'Exosome Communication | Perturbation: {pert_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return output_path
        plt.close(fig)
        return None


# ============================================================
# 模块级辅助函数
# ============================================================

def run_all_perturbations(model: Optional[SubcellularCompartmentsModel] = None) -> Dict:
    if model is None:
        model = SubcellularCompartmentsModel()

    types = ['ER_stress', 'nuclear_damage', 'autophagy_block',
             'exosome_boost', 'severe_degeneration']
    results = {}
    for pt in types:
        print(f"  Running '{pt}'...")
        results[pt] = model.simulate_perturbation(pt)
    print("  Running 'healthy_baseline'...")
    results['healthy_baseline'] = model.simulate()

    summary = {}
    for name, res in results.items():
        yf = res['y'][:, -1]
        summary[name] = model.get_subcellular_state(yf)
    return {'results': results, 'summary': summary}


# ============================================================
# 快速测试入口
# ============================================================

if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)

    print("=" * 60)
    print("亚细胞区室整合模型 — 快速测试")
    print("=" * 60)

    model = SubcellularCompartmentsModel()
    print(f"\n[1] 初始化模型: {N_VARS} variables, 5 subsystems")

    # 2. 健康基线
    print("\n[2] 健康基线仿真...")
    res_base = model.simulate()
    s = model.get_subcellular_state(res_base['y'][:, -1])
    print(f"  Overall health: {s['overall_subcellular_health']:.3f}")
    for sys_name in model.system_groups:
        print(f"  {sys_name:30s} health: {s[f'{sys_name}_health']:.3f}")
    print(f"  Success: {res_base['success']}")

    # 变量终值
    print("\n  Variable final values:")
    for name in model.state_names:
        v = res_base['y'][model.var_indices[name], -1]
        print(f"    {name:30s} = {v:.4f}")

    # 3. 扰动
    print("\n[3] 扰动仿真...")
    for pert in ['ER_stress', 'nuclear_damage', 'autophagy_block',
                 'exosome_boost', 'severe_degeneration']:
        res = model.simulate_perturbation(pert)
        s = model.get_subcellular_state(res['y'][:, -1])
        print(f"  {pert:25s} | Overall={s['overall_subcellular_health']:.3f} "
              f"| Inflam={s['inflammatory_burden']:.3f} "
              f"| UPR={s['UPR_activation']:.3f}")

    # 4. 独立通路
    print("\n[4] 独立通路模拟...")
    for pw in ['ER_stress', 'nuclear_damage', 'autophagy_block', 'exosome_boost']:
        res = model.simulate_pathway(pw, severity=0.6)
        s = model.get_subcellular_state(res['y'][:, -1])
        print(f"  {pw:25s} | {pw}_health={s[f'{pw}_health']:.3f} "
              f"| Overall={s['overall_subcellular_health']:.3f}")

    # 5. 绘图
    print("\n[5] 绘图...")
    model.plot_subcellular_overview(res_base, 'output/subcellular_overview.png')
    model.plot_upr_pathway(res_base, 'output/upr_pathway.png')
    model.plot_autophagy_lysosome(res_base, 'output/autophagy_lysosome.png')
    model.plot_nuclear_damage(res_base, 'output/nuclear_damage.png')
    rd = model.simulate_perturbation('severe_degeneration')
    model.plot_exosome_communication(rd, 'output/exosome_communication.png')
    print("  All plots saved.")

    # 6. 对比
    print("\n[6] 全面扰动对比...")
    all_res = run_all_perturbations(model)
    print("\n  Perturbation comparison:")
    hdr = f"  {'Condition':25s} | {'Overall':8s} | {'ER Stress':10s} | {'Nuclear':9s} | {'A-L':9s} | {'Inflam':8s}"
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    for name, ss in all_res['summary'].items():
        print(f"  {name:25s} | {ss['overall_subcellular_health']:.4f}  | "
              f"{ss['ER_stress_health']:.4f}   | "
              f"{ss['Nuclear_health']:.4f}  | "
              f"{ss['Autophagy_Lysosome_health']:.4f}  | "
              f"{ss['inflammatory_burden']:.4f}")

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
