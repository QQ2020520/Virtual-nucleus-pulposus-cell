"""
Multi-Scale Integrator for Virtual NP Cell
============================================
Iterative feedback bridge connecting 4 independent ODE systems:

  1. coupled_engine.py  — 23D (mechanotransduction + signaling + metabolism + senescence + ECM)
  2. mitochondrial_dynamics.py — 14D (fission/fusion/Δψm/SIRT3/PINK1/apoptosis)
  3. subcellular_compartments.py — 20D (ER stress/UPR/nuclear/cfDNA/autophagy/exosome)
  4. rna_dynamics.py — 16D (stress granules/RBP/NEAT1/LLPS/translation)

Design: Iterative coupled feedback — run each module sequentially,
map outputs → inputs, repeat until convergence.

Author: Virtual NP Cell Team
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, Tuple, List, Union, Any
from copy import deepcopy
import warnings
import os
import sys

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Import 4 models (with path setup)
# ============================================================
_sim_dir = os.path.dirname(os.path.abspath(__file__))
_reg_dir = os.path.join(os.path.dirname(_sim_dir), 'regulation')
if _sim_dir not in sys.path:
    sys.path.insert(0, _sim_dir)
if _reg_dir not in sys.path:
    sys.path.insert(0, _reg_dir)

from coupled_engine import NPCoupledModel, Var as CoupledVar, VAR_CN as COUPLED_CN
from mitochondrial_dynamics import (
    MitochondrialDynamicsModel,
    IDX as MitoIDX,
    VAR_NAMES as MITO_NAMES,
)
from subcellular_compartments import (
    SubcellularCompartmentsModel,
    IDX as SubcellIDX,
    VAR_NAMES as SUBCELL_NAMES,
)
from rna_dynamics import (
    RNADynamicsModel,
    VAR_NAMES as RNA_NAMES,
)


# ============================================================
# 2. Interface Mapping (documented constants)
# ============================================================

# coupled_engine outputs → other modules:
#   ATP (coupled y[9])     → mito: Δψm recovery,  subcell: external_state['atp']
#   NFKB (coupled y[5])    → subcell: external_state['nfkb'],  RNA: SASP_mRNA transcription
#   ROS_mito (coupled y[10]) → mito: oxidative stress↑,  subcell: external_state['ros']
#   SASP_score (coupled y[15]) → subcell: external_state['sasp'],  RNA: SASP burden
#
# Other modules → coupled_engine:
#   Mito.miROS                          → coupled ROS_mito (oxidative_stress↑)
#   Mito.Mito_membrane_potential        → coupled ATP (ψm↓ → ATP↓)
#   Subcell.CHOP                        → coupled senescence (ER stress → aging)
#   Subcell.cfDNA_release / DAMPs       → coupled NFKB (cGAS-STING → NF-κB)
#   Subcell.Autophagy_flux              → coupled AMPK / senescence
#   RNA.ECM_mRNA                        → coupled ECM_synth / Aggrecan
#   RNA.Stress_granule                  → coupled ECM_synth (SG traps ECM mRNA)

# coupled engine perturbation dict keys:
PERT_KEYS = [
    'condition_factor', 'ampk_act', 'nfkb_inh', 'rox_inh',
    'senolytic', 'mmp_inh', 'tnf_stim', 'il1b_stim', 'oxidative_stress'
]

# Coupled engine variable indices for reference:
C_ATP = 9
C_NFKB = 5
C_ROS = 10
C_SASP = 15
C_SENESCENCE = 18
C_ECM_SYNTH = 8
C_AGGRECAN = 21
C_COL2 = 22
C_SOX9 = 6
C_AMPK = 3
C_MMP = 7

# Mito indices (from IDX):
M_MIROS = 5
M_MEMBRANE_POTENTIAL = 4
M_FRAGMENTATION = 3
M_SIRT3 = 8
M_PINK1 = 9
M_APOPTOSIS = 12  # CytC_release
M_MITO_MASS = 0

# Subcell indices:
S_CHOP = 5
S_CFDNA = 16
S_DAMPS = 19
S_AUTOPHAGY = 10
S_NLRP3 = 13
S_CGAS_STING = 9
S_NUCLEAR_RUPTURE = 8
S_EXOSOME = 14
S_SASP_PROP = 17

# RNA indices:
R_ECM_MRNA = 1
R_SASP_MRNA = 2
R_STRESS_GRANULE = 6
R_HUR = 8
R_TTP = 9
R_NEAT1 = 4
R_LLPS = 11
R_TRANSLATION = 13


class MultiScaleIntegrator:
    """
    Multi-Scale Integrator

    Iteratively runs 4 NP cell ODE modules (coupled_engine, mitochondrial_dynamics,
    subcellular_compartments, rna_dynamics) with feedback coupling.

    The core method run_coupled_iteration() implements:
      Step 1: Run coupled_engine → baseline steady state
      Step 2: Map coupled outputs → mito/subcell/RNA inputs, run all 3 modules
      Step 3: Map mito/subcell/RNA outputs → coupled perturbation parameters
      Step 4: Repeat until convergence (iterations times)
    """

    def __init__(self):
        """Load all 4 models with try/except"""
        self.models_loaded = {}

        try:
            self.coupled = NPCoupledModel()
            self.models_loaded['coupled'] = True
        except Exception as e:
            self.coupled = None
            self.models_loaded['coupled'] = False
            print(f"[MSI] Warning: coupled_engine not loaded: {e}")

        try:
            self.mito = MitochondrialDynamicsModel()
            self.models_loaded['mito'] = True
        except Exception as e:
            self.mito = None
            self.models_loaded['mito'] = False
            print(f"[MSI] Warning: mitochondrial_dynamics not loaded: {e}")

        try:
            self.subcell = SubcellularCompartmentsModel()
            self.models_loaded['subcell'] = True
        except Exception as e:
            self.subcell = None
            self.models_loaded['subcell'] = False
            print(f"[MSI] Warning: subcellular_compartments not loaded: {e}")

        try:
            self.rna = RNADynamicsModel()
            self.models_loaded['rna'] = True
        except Exception as e:
            self.rna = None
            self.models_loaded['rna'] = False
            print(f"[MSI] Warning: rna_dynamics not loaded: {e}")

        loaded = [k for k, v in self.models_loaded.items() if v]
        print(f"[MSI] Models loaded: {loaded}")

    # ============================================================
    # 3. Interface Mapping Methods
    # ============================================================

    def _coupled_to_mito_params(self, coupled_ss: np.ndarray) -> Dict[str, float]:
        """
        Map coupled_engine steady state → mito model parameter adjustments.

        ATP (coupled y9):  high → faster Δψm recovery & fusion
        ROS (coupled y10): high → increased miROS production & fission
        """
        if self.mito is None:
            return {}
        atp = np.clip(coupled_ss[C_ATP] / 1.0, 0.2, 1.5)
        ros = np.clip(coupled_ss[C_ROS] / 0.15, 0.5, 4.0)

        params = dict(self.mito.params)
        # ATP → Δψm recovery & fusion
        params['k_mmp_recovery'] = self.mito.params['k_mmp_recovery'] * atp
        params['k_atp_fusion'] = self.mito.params['k_atp_fusion'] * atp
        # ROS → miROS production, fission drive, mtDNA damage
        params['k_ros_basal'] = self.mito.params['k_ros_basal'] * ros
        params['k_ros_dysfunction'] = self.mito.params['k_ros_dysfunction'] * np.clip(
            coupled_ss[C_ROS] / 0.2, 0.5, 3.0
        )
        params['k_ros_fission'] = self.mito.params['k_ros_fission'] * np.clip(
            coupled_ss[C_ROS] / 0.15, 0.5, 3.0
        )
        return params

    def _coupled_to_subcell_external_state(self, coupled_ss: np.ndarray) -> Dict[str, float]:
        """
        Map coupled_engine steady state → subcellular external_state dict.
        """
        return {
            'atp': np.clip(coupled_ss[C_ATP], 0.01, 1.2),
            'nfkb': np.clip(coupled_ss[C_NFKB], 0.0, 1.0),
            'ros': np.clip(coupled_ss[C_ROS], 0.0, 2.0),
            'sasp': np.clip(coupled_ss[C_SASP], 0.0, 1.0),
        }

    def _coupled_to_rna_params(self, coupled_ss: np.ndarray) -> Dict[str, float]:
        """
        Map coupled_engine steady state → RNA model parameter adjustments.

        NFKB (y5):  drives SASP_mRNA transcription
        SASP (y15): adds to stress input
        SOX9 (y6): drives ECM_mRNA transcription
        """
        if self.rna is None:
            return {}
        nfkb = np.clip(coupled_ss[C_NFKB], 0.01, 1.0)
        sasp = np.clip(coupled_ss[C_SASP], 0.0, 1.0)
        sox9 = np.clip(coupled_ss[C_SOX9], 0.01, 1.0)
        ros = np.clip(coupled_ss[C_ROS], 0.0, 2.0)

        params = dict(self.rna.params)
        params['nfkb_signal'] = nfkb
        params['sox9_activity'] = sox9
        # Stress input from ROS + SASP
        params['stress_input'] = np.clip(0.03 + 0.2 * sasp + 0.15 * ros, 0.01, 0.8)
        # p38 activity modulated by SASP (inflammatory signaling)
        params['p38_activity'] = np.clip(0.2 + 0.4 * nfkb, 0.05, 0.8)
        return params

    # ---- Feedback: other modules → coupled_engine perturbation ----

    def _mito_feedback(self, mito_ss: np.ndarray) -> Dict[str, float]:
        """
        Mito → coupled perturbation:
          miROS → oxidative_stress (additive)
          Low Δψm → energy stress → ampk_act (compensatory)
          CytC/apoptosis → pro-inflammatory (il1b_stim)
        """
        pert = {}
        if mito_ss is None:
            return pert

        miros = np.clip(mito_ss[M_MIROS], 0.0, 2.0)
        psi = np.clip(mito_ss[M_MEMBRANE_POTENTIAL], 0.05, 1.0)
        cytc = np.clip(mito_ss[M_APOPTOSIS], 0.0, 1.0)

        # miROS → coupled ROS (oxidative stress)
        pert['oxidative_stress'] = np.clip(miros * 0.4, 0.0, 0.7)

        # Low Δψm → ATP deficit → AMPK activation (compensatory)
        if psi < 0.7:
            pert['ampk_act'] = np.clip((0.7 - psi) * 0.5, 0.0, 0.5)

        # Severe mito stress → inflammatory signaling
        if cytc > 0.3:
            pert['il1b_stim'] = np.clip(cytc * 0.3, 0.0, 0.4)
        if miros > 0.8:
            pert['il1b_stim'] = np.clip(
                pert.get('il1b_stim', 0.0) + miros * 0.1, 0.0, 0.5
            )

        return pert

    def _subcell_feedback(self, subcell_ss: np.ndarray) -> Tuple[Dict[str, float], float]:
        """
        Subcellular → coupled perturbation:
          CHOP               → ER stress → senescence (return chop_val)
          cfDNA/DAMPs         → cGAS-STING → NF-κB activation (il1b_stim)
          Low autophagy       → less AMPK, more senescence
          NLRP3               → IL-1β signaling
        """
        pert = {}
        chop_val = 0.0
        if subcell_ss is None:
            return pert, chop_val

        chop = np.clip(subcell_ss[S_CHOP], 0.0, 1.0)
        cfdna = np.clip(subcell_ss[S_CFDNA], 0.0, 1.0)
        damp = np.clip(subcell_ss[S_DAMPS], 0.0, 1.0)
        autophagy = np.clip(subcell_ss[S_AUTOPHAGY], 0.0, 1.0)
        nlrp3 = np.clip(subcell_ss[S_NLRP3], 0.0, 1.0)

        chop_val = chop

        # cfDNA/DAMPs → IL-1β / NF-κB (cGAS-STING pathway)
        cgas_drive = np.clip(cfdna * 0.4 + damp * 0.2, 0.0, 0.5)
        pert['il1b_stim'] = np.clip(cgas_drive + nlrp3 * 0.3, 0.0, 0.6)

        # Low autophagy → less AMPK → more senescence
        if autophagy < 0.4:
            existing_ampk = pert.get('ampk_act', 0.0)
            pert['ampk_act'] = np.clip(existing_ampk - (0.4 - autophagy) * 0.2, -0.3, 0.5)

        # High CHOP → increase effective condition_factor (ER stress → senescence)
        # This is returned separately and applied to condition_factor

        return pert, chop_val

    def _rna_feedback(self, rna_ss: np.ndarray) -> Dict[str, float]:
        """
        RNA → coupled perturbation:
          Low ECM_mRNA      → reduced ECM synthesis capacity (simulated via mmp_inh mod)
          Stress_granule    → SG traps ECM mRNA → reduce ECM capacity
          High SASP_mRNA    → amplifies inflammatory signaling
        """
        pert = {}
        if rna_ss is None:
            return pert

        ecm_mrna = np.clip(rna_ss[R_ECM_MRNA], 0.0, 5.0)
        sasp_mrna = np.clip(rna_ss[R_SASP_MRNA], 0.0, 5.0)
        sg = np.clip(rna_ss[R_STRESS_GRANULE], 0.0, 1.0)

        # Low ECM_mRNA → ECM deficiency
        if ecm_mrna < 0.5:
            pert['mmp_inh'] = np.clip((1.0 - ecm_mrna / 0.5) * 0.2, -0.4, 0.0)

        # Stress granule → increased stress/ECM deficit
        if sg > 0.2:
            pert['oxidative_stress'] = np.clip(
                pert.get('oxidative_stress', 0.0) + sg * 0.2, 0.0, 0.6
            )

        # High SASP_mRNA → inflammatory burden
        if sasp_mrna > 1.0:
            pert['il1b_stim'] = np.clip(
                pert.get('il1b_stim', 0.0) + (sasp_mrna / 5.0) * 0.15, 0.0, 0.5
            )

        return pert

    def _apply_perturbation(self, base_pert: Dict[str, float],
                            feedback: Dict[str, float],
                            alpha: float = 0.3) -> Dict[str, float]:
        """
        Merge feedback perturbation into base perturbation with damping factor alpha.
        """
        result = dict(base_pert)
        for k, v in feedback.items():
            if k in result and k != 'condition_factor':
                result[k] = np.clip(result.get(k, 0.0) + v * alpha, -0.5, 1.0)
            elif k != 'condition_factor':
                result[k] = np.clip(v * alpha, -0.5, 1.0)
        return result

    # ============================================================
    # 4. Core Method: run_coupled_iteration
    # ============================================================

    def run_coupled_iteration(self, condition: str = 'normal',
                              iterations: int = 3) -> Dict[str, Any]:
        """
        Core iterative feedback coupling method.

        Parameters
        ----------
        condition : str
            'normal', 'early_degeneration', or 'late_degeneration'
        iterations : int
            Number of feedback iterations (default 3)

        Returns
        -------
        dict with iteration-by-iteration state history
        """
        cf_map = {
            'normal': 0.0,
            'early_degeneration': 0.4,
            'late_degeneration': 0.8
        }
        base_cf = cf_map.get(condition, 0.0)

        # Initialize perturbation dict
        coupled_pert = {
            'condition_factor': base_cf,
            'ampk_act': 0.0, 'nfkb_inh': 0.0, 'rox_inh': 0.0,
            'senolytic': 0.0, 'mmp_inh': 0.0,
            'tnf_stim': 0.0, 'il1b_stim': 0.0, 'oxidative_stress': 0.0,
        }

        # Store iteration-by-iteration snapshots
        history = {
            'condition': condition,
            'iterations': iterations,
            'coupled_ss': [],
            'mito_ss': [],
            'subcell_ss': [],
            'rna_ss': [],
            'coupled_pert': [],
            'convergence_metrics': [],
            'error_log': [],
            'feedback_matrix': [],  # tracks what feedback was applied each iteration
        }

        # Start from healthy initial state for iteration 1
        init_coupled = None

        for it in range(iterations):
            it_label = f"[{condition}] Iter {it + 1}/{iterations}"
            fb_record = {}

            # ---- Step 1: Run coupled engine ----
            try:
                coupled_res = self.coupled.simulate(
                    sim_type=condition,
                    perturbation=coupled_pert,
                    initial_state=init_coupled
                )
                coupled_ss = coupled_res[condition]['steady_state']
                init_coupled = coupled_ss  # carry forward for next iteration's initial state
                history['coupled_ss'].append(coupled_ss)
                history['coupled_pert'].append(dict(coupled_pert))
            except Exception as e:
                history['error_log'].append(f"Iter {it + 1} coupled: {e}")
                print(f"  {it_label} ⚠ coupled error: {e}")
                break

            # ---- Step 2a: Run mitochondria ----
            mito_ss = None
            try:
                mito_params = self._coupled_to_mito_params(coupled_ss)
                mito_temp = MitochondrialDynamicsModel(params=mito_params)
                mito_res = mito_temp.simulate(t_span=(0, 200), n_points=500)
                mito_ss = mito_res['y'][:, -1]
                history['mito_ss'].append(mito_ss)
            except Exception as e:
                history['error_log'].append(f"Iter {it + 1} mito: {e}")
                mito_ss = None
                history['mito_ss'].append(None)

            # ---- Step 2b: Run subcellular compartments ----
            subcell_ss = None
            try:
                ext_state = self._coupled_to_subcell_external_state(coupled_ss)
                # Choose subcell perturbation based on condition severity
                subcell_pert = None
                if base_cf > 0.6:
                    subcell_pert = 'severe_degeneration'
                elif base_cf > 0.2:
                    subcell_pert = 'ER_stress'
                subcell_res = self.subcell.simulate(
                    t_span=(0, 200), n_points=500,
                    perturbation=subcell_pert,
                    external_state=ext_state,
                )
                subcell_ss = subcell_res['y'][:, -1]
                history['subcell_ss'].append(subcell_ss)
            except Exception as e:
                history['error_log'].append(f"Iter {it + 1} subcell: {e}")
                subcell_ss = None
                history['subcell_ss'].append(None)

            # ---- Step 2c: Run RNA dynamics ----
            rna_ss = None
            try:
                rna_params = self._coupled_to_rna_params(coupled_ss)
                rna_temp = RNADynamicsModel(params=rna_params)
                rna_res = rna_temp.simulate(t_span=(0, 200), n_points=500)
                rna_ss = rna_res['y'][:, -1]
                history['rna_ss'].append(rna_ss)
            except Exception as e:
                history['error_log'].append(f"Iter {it + 1} rna: {e}")
                rna_ss = None
                history['rna_ss'].append(None)

            # ---- Step 3: Feedback - maps module outputs → coupled perturbation ----
            feedback_merged = {}

            # 3a: Mito feedback
            mito_fb = self._mito_feedback(mito_ss)
            feedback_merged.update(mito_fb)
            fb_record['mito'] = dict(mito_fb)

            # 3b: Subcell feedback
            subcell_fb, chop_val = self._subcell_feedback(subcell_ss)
            feedback_merged.update(subcell_fb)
            fb_record['subcell'] = dict(subcell_fb)
            fb_record['chop_val'] = chop_val

            # CHOP → senescence boost (modulate condition_factor)
            if subcell_ss is not None and chop_val > 0.3:
                chop_sen_boost = np.clip((chop_val - 0.3) * 0.15, 0.0, 0.4)
                coupled_pert['condition_factor'] = np.clip(base_cf + chop_sen_boost, 0.0, 1.0)
                fb_record['chop_sen_boost'] = chop_sen_boost

            # 3c: RNA feedback
            rna_fb = self._rna_feedback(rna_ss)
            feedback_merged.update(rna_fb)
            fb_record['rna'] = dict(rna_fb)

            # Apply accumulated feedback with damping
            coupled_pert = self._apply_perturbation(coupled_pert, feedback_merged, alpha=0.3)
            history['feedback_matrix'].append(fb_record)

            # ---- Compute convergence metrics ----
            if it > 0:
                prev_ss = history['coupled_ss'][-2]
                curr_ss = coupled_ss
                delta = np.max(np.abs(curr_ss - prev_ss))
                history['convergence_metrics'].append(delta)
            else:
                history['convergence_metrics'].append(1.0)  # first iteration

            print(f"  {it_label} | "
                  f"cf={coupled_pert['condition_factor']:.3f} "
                  f"ox_stress={coupled_pert['oxidative_stress']:.3f} "
                  f"il1b={coupled_pert['il1b_stim']:.3f} "
                  f"ampk={coupled_pert['ampk_act']:.3f}")

        return history

    # ============================================================
    # 5. get_integrated_state — 50+ dimensional state vector
    # ============================================================

    def get_integrated_state(self, iteration_results: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Return a 50+ dimensional integrated state dictionary from iteration results.

        Uses the final iteration's steady states.
        """
        coupled_ss = iteration_results['coupled_ss'][-1] if iteration_results['coupled_ss'] else None
        mito_ss = iteration_results['mito_ss'][-1] if iteration_results['mito_ss'] else None
        subcell_ss = iteration_results['subcell_ss'][-1] if iteration_results['subcell_ss'] else None
        rna_ss = iteration_results['rna_ss'][-1] if iteration_results['rna_ss'] else None

        state = {}

        # --- Mechano (force transduction) ---
        if coupled_ss is not None:
            state['mechano'] = {
                'stiffness_signal': float(coupled_ss[0]),
                'MRTFA_nuclear': float(coupled_ss[1]),
                'Kidins220': float(coupled_ss[2]),
                'AMPK_P': float(coupled_ss[3]),
                'PFKFB3': float(coupled_ss[4]),
            }
            state['inflammation'] = {
                'NFKB': float(coupled_ss[5]),
                'SOX9': float(coupled_ss[6]),
                'MMP': float(coupled_ss[7]),
            }
            state['metabolic'] = {
                'ATP': float(coupled_ss[9]),
                'ROS_mito': float(coupled_ss[10]),
                'Lactate': float(coupled_ss[11]),
                'HIF1_alpha': float(coupled_ss[12]),
            }
            state['senescence'] = {
                'p53': float(coupled_ss[13]),
                'p21': float(coupled_ss[14]),
                'SASP_score': float(coupled_ss[15]),
                'IL1B': float(coupled_ss[16]),
                'TNF': float(coupled_ss[17]),
                'senescence_score': float(coupled_ss[18]),
            }
            state['ecm'] = {
                'ECM_stiffness': float(coupled_ss[19]),
                'ECM_degradation': float(coupled_ss[20]),
                'Aggrecan': float(coupled_ss[21]),
                'Col2': float(coupled_ss[22]),
            }
        else:
            state['mechano'] = state['inflammation'] = state['metabolic'] = {}
            state['senescence'] = state['ecm'] = {}

        # --- Mitochondrial ---
        if mito_ss is not None:
            state['mitochondrial'] = {
                'mito_mass': float(mito_ss[M_MITO_MASS]),
                'fission': float(mito_ss[MitoIDX['Mito_fission']]),
                'fusion': float(mito_ss[MitoIDX['Mito_fusion']]),
                'fragmentation': float(mito_ss[M_FRAGMENTATION]),
                'membrane_potential': float(mito_ss[M_MEMBRANE_POTENTIAL]),
                'miROS': float(mito_ss[M_MIROS]),
                'mtDNA_damage': float(mito_ss[MitoIDX['mtDNA_damage']]),
                'NAD_NADH_ratio': float(mito_ss[MitoIDX['NAD_NADH_ratio']]),
                'SIRT3_activity': float(mito_ss[M_SIRT3]),
                'PINK1_level': float(mito_ss[M_PINK1]),
                'Parkin_recruit': float(mito_ss[MitoIDX['Parkin_recruit']]),
                'mitophagy_flux': float(mito_ss[MitoIDX['Mitophagy_flux']]),
                'apoptosis_risk': float(mito_ss[M_APOPTOSIS]),
                'biogenesis': float(mito_ss[MitoIDX['Mito_biogenesis']]),
            }
        else:
            state['mitochondrial'] = {}

        # --- Subcellular ---
        if subcell_ss is not None:
            state['subcellular'] = {
                'ER_fold_load': float(subcell_ss[0]),
                'ER_misfolded': float(subcell_ss[1]),
                'BiP_GRP78': float(subcell_ss[2]),
                'PERK_act': float(subcell_ss[3]),
                'ATF6_act': float(subcell_ss[4]),
                'CHOP': float(subcell_ss[5]),
                'Lamin_A_C': float(subcell_ss[6]),
                'Heterochromatin': float(subcell_ss[7]),
                'NE_rupture': float(subcell_ss[8]),
                'cGAS_STING': float(subcell_ss[9]),
                'Autophagy_flux': float(subcell_ss[10]),
                'Lysosomal_function': float(subcell_ss[11]),
                'p62_accumulation': float(subcell_ss[12]),
                'NLRP3': float(subcell_ss[13]),
                'Exosome_secretion': float(subcell_ss[14]),
                'EV_miRNA_cargo': float(subcell_ss[15]),
                'cfDNA_release': float(subcell_ss[16]),
                'SASP_propagation': float(subcell_ss[17]),
                'cfRNA_level': float(subcell_ss[18]),
                'DAMPs_activation': float(subcell_ss[19]),
            }
        else:
            state['subcellular'] = {}

        # --- RNA Dynamics ---
        if rna_ss is not None:
            state['rna_dynamics'] = {
                'Total_mRNA_pool': float(rna_ss[0]),
                'ECM_mRNA': float(rna_ss[1]),
                'SASP_mRNA': float(rna_ss[2]),
                'miRNA_machinery': float(rna_ss[3]),
                'NEAT1': float(rna_ss[4]),
                'P_body': float(rna_ss[5]),
                'Stress_granule': float(rna_ss[6]),
                'SG_mRNA_trapping': float(rna_ss[7]),
                'HuR': float(rna_ss[8]),
                'TTP': float(rna_ss[9]),
                'IGF2BP': float(rna_ss[10]),
                'LLPS_condensates': float(rna_ss[11]),
                'Nuclear_export': float(rna_ss[12]),
                'Translation_efficiency': float(rna_ss[13]),
                'circRNA_buffer': float(rna_ss[14]),
                'eIF2a_phos': float(rna_ss[15]),
            }
        else:
            state['rna_dynamics'] = {}

        return state

    # ============================================================
    # 6. compare_conditions
    # ============================================================

    def compare_conditions(self, conditions: List[str] = None):
        """
        Run all specified conditions and compare 50+ dimensional states.
        Prints a summary comparison table.

        Parameters
        ----------
        conditions : list of str or None
            Default: ['normal', 'early_degeneration', 'late_degeneration']
        """
        if conditions is None:
            conditions = ['normal', 'early_degeneration', 'late_degeneration']

        print("\n" + "=" * 90)
        print("Multi-Scale Condition Comparison")
        print("=" * 90)

        all_results = {}
        for cond in conditions:
            print(f"\n  Running: {cond}...")
            res = self.run_coupled_iteration(condition=cond, iterations=3)
            state = self.get_integrated_state(res)
            all_results[cond] = state

        # Build comparison table by category
        categories = ['mechano', 'inflammation', 'metabolic', 'senescence',
                      'ecm', 'mitochondrial', 'subcellular', 'rna_dynamics']
        key_metrics = {
            'mechano': ['AMPK_P', 'MRTFA_nuclear'],
            'inflammation': ['NFKB', 'SOX9', 'MMP'],
            'metabolic': ['ATP', 'ROS_mito'],
            'senescence': ['senescence_score', 'SASP_score', 'p53'],
            'ecm': ['ECM_stiffness', 'Aggrecan', 'Col2'],
            'mitochondrial': ['fragmentation', 'membrane_potential', 'miROS', 'SIRT3_activity'],
            'subcellular': ['CHOP', 'Autophagy_flux', 'cfDNA_release', 'NLRP3'],
            'rna_dynamics': ['ECM_mRNA', 'SASP_mRNA', 'Stress_granule', 'HuR'],
        }

        print(f"\n{'Category':20s}", end='')
        for cond in conditions:
            print(f" | {cond:25s}", end='')
        print()
        print("-" * (20 + (27 * len(conditions))))

        for cat in categories:
            if cat not in key_metrics:
                continue
            for metric in key_metrics[cat]:
                row = f"  {cat}.{metric:25s}"
                for cond in conditions:
                    val = all_results[cond].get(cat, {}).get(metric, 'N/A')
                    if isinstance(val, float):
                        row += f" | {val:8.4f}          "
                    else:
                        row += f" | {'N/A':>8}          "
                print(row)
            print("-" * (20 + (27 * len(conditions))))

        return all_results

    # ============================================================
    # 7. get_cross_talk_network
    # ============================================================

    def get_cross_talk_network(self) -> Dict[str, List[Dict]]:
        """
        Return the cross-talk information flow graph between modules.

        Returns dict with 'nodes' and 'edges' describing what variables
        flow from which module to which module.
        """
        nodes = [
            {'id': 'coupled', 'label': 'Coupled Engine (23D)', 'color': '#3498DB'},
            {'id': 'mito', 'label': 'Mitochondria (14D)', 'color': '#E74C3C'},
            {'id': 'subcell', 'label': 'Subcellular (20D)', 'color': '#2ECC71'},
            {'id': 'rna', 'label': 'RNA Dynamics (16D)', 'color': '#9B59B6'},
        ]

        edges = [
            # coupled → others
            {'source': 'coupled', 'target': 'mito', 'vars': 'ATP, ROS_mito',
             'effect': 'ATP→Δψm↑, ROS→miROS↑', 'direction': 'forward'},
            {'source': 'coupled', 'target': 'subcell', 'vars': 'ATP, NFKB, ROS, SASP',
             'effect': 'external_state: atp, nfkb, ros, sasp', 'direction': 'forward'},
            {'source': 'coupled', 'target': 'rna', 'vars': 'NFKB, SASP, SOX9, ROS',
             'effect': 'NFKB→SASP_mRNA↑, SOX9→ECM_mRNA↑, SASP→stress', 'direction': 'forward'},
            # others → coupled
            {'source': 'mito', 'target': 'coupled', 'vars': 'miROS, Δψm, CytC',
             'effect': 'miROS→oxidative_stress, Δψm↓→ATP↓, CytC→IL-1β',
             'direction': 'feedback'},
            {'source': 'subcell', 'target': 'coupled', 'vars': 'CHOP, cfDNA, DAMPs, Autophagy, NLRP3',
             'effect': 'CHOP→senescence, cfDNA→NF-κB, Autophagy→AMPK/senescence',
             'direction': 'feedback'},
            {'source': 'rna', 'target': 'coupled', 'vars': 'ECM_mRNA, SASP_mRNA, SG',
             'effect': 'ECM_mRNA↓→ECM deficit, SASP_mRNA→inflammation, SG→ECM trapping',
             'direction': 'feedback'},
            # cross-talk between non-coupled modules
            {'source': 'mito', 'target': 'subcell', 'vars': 'miROS, Δψm, CytC',
             'effect': 'miROS→ER stress, Δψm→ATP→subcell', 'direction': 'indirect'},
            {'source': 'subcell', 'target': 'rna', 'vars': 'ER stress, cfDNA, DAMPs',
             'effect': 'ER stress→eIF2α-P→SG, DAMPs→NF-κB→RNA', 'direction': 'indirect'},
            {'source': 'rna', 'target': 'subcell', 'vars': 'SG, LLPS, HuR',
             'effect': 'SG→stress signals→subcell', 'direction': 'indirect'},
        ]

        return {'nodes': nodes, 'edges': edges}

    # ============================================================
    # 8. simulate_intervention
    # ============================================================

    def simulate_intervention(self, target: str, strength: float = 1.0,
                              condition: str = 'late_degeneration',
                              iterations: int = 5) -> Dict[str, Any]:
        """
        Simulate a therapeutic intervention in the integrated system.

        Supported interventions:
          'AMPK_act'    → AMPK activation (metformin analog)
          'Senolytic'   → senescence clearance (dasatinib + quercetin)
          'MitoQ'       → mitochondrial ROS scavenger
          'NFKB_inh'    → NF-κB inhibition
          'MMP_inh'     → MMP inhibition
          'Autophagy'   → autophagy activation (rapamycin analog)
          'HuR_boost'   → HuR stabilization (ECM mRNA protection)
          'NEAT1_KD'    → NEAT1 knockdown (reduce miRNA sponge)

        Returns control vs intervention comparison.
        """
        control_res = self.run_coupled_iteration(condition=condition, iterations=iterations)
        control_state = self.get_integrated_state(control_res)

        # Build intervention perturbation
        pert_overlay = {}
        target_desc = target

        if target == 'AMPK_act' or target == 'AMPK':
            pert_overlay['ampk_act'] = np.clip(strength * 0.8, 0.0, 1.0)
            target_desc = 'AMPK激活 (Metformin)'
        elif target == 'Senolytic':
            pert_overlay['senolytic'] = np.clip(strength * 0.5, 0.0, 1.0)
            target_desc = 'Senolytic (D+Q)'
        elif target == 'MitoQ':
            pert_overlay['rox_inh'] = np.clip(strength * 0.8, 0.0, 1.0)
            target_desc = 'MitoQ (线粒体ROS清除)'
        elif target == 'NFKB_inh' or target == 'NFKB':
            pert_overlay['nfkb_inh'] = np.clip(strength * 0.6, 0.0, 1.0)
            target_desc = 'NF-κB抑制 (TAK1i)'
        elif target == 'MMP_inh' or target == 'MMP':
            pert_overlay['mmp_inh'] = np.clip(strength * 0.7, 0.0, 1.0)
            target_desc = 'MMP抑制 (Doxycycline)'
        elif target == 'Autophagy':
            pert_overlay['ampk_act'] = np.clip(strength * 0.6, 0.0, 1.0)
            target_desc = '自噬激活 (Rapamycin)'
        elif target == 'HuR_boost':
            # Boost ECM_mRNA by modifying rna params; simulate in coupled via condition_factor reduction
            pert_overlay['mmp_inh'] = np.clip(strength * 0.2, 0.0, 0.5)
            target_desc = 'HuR增强 (ECM mRNA保护)'
        elif target == 'NEAT1_KD':
            # NEAT1 knockdown → more available miRNA → less ECM degradation
            pert_overlay['mmp_inh'] = np.clip(strength * 0.3, 0.0, 0.5)
            target_desc = 'NEAT1敲低 (miRNA海绵↓)'
        else:
            raise ValueError(f"Unknown intervention: {target}. "
                             f"Supported: AMPK_act, Senolytic, MitoQ, NFKB_inh, "
                             f"MMP_inh, Autophagy, HuR_boost, NEAT1_KD")

        # Run intervention
        cf_map = {'normal': 0.0, 'early_degeneration': 0.4, 'late_degeneration': 0.8}
        base_cf = cf_map.get(condition, 0.8)

        coupled_pert = {
            'condition_factor': base_cf,
            'ampk_act': pert_overlay.get('ampk_act', 0.0),
            'nfkb_inh': pert_overlay.get('nfkb_inh', 0.0),
            'rox_inh': pert_overlay.get('rox_inh', 0.0),
            'senolytic': pert_overlay.get('senolytic', 0.0),
            'mmp_inh': pert_overlay.get('mmp_inh', 0.0),
            'tnf_stim': 0.0, 'il1b_stim': 0.0, 'oxidative_stress': 0.0,
        }

        # For interventions that aren't directly in coupled_pert, we need
        # to modify the other modules. Create custom intervention.
        custom_mito_params = None
        custom_rna_params = None

        if target == 'MitoQ':
            # Also affect mito module directly
            custom_mito_params = {'k_ros_scavenge': self.mito.params['k_ros_scavenge'] * (1.0 + strength)}
        elif target == 'HuR_boost':
            custom_rna_params = {'k_hur_syn': self.rna.params['k_hur_syn'] * (1.0 + strength * 0.5)}
        elif target == 'NEAT1_KD':
            custom_rna_params = {'k_neat1_syn': self.rna.params['k_neat1_syn'] * max(0.1, 1.0 - strength * 0.7)}

        intervention_state = self._run_with_custom_params(
            condition=condition,
            coupled_pert=coupled_pert,
            custom_mito_params=custom_mito_params,
            custom_rna_params=custom_rna_params,
            iterations=iterations,
        )
        intervention_state = self.get_integrated_state(intervention_state)

        # Compute delta between control and intervention
        delta = {}
        for category in control_state:
            delta[category] = {}
            for var_name in control_state.get(category, {}):
                c_val = control_state[category].get(var_name, 0)
                i_val = intervention_state.get(category, {}).get(var_name, 0)
                if isinstance(c_val, float) and isinstance(i_val, float) and abs(c_val) > 1e-8:
                    delta[category][var_name] = (i_val - c_val) / max(abs(c_val), 0.001) * 100
                else:
                    delta[category][var_name] = 0.0

        return {
            'target': target,
            'target_desc': target_desc,
            'strength': strength,
            'condition': condition,
            'control_state': control_state,
            'intervention_state': intervention_state,
            'delta_pct': delta,
        }

    def _run_with_custom_params(self, condition, coupled_pert,
                                custom_mito_params=None,
                                custom_rna_params=None,
                                iterations=3) -> Dict:
        """Helper: run iteration with custom model parameters."""
        cf_map = {'normal': 0.0, 'early_degeneration': 0.4, 'late_degeneration': 0.8}
        base_cf = cf_map.get(condition, 0.8)

        history = {
            'condition': condition,
            'iterations': iterations,
            'coupled_ss': [],
            'mito_ss': [],
            'subcell_ss': [],
            'rna_ss': [],
            'coupled_pert': [],
            'convergence_metrics': [],
            'error_log': [],
            'feedback_matrix': [],
        }
        init_coupled = None

        for it in range(iterations):
            # --- Coupled engine ---
            try:
                coupled_res = self.coupled.simulate(
                    sim_type=condition,
                    perturbation=coupled_pert,
                    initial_state=init_coupled
                )
                coupled_ss = coupled_res[condition]['steady_state']
                init_coupled = coupled_ss
                history['coupled_ss'].append(coupled_ss)
                history['coupled_pert'].append(dict(coupled_pert))
            except Exception as e:
                history['error_log'].append(f"Iter {it+1} coupled: {e}")
                break

            # --- Mito ---
            mito_ss = None
            try:
                mito_params = self._coupled_to_mito_params(coupled_ss)
                if custom_mito_params:
                    mito_params.update(custom_mito_params)
                mito_temp = MitochondrialDynamicsModel(params=mito_params)
                mito_res = mito_temp.simulate(t_span=(0, 200), n_points=500)
                mito_ss = mito_res['y'][:, -1]
                history['mito_ss'].append(mito_ss)
            except Exception as e:
                history['mito_ss'].append(None)

            # --- Subcell ---
            subcell_ss = None
            try:
                ext_state = self._coupled_to_subcell_external_state(coupled_ss)
                subcell_pert = 'severe_degeneration' if base_cf > 0.6 else ('ER_stress' if base_cf > 0.2 else None)
                subcell_res = self.subcell.simulate(
                    t_span=(0, 200), n_points=500,
                    perturbation=subcell_pert,
                    external_state=ext_state,
                )
                subcell_ss = subcell_res['y'][:, -1]
                history['subcell_ss'].append(subcell_ss)
            except Exception as e:
                history['subcell_ss'].append(None)

            # --- RNA ---
            rna_ss = None
            try:
                rna_params = self._coupled_to_rna_params(coupled_ss)
                if custom_rna_params:
                    rna_params.update(custom_rna_params)
                rna_temp = RNADynamicsModel(params=rna_params)
                rna_res = rna_temp.simulate(t_span=(0, 200), n_points=500)
                rna_ss = rna_res['y'][:, -1]
                history['rna_ss'].append(rna_ss)
            except Exception as e:
                history['rna_ss'].append(None)

            # --- Feedback ---
            mito_fb = self._mito_feedback(mito_ss)
            subcell_fb, chop_val = self._subcell_feedback(subcell_ss)
            rna_fb = self._rna_feedback(rna_ss)
            all_fb = {}
            all_fb.update(mito_fb)
            all_fb.update(subcell_fb)
            all_fb.update(rna_fb)

            if subcell_ss is not None and chop_val > 0.3:
                coupled_pert['condition_factor'] = np.clip(base_cf + (chop_val - 0.3) * 0.15, 0.0, 1.0)

            coupled_pert = self._apply_perturbation(coupled_pert, all_fb, alpha=0.25)

            if it > 0:
                delta = np.max(np.abs(coupled_ss - history['coupled_ss'][-2]))
                history['convergence_metrics'].append(delta)
            else:
                history['convergence_metrics'].append(1.0)

        return history

    # ============================================================
    # 9. quick_summary
    # ============================================================

    def quick_summary(self, iteration_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Print and return key integrated metrics from the iteration results.
        """
        state = self.get_integrated_state(iteration_results)

        summary = {}

        # Key indicators
        indicators = [
            ('ECM Stiffness', 'ecm', 'ECM_stiffness'),
            ('NF-κB', 'inflammation', 'NFKB'),
            ('SOX9', 'inflammation', 'SOX9'),
            ('MMP', 'inflammation', 'MMP'),
            ('ATP', 'metabolic', 'ATP'),
            ('ROS', 'metabolic', 'ROS_mito'),
            ('Senescence', 'senescence', 'senescence_score'),
            ('SASP', 'senescence', 'SASP_score'),
            ('Aggrecan', 'ecm', 'Aggrecan'),
            ('Col2', 'ecm', 'Col2'),
            ('AMPK', 'mechano', 'AMPK_P'),
            ('Mito Δψm', 'mitochondrial', 'membrane_potential'),
            ('Mito Frag.', 'mitochondrial', 'fragmentation'),
            ('miROS', 'mitochondrial', 'miROS'),
            ('SIRT3', 'mitochondrial', 'SIRT3_activity'),
            ('Apoptosis', 'mitochondrial', 'apoptosis_risk'),
            ('CHOP', 'subcellular', 'CHOP'),
            ('cfDNA', 'subcellular', 'cfDNA_release'),
            ('Autophagy', 'subcellular', 'Autophagy_flux'),
            ('NLRP3', 'subcellular', 'NLRP3'),
            ('ECM mRNA', 'rna_dynamics', 'ECM_mRNA'),
            ('SASP mRNA', 'rna_dynamics', 'SASP_mRNA'),
            ('Stress Granule', 'rna_dynamics', 'Stress_granule'),
            ('HuR', 'rna_dynamics', 'HuR'),
            ('NEAT1', 'rna_dynamics', 'NEAT1'),
        ]

        print("\n" + "=" * 60)
        print(f"Multi-Scale Integrated State — {iteration_results.get('condition', 'unknown')}")
        print("=" * 60)

        sections = {
            'mechano': 'Force Transduction',
            'inflammation': 'Inflammation',
            'metabolic': 'Metabolic',
            'senescence': 'Senescence',
            'ecm': 'ECM',
            'mitochondrial': 'Mitochondrial',
            'subcellular': 'Subcellular',
            'rna_dynamics': 'RNA Dynamics',
        }

        current_section = ''
        for label, cat, var in indicators:
            val = state.get(cat, {}).get(var, None)
            if val is None:
                continue
            if cat != current_section:
                print(f"\n  --- {sections.get(cat, cat)} ---")
                current_section = cat
            print(f"  {label:20s} = {val:.4f}")
            summary[f'{cat}.{var}'] = float(val)

        n_iters = len(iteration_results.get('coupled_ss', []))
        n_errors = len(iteration_results.get('error_log', []))
        convergence = iteration_results.get('convergence_metrics', [])
        final_delta = convergence[-1] if len(convergence) > 1 else 'N/A'
        print(f"\n  --- Summary ---")
        print(f"  Iterations completed: {n_iters}")
        print(f"  Errors: {n_errors}")
        if isinstance(final_delta, float):
            print(f"  Final convergence Δ: {final_delta:.6f}")
        print("=" * 60)

        return summary

    # ============================================================
    # 10. plot_integrated_heatmap
    # ============================================================

    def plot_integrated_heatmap(self, results: Dict[str, Dict],
                                output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (18, 14)) -> plt.Figure:
        """
        Hierarchical heatmap of 50+ variables grouped by category.
        Compares across conditions stored in results dict.

        Parameters
        ----------
        results : dict mapping condition_name → iteration_results
        output_path : str or None
        figsize : tuple
        """
        # Extract integrated states for each condition
        conditions = list(results.keys())
        n_conds = len(conditions)

        states = {}
        for cond in conditions:
            states[cond] = self.get_integrated_state(results[cond])

        # Define variable groups with display names
        var_groups = [
            ('Mechano', 'mechano',
             [(0, 'Stiffness'), (1, 'MRTFA'), (2, 'Kidins220'), (3, 'AMPK'), (4, 'PFKFB3')]),
            ('Inflammatory', 'inflammation',
             [(5, 'NF-κB'), (6, 'SOX9'), (7, 'MMP')]),
            ('Metabolic', 'metabolic',
             [(9, 'ATP'), (10, 'ROS'), (11, 'Lactate'), (12, 'HIF1a')]),
            ('Senescence', 'senescence',
             [(13, 'p53'), (14, 'p21'), (15, 'SASP'), (16, 'IL1B'), (17, 'TNF'), (18, 'SenSc')]),
            ('ECM', 'ecm',
             [(19, 'ECM Stiff'), (20, 'ECM Deg'), (21, 'Aggrecan'), (22, 'Col2')]),
            ('Mito', 'mitochondrial',
             [(0, 'Mass'), (1, 'Fission'), (2, 'Fusion'), (3, 'Frag'),
              (4, 'Δψm'), (5, 'miROS'),
              (6, 'mtDNA'), (7, 'NAD/NADH'), (8, 'SIRT3'),
              (9, 'PINK1'), (11, 'Mitophagy'), (12, 'Apoptosis'), (13, 'Biogen')]),
            ('Subcell', 'subcellular',
             [(0, 'ER Load'), (1, 'Misfold'), (2, 'BiP'), (3, 'PERK'),
              (4, 'ATF6'), (5, 'CHOP'),
              (6, 'Lamin'), (8, 'NE Rupt'), (9, 'cGAS'),
              (10, 'Autophagy'), (11, 'Lyso'), (12, 'p62'),
              (13, 'NLRP3'), (14, 'Exosome'), (16, 'cfDNA'), (17, 'SASP Prop'),
              (18, 'cfRNA'), (19, 'DAMPs')]),
            ('RNA', 'rna_dynamics',
             [(0, 'mRNA Pool'), (1, 'ECM mRNA'), (2, 'SASP mRNA'),
              (3, 'miRNA'), (4, 'NEAT1'),
              (5, 'P-body'), (6, 'SG'), (7, 'SG Trap'),
              (8, 'HuR'), (9, 'TTP'), (10, 'IGF2BP'),
              (11, 'LLPS'), (12, 'Export'), (13, 'Transl'),
              (14, 'circRNA'), (15, 'eIF2a-P')]),
        ]

        # Build data matrix
        rows = []
        row_labels = []
        row_colors = []
        group_colors = {
            'Mechano': '#3498DB', 'Inflammatory': '#E74C3C', 'Metabolic': '#F39C12',
            'Senescence': '#8E44AD', 'ECM': '#2ECC71',
            'Mito': '#E74C3C', 'Subcell': '#2ECC71', 'RNA': '#9B59B6',
        }

        for group_name, cat, var_list in var_groups:
            for idx, short_label in var_list:
                row_label = f"{group_name} | {short_label}"

                # Get values from each condition
                values = []
                for cond in conditions:
                    if cat == 'mechano':
                        keys = ['stiffness_signal', 'MRTFA_nuclear', 'Kidins220', 'AMPK_P', 'PFKFB3']
                        s = states[cond].get(cat, {})
                        val = s.get(keys[idx]) if idx < len(keys) else None
                    elif cat == 'inflammation':
                        keys = ['NFKB', 'SOX9', 'MMP']
                        s = states[cond].get(cat, {})
                        val = s.get(keys[idx]) if idx < len(keys) else None
                    elif cat == 'metabolic':
                        keys = ['ATP', 'ROS_mito', 'Lactate', 'HIF1_alpha']
                        s = states[cond].get(cat, {})
                        val = s.get(keys[idx - 9]) if idx - 9 < len(keys) else None
                    elif cat == 'senescence':
                        keys = ['p53', 'p21', 'SASP_score', 'IL1B', 'TNF', 'senescence_score']
                        s = states[cond].get(cat, {})
                        val = s.get(keys[idx - 13]) if idx - 13 < len(keys) else None
                    elif cat == 'ecm':
                        keys = ['ECM_stiffness', 'ECM_degradation', 'Aggrecan', 'Col2']
                        s = states[cond].get(cat, {})
                        val = s.get(keys[idx - 19]) if idx - 19 < len(keys) else None
                    elif cat == 'mitochondrial':
                        keys_map = [
                            'mito_mass', 'fission', 'fusion', 'fragmentation',
                            'membrane_potential', 'miROS', 'mtDNA_damage',
                            'NAD_NADH_ratio', 'SIRT3_activity', 'PINK1_level',
                            None, 'mitophagy_flux', 'apoptosis_risk', 'biogenesis',
                        ]
                        s = states[cond].get(cat, {})
                        k = keys_map[idx] if idx < len(keys_map) else None
                        val = s.get(k) if k else None
                    elif cat == 'subcellular':
                        keys_map = [
                            'ER_fold_load', 'ER_misfolded', 'BiP_GRP78',
                            'PERK_act', 'ATF6_act', 'CHOP',
                            'Lamin_A_C', None, 'NE_rupture', 'cGAS_STING',
                            'Autophagy_flux', 'Lysosomal_function', 'p62_accumulation',
                            'NLRP3', 'Exosome_secretion', None,
                            'cfDNA_release', 'SASP_propagation', 'cfRNA_level',
                            'DAMPs_activation',
                        ]
                        s = states[cond].get(cat, {})
                        k = keys_map[idx] if idx < len(keys_map) else None
                        val = s.get(k) if k else None
                    elif cat == 'rna_dynamics':
                        keys_map = [
                            'Total_mRNA_pool', 'ECM_mRNA', 'SASP_mRNA',
                            'miRNA_machinery', 'NEAT1', 'P_body',
                            'Stress_granule', 'SG_mRNA_trapping',
                            'HuR', 'TTP', 'IGF2BP',
                            'LLPS_condensates', 'Nuclear_export',
                            'Translation_efficiency', 'circRNA_buffer', 'eIF2a_phos',
                        ]
                        s = states[cond].get(cat, {})
                        k = keys_map[idx] if idx < len(keys_map) else None
                        val = s.get(k) if k else None
                    else:
                        val = None

                    if val is not None and isinstance(val, (int, float)):
                        values.append(val)
                    else:
                        values.append(np.nan)

                # Skip all-NaN rows
                if any(not np.isnan(v) for v in values):
                    rows.append(values)
                    row_labels.append(row_label)
                    row_colors.append(group_colors.get(group_name, '#999'))

        data = np.array(rows)

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5], hspace=0.08)

        # -- Top panel: color bar for groups --
        ax_cbar = fig.add_subplot(gs[0])
        from matplotlib.patches import Rectangle
        unique_groups = list(group_colors.keys())
        x_pos = np.linspace(0.05, 0.95, len(unique_groups))
        for i, (grp, color) in enumerate(group_colors.items()):
            rect = Rectangle((x_pos[i] - 0.04, 0.1), 0.08, 0.8,
                             facecolor=color, edgecolor='white', lw=1)
            ax_cbar.add_patch(rect)
            ax_cbar.text(x_pos[i], 0.5, grp, ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
        ax_cbar.set_xlim(0, 1)
        ax_cbar.set_ylim(0, 1)
        ax_cbar.axis('off')

        # -- Main heatmap --
        ax = fig.add_subplot(gs[1])

        # Normalize each row to [0, 1] for coloring
        data_norm = np.zeros_like(data)
        for i in range(data.shape[0]):
            row = data[i, :]
            valid = ~np.isnan(row)
            if np.sum(valid) > 0:
                rmin, rmax = np.nanmin(row), np.nanmax(row)
                if rmax > rmin:
                    data_norm[i, valid] = (row[valid] - rmin) / (rmax - rmin)
                else:
                    data_norm[i, valid] = 0.5

        im = ax.imshow(data_norm, aspect='auto', cmap='RdYlBu_r',
                       interpolation='nearest')

        # Annotate with actual values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=6, color='black' if 0.3 < data_norm[i, j] < 0.7 else 'white')

        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xticks(range(n_conds))
        ax.set_xticklabels(conditions, fontsize=9, rotation=30, ha='right')

        # Color row labels by group
        for i, color in enumerate(row_colors):
            ax.get_yticklabels()[i].set_color(color)

        ax.set_title('Multi-Scale Integrated State — Hierarchical Heatmap',
                    fontsize=13, fontweight='bold', pad=10)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[MSI] Heatmap saved: {output_path}")
        plt.close(fig)
        return fig

    # ============================================================
    # 11. plot_feedback_convergence
    # ============================================================

    def plot_feedback_convergence(self, results: Dict[str, Dict],
                                  output_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot convergence of iterative feedback across iterations.

        If results contains multiple conditions, overlay them.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        colors = {'normal': '#2ECC71', 'early_degeneration': '#F39C12',
                  'late_degeneration': '#E74C3C'}

        # Collect data
        if 'condition' in results:
            # Single result
            results = {results['condition']: results}

        for cond, res in results.items():
            conv = res.get('convergence_metrics', [])
            iters = range(1, len(conv) + 1)
            c = colors.get(cond, '#888')

            # Plot convergence Δ
            ax = axes[0]
            ax.plot(iters, conv, 'o-', color=c, lw=2, label=cond, markersize=6)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Max Δ State', fontsize=11)
            ax.set_title('Convergence Rate', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            # Coupled perturbation evolution
            pert_history = res.get('coupled_pert', [])
            if len(pert_history) > 0:
                ax2 = axes[1]
                p_iters = range(1, len(pert_history) + 1)
                for p_key in ['condition_factor', 'oxidative_stress', 'il1b_stim', 'ampk_act']:
                    vals = [p.get(p_key, 0.0) for p in pert_history]
                    ax2.plot(p_iters, vals, 'o-', lw=1.5, markersize=5,
                            label=f'{cond}:{p_key}')
                ax2.set_xlabel('Iteration', fontsize=11)
                ax2.set_ylabel('Perturbation Value', fontsize=11)
                ax2.set_title('Perturbation Evolution', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=7, ncol=2)

            # Key variable trajectory across iterations
            ss_history = res.get('coupled_ss', [])
            if len(ss_history) > 0:
                ax3 = axes[2]
                var_indices = [C_ATP, C_NFKB, C_SASP, C_SENESCENCE, C_AGGRECAN]
                var_labels = ['ATP', 'NFKB', 'SASP', 'Senescence', 'Aggrecan']
                markers = ['o', 's', '^', 'D', 'v']
                for idx, label, marker in zip(var_indices, var_labels, markers):
                    vals = [ss[idx] for ss in ss_history]
                    ax3.plot(range(1, len(vals) + 1), vals, f'-{marker}',
                            lw=1.5, markersize=6, label=f'{cond}:{label}')
                ax3.set_xlabel('Iteration', fontsize=11)
                ax3.set_ylabel('Variable Value', fontsize=11)
                ax3.set_title('Key Variable Convergence', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=8)

        fig.suptitle('Feedback Coupling Convergence', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[MSI] Convergence plot saved: {output_path}")
        plt.close(fig)
        return fig


# ============================================================
# 12. Main entry point / self-test
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Multi-Scale Integrator — Self-Test")
    print("=" * 70)

    ms = MultiScaleIntegrator()

    # Test 1: Normal condition
    print("\n[1] Normal condition (3 iterations)...")
    res_normal = ms.run_coupled_iteration('normal', iterations=3)
    state = ms.get_integrated_state(res_normal)
    print(f"    Mechano module:  {len(state.get('mechano', {}))} vars")
    print(f"    Inflammation:    {len(state.get('inflammation', {}))} vars")
    print(f"    Metabolic:       {len(state.get('metabolic', {}))} vars")
    print(f"    Senescence:      {len(state.get('senescence', {}))} vars")
    print(f"    ECM:             {len(state.get('ecm', {}))} vars")
    print(f"    Mitochondrial:   {len(state.get('mitochondrial', {}))} vars")
    print(f"    Subcellular:     {len(state.get('subcellular', {}))} vars")
    print(f"    RNA Dynamics:    {len(state.get('rna_dynamics', {}))} vars")

    total_vars = sum(len(v) for v in state.values())
    print(f"    Total: {total_vars}+ dimensions")

    # Test 2: quick_summary
    print("\n[2] Quick summary...")
    ms.quick_summary(res_normal)

    # Test 3: Compare conditions
    print("\n[3] Compare conditions...")
    ms.compare_conditions(['normal', 'early_degeneration', 'late_degeneration'])

    # Test 4: Cross-talk network
    print("\n[4] Cross-talk network...")
    network = ms.get_cross_talk_network()
    print(f"    Nodes: {len(network['nodes'])}")
    for n in network['nodes']:
        print(f"      {n['id']:12s} — {n['label']}")
    print(f"    Edges: {len(network['edges'])}")
    for e in network['edges']:
        print(f"      {e['source']:10s} → {e['target']:10s}: {e['vars']}")

    # Test 5: Intervention
    print("\n[5] Intervention simulation...")
    int_res = ms.simulate_intervention(target='Senolytic', strength=1.0,
                                       condition='late_degeneration', iterations=3)
    print(f"    Target: {int_res['target_desc']}")
    print(f"    Key Δ (senescence): {int_res['delta_pct'].get('senescence', {}).get('senescence_score', 0):.1f}%")
    print(f"    Key Δ (SASP):       {int_res['delta_pct'].get('senescence', {}).get('SASP_score', 0):.1f}%")
    print(f"    Key Δ (Aggrecan):   {int_res['delta_pct'].get('ecm', {}).get('Aggrecan', 0):.1f}%")

    # Test 6: MitoQ intervention
    print("\n[6] MitoQ intervention...")
    int_res2 = ms.simulate_intervention(target='MitoQ', strength=1.0,
                                        condition='late_degeneration', iterations=3)
    print(f"    Target: {int_res2['target_desc']}")
    print(f"    Key Δ (ROS):        {int_res2['delta_pct'].get('metabolic', {}).get('ROS_mito', 0):.1f}%")
    print(f"    Key Δ (miROS):      {int_res2['delta_pct'].get('mitochondrial', {}).get('miROS', 0):.1f}%")

    print("\n" + "=" * 70)
    print("✅ Multi-scale integrator OK")
    print("=" * 70)
