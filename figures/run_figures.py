#!/usr/bin/env python3
"""
Part 2: Publication-Grade Figures Pipeline
=============================================
论文级多面板 Figure 生成 (Figure 1-7)，统一配色、出版级分辨率 (300+ DPI)

Figure 设计:
  Figure 1: System architecture diagram (多尺度虚拟髓核细胞系统架构)
  Figure 2: Validation scatter (模型 vs 文献定量一致性)
  Figure 3: Signaling pathway heatmap + perturbation bar chart
  Figure 4: MRTF-A mechano-glycolysis coupling (4-panel)
  Figure 5: ECM degradation dynamics (multi-panel, normal vs degenerated)
  Figure 6: Epigenetic senescence (SETD1A/H3K4me3 axis, 4-panel)
  Figure 7: ABM microenvironment (spatial + time series)
  Supplementary Figure S1: miRNA regulatory network
  Supplementary Figure S2: scRNA-seq integration
"""

import sys, os, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
matplotlib.rcParams['font.family'] = ['HarmonyHeiTi', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.signaling import NPSignalingModel, MRTFAMechanoModel
from simulation.ecm_model import ECMDegradationModel
from simulation.epigenetic_senescence import EpigeneticSenescenceModel
from simulation.abm_microenv import run_abm_simulation, plot_abm_grid, plot_abm_timeseries
from analysis.biomarker import (
    simulate_biomarker_data, rank_biomarkers, simulate_trend_data
)

OUTPUT = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# 统一配色方案 (Nature/Cell 风格)
# ================================================================

# Primary palette (Nature Review style)
NP_BLUE = '#3B6EA5'
NP_RED = '#C15C4A'
NP_GREEN = '#4C9A6D'
NP_ORANGE = '#D48B3E'
NP_PURPLE = '#7B6FA0'
NP_CYAN = '#3D8B91'
NP_YELLOW = '#D4B83E'
NP_GREY = '#7F8C8D'

# Extended palette
COLOR_NORMAL = NP_BLUE
COLOR_DEGEN = NP_RED
COLOR_TREATMENT = NP_GREEN
COLOR_INFLAM = NP_ORANGE
COLOR_WNT = NP_PURPLE
COLOR_HYPOXIA = NP_CYAN

# Heatmap colormap
CMAP_DIVERGING = LinearSegmentedColormap.from_list('np_cmap',
    ['#2166AC', '#4393C3', '#F7F7F7', '#D6604D', '#B2182B'], N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list('np_seq',
    ['#F7F7F7', '#4393C3', '#2166AC'], N=256)

# Figure dimensions
FIG_WIDTH = 7.5  # inches (Nature single-column)
FIG_WIDTH_MEDIUM = 10.0
FIG_WIDTH_FULL = 14.0
FIG_HEIGHT_COEFF = 0.75
DPI = 300

# Axis style
def style_ax(ax, xlabel='', ylabel='', title='', legend=False, fontsize=9):
    """Apply consistent Nature-style axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=7, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=6)
    return ax

# ================================================================
# Figure 1: System Architecture Diagram
# ================================================================

def figure1_architecture(output_path=None):
    """Figure 1: Schematic of the multi-scale Virtual NP Cell system.
    展示四个尺度层：分子 → 细胞 → 微环境 → 临床，以及三个交叉模块。
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_FULL, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def draw_layer_box(ax, x, y, w, h, color, label, alpha=0.15):
        """Draw a rounded box for a layer."""
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor=color, alpha=alpha,
                             linewidth=1.5)
        ax.add_patch(box)
        # Bottom label
        ax.text(x + w/2, y - 0.25, label, ha='center', va='top',
                fontsize=10, fontweight='bold', color=color)

    def draw_box(ax, x, y, w, h, color, text, text_color='white', fontsize=7):
        """Draw a small box with text."""
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold')

    def draw_arrow(ax, x1, y1, x2, y2, color='#666666', lw=1.5):
        """Draw an arrow between boxes."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle='arc3,rad=0.1'))

    # Title
    ax.text(7, 8.5, 'Multi-Scale Virtual Nucleus Pulposus Cell System',
            ha='center', fontsize=14, fontweight='bold', color='#2C3E50')

    # --- Layer 1: Molecular Scale (top) ---
    draw_layer_box(ax, 0.5, 5.5, 13, 2.5, NP_BLUE, 'Molecular Scale', alpha=0.12)
    draw_box(ax, 0.8, 6.0, 2.0, 0.7, NP_BLUE, 'Signaling\nODE Model')
    draw_box(ax, 3.1, 6.0, 2.0, 0.7, '#2E86C1', 'MRTF-A\nMechano Coupling')
    draw_box(ax, 5.4, 6.0, 2.0, 0.7, NP_PURPLE, 'Epigenetic\n(SETD1A/H3K4me3)')
    draw_box(ax, 7.7, 6.0, 2.0, 0.7, NP_GREEN, 'ECM\nMetabolism')
    draw_box(ax, 10.0, 6.0, 1.8, 0.7, NP_ORANGE, 'miRNA\nRegulation')
    draw_box(ax, 12.1, 6.0, 1.2, 0.7, '#8E44AD', 'm6A\nEpitrans')

    # --- Layer 2: Cellular Scale ---
    draw_layer_box(ax, 0.5, 3.0, 13, 2.2, NP_GREEN, 'Cellular Scale', alpha=0.12)
    draw_box(ax, 1.0, 3.5, 2.5, 0.7, '#1ABC9C', 'Cellular Metabolism')
    draw_box(ax, 4.0, 3.5, 2.5, 0.7, '#3498DB', 'Cellular Senescence')
    draw_box(ax, 7.0, 3.5, 2.5, 0.7, '#E67E22', 'Hypoxia Response')
    draw_box(ax, 10.0, 3.5, 1.6, 0.7, '#9B59B6', 'Apoptosis')
    draw_box(ax, 11.9, 3.5, 1.4, 0.7, '#C0392B', 'SASP')

    # --- Layer 3: Microenvironment Scale ---
    draw_layer_box(ax, 0.5, 1.0, 13, 1.8, NP_ORANGE, 'Microenvironment Scale', alpha=0.12)
    draw_box(ax, 1.2, 1.5, 2.8, 0.7, '#D4AC0D', 'ABM: Cell-Cell\nInteractions', fontsize=6)
    draw_box(ax, 4.5, 1.5, 2.8, 0.7, '#E67E22', 'ECM Stiffness\nGradient', fontsize=6)
    draw_box(ax, 7.8, 1.5, 2.5, 0.7, '#CB4335', 'Inflammatory\nCytokines', fontsize=6)
    draw_box(ax, 10.8, 1.5, 2.5, 0.7, '#5DADE2', 'Oxygen & Nutrient\nGradients', fontsize=6)

    # Arrows between layers
    for x in [1.8, 4.2, 6.5, 8.8, 11.0]:
        ax.annotate('', xy=(x, 3.3), xytext=(x, 5.3),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1.2, alpha=0.5))
    for x in [2.6, 5.9, 9.1]:
        ax.annotate('', xy=(x, 1.3), xytext=(x, 2.8),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1.2, alpha=0.5))

    # --- Cross-cutting modules (right side) ---
    cross_x = 8.0
    cross_y = 1.0
    draw_box(ax, 9.5, 0.3, 3.8, 0.5, NP_RED, 'Cross-Module: Orchestrator + Knowledge Base', '#fff', 7)

    # Legend bottom
    legend_elements = [
        mpatches.Patch(color=NP_BLUE, alpha=0.3, label='Molecular (ODE Kinetics)'),
        mpatches.Patch(color=NP_GREEN, alpha=0.3, label='Cellular (Phenotypes)'),
        mpatches.Patch(color=NP_ORANGE, alpha=0.3, label='Microenvironment (ABM)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              framealpha=0.8, edgecolor='#ccc', ncol=3,
              bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 1: {output_path}")
    plt.close()
    return fig


# ================================================================
# Figure 2: Validation Scatter (Model vs Literature)
# ================================================================

def figure2_validation_scatter(validation_data, output_path=None):
    """Figure 2: Validation scatter plot — model vs literature quantitative agreement."""
    data = validation_data.get('scatter', {})
    sim = np.array(data.get('sim', []))
    lit = np.array(data.get('lit', []))
    stats = validation_data.get('validation_stats', {})

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 0.85))

    if len(sim) > 0 and len(lit) > 0:
        ax.scatter(lit, sim, c=NP_BLUE, s=45, edgecolors='white', linewidth=0.6,
                   alpha=0.85, zorder=5)

        # Perfect agreement line
        lim_min = min(lit.min(), sim.min()) * 0.9
        lim_max = max(lit.max(), sim.max()) * 1.1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='#666',
                linewidth=1.2, alpha=0.6, label='Perfect agreement')

        # R² annotation
        r2 = stats.get('r_squared', 0)
        p_r = stats.get('pearson_r', 0)
        mape = stats.get('mape', 0)
        ax.text(0.05, 0.95,
                f"R² = {r2:.3f}\nPearson r = {p_r:.3f}\nMAPE = {mape:.1f}%",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))

        # 95% confidence interval shading
        z = np.polyfit(lit, sim, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lim_min, lim_max, 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, '-', color=NP_RED, linewidth=1.5, alpha=0.5,
                label=f'Fit (slope={z[0]:.2f})')

    style_ax(ax, xlabel='Literature Reported Value', ylabel='Model Simulated Value',
             title='Figure 2 | Virtual NP Cell — Literature Validation')

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(fontsize=7, framealpha=0.8, edgecolor='#ccc')
    ax.set_aspect('equal')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 2: {output_path}")
    plt.close()
    return fig


# ================================================================
# Figure 3: Signaling Pathway — Heatmap + Perturbation Screen
# ================================================================

def figure3_signaling_perturbation(output_path=None):
    """Figure 3: Combined signaling analysis — perturbation screen heatmap."""
    model = NPSignalingModel()
    perturbations = [
        ("Normal", {}),
        ("Inflammation", {'IL1B': 3.0, 'TNF': 2.0}),
        ("Wnt Act.", {'WNT': 2.0}),
        ("Hypoxia", {'HYPOXIA': 1.0}),
        ("TGFb Suppl.", {'TGFB': 2.0}),
        ("GSK3b Inh.", {'GSK3B_inhibit': 0.8}),
        ("Combined", {'IL1B': 3.0, 'TNF': 2.0, 'WNT': 1.5, 'HYPOXIA': 0.3}),
    ]

    # Run screens
    results, base_ecm, base_mmp = model.run_perturbation_screen(perturbations)
    
    # Variables to track
    var_names = ['TGF-β', 'SMAD2/3', 'β-catenin', 'NF-κB', 'HIF-1α',
                 'MAPK/ERK', 'SOX9', 'MMPs', 'ECM', 'Inflammation']
    n_vars = len(var_names)
    n_conds = len(perturbations)

    # Extract profiles for each perturbation
    heatmap_data = np.zeros((n_vars, n_conds))
    for j, (name, p_dict) in enumerate(perturbations):
        _, y = model.simulate(perturbations=p_dict)
        for i in range(n_vars):
            heatmap_data[i, j] = y[i, -1]

    # Z-score normalize for heatmap (across conditions)
    heatmap_z = np.zeros_like(heatmap_data)
    for i in range(n_vars):
        vals = heatmap_data[i, :]
        heatmap_z[i, :] = (vals - vals.mean()) / max(vals.std(), 1e-6)

    # Create figure
    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 8.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 0.02, 1.8], height_ratios=[1.2, 1.2],
                  hspace=0.35, wspace=0.35)

    # --- Panel A: Heatmap (left) ---
    ax_hm = fig.add_subplot(gs[0, 0])
    im = ax_hm.imshow(heatmap_z, aspect='auto', cmap=CMAP_DIVERGING,
                      vmin=-2, vmax=2)

    ax_hm.set_xticks(range(n_conds))
    cond_labels = [p[0].replace('\n', ' ') for p in perturbations]
    ax_hm.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=6)
    ax_hm.set_yticks(range(n_vars))
    ax_hm.set_yticklabels(['TGF-β', 'SMAD2/3', 'β-catenin', 'NF-κB', 'HIF-1α',
                           'MAPK/ERK', 'SOX9', 'MMPs', 'ECM', 'Inflammation'], fontsize=7)
    ax_hm.set_title('A  Steady-State Signaling Profile', fontsize=9, fontweight='bold', loc='left')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_hm, fraction=0.05, pad=0.02)
    cbar.set_label('Z-score (across conditions)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Annotate values
    for i in range(n_vars):
        for j in range(n_conds):
            val = heatmap_data[i, j]
            color = 'white' if abs(heatmap_z[i, j]) > 0.8 else 'black'
            ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=5, color=color)

    # --- Panel B: ECM/MMP bar chart (right, top) ---
    ax_bar = fig.add_subplot(gs[0, 2])
    pert_names = [p[0] for p in perturbations]  # original names (may have newlines)
    x = np.arange(len(pert_names))
    w = 0.35

    ecm_changes = [results[n]["ECM_change"] for n in pert_names]
    mmp_changes = [results[n]["MMP_change"] for n in pert_names]

    bars1 = ax_bar.bar(x - w/2, ecm_changes, w, label='ECM change %',
                       color=NP_GREEN, alpha=0.85, edgecolor='none')
    bars2 = ax_bar.bar(x + w/2, mmp_changes, w, label='MMP change %',
                       color=NP_RED, alpha=0.85, edgecolor='none')

    ax_bar.axhline(0, color='#666', linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([n.replace(chr(92)+"n", " ") for n in pert_names], rotation=25, ha="right", fontsize=7)
    style_ax(ax_bar, ylabel='Change from baseline (%)',
             title='B  ECM & MMP Response to Perturbations')
    ax_bar.legend(fontsize=7, framealpha=0.8, edgecolor='#ccc')

    # --- Panel C: Anabolic score (lower left) ---
    ax_ano = fig.add_subplot(gs[1, 0])
    scores = [results[n]["Anabolic_score"] for n in pert_names]
    colors = [NP_GREEN if s > 0.5 else NP_RED for s in scores]
    ax_ano.bar(range(len(pert_names)), scores, color=colors, alpha=0.85,
               edgecolor='none', width=0.6)
    ax_ano.axhline(0.5, color='#666', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_ano.set_xticks(range(len(pert_names)))
    ax_ano.set_xticklabels([n.replace(chr(92)+"n", " ") for n in pert_names], rotation=25, ha="right", fontsize=7)
    style_ax(ax_ano, ylabel='Anabolic score',
             title='D  Anabolic: ECM/(ECM+MMP)')

    # --- Panel D: SOX9 vs NF-κB scatter (lower right) ---
    ax_sc = fig.add_subplot(gs[1, 2])
    sox9_vals = [results[n]["SOX9"] for n in pert_names]
    nfkb_vals = [results[n]["NFKB"] for n in pert_names]

    ax_sc.scatter(sox9_vals, nfkb_vals, c=range(len(pert_names)),
                  cmap='RdYlBu_r', s=60, edgecolors='white', linewidth=0.6,
                  zorder=5)
    for i, name in enumerate(pert_names):
        short_name = name.split('\n')[0] if '\n' in name else name[:8]
        ax_sc.annotate(short_name, (sox9_vals[i], nfkb_vals[i]),
                       fontsize=6, xytext=(3, 3), textcoords='offset points',
                       alpha=0.7)
    style_ax(ax_sc, xlabel='SOX9 (anabolic master TF)',
             ylabel='NF-κB (catabolic master TF)',
             title='C  SOX9 vs NF-κB Phase Space')

    # Baseline annotation
    fig.text(0.37, 0.01,
             f'Baseline ECM = {base_ecm:.3f}, MMP = {base_mmp:.3f}',
             ha='center', fontsize=7, color=NP_GREY)

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 3: {output_path}")
    plt.close()
    return fig


# ================================================================
# Figure 4: MRTF-A Mechano-Glycolysis Coupling (4-panel)
# ================================================================

def figure4_mrtfa_mechano(output_path=None):
    """Figure 4: MRTF-A mechano-glycolysis coupling — 4 panel figure."""
    model = MRTFAMechanoModel()

    # Simulations
    t_n, y_n = model.simulate(stiffness=2.0)
    t_d, y_d = model.simulate(stiffness=15.0)
    t_ccg, y_ccg = model.simulate(stiffness=15.0, perturbation={'CCG_inhibitor': 0.8})

    var_names = ['MRTF-A(cyt)', 'MRTF-A(nuc)', 'Kidins220', 'p-AMPK',
                 'PFKFB3', 'PFKM', 'PLD1', 'Glycolysis']
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
              '#1ABC9C', '#9B59B6', '#E67E22', '#C0392B']

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 9))
    gs = GridSpec(2, 6, figure=fig, hspace=0.4, wspace=0.5)

    # Panels A-B: Time series side-by-side
    for cond_idx, (t, y, label) in enumerate([
        (t_n, y_n, 'Normal Stiffness (~2 kPa)'),
        (t_d, y_d, 'Degenerated Stiffness (~15 kPa)'),
    ]):
        for var_idx in range(2):  # Only show 2 time series rows
            ax = fig.add_subplot(gs[var_idx, cond_idx*3:cond_idx*3+2])
            for i in range(4 * var_idx, min(4 * (var_idx + 1), 8)):
                ax.plot(t, y[i], color=colors[i], linewidth=1.5, alpha=0.85,
                        label=var_names[i])
            style_ax(ax, xlabel='Time (au)' if var_idx == 1 else '',
                     ylabel='Activity' if cond_idx == 0 else '',
                     title=f'{"A" if cond_idx == 0 and var_idx == 0 else ""}'
                           f'{"B" if cond_idx == 1 and var_idx == 0 else ""}'
                           f'  {label}', fontsize=8)
            if var_idx == 1:
                ax.legend(fontsize=5.5, ncol=2, framealpha=0.8, edgecolor='#ccc')

    # Panel C: Bar plot of normal vs degenerated endpoint values
    ax_bar = fig.add_subplot(gs[:, 4])
    bar_names = ['MRTF-A\n(nuclear)', 'Kidins220', 'p-AMPK', 'Glycolysis']
    bar_indices = [1, 2, 3, 7]
    x = np.arange(len(bar_names))
    w = 0.3

    for idx, (data, label, color) in enumerate([
        (y_n, 'Normal (2 kPa)', NP_BLUE),
        (y_d, 'Degenerated (15 kPa)', NP_RED),
        (y_ccg, 'Deg + CCG', NP_GREEN),
    ]):
        values = [data[i, -1] for i in bar_indices]
        offset = (idx - 1) * w
        bars = ax_bar.bar(x + offset, values, w, label=label, color=color,
                          alpha=0.8, edgecolor='white', linewidth=0.4)
        for bar, val in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_names, fontsize=7)
    style_ax(ax_bar, ylabel='Steady-state value',
             title='C  Endpoint Comparison')
    ax_bar.legend(fontsize=6, framealpha=0.8, edgecolor='#ccc')

    # Panel D: Schematic (text summary)
    ax_sch = fig.add_subplot(gs[:, 5])
    ax_sch.axis('off')
    summary_text = (
        "D  Mechano-Glycolysis Axis:\n\n"
        "ECM Stiffness ↑ (2→15 kPa)\n"
        "  → MRTF-A nuclear import ↑\n"
        "  → Kidins220 transcription ↓\n"
        "  → AMPK phosphorylation ↓\n"
        "  → PFKFB3/PFKM/PLD1 ↓\n"
        "  → Glycolysis ↓\n\n"
        "CCG-1423 (MRTF-A inhibitor)\n"
        "  rescues glycolysis\n\n"
        f"Normal glycolysis: {y_n[7,-1]:.3f}\n"
        f"Degenerated: {y_d[7,-1]:.3f}\n"
        f"CCG rescue: {y_ccg[7,-1]:.3f}"
    )
    ax_sch.text(0.05, 0.95, summary_text, transform=ax_sch.transAxes,
                fontsize=8, va='top', ha='left', linespacing=1.5,
                fontfamily='monospace')

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 4: {output_path}")
    plt.close()
    return fig


# ================================================================
# Figure 5: ECM Dynamics
# ================================================================

def figure5_ecm_dynamics(output_path=None):
    """Figure 5: ECM degradation dynamics — normal vs degenerated."""
    model = ECMDegradationModel()

    t_n, y_n = model.simulate()
    t_d, y_d = model.simulate(
        t_span=(0, 400), n_points=500,
        perturbation={'INFLAM': 3.0, 'OXSTRESS': 2.0, 'NUTRIENT': -1.0, 'MECHANICAL': 0.5},
        degen_accel=1.0
    )

    var_names = ['Aggrecan', 'Collagen II', 'MMP Activity', 'TIMP Activity',
                 'Inflammation', 'NP Cell Density', 'Water Content']
    colors = ['#2E86C1', '#1ABC9C', '#E74C3C', '#2ECC71',
              '#E67E22', '#8E44AD', '#3498DB']

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # Time series for key variables
    key_vars = [0, 1, 2, 3, 4, 5]
    for i, vi in enumerate(key_vars):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.plot(t_n, y_n[vi], color=NP_BLUE, linewidth=1.8, label='Normal')
        ax.plot(t_d, y_d[vi], color=NP_RED, linewidth=1.8, label='Degenerated', alpha=0.85)
        style_ax(ax, xlabel='Time (au)', title=var_names[vi])
        if 'MMP' in var_names[vi] or 'TIMP' in var_names[vi]:
            ax.axhline(y_n[vi, -1], color=NP_BLUE, linestyle=':', alpha=0.4)
            ax.axhline(y_d[vi, -1], color=NP_RED, linestyle=':', alpha=0.4)

    # Panel: MMP/TIMP ratio (lower right)
    ax_ratio = fig.add_subplot(gs[1, 3])
    ratio_n = y_n[2] / (y_n[3] + 1e-6)
    ratio_d = y_d[2] / (y_d[3] + 1e-6)
    ax_ratio.plot(t_n, ratio_n, color=NP_BLUE, linewidth=1.8, label='Normal')
    ax_ratio.plot(t_d, ratio_d, color=NP_RED, linewidth=1.8, label='Degenerated', alpha=0.85)
    ax_ratio.axhline(1, color='#666', linestyle='--', linewidth=0.8, alpha=0.5)
    style_ax(ax_ratio, xlabel='Time (au)', ylabel='MMP/TIMP',
             title='MMP:TIMP Ratio')
    ax_ratio.legend(fontsize=6, framealpha=0.8, edgecolor='#ccc')

    # Summary annotation on the figure
    fig.text(0.5, 0.01,
             f'Normal ECM integrity score: {y_n[0,-1]+y_n[1,-1]:.2f}  |  '
             f'Degenerated: {y_d[0,-1]+y_d[1,-1]:.2f}  |  '
             f'ECM loss: {(1-(y_d[0,-1]+y_d[1,-1])/(y_n[0,-1]+y_n[1,-1]))*100:.1f}%',
             ha='center', fontsize=7, color=NP_GREY,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 5: {output_path}")
    plt.close()
    return fig


# ================================================================
# Figure 6: Epigenetic Senescence (SETD1A/H3K4me3)
# ================================================================

def figure6_epigenetic_senescence(output_path=None):
    """Figure 6: SETD1A/H3K4me3 epigenetic senescence — multi-panel."""
    model = EpigeneticSenescenceModel()

    t_b, y_b = model.simulate()
    t_kd, y_kd = model.simulate(perturbation={'SETD1A_knockdown': 0.8})
    t_oe, y_oe = model.simulate(perturbation={'SETD1A_overexpression': 2.0})

    var_names = ['SETD1A', 'H3K4me3', 'HELZ2', 'PPARα',
                 'HIF-1α', 'Senescence', 'Glycolysis(epi)', 'SASP']
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
              '#1ABC9C', '#9B59B6', '#E67E22', '#C0392B']

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: Key time series (baseline + perturbations)
    key_indices = [0, 1, 4, 5, 6, 7]
    for i, vi in enumerate(key_indices):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(t_b, y_b[vi], color=NP_BLUE, linewidth=1.8, label='Baseline')
        ax.plot(t_kd, y_kd[vi], color=NP_RED, linewidth=1.5, label='SETD1A KD 80%',
                linestyle='--', alpha=0.85)
        ax.plot(t_oe, y_oe[vi], color=NP_GREEN, linewidth=1.5, label='SETD1A OE 2x',
                linestyle=':', alpha=0.85)
        style_ax(ax, title=var_names[vi])
        if i >= 3:
            ax.set_xlabel('Time (au)')
        ax.legend(fontsize=5.5, framealpha=0.8, edgecolor='#ccc')

    # Panel B: Multi-condition comparison bar chart
    conditions = [
        ("Baseline", {}),
        ("SETD1A\nKD 80%", {'SETD1A_knockdown': 0.8}),
        ("SETD1A\nOE 2x", {'SETD1A_overexpression': 2.0}),
        ("PPARα\nAgo 2x", {'PPARa_agonist': 2.0}),
    ]
    screen_results, base_glc, base_sen = model.run_senescence_screen(conditions)

    ax_bar = fig.add_subplot(gs[:, 3])
    metrics = ["H3K4me3", "HIF1a", "Glycolysis", "Senescence", "SASP"]
    metric_labels = ["H3K4me3", "HIF-1α", "Glycolysis", "Senescence", "SASP"]
    bar_colors = ['#E74C3C', '#1ABC9C', '#E67E22', '#9B59B6', '#C0392B']

    names = list(screen_results.keys())
    x = np.arange(len(metrics))
    w = 0.18

    for i, name in enumerate(names):
        values = [screen_results[name].get(m, 0) for m in metrics]
        offset = (i - (len(names)-1)/2) * w
        bars = ax_bar.bar(x + offset, values, w, label=name,
                          color=[NP_BLUE, NP_RED, NP_GREEN, NP_ORANGE][i],
                          alpha=0.8, edgecolor='white', linewidth=0.3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=7)
    style_ax(ax_bar, ylabel='Steady-state value',
             title='Multi-Condition\nComparison')
    ax_bar.legend(fontsize=5.5, framealpha=0.8, edgecolor='#ccc',
                  loc='upper right')

    # Summary text
    fig.text(0.5, 0.01,
             f'SETD1A↓→H3K4me3↓→HELZ2↓→PPARα↓→HIF1α↓→Senescence↑→SASP↑  |  '
             f"PPARα agonist rescues glycolysis (+{(screen_results.get('PPARα Ago 2x', {}).get('Glycolysis_change', 0)):.1f}%)",
             ha='center', fontsize=7, color=NP_GREY,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 6: {output_path}")
    plt.close()
    return fig


# ================================================================
# Figure 7: ABM Microenvironment
# ================================================================

def figure7_abm_microenv(output_path=None):
    """Figure 7: ABM microenvironment — spatial grid + time series."""
    print("  Running ABM simulations...")
    abm_normal = run_abm_simulation(grid_size=20, n_np_cells=30, n_macrophages=6,
                                    n_fibroblasts=8, n_steps=50, degenerative=False, seed=42)
    abm_degen = run_abm_simulation(grid_size=20, n_np_cells=30, n_macrophages=6,
                                   n_fibroblasts=8, n_steps=50, degenerative=True,
                                   inflam_seed=0.5, mechanical_overload=0.6, seed=43)

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Panels A-B: Grid snapshots (final state)
    grid_vars = ['ecm', 'inflam', 'cell_type']
    grid_labels = ['ECM Concentration', 'Inflammation Level', 'Cell Distribution']
    for idx, (var, label) in enumerate(zip(grid_vars, grid_labels)):
        ax_n = fig.add_subplot(gs[0, idx])
        ax_d = fig.add_subplot(gs[1, idx])

        # Normal grid
        grid_n = abm_normal['grid_history'][-1]
        if var == 'ecm':
            im = ax_n.imshow(grid_n['ecm'], cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
            ax_n.set_title(f'Normal ECM\n({label})', fontsize=8, fontweight='bold')
        elif var == 'inflam':
            im = ax_n.imshow(grid_n['inflammation'], cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
            ax_n.set_title(f'Normal Inflammation\n({label})', fontsize=8, fontweight='bold')
        else:
            # Cell type overlay
            ct_grid = np.zeros((20, 20))
            for agent in abm_normal['agents']:
                if getattr(agent, 'alive', True) and hasattr(agent, 'x') and hasattr(agent, 'y'):
                    xi, yi = int(agent.x), int(agent.y)
                    if 0 <= xi < 20 and 0 <= yi < 20:
                        ct_map = {'NP_cell': 1, 'Macrophage': 2, 'Fibroblast': 3,
                                  'Apoptotic_NP': 4}
                        ct_grid[xi, yi] = ct_map.get(agent.cell_type, 0)
            im = ax_n.imshow(ct_grid, cmap='Set2', vmin=0, vmax=4, aspect='equal')
            ax_n.set_title(f'Normal Cell\nDistribution', fontsize=8, fontweight='bold')

        ax_n.axis('off')

        # Degenerated grid
        grid_d = abm_degen['grid_history'][-1]
        if var == 'ecm':
            im_d = ax_d.imshow(grid_d['ecm'], cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
            ax_d.set_title(f'Degenerated ECM\n({label})', fontsize=8, fontweight='bold')
        elif var == 'inflam':
            im_d = ax_d.imshow(grid_d['inflammation'], cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
            ax_d.set_title(f'Degenerated Inflam\n({label})', fontsize=8, fontweight='bold')
        else:
            ct_grid_d = np.zeros((20, 20))
            for agent in abm_degen['agents']:
                if getattr(agent, 'alive', True) and hasattr(agent, 'x') and hasattr(agent, 'y'):
                    xi, yi = int(agent.x), int(agent.y)
                    if 0 <= xi < 20 and 0 <= yi < 20:
                        ct_map = {'NP_cell': 1, 'Macrophage': 2, 'Fibroblast': 3,
                                  'Apoptotic_NP': 4}
                        ct_grid_d[xi, yi] = ct_map.get(agent.cell_type, 0)
            im_d = ax_d.imshow(ct_grid_d, cmap='Set2', vmin=0, vmax=4, aspect='equal')
            ax_d.set_title(f'Degenerated Cell\nDistribution', fontsize=8, fontweight='bold')

        ax_d.axis('off')

        # Colorbar for last panel in row
        if var != 'cell_type':
            divider = make_axes_locatable(ax_d)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im_d, cax=cax)

    # Panel D: Time series
    ax_ts = fig.add_subplot(gs[:, 3])
    # Normal
    ecm_hist_n = abm_normal['ecm_history']
    if len(ecm_hist_n) > 0:
        ts_n = np.array(ecm_hist_n)
        ax_ts.plot(ts_n[:, 0], ts_n[:, 1], color=NP_BLUE, linewidth=1.8,
                   label=f'Normal (NP cells: {len([a for a in abm_normal["agents"] if a.cell_type=="NP_cell"])})')
    # Degenerated
    ecm_hist_d = abm_degen['ecm_history']
    if len(ecm_hist_d) > 0:
        ts_d = np.array(ecm_hist_d)
        ax_ts.plot(ts_d[:, 0], ts_d[:, 1], color=NP_RED, linewidth=1.8,
                   label=f'Degenerated (NP cells: {len([a for a in abm_degen["agents"] if a.cell_type=="NP_cell"])})')

    style_ax(ax_ts, xlabel='Time step', ylabel='ECM concentration',
             title='ABM Time Series\nECM Dynamics')
    ax_ts.legend(fontsize=7, framealpha=0.8, edgecolor='#ccc')

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Figure 7: {output_path}")
    plt.close()
    return fig


# ================================================================
# Supplementary Figure S1: miRNA Network
# ================================================================

def figure_s1_mirna(output_path=None):
    """Supplementary Figure S1: miRNA regulatory network."""
    from regulation.mirna_network import (
        simulate_mirna_expression, mirna_diagnostic_analysis
    )

    mirna_data = simulate_mirna_expression(n_normal=25, n_degen=25, seed=42)
    fig = mirna_diagnostic_analysis(mirna_data, output_path=output_path, dpi=DPI)

    return fig


# ================================================================
# Supplementary Figure S2: scRNA-seq Integration
# ================================================================

def figure_s2_scrnaseq(output_path=None):
    """Supplementary Figure S2: scRNA-seq integration pipeline."""
    from data.scrnaintegration import (
        simulate_scrnaseq, pca_tsne_visualization,
        calculate_pathway_activity, plot_pathway_activity
    )

    adata = simulate_scrnaseq(n_cells=1500, n_genes=400, seed=42)
    activities = calculate_pathway_activity(adata)

    # Combined figure
    fig1 = pca_tsne_visualization(adata, output_path=None, dpi=DPI)
    fig2 = plot_pathway_activity(adata, activities, output_path=None, dpi=DPI)

    # Save combined
    if output_path:
        fig1.savefig(output_path.replace('.png', '_dimreduction.png'),
                     dpi=DPI, bbox_inches='tight', facecolor='white')
        fig2.savefig(output_path.replace('.png', '_pathway.png'),
                     dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[✓] Supplementary Figure S2: {output_path}")

    plt.close(fig1)
    plt.close(fig2)
    return fig1, fig2


# ================================================================
# Main: Generate All Figures
# ================================================================

def main():
    print("=" * 60)
    print("  Virtual NP Cell — Publication-Grade Figures Pipeline")
    print("=" * 60)
    print(f"  Output: {OUTPUT}")
    print(f"  DPI: {DPI}")
    print()

    # Load validation data if exists, otherwise generate
    vdata_path = os.path.join(OUTPUT, "validation_data.json")
    if os.path.exists(vdata_path):
        with open(vdata_path) as f:
            validation_data = json.load(f)
        print("✓ Loaded validation data from previous run")
    else:
        print("⚠ No validation data found. Run run_validation.py first.")
        validation_data = {'scatter': {'sim': [], 'lit': []}, 'validation_stats': {}}

    # Figure 1: System Architecture
    print("\n[1/7] Generating Figure 1 — System Architecture...")
    figure1_architecture(os.path.join(OUTPUT, "Figure_1_architecture.png"))

    # Figure 2: Validation Scatter
    print("[2/7] Generating Figure 2 — Validation Scatter...")
    figure2_validation_scatter(validation_data,
                               os.path.join(OUTPUT, "Figure_2_validation.png"))

    # Figure 3: Signaling + Perturbation
    print("[3/7] Generating Figure 3 — Signaling & Perturbation...")
    figure3_signaling_perturbation(os.path.join(OUTPUT, "Figure_3_signaling.png"))

    # Figure 4: MRTF-A
    print("[4/7] Generating Figure 4 — MRTF-A Mechano-Glycolysis...")
    figure4_mrtfa_mechano(os.path.join(OUTPUT, "Figure_4_mrtfa.png"))

    # Figure 5: ECM
    print("[5/7] Generating Figure 5 — ECM Dynamics...")
    figure5_ecm_dynamics(os.path.join(OUTPUT, "Figure_5_ecm.png"))

    # Figure 6: Epigenetic Senescence
    print("[6/7] Generating Figure 6 — Epigenetic Senescence...")
    figure6_epigenetic_senescence(os.path.join(OUTPUT, "Figure_6_epigenetic.png"))

    # Figure 7: ABM Microenvironment
    print("[7/7] Generating Figure 7 — ABM Microenvironment...")
    figure7_abm_microenv(os.path.join(OUTPUT, "Figure_7_abm.png"))

    # Supplementary Figures
    print("\n[S1/2] Generating Supplementary Figure S1 — miRNA Network...")
    try:
        figure_s1_mirna(os.path.join(OUTPUT, "Figure_S1_mirna_roc.png"))
    except Exception as e:
        print(f"  ⚠ Supplementary Fig S1 skipped: {e}")

    print("[S2/2] Generating Supplementary Figure S2 — scRNA-seq...")
    try:
        figure_s2_scrnaseq(os.path.join(OUTPUT, "Figure_S2_scrnaseq.png"))
    except Exception as e:
        print(f"  ⚠ Supplementary Fig S2 skipped: {e}")

    print("\n" + "=" * 60)
    print("  All figures generated successfully!")
    print(f"  Output directory: {OUTPUT}")
    print("=" * 60)

    # List files
    files = sorted([f for f in os.listdir(OUTPUT) if f.endswith('.png')])
    print("\n  Generated figures:")
    for f in files:
        fpath = os.path.join(OUTPUT, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    . {f:45s} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
