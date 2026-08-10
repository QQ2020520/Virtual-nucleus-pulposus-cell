#!/usr/bin/env python3
"""
gene_validation_reproduce.py — REPRODUCIBLE 13-gene validation table (Table 2)
for the Virtual NP Cell manuscript. Day-6 "high-quality supplemental" closure:
the original Table 2 in data/validation_report.md had no code-faithful generator
(normalization mixed, e.g. MRTFA set degen=1.0, HELZ2 mismatched real model).

This script regenerates the 13-gene "Model Δ%" column DIRECTLY from the current
ODE models using ONE consistent rule applied to every gene:

    fold_change = degenerate_steady / healthy_steady
    model_delta% = (fold_change - 1) * 100

Healthy state is normalized to 1.000 for every gene (reference frame used in the
paper). Genes that have a direct state variable in an ODE model are computed
exactly; genes with only a pathway-protein proxy are tagged 'proxy' and computed
from the closest model variable, with the limitation stated explicitly.

Run:  python3 figures/gene_validation_reproduce.py
Outputs:
  - output/gene_validation_data.json     (all numbers, machine-readable)
  - output/gene_validation_report.md     (formatted Table 2 + MAPE/R2)
"""

import os, sys, json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.signaling import MRTFAMechanoModel
from simulation.epigenetic_senescence import EpigeneticSenescenceModel
from simulation.ecm_model import ECMDegradationModel

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT, exist_ok=True)

ECM_PERT = {"INFLAM": 3.0, "OXSTRESS": 2.0, "NUTRIENT": -1.0, "MECHANICAL": 0.5}


def run_models():
    """Return all healthy/degenerate steady-state vectors (same signatures as
    validation_reproduce.py / run_validation.py)."""
    mrtf = MRTFAMechanoModel()
    _, yn_m = mrtf.simulate(stiffness=2.0)
    _, yd_m = mrtf.simulate(stiffness=15.0)

    epi = EpigeneticSenescenceModel()
    _, yn_e = epi.simulate()
    _, yd_e = epi.simulate(perturbation={"SETD1A_knockdown": 0.8})

    ec = ECMDegradationModel()
    _, yn_c = ec.simulate()
    _, yd_c = ec.simulate(t_span=(0, 400), n_points=500,
                          perturbation=ECM_PERT, degen_accel=1.0)
    return (mrtf, epi, ec), (yn_m, yn_e, yn_c), (yd_m, yd_e, yd_c)


# gene -> (model_key, state_index, healthy_slice, degenerate_slice)
# state indices verified against source: MRTFAMechanoModel=8 vars,
# EpigeneticSenescenceModel=8 vars, ECMDegradationModel=7 vars.
GENE_MAP = {
    #             model  idx  healthy        degenerate       proxy_note
    "ACAN":    ("ecm", 0, "normal", "degenerate", None),
    "COL2A1":  ("ecm", 1, "normal", "degenerate", None),
    "SOX9":    ("ecm", 0, "normal", "degenerate", "anabolic regulator; proxied by Aggrecan (matrix synthesis)"),
    "MMP3":    ("ecm", 2, "normal", "degenerate", None),
    "IL6":     ("ecm", 4, "normal", "degenerate", "cytokine; proxied by ECM inflammation level"),
    "HIF1A":   ("epi", 4, "baseline", "setd1a_KD80", None),
    "SETD1A":  ("epi", 0, "baseline", "setd1a_KD80", None),
    "HELZ2":   ("epi", 2, "baseline", "setd1a_KD80", None),
    "MRTFA":   ("mrtf", 1, "2kPa", "15kPa", None),
    "KIDINS220": ("mrtf", 2, "2kPa", "15kPa", None),
    "PFKFB3":  ("mrtf", 4, "2kPa", "15kPa", None),
    "PFKM":    ("mrtf", 5, "2kPa", "15kPa", None),
    "PLD1":    ("mrtf", 6, "2kPa", "15kPa", None),
}

# Literature values from data/validation_report.md (2026-07-10), given as
# (healthy, degenerate) absolute reference values in that report's scale.
LIT = {
    "ACAN": (1.000, 0.350), "COL2A1": (1.000, 0.420), "SOX9": (0.800, 0.380),
    "MMP3": (0.200, 0.850), "IL6": (0.100, 0.720), "HIF1A": (0.700, 0.300),
    "SETD1A": (0.800, 0.350), "HELZ2": (0.650, 0.350), "MRTFA": (0.400, 0.850),
    "KIDINS220": (0.750, 0.250), "PFKFB3": (0.800, 0.150),
    "PFKM": (0.700, 0.200), "PLD1": (0.600, 0.250),
}


def main():
    _, yn, yd = run_models()   # tuples: (mrtf, epi, ecm) for (healthy, degenerate)
    vec = {"mrtf": 0, "epi": 1, "ecm": 2}

    rows, model_deltas, lit_deltas = [], [], []
    for gene, (model, idx, hk, dk, proxy) in GENE_MAP.items():
        hi = vec[model]
        hv = float(yn[hi][idx, -1])   # healthy steady-state (already the model's own healthy cond)
        dv = float(yd[hi][idx, -1])   # degenerate steady-state (model's own degen cond)
        ratio = (dv / hv) if hv != 0 else float("nan")
        m_delta = (ratio - 1.0) * 100.0
        lh, ld = LIT[gene]
        l_delta = (ld - lh) / lh * 100.0
        model_deltas.append(m_delta)
        lit_deltas.append(l_delta)
        rows.append({
            "gene": gene, "model_state": {"model": model, "index": idx,
                                          "healthy_cond": hk, "degenerate_cond": dk},
            "model_healthy": round(hv, 4), "model_degenerate": round(dv, 4),
            "model_ratio": round(ratio, 3), "model_delta_pct": round(m_delta, 2),
            "lit_healthy": lh, "lit_degenerate": ld, "lit_delta_pct": round(l_delta, 2),
            "direction_match": (m_delta < 0) == (l_delta < 0),
            "proxy": proxy,
        })

    n_match = sum(r["direction_match"] for r in rows)
    # Directional accuracy on raw model ratios (un-rounded).
    acc = n_match / len(rows) * 100.0
    # MAPE on |delta%| (percent-scale, as in original report).
    mape = float(np.mean([abs(md - ld) for md, ld in
                          zip([r["model_delta_pct"] for r in rows], lit_deltas)]))
    # R2 of model vs literature delta% (coefficient of determination).
    md = np.array([r["model_delta_pct"] for r in rows])
    ld = np.array(lit_deltas)
    ss_res = float(np.sum((md - ld) ** 2))
    ss_tot = float(np.sum((ld - ld.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    out = {
        "rule": "fold_change = degen_steady/healthy_steady; delta%=(ratio-1)*100; healthy normalized to 1.000",
        "n_genes": len(rows), "directional_accuracy_pct": round(acc, 1),
        "mape_delta_pct": round(mape, 1), "r_squared_delta_pct": round(r2, 3),
        "genes": rows,
    }
    with open(os.path.join(OUTPUT, "gene_validation_data.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown report ----
    L = ["# Table 2 (reproduced) — 13-gene Model vs Literature Direction & Delta",
         "",
         f"> **Rule**: fold-change = degenerate_steady/healthy_steady; Δ% = (fold-change − 1) × 100; "
         f"healthy state normalized to 1.000 for every gene.",
         f"> **Stats**: Directional accuracy = {acc:.1f}% ({n_match}/{len(rows)}); "
         f"MAPE(Δ%) = {mape:.1f} pts; R²(model vs literature Δ%) = {r2:.3f}.",
         f"> **Provenance**: all model values regenerated by `figures/gene_validation_reproduce.py`.",
         "",
         "| Gene | Model state mapped | Model healthy | Model degenerate | Model Δ% | Lit Δ% | Direction | Proxy? |",
         "|------|--------------------|--------------:|-----------------:|---------:|-------:|-----------|--------|"]
    for r in rows:
        proxy = "yes" if r["proxy"] else "no"
        state = f"{r['model_state']['model']}[{r['model_state']['index']}] ({r['model_state']['healthy_cond']}→{r['model_state']['degenerate_cond']})"
        L.append(f"| {r['gene']} | {state} | {r['model_healthy']:.3f} | {r['model_degenerate']:.3f} | "
                 f"{r['model_delta_pct']:+.1f}% | {r['lit_delta_pct']:+.1f}% | "
                 f"{'✅' if r['direction_match'] else '❌'} | {proxy} |")
    L += [
        "",
        "### Proxy genes (no direct transcript in an ODE model)",
        "",
    ]
    for r in rows:
        if r["proxy"]:
            L.append(f"- **{r['gene']}** — {r['proxy']} (computed from that variable).")
    L += [
        "",
        "### Quantitative caveats",
        "",
        "- SETD1A/HELZ2/HIF1A use the EpigeneticSenescenceModel under SETD1A "
        "knockdown (80%), the paper's central degeneration driver.",
        "- MRTF-A gene family uses MRTFAMechanoModel at 15 kPa (degenerate) vs 2 kPa.",
        "- ECM genes (ACAN/COL2A1/SOX9/MMP3/IL6) use ECMDegradationModel under the "
        "degenerative stress perturbation (INFLAM 3.0/OXSTRESS 2.0/NUTRIENT −1.0/MECH 0.5).",
        "",
    ]
    with open(os.path.join(OUTPUT, "gene_validation_report.md"), "w") as f:
        f.write("\n".join(L))

    print(f"DONE: {n_match}/{len(rows)} directional match ({acc:.1f}%), "
          f"MAPE(Δ%)={mape:.1f}, R²(Δ%)={r2:.3f}")
    print("Wrote: output/gene_validation_data.json, output/gene_validation_report.md")


if __name__ == "__main__":
    main()
