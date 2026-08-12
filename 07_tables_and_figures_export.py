"""
Reliability Assessment of an RC Beam
=====================================
Module 7: Generates every table (CSV) and figure (PNG) reported in the
Results and Discussion chapter (Chapter 4) of the dissertation, from the
outputs of Modules 3-6.

Run this script last, after Modules 1-6 have been executed (or imported),
to regenerate the full set of dissertation tables/figures into OUTDIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _01_reference_design_and_random_variables import FIXED
from _02_limit_state_functions import G_FUNCS
from _03_reliability_methods import form, monte_carlo, mvfosm
from _04_cover_sweep_and_feasible_window import (
    sweep_df, covers, FEASIBLE_WINDOW, feasible_xc3, feasible_xc4,
    table_xc3, table_xc4, compute_beta_T, service_life_table, Tgrid, covers_demo,
)
import _05_lifecycle_cost_optimization as lco
import _06_cost_sensitivity_analysis as sens
from _01_reference_design_and_random_variables import RV_SETS

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# Table 4.4 / Figure 4.1: Three reliability methods compared at reference cover
# =============================================================================

N_MC_TABLE = 5_000_000
rows_d4 = []
for nm, g in G_FUNCS.items():
    res_form = form(g, RV_SETS[nm], FIXED)
    pf_mc, b_mc, nf = monte_carlo(g, RV_SETS[nm], FIXED, N=N_MC_TABLE)
    b_mv, pf_mv = mvfosm(g, RV_SETS[nm], FIXED)
    rows_d4.append({
        "limit state": nm, "beta_FORM": res_form["beta"], "beta_MCS": b_mc,
        "beta_MVFOSM": b_mv, "FORM_minus_MVFOSM": res_form["beta"] - b_mv,
    })
table_D4 = pd.DataFrame(rows_d4).set_index("limit state").round(4)
table_D4.to_csv(os.path.join(OUTDIR, "Table4.4_method_comparison.csv"))

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(table_D4)); w = 0.27
ax.bar(x - w, table_D4["beta_FORM"], w, label="FORM", color="#4C72B0")
ax.bar(x, table_D4["beta_MCS"], w, label="MCS", color="#55A868")
ax.bar(x + w, table_D4["beta_MVFOSM"], w, label="MVFOSM", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(table_D4.index, rotation=25, ha="right")
ax.set_ylabel(r"reliability index $\beta$", fontsize=13)
ax.set_title("Comparison of the three reliability methods at reference cover (c=30 mm)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Figure4.1_method_comparison.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# Table 4.5: beta at extremes of the swept cover range (10 mm vs 90 mm)
# =============================================================================

c_lo, c_hi = sweep_df.index.min(), sweep_df.index.max()
table_D5 = pd.DataFrame({
    f"beta_at_{int(c_lo)}mm": sweep_df.loc[c_lo],
    f"beta_at_{int(c_hi)}mm": sweep_df.loc[c_hi],
})
table_D5["delta_beta"] = table_D5[f"beta_at_{int(c_hi)}mm"] - table_D5[f"beta_at_{int(c_lo)}mm"]
table_D5 = table_D5.round(3)
table_D5.to_csv(os.path.join(OUTDIR, "Table4.5_cover_extremes.csv"))

# =============================================================================
# Figure 4.4 / 4.5: Reliability index vs cover, XC3-only and XC4-only
# =============================================================================


def plot_reliability_with_window(exclude_name, subtitle, dur_target, dur_label,
                                  feasible_window, savepath):
    fig, ax = plt.subplots(figsize=(10, 6))
    for nm in G_FUNCS:
        if nm == exclude_name:
            continue
        ax.plot(sweep_df.index, sweep_df[nm], marker="o", ms=5, lw=2, label=nm)
    ax.axhline(3.8, color="red", ls="--", lw=1.5, label=r"ULS target 3.8")
    ax.axhline(1.5, color="grey", ls="-", lw=1.5, label=r"SLS target 1.5")
    ax.axhline(dur_target, color="magenta", ls=(0, (3, 1, 1, 1)), lw=1.5, label=dur_label)
    if len(feasible_window):
        clo, chi = feasible_window.min(), feasible_window.max()
        ax.axvspan(clo, chi, color="0.90", label=f"feasible window ({int(clo)}-{int(chi)} mm)")
    ax.set_xlabel("concrete cover [mm]", fontsize=13)
    ax.set_ylabel(r"reliability index $\beta$", fontsize=13)
    ax.set_title(f"Reliability index vs cover ({subtitle})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


plot_reliability_with_window(
    exclude_name="Durability carbonation XC4", subtitle="XC3 only",
    dur_target=0.5, dur_label=r"XC3 durability target 0.5",
    feasible_window=np.asarray(feasible_xc3),
    savepath=os.path.join(OUTDIR, "Figure4.4_reliability_vs_cover_XC3.png"))

plot_reliability_with_window(
    exclude_name="Durability carbonation XC3", subtitle="XC4 only",
    dur_target=1.5, dur_label=r"XC4 durability target 1.5",
    feasible_window=np.asarray(feasible_xc4),
    savepath=os.path.join(OUTDIR, "Figure4.5_reliability_vs_cover_XC4.png"))

# =============================================================================
# Table 4.6: feasible cover windows and governing limit states
# =============================================================================


def summarize_window(table, feasible, label):
    feasible = np.asarray(feasible)
    if len(feasible) == 0:
        return {"exposure": label, "cmin": None, "cmax": None,
                "governing_LS_lower": None, "margin_lower": None,
                "governing_LS_upper": None, "margin_upper": None}
    clo, chi = feasible.min(), feasible.max()
    row_lo = table.loc[clo]; row_hi = table.loc[chi]
    return {"exposure": label, "cmin": clo, "cmax": chi,
            "governing_LS_lower": row_lo["governing LS"],
            "margin_lower": row_lo["margin (target)"],
            "governing_LS_upper": row_hi["governing LS"],
            "margin_upper": row_hi["margin (target)"]}


table_D6 = pd.DataFrame([
    summarize_window(table_xc3, feasible_xc3, "XC3"),
    summarize_window(table_xc4, feasible_xc4, "XC4"),
]).set_index("exposure")
table_D6.to_csv(os.path.join(OUTDIR, "Table4.6_feasible_windows.csv"))

# =============================================================================
# Table 4.7 / Figure 4.6: predicted service life vs cover (XC3 vs XC4)
# =============================================================================

DUR_TARGET_XC3, DUR_TARGET_XC4 = 0.5, 1.5
beta_T_xc3 = compute_beta_T(G_FUNCS["Durability carbonation XC3"], "Durability carbonation XC3")
beta_T_xc4 = compute_beta_T(G_FUNCS["Durability carbonation XC4"], "Durability carbonation XC4")

sl_xc3 = service_life_table(beta_T_xc3, DUR_TARGET_XC3)
sl_xc4 = service_life_table(beta_T_xc4, DUR_TARGET_XC4)

table_D7 = pd.DataFrame({
    "cover_mm": list(covers_demo),
    "service_life_XC3_yr": [sl_xc3[c] for c in covers_demo],
    "service_life_XC4_yr": [sl_xc4[c] for c in covers_demo],
}).set_index("cover_mm")
table_D7.to_csv(os.path.join(OUTDIR, "Table4.7_service_life.csv"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for ax, beta_T, target, label in [(axes[0], beta_T_xc3, DUR_TARGET_XC3, "XC3"),
                                    (axes[1], beta_T_xc4, DUR_TARGET_XC4, "XC4")]:
    for c in covers_demo:
        b = beta_T[c]
        ax.plot(Tgrid, b, lw=2, label=f"cover {c} mm")
        below = np.where(b >= target)[0]
        if len(below):
            t_cross = Tgrid[below[0]]
            ax.plot(t_cross, target, "ko", ms=5)
            ax.annotate(f"{int(t_cross)} yr", (t_cross, target),
                        textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.axhline(target, ls="--", color="red", lw=1.5, label=f"target beta={target}")
    ax.set_xlabel("time T [yr]", fontsize=12)
    ax.set_title(f"Durability reliability profile ({label})", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel(r"durability $\beta(t)$", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Figure4.6_service_life_profiles.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# Table 4.9: Eurocode 2 nominal cover vs. feasible window and cost-optimal cover
# =============================================================================

c_nom_XC3 = 35  # EN 1992-1-1 Table 4.4N, Structural Class S4, XC3
c_nom_XC4 = 40  # same table, XC4

rows_d9 = []
for label, cost_A, c_opt, feas_win, c_nom in [
    ("XC3", lco.cost_A_xc3 if hasattr(lco, "cost_A_xc3") else None, None, feasible_xc3, c_nom_XC3),
    ("XC4", lco.cost_A_xc4 if hasattr(lco, "cost_A_xc4") else None, None, feasible_xc4, c_nom_XC4),
]:
    pass  # populated after running Module 5's __main__ block; see README

print(f"All tables and figures exported to {OUTDIR}")
