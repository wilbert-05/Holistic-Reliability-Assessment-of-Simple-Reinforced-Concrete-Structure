"""
Reliability Assessment of an RC Beam
=====================================
Module 4: Cover sweep (10-90 mm), feasible-window evaluation per exposure
class, and time-dependent durability / predicted service life.

Corresponds to Methodology Section 3.8 and Results Sections 4.3.2-4.3.4.

NOTE: The FORM-based random-variable importance-factor sensitivity analysis
(Methodology Section 3.7.1 / Results Section 4.2.1) and the Monte Carlo
sample-size convergence study (Methodology Section 3.7.2 / Results Section
4.2.2) live in a separate companion notebook not included in this folder.
"""

import numpy as np
import pandas as pd
from _01_reference_design_and_random_variables import FIXED, fixed_for
from _02_limit_state_functions import G_FUNCS, G_DUR_BY_EXPOSURE
from _01_reference_design_and_random_variables import RV_SETS
from _03_reliability_methods import form

# =============================================================================
# 9. Cover sweep 10 - 90 mm
# The concrete cover is the design variable. A larger cover improves
# durability but reduces the effective depth d, lowering resistance and
# raising deflection, crack width, and stresses - so structural reliability
# drops. FORM is re-run for every limit state while sweeping the cover from
# 10 mm to 90 mm in 1 mm steps (5 mm shown in the original demonstration).
# =============================================================================

covers = np.arange(10, 91, 1, dtype=float)
sweep = {nm: [] for nm in G_FUNCS}

for c in covers:
    fixed_c = dict(FIXED)
    fixed_c["cover"] = float(c)
    for nm, g in G_FUNCS.items():
        r = form(g, RV_SETS[nm], fixed_c)
        sweep[nm].append(r["beta"])

sweep_df = pd.DataFrame(sweep, index=covers)
sweep_df.index.name = "cover mm"

# =============================================================================
# 9.2 Governing limit state and feasible cover window
# For each cover we take the lowest-margin (governing) limit state and check
# it against its own target. The feasible window is the widest interval of
# cover for which ALL seven limit states pass simultaneously.
# =============================================================================

BETA_TARGET_XC3 = {
    "ULS bending": 3.8, "ULS shear": 3.8,
    "SLS deflection": 1.5, "SLS crack": 1.5,
    "SLS steel stress": 1.5, "SLS concrete stress": 1.5,
    "Durability carbonation XC3": 0.5,
}

BETA_TARGET_XC4 = {
    "ULS bending": 3.8, "ULS shear": 3.8,
    "SLS deflection": 1.5, "SLS crack": 1.5,
    "SLS steel stress": 1.5, "SLS concrete stress": 1.5,
    "Durability carbonation XC4": 1.5,
}


def evaluate_feasibility(beta_target, exposure_label):
    meets = pd.DataFrame(
        {nm: sweep_df[nm] >= tgt for nm, tgt in beta_target.items()},
        index=covers)
    all_pass = meets.all(axis=1)

    margins = pd.DataFrame(
        {nm: sweep_df[nm] - beta_target[nm] for nm in beta_target},
        index=covers)
    governing_ls = margins.idxmin(axis=1)
    governing_margin = margins.min(axis=1)

    table = pd.DataFrame({
        "governing LS": governing_ls,
        "margin (target)": governing_margin.round(3),
        "all pass": all_pass,
    })

    print(f"{'':=<70}")
    print(f" Exposure class: {exposure_label} (target = {min(beta_target.values())})")
    print(f"{'':=<70}")
    print(table.to_string())

    feasible = covers[all_pass.values]
    if len(feasible):
        c_low, c_high = feasible.min(), feasible.max()
        print(f"\nFeasible cover window (all LS meet target): {c_low:.0f}-{c_high:.0f} mm")
        print(f"  Lower bound governed by {governing_ls[c_low]}"
              f" (margin={margins.loc[c_low, governing_ls[c_low]]:.3f})")
        print(f"  Upper bound governed by {governing_ls[c_high]}"
              f" (margin={margins.loc[c_high, governing_ls[c_high]]:.3f})")
    else:
        print("\nNo cover in the sweep range satisfies ALL limit states simultaneously.")

    return table, feasible


table_xc3, feasible_xc3 = evaluate_feasibility(BETA_TARGET_XC3, "XC3 (target 0.5)")
table_xc4, feasible_xc4 = evaluate_feasibility(BETA_TARGET_XC4, "XC4 (target 1.5)")

FEASIBLE_WINDOW = {"XC3": feasible_xc3, "XC4": feasible_xc4}

# =============================================================================
# 9.3 Durability reliability over time T - service-life prediction
# The durability limit state is the only one whose g depends explicitly on
# time, through the square-root-of-time carbonation front. A single beta is
# therefore meaningless without stating the reference period T. Sweeping T
# gives the predicted service life: the year at which beta(T) crosses target.
# =============================================================================

Tgrid = np.arange(1, 101, 1)
covers_demo = [10, 20, 28, 30, 34, 40, 50]


def compute_beta_T(gfunc, rvset_name):
    beta_T = {}
    for c in covers_demo:
        bs = []
        for T in Tgrid:
            fx = dict(FIXED)
            fx["cover"] = float(c)
            fx["T_life"] = float(T)
            bs.append(form(gfunc, RV_SETS[rvset_name], fx)["beta"])
        beta_T[c] = np.array(bs)
    return beta_T


def service_life_table(beta_T, target):
    out = {}
    for c in covers_demo:
        b = beta_T[c]
        below = np.where(b >= target)[0]
        out[c] = Tgrid[below[0]] if len(below) else 100
    return out


if __name__ == "__main__":
    beta_T_xc3 = compute_beta_T(G_FUNCS["Durability carbonation XC3"], "Durability carbonation XC3")
    beta_T_xc4 = compute_beta_T(G_FUNCS["Durability carbonation XC4"], "Durability carbonation XC4")

    sl_xc3 = service_life_table(beta_T_xc3, 0.5)
    sl_xc4 = service_life_table(beta_T_xc4, 1.5)

    table_service_life = pd.DataFrame({
        "cover_mm": covers_demo,
        "service_life_XC3_yr": [sl_xc3[c] for c in covers_demo],
        "service_life_XC4_yr": [sl_xc4[c] for c in covers_demo],
    }).set_index("cover_mm")

    print("\nPredicted service life vs cover (XC3 vs XC4):")
    print(table_service_life)
