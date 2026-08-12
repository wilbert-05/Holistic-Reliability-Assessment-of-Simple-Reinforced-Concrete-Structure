"""
Reliability Assessment of an RC Beam
=====================================
Module 6: Cost sensitivity analysis.

Two one-at-a-time sensitivity sweeps around the two calibration inputs that
are taken from the literature rather than measured directly:

  A) Discount rate r, swept from 1% to 10% in steps of 1%.
  B) Severity exponent (power) in severity_mult(), swept from 0.1 to 1.0.

For each sweep value, run_cost_optimization() is re-run for BOTH exposure
classes and the resulting optimal cover and minimum total cost are recorded.

IMPORTANT: this temporarily overwrites the module-level r_disc / severity_mult
used by disc()/get_C_STRUCT()/get_Ff()/get_rep_costs(), then restores the
original values at the end so the rest of the analysis is unaffected.

N=6000 matches the headline Monte Carlo sample size used in Module 5 (Table
4.7/4.9 in the dissertation). An earlier run at N=2000 produced slightly
different optimal covers purely due to sampling noise near the very flat
cost minimum (total cost differs by only ~0.03-0.07% between neighbouring
covers) -- this is expected behaviour for a shallow objective, not a bug,
and N=6000 was adopted throughout to match the headline results exactly.

Corresponds to Results Section 4.4.3.
"""

import numpy as np
import pandas as pd

import _05_lifecycle_cost_optimization as lco
from _04_cover_sweep_and_feasible_window import FEASIBLE_WINDOW

N_SENSITIVITY = 6000  # matches headline Table 4.7/4.9 sample size

# =============================================================================
# A. Discount rate sensitivity, r = 1% ... 10%
# =============================================================================


def run_discount_rate_sweep():
    r_disc_original = lco.r_disc
    rgrid = np.arange(0.01, 0.101, 0.01)
    rows = []
    for r_test in rgrid:
        lco.r_disc = float(r_test)
        cost_A_xc3, c_opt_xc3, _ = lco.run_cost_optimization("XC3", N=N_SENSITIVITY)
        cost_A_xc4, c_opt_xc4, _ = lco.run_cost_optimization("XC4", N=N_SENSITIVITY)
        rows.append({
            "r_disc": round(r_test, 6),
            "c_opt_XC3": c_opt_xc3, "TOTAL_XC3": cost_A_xc3.loc[c_opt_xc3, "TOTAL"],
            "c_opt_XC4": c_opt_xc4, "TOTAL_XC4": cost_A_xc4.loc[c_opt_xc4, "TOTAL"],
        })
    lco.r_disc = r_disc_original  # restore baseline (0.02)
    return pd.DataFrame(rows).set_index("r_disc")


# =============================================================================
# B. Severity exponent sensitivity, power = 0.1 ... 1.0
# XC3 is structurally unaffected by this sweep (severity_mult only scales
# XC4's consequence-side parameters relative to the XC3 reference), so its
# optimum should stay fixed across every tested value -- used as a sanity
# check below.
# =============================================================================


def run_severity_exponent_sweep():
    severity_mult_original = lco.severity_mult
    powergrid = np.arange(0.1, 1.01, 0.1)
    rows = []
    for p_test in powergrid:
        def severity_mult_patched(exposure, power=p_test):
            mean_ic = lco.ICORR_BY_EXPOSURE[exposure][1]
            return (mean_ic / lco.ICORR_REF) ** power
        lco.severity_mult = severity_mult_patched

        cost_A_xc3, c_opt_xc3, _ = lco.run_cost_optimization("XC3", N=N_SENSITIVITY)
        cost_A_xc4, c_opt_xc4, _ = lco.run_cost_optimization("XC4", N=N_SENSITIVITY)
        rows.append({
            "severity_power": round(p_test, 6),
            "c_opt_XC3": c_opt_xc3, "TOTAL_XC3": cost_A_xc3.loc[c_opt_xc3, "TOTAL"],
            "c_opt_XC4": c_opt_xc4, "TOTAL_XC4": cost_A_xc4.loc[c_opt_xc4, "TOTAL"],
        })
    lco.severity_mult = severity_mult_original  # restore original function
    return pd.DataFrame(rows).set_index("severity_power")


if __name__ == "__main__":
    print("Running discount-rate sensitivity sweep (1%-10%)...")
    table_discount_rate = run_discount_rate_sweep()
    print(table_discount_rate)

    print("\nRunning severity-exponent sensitivity sweep (0.1-1.0)...")
    table_severity_exponent = run_severity_exponent_sweep()
    print(table_severity_exponent)

    # Sanity check 1: XC3 TOTAL must be constant across the severity sweep.
    xc3_variation = table_severity_exponent["TOTAL_XC3"].std()
    print(f"\nSanity check 1: XC3 TOTAL std-dev across power sweep = "
          f"{xc3_variation:.8f} (should be 0.0)")

    # Sanity check 2: the power=0.4 row should match the Module 5 headline exactly
    # (same N, same seed, same r_disc -- power=0.4 IS the headline configuration).
    row_04 = table_severity_exponent.loc[0.4]
    print(f"Sanity check 2 (power=0.4 vs headline):")
    print(f"  Sweep : c_opt_XC3={row_04.c_opt_XC3}, TOTAL_XC3={row_04.TOTAL_XC3:.6f}, "
          f"c_opt_XC4={row_04.c_opt_XC4}, TOTAL_XC4={row_04.TOTAL_XC4:.6f}")
