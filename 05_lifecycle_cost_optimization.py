"""
Reliability Assessment of an RC Beam
=====================================
Module 5: Lifecycle cost optimization.

Among the reliability-feasible covers (Module 4), choose the cover with the
lowest expected lifecycle cost. Every consequence is counted once:

    C_total(c) = C_build(c) + C_inspection(c) + C_repair_minor(c)
                 + C_repair_major(c) + C_fail(c) + C_risk(c)

Structural risk uses the undamaged reliability from the Module-4 cover sweep
(t=0), justified because the maintenance rule keeps the section healthy
(perfect repair). Durability maintenance is an inspect-repair Monte Carlo
lifecycle simulation driven by the fib carbonation CDF, fully consistent
with the reliability assessment of Modules 1-4. Corrosion appears only in
the maintenance term; structural failure only in the risk term.

Corresponds to Methodology Section 3.9 and Results Section 4.4.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d, RegularGridInterpolator

from _01_reference_design_and_random_variables import (
    FIXED, RV_SETS, As_tot, bar_diam, n_bars, ICORR_BY_EXPOSURE, x_from_u,
)
from _02_limit_state_functions import G_FUNCS, G_DUR_BY_EXPOSURE
from _03_reliability_methods import form
from _04_cover_sweep_and_feasible_window import FEASIBLE_WINDOW, covers

# =============================================================================
# 10.0 Common setup: feasible window, unit costs, durability lifecycle engine
# =============================================================================

STRUCT_LS = [nm for nm in G_FUNCS if not nm.startswith("Durability carbonation")]

RDISC_BASELINE = 0.02
r_disc = RDISC_BASELINE
c_insp = 0.04
dt_insp = 5.0
T_life = 50.0

C_build0 = 1.00
build_per_mm = 0.00167
c_ref = 30.0

Ff_BASE = 0.4
c_rep_minor_BASE = 0.3
c_rep_medium_BASE = 0.3 * 1.23
c_rep_major_BASE = 0.3 * 1.34
C_STRUCT_BASE = {
    "ULS bending": 5.0, "ULS shear": 7.0,
    "SLS deflection": 0.10, "SLS crack": 0.10,
    "SLS steel stress": 0.10, "SLS concrete stress": 0.10,
}

ICORR_REF = ICORR_BY_EXPOSURE["XC3"][1]


def severity_mult(exposure, power=0.4):
    """Exposure severity multiplier m(e) = (i_corr,e / i_corr,XC3)^power."""
    mean_ic = ICORR_BY_EXPOSURE[exposure][1]
    return (mean_ic / ICORR_REF) ** power


def get_C_STRUCT(exposure):
    m = severity_mult(exposure)
    return {k: v * m for k, v in C_STRUCT_BASE.items()}


def get_Ff(exposure):
    return Ff_BASE * severity_mult(exposure)


def get_rep_costs(exposure):
    m = severity_mult(exposure)
    return {
        "minor": c_rep_minor_BASE,
        "medium": c_rep_medium_BASE * m,
        "major": c_rep_major_BASE * m,
    }


def disc(t):
    return 1.0 / (1.0 + r_disc) ** t


def build_cost(c):
    return C_build0 + build_per_mm * (c - c_ref)


C_insp_pv = float(np.sum(c_insp * disc(np.arange(dt_insp, T_life, dt_insp))))


def durability_cdf(cover_mm, exposure, tmax=200.0, n=160):
    """Time-dependent probability of depassivation F(t) = Phi(-beta(T))."""
    g_dur = G_DUR_BY_EXPOSURE[exposure]
    rv_dur = RV_SETS[f"Durability carbonation {exposure}"]
    fx = dict(FIXED)
    fx["cover"] = float(cover_mm)
    ts = np.linspace(0.5, tmax, n)
    betas = np.array([form(g_dur, rv_dur, {**fx, "T_life": float(T)})["beta"] for T in ts])
    F = np.maximum.accumulate(np.clip(norm.cdf(-betas), 1e-9, 1 - 1e-9))
    return ts, F


# =============================================================================
# 10.1 Structural response surfaces (precomputed via FORM on a grid of
# exposure time x corrosion rate) so the lifecycle Monte Carlo loop can use
# cheap bilinear interpolation instead of thousands of repeated FORM solves.
# =============================================================================

BETA_TARGET_STRUCT = {
    "ULS bending": 3.8, "ULS shear": 3.8,
    "SLS deflection": 1.5, "SLS crack": 1.5,
    "SLS steel stress": 1.5, "SLS concrete stress": 1.5,
}


def draw_icorr(rng, exposure):
    dist, mean, cov = ICORR_BY_EXPOSURE[exposure]
    return float(x_from_u(rng.standard_normal(), dist, mean, mean * cov))


def bar_loss_from_icorr(icorr, t_exposure):
    """Uniform corrosion rate: Delta_phi(t) = 0.0232 * i_corr * t_exposure [mm]
    (Vu & Stewart, 2000; Val et al., 2025)."""
    return 0.0232 * icorr * t_exposure


def residual_As(t_exposure, icorr):
    phi = np.maximum(bar_diam - bar_loss_from_icorr(icorr, t_exposure), 0.0)
    return n_bars * np.pi * phi ** 2 / 4.0


def struct_beta_vs_time(cover_mm, ls_name, texp_grid, icorr_grid):
    dist, mean, cov = RV_SETS[ls_name]["As"]
    fixed_c = dict(FIXED)
    fixed_c["cover"] = float(cover_mm)
    B = np.empty((len(texp_grid), len(icorr_grid)))
    for i, te in enumerate(texp_grid):
        for j, ic in enumerate(icorr_grid):
            rv = dict(RV_SETS[ls_name])
            rv["As"] = (dist, float(residual_As(te, ic)), cov)
            B[i, j] = form(G_FUNCS[ls_name], rv, fixed_c)["beta"]
    return RegularGridInterpolator(
        (np.asarray(texp_grid), np.asarray(icorr_grid)), B,
        bounds_error=False, fill_value=None)


def struct_pf_curves(cover_mm, exposure, texp_max=50.0, n_texp=12, n_icorr=7):
    dist, mean, cov = ICORR_BY_EXPOSURE[exposure]
    tgrid = np.linspace(0.0, texp_max, n_texp)
    icgrid = np.linspace(
        float(x_from_u(norm.ppf(0.05), dist, mean, mean * cov)),
        float(x_from_u(norm.ppf(0.95), dist, mean, mean * cov)), n_icorr)
    return {ls: struct_beta_vs_time(cover_mm, ls, tgrid, icgrid) for ls in STRUCT_LS}


# =============================================================================
# 10.2 Lifecycle Monte Carlo maintenance simulation.
# Each of N simulated 50-year trajectories samples a random corrosion rate
# and a random depassivation time (inverse-transform from the durability
# CDF). Whenever depassivation occurs, the structure is exposed until the
# next scheduled inspection (every dt_insp years); at that inspection, the
# residual steel section is evaluated through the response surfaces and a
# three-tier maintenance decision is made:
#   a) all structural LS still pass  -> minor repair (cover reinstatement)
#   b) an SLS (not ULS) has failed   -> medium repair
#   c) a ULS has failed              -> major repair (+ reinforcement replace)
# Each repair resets the carbonation front and restores nominal condition.
# =============================================================================


def maintenance_cost_v3(cover_mm, exposure, interps, N=6000, seed=1):
    Ff_e = get_Ff(exposure)
    rep_e = get_rep_costs(exposure)
    C_STRUCT_e = get_C_STRUCT(exposure)

    ULS_LS = [ls for ls in STRUCT_LS if ls.startswith("ULS")]
    SLS_LS = [ls for ls in STRUCT_LS if ls.startswith("SLS")]

    ts, F = durability_cdf(cover_mm, exposure)
    inv = interp1d(F, ts, bounds_error=False, fill_value=(ts[0], np.inf))
    rng = np.random.default_rng(seed)

    te_ax, ic_ax = interps[STRUCT_LS[0]].grid
    te_lo, te_hi = float(te_ax[0]), float(te_ax[-1])
    ic_lo, ic_hi = float(ic_ax[0]), float(ic_ax[-1])

    years = np.arange(0.0, T_life + 1.0, dt_insp / dt_insp)  # annual grid
    disc_years = disc(years)

    rep_minor = np.zeros(N); rep_medium = np.zeros(N); rep_major = np.zeros(N)
    fail = np.zeros(N); srisk = np.zeros(N)
    n_major_events = np.zeros(N); n_medium_events = np.zeros(N)

    def betas_at(te, ic):
        pt = np.array([np.clip(te, te_lo, te_hi), np.clip(ic, ic_lo, ic_hi)])
        return {ls: float(interps[ls]([pt])[0]) for ls in STRUCT_LS}

    for k in range(N):
        icorr = draw_icorr(rng, exposure)
        origin = 0.0
        intervals = []
        while True:
            tf = origin + float(inv(rng.random()))
            if not np.isfinite(tf) or tf >= T_life:
                intervals.append((origin, T_life + 1.0))
                break
            t_rep = np.ceil(tf / dt_insp) * dt_insp
            if t_rep > T_life:
                intervals.append((origin, T_life + 1.0))
                break
            fail[k] += (t_rep - tf) * Ff_e * disc(t_rep)
            b = betas_at(t_rep - tf, icorr)
            uls_fail = any(b[ls] < BETA_TARGET_STRUCT[ls] for ls in ULS_LS)
            sls_fail = any(b[ls] < BETA_TARGET_STRUCT[ls] for ls in SLS_LS)
            if uls_fail:
                rep_major[k] += rep_e["major"] * disc(t_rep)
                n_major_events[k] += 1
            elif sls_fail:
                rep_medium[k] += rep_e["medium"] * disc(t_rep)
                n_medium_events[k] += 1
            else:
                rep_minor[k] += rep_e["minor"] * disc(t_rep)
            intervals.append((origin, t_rep))
            origin = t_rep

        exposure_clk = np.zeros_like(years)
        for t0, t1 in intervals:
            m = (years >= t0) & (years < t1)
            exposure_clk[m] = years[m] - t0

        rk = np.zeros_like(years)
        for ls in STRUCT_LS:
            pf = norm.cdf(-interps[ls](np.column_stack([
                np.clip(exposure_clk, te_lo, te_hi),
                np.full(years.shape, np.clip(icorr, ic_lo, ic_hi))])))
            dpf = np.clip(np.diff(pf, prepend=pf[0]), 0.0, None)
            rk += dpf * C_STRUCT_e[ls]
        srisk[k] = np.sum(rk * disc_years)

    return dict(
        inspection=C_insp_pv,
        repair_minor=rep_minor.mean(), repair_medium=rep_medium.mean(),
        repair_major=rep_major.mean(),
        repair=rep_minor.mean() + rep_medium.mean() + rep_major.mean(),
        failure=fail.mean(), structrisk=srisk.mean(),
        frac_major=float((n_major_events > 0).mean()),
        frac_medium=float((n_medium_events > 0).mean()),
    )


# =============================================================================
# 10.3 Cost-optimization driver: evaluate every cover in the feasible window,
# cache the structural response surfaces per (exposure, cover), and return
# the cover that minimizes total expected lifecycle cost.
# =============================================================================

INTERPS_CACHE = {}


def run_cost_optimization(exposure, N=6000, seed=1):
    feas_set = np.asarray(FEASIBLE_WINDOW[exposure]).astype(int)
    c_lo, c_hi = int(feas_set.min()), int(feas_set.max())
    covers_disp = np.arange(max(18, int(covers.min())), c_hi + 1)
    rows = []
    for c in covers_disp:
        key = (exposure, int(c))
        if key not in INTERPS_CACHE:
            INTERPS_CACHE[key] = struct_pf_curves(int(c), exposure)
        interps = INTERPS_CACHE[key]
        d = maintenance_cost_v3(int(c), exposure, interps, N=N, seed=seed)
        b = build_cost(c)
        rows.append(dict(
            cover=int(c), build=b, inspection=d["inspection"],
            repair_minor=d["repair_minor"], repair_medium=d["repair_medium"],
            repair_major=d["repair_major"], repair=d["repair"],
            failure=d["failure"], structrisk=d["structrisk"],
            frac_major=d["frac_major"], frac_medium=d["frac_medium"],
            TOTAL=b + d["inspection"] + d["repair"] + d["failure"] + d["structrisk"],
        ))
    cost_df = pd.DataFrame(rows).set_index("cover")
    feas = cost_df.loc[cost_df.index.isin(feas_set)]
    c_opt = int(feas["TOTAL"].idxmin())
    return cost_df, c_opt, (c_lo, c_hi)


if __name__ == "__main__":
    cost_A_xc3, c_opt_xc3, feas_window_xc3 = run_cost_optimization("XC3")
    print(f"XC3 optimum: {c_opt_xc3} mm  (window {feas_window_xc3})")

    cost_A_xc4, c_opt_xc4, feas_window_xc4 = run_cost_optimization("XC4")
    print(f"XC4 optimum: {c_opt_xc4} mm  (window {feas_window_xc4})")

    print("\nCost decomposition at optimum (XC3 vs XC4):")
    cols = ["build", "inspection", "repair_minor", "repair_medium",
            "repair_major", "repair", "failure", "structrisk", "TOTAL"]
    table_D8 = pd.DataFrame({
        f"XC3 (c={c_opt_xc3}mm)": cost_A_xc3.loc[c_opt_xc3, cols],
        f"XC4 (c={c_opt_xc4}mm)": cost_A_xc4.loc[c_opt_xc4, cols],
    }).round(6)
    print(table_D8)
