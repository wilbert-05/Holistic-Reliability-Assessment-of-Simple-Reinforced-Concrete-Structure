"""
Reliability Assessment of an RC Beam
=====================================
Module 9: Monte Carlo sample-size convergence study.

Sweeps the Monte Carlo sample size N from 1,000,000 to 15,000,000, running
10 independent replications (distinct random seeds) at each N, to quantify
how the scatter of the estimated reliability index beta shrinks as N grows,
for all seven limit states (six structural + one durability, evaluated
under XC3 parameters as representative of both exposure classes -- see
Methodology Section 3.7.2).

The half-width of the 95% confidence interval of beta (1.96 * std across
replications) is used as the convergence criterion, with a target threshold
of hw <= 0.05. The minimum N at which all limit states pass this threshold
is adopted as the sample size for the main Monte Carlo reliability model
(N = 5,000,000 in the headline results, Module 3/Results Section 4.2.2).

Corresponds to Methodology Section 3.7.2 and Results Section 4.2.2
(Tables 4.2/4.3, Figures 4.2/4.3).
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# =============================================================================
# 1. Deterministic baseline (same reference beam as Module 1)
# =============================================================================

L = 9
bar_diam = 20
n_bars = 3
fck = 30
fyk = 500
cover = 30
Es = 200000
gk = 6
qk = 6

As_tot = n_bars * math.pi * bar_diam**2 / 4
d_eff = 600 - cover - bar_diam / 2

# =============================================================================
# 2. Random-variable library (identical values to Module 1)
# =============================================================================

RV = {
    "b": ("normal", 250, 0.019),
    "h": ("normal", 600, 0.013),
    "As": ("normal", As_tot, 0.02),
    "fc": ("lognormal", 1.22 * fck, 0.15),
    "fy": ("normal", 1.22 * fyk, 0.04),
    "fctm": ("lognormal", 0.3 * fck**(2/3), 0.30),
    "Ec_eff": ("lognormal", Es / 15, 0.15),
    "Es": ("normal", Es, 0.03),
    "G": ("normal", 1.06 * gk, 0.12),
    "Q": ("gumbel", 0.9 * qk, 0.24),
    "theta_R_bend": ("lognormal", 1.2, 0.15),
    "theta_R_shear": ("lognormal", 1.00, 0.1),
    "theta_E": ("lognormal", 1.00, 0.10),
    "theta_defl": ("lognormal", 1.29, 0.24),
    "theta_crack": ("lognormal", 1.00, 0.30),
    "theta_stress": ("lognormal", 1.00, 0.10),
    "RACC0": ("normal", 2145.0, 0.45),
    "eps_t": ("normal", 315.5, 0.152),
    "thetacarb": ("lognormal", 1.0, 0.150),
    "kt": ("normal", 1.25, 0.280),
    "Cs": ("normal", 8.2e-4, 0.122),
}


def x_from_u(u, dist, mean, std):
    if dist == "normal":
        return mean + std * u
    if dist == "lognormal":
        s = np.sqrt(np.log1p((std / mean) ** 2))
        m = np.log(mean) - 0.5 * s * s
        return np.exp(m + s * u)
    if dist == "gumbel":
        scale = std * np.sqrt(6) / np.pi
        loc = mean - 0.5772156649 * scale
        p = np.clip(norm.cdf(u), 1e-12, 1.0 - 1e-12)
        return loc - scale * np.log(-np.log(p))
    raise ValueError(f"unknown distribution: {dist}")


# =============================================================================
# 3. Random-variable sets per limit state (post FORM-screening; geometry kept
# Random is b, h, As -- bar_diam and L are fixed, per Module 8's decision rule).
# Durability here is evaluated once under XC3 parameters, since Module 8's
# screening showed both exposure classes share the same governing variables
# up to the carbonation coefficients (Results Section 4.2.2).
# =============================================================================

GEOM = ["b", "h", "As"]

RV_SETS = {
    "ULS bending": {
        **{k: RV[k] for k in GEOM}, "fy": RV["fy"], "fc": RV["fc"], "Es": RV["Es"],
        "G": RV["G"], "Q": RV["Q"], "theta_R": RV["theta_R_bend"], "theta_E": RV["theta_E"],
    },
    "ULS shear": {
        **{k: RV[k] for k in GEOM}, "fc": RV["fc"], "G": RV["G"], "Q": RV["Q"],
        "theta_R": RV["theta_R_shear"], "theta_E": RV["theta_E"],
    },
    "SLS deflection": {
        **{k: RV[k] for k in GEOM}, "fctm": RV["fctm"], "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"], "theta_defl": RV["theta_defl"],
    },
    "SLS crack": {
        **{k: RV[k] for k in GEOM}, "fctm": RV["fctm"], "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"], "theta_crack": RV["theta_crack"],
    },
    "SLS steel stress": {
        **{k: RV[k] for k in GEOM}, "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"], "theta_stress": RV["theta_stress"],
    },
    "SLS concrete stress": {
        **{k: RV[k] for k in GEOM}, "fc": RV["fc"], "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"], "theta_stress": RV["theta_stress"],
    },
    "Durability carbonation": {
        k: RV[k] for k in ("RACC0", "eps_t", "thetacarb", "kt", "Cs")
    },
}

FIXED = {
    "cover": cover, "bar_diam": bar_diam, "L": L, "fyk": fyk, "fck": fck,
    "psi2": 0.30, "gamma_G": 1.0, "gamma_Q": 1.0, "C_Rd_c": 0.18,
    "wk_limit": 0.30, "T_life": 50.0, "ke_carb": 1.0, "Wt_carb": 1.0, "kc": 1.60,
}

# --- Limit-state functions (identical to Module 2, durability = XC3 form) ---


def _d(X):
    return X["h"] - X["cover"] - X["bar_diam"] / 2


def _load_effects(X):
    g_perm = X["G"]
    w_uls = X["gamma_G"] * g_perm + X["gamma_Q"] * X["Q"]
    w_qp = g_perm + X["psi2"] * X["Q"]
    w_char = g_perm + X["Q"]
    return (w_uls * X["L"]**2 / 8, w_uls * X["L"] / 2,
            w_qp * X["L"]**2 / 8, w_char * X["L"]**2 / 8)


def _state_I(X, d):
    ae = X["Es"] / X["Ec_eff"]
    A_I = X["b"] * X["h"] + (ae - 1) * X["As"]
    y_I = (X["b"] * X["h"] * X["h"] / 2 + (ae - 1) * X["As"] * d) / A_I
    I_I = (X["b"] * X["h"]**3 / 12 + X["b"] * X["h"] * (X["h"] / 2 - y_I)**2
           + (ae - 1) * X["As"] * (d - y_I)**2)
    return y_I, I_I


def _state_II(X, d):
    ae = X["Es"] / X["Ec_eff"]
    P, Q = X["b"] / 2, ae * X["As"]
    R = -ae * X["As"] * d
    y_II = (-Q + np.sqrt(Q**2 - 4 * P * R)) / (2 * P)
    I_cr = X["b"] * y_II**3 / 3 + ae * X["As"] * (d - y_II)**2
    return y_II, I_cr


def g_bending(X):
    d = _d(X)
    alpha_cc, lam, eps_cu, eps_su = 1.0, 0.8, 0.0035, 0.025
    fcd, fyd = alpha_cc * X["fc"], X["fy"]
    x_equil = X["As"] * fyd / (lam * fcd * X["b"])
    x_bal = d * eps_cu / (eps_cu + eps_su)
    x_c = np.maximum(x_equil, x_bal)
    sig_s = np.minimum(eps_cu * (d / x_c - 1.0) * X["Es"], fyd)
    z = d - x_c / 2.0
    M_Rd = X["As"] * sig_s * z / 1e6
    M_Ed = _load_effects(X)[0]
    return X["theta_R"] * M_Rd - X["theta_E"] * M_Ed


def g_shear(X):
    d = _d(X)
    k = np.minimum(1 + np.sqrt(200 / d), 2)
    rho_l = np.minimum(X["As"] / (X["b"] * d), 0.02)
    V_Rd_c = X["C_Rd_c"] * k * (100 * rho_l * X["fc"]) ** (1 / 3) * X["b"] * d / 1000
    V_Ed = _load_effects(X)[1]
    return X["theta_R"] * V_Rd_c - X["theta_E"] * V_Ed


def g_deflection(X):
    d = _d(X)
    y_I, I_I = _state_I(X, d)
    M_cr = X["fctm"] * I_I / (X["h"] - y_I) / 1e6
    y_II, I_cr = _state_II(X, d)
    M_Ed = _load_effects(X)[2]
    zeta = np.maximum(1 - 0.5 * (M_cr / M_Ed) ** 2, 0.0)
    k_I = M_Ed * 1e6 / (X["Ec_eff"] * I_I)
    k_II = M_Ed * 1e6 / (X["Ec_eff"] * I_cr)
    k_eff = zeta * k_II + (1 - zeta) * k_I
    delta = 5.0 / 48.0 * k_eff * (X["L"] * 1000) ** 2
    delta_lim = X["L"] * 1000 / 250
    return delta_lim - X["theta_defl"] * delta


def g_crack(X):
    d = _d(X)
    ae = X["Es"] / X["Ec_eff"]
    y_II, I_cr = _state_II(X, d)
    M_Ed = _load_effects(X)[2]
    sig_s = ae * M_Ed * 1e6 * (d - y_II) / I_cr
    hc_eff = np.minimum(np.minimum(2.5 * (X["h"] - d), (X["h"] - y_II) / 3.0), X["h"] / 2.0)
    rho_p_eff = X["As"] / (hc_eff * X["b"])
    kt_ = 0.4
    eps = (sig_s - kt_ * (X["fctm"] / rho_p_eff) * (1 + ae * rho_p_eff)) / X["Es"]
    eps = np.maximum(eps, 0.6 * sig_s / X["Es"])
    k1, k2, k3, k4 = 0.8, 0.5, 3.4, 0.425
    sr_max = k3 * X["cover"] + k1 * k2 * k4 * X["bar_diam"] / rho_p_eff
    wk = sr_max * eps
    return X["wk_limit"] - X["theta_crack"] * wk


def g_steel_stress(X):
    d = _d(X)
    ae = X["Es"] / X["Ec_eff"]
    y_II, I_cr = _state_II(X, d)
    M_char = _load_effects(X)[3]
    sig_s = ae * M_char * 1e6 * (d - y_II) / I_cr
    return 0.8 * X["fyk"] - X["theta_stress"] * sig_s


def g_conc_stress(X):
    d = _d(X)
    y_II, I_cr = _state_II(X, d)
    M_qp = _load_effects(X)[2]
    sig_c = M_qp * 1e6 * y_II / I_cr
    return 0.45 * X["fck"] - X["theta_stress"] * sig_c


def g_durability(X):
    """Durability SLS - depassivation by carbonation (fib Bulletin 34, Eq. B1.1-2)."""
    xc = (X["thetacarb"] * np.sqrt(2 * X["ke_carb"] * X["kc"]
          * (X["kt"] * X["RACC0"] + X["eps_t"]) * X["Cs"] * X["T_life"]) * X["Wt_carb"])
    return X["cover"] - xc


G_FUNCS = {
    "ULS bending": g_bending, "ULS shear": g_shear,
    "SLS deflection": g_deflection, "SLS crack": g_crack,
    "SLS steel stress": g_steel_stress, "SLS concrete stress": g_conc_stress,
    "Durability carbonation": g_durability,
}

# =============================================================================
# 4. Crude Monte Carlo estimator
# =============================================================================


def monte_carlo(g_func, rv_set, fixed, N=1_000_000, seed=12345):
    rng = np.random.default_rng(seed)
    X = dict(fixed)
    for nm, (dist, mean, cov) in rv_set.items():
        u = rng.standard_normal(N)
        X[nm] = x_from_u(u, dist, mean, mean * cov)
    g = g_func(X)
    nf = int(np.sum(g <= 0))
    pf = nf / N
    if pf == 0:
        beta = np.inf
    elif pf >= 1:
        beta = -np.inf
    else:
        beta = -norm.ppf(pf)
    return pf, beta, nf


# =============================================================================
# 5. Sample-size study configuration.
# Sweep N over five points; 10 independent replications per N to quantify
# the scatter of beta (not just its mean).
# =============================================================================

N_GRID = [1_000_000, 2_000_000, 5_000_000, 7_000_000, 15_000_000]
N_REPEAT = 10
BASE_SEED = 2025
STUDY_STATES = list(G_FUNCS.keys())
TARGET_HALFWIDTH = 0.05  # desired 95% half-width on beta


def run_sample_size_study():
    rows = []
    for nm in STUDY_STATES:
        g = G_FUNCS[nm]
        for N in N_GRID:
            for r in range(N_REPEAT):
                pf, beta, nf = monte_carlo(g, RV_SETS[nm], FIXED, N=N, seed=BASE_SEED + r)
                rows.append({"limit state": nm, "N": N, "rep": r,
                             "pf": pf, "beta": beta, "n_fail": nf})
    return pd.DataFrame(rows)


def summarise_group(group):
    beta = group["beta"].to_numpy()
    finite = np.isfinite(beta)
    bf = beta[finite]
    n_ok = bf.size
    mean = bf.mean() if n_ok else np.nan
    std = bf.std(ddof=1) if n_ok > 1 else np.nan
    sem = std / np.sqrt(n_ok) if n_ok > 1 else np.nan
    return pd.Series({
        "beta_mean": mean, "beta_std": std, "beta_sem": sem,
        "ci95_lo": mean - 1.96 * std if n_ok > 1 else np.nan,
        "ci95_hi": mean + 1.96 * std if n_ok > 1 else np.nan,
        "pf_mean": group["pf"].mean(), "n_inf": int((~finite).sum()),
    })


def fit_convergence_slopes(summary):
    """Log-log fit of beta_std vs N; theoretical slope under 1/sqrt(N) scaling is -0.5."""
    slopes = {}
    for nm in STUDY_STATES:
        sub = summary[summary["limit state"] == nm].copy()
        sub = sub[sub["beta_std"].notna() & (sub["beta_std"] > 0)]
        if len(sub) < 2:
            slopes[nm] = np.nan
            continue
        logN = np.log(sub["N"].to_numpy(dtype=float))
        logS = np.log(sub["beta_std"].to_numpy(dtype=float))
        slope, _ = np.polyfit(logN, logS, 1)
        slopes[nm] = slope
    return slopes


def recommend_sample_size(summary):
    """Minimum N at which the half-width of the 95% CI first drops below the target."""
    rec_rows = []
    for nm in STUDY_STATES:
        sub = summary[summary["limit state"] == nm].copy()
        sub["hw"] = 1.96 * sub["beta_std"]
        ok = sub[sub["hw"] <= TARGET_HALFWIDTH]
        if ok.empty:
            n_rec, hw_rec = f">{max(N_GRID):,}", float("nan")
        else:
            n_rec, hw_rec = int(ok.iloc[0]["N"]), ok.iloc[0]["hw"]
        rec_rows.append({"limit state": nm, f"min N (hw<={TARGET_HALFWIDTH})": n_rec,
                          "hw @ min N": round(hw_rec, 4) if not pd.isna(hw_rec) else "-"})
    return pd.DataFrame(rec_rows)


if __name__ == "__main__":
    print(f"Sample sizes: {', '.join(f'{n:,}' for n in N_GRID)}")
    print(f"Replications: {N_REPEAT} per sample size")
    print(f"Study states: {', '.join(STUDY_STATES)}")
    print(f"Total MC runs: {len(N_GRID) * N_REPEAT * len(STUDY_STATES)}\n")

    mc_runs = run_sample_size_study()
    print(f"Collected {len(mc_runs)} Monte Carlo runs.\n")

    summary = (mc_runs.groupby(["limit state", "N"], sort=False)
               .apply(summarise_group, include_groups=False)
               .reset_index())
    print(summary.round(5))

    print("\nDoes the scatter shrink like 1/sqrt(N)? (theoretical slope: -0.5)")
    print("-" * 60)
    slopes = fit_convergence_slopes(summary)
    for nm, slope in slopes.items():
        print(f"{nm:<26}{slope:.3f}" if not np.isnan(slope) else f"{nm:<26}N/A")

    print("\nRecommended sample size (target half-width <= 0.05):")
    print(recommend_sample_size(summary))
