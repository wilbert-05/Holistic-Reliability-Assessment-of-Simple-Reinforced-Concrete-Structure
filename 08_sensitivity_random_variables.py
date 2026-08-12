"""
Reliability Assessment of an RC Beam
=====================================
Module 8: FORM-based random-variable importance-factor sensitivity analysis.

Runs FORM on every limit-state function to obtain the importance factors
alpha^2 for every candidate random variable, in order to decide, per limit
state, which variables must stay Random and which can be treated as Fixed
at their mean value.

Beam section. All inputs are taken from the deterministic verification in
Al-Mosawe et al. (2024): span L = 9 m, section b x h = 250 x 600 mm, 3 phi 20
tension bars (As = 942.5 mm^2), effective depth d ~ 560 mm, C30/37, B500,
cover 30 mm, characteristic loads gk = 6 kN/m (permanent) and qk = 6 kN/m
(variable). This is the same reference beam used throughout Modules 1-7 and
Module 9.

Decision rule: variables with alpha^2 >= 0.05 are considered influential and
retained as random; variables with alpha^2 < 0.01 contribute negligibly and
are fixed at their mean in the main reliability model (Methodology Section
3.7.1). Variables in between (0.01 <= alpha^2 < 0.05) are judged case-by-case.

The reduced random-variable sets resulting from this screening are the ones
hard-coded into RV_SETS in Module 1 (01_reference_design_and_random_variables.py).

Corresponds to Methodology Section 3.7.1 and Results Section 4.2.1.
"""

import math
import numpy as np
from scipy.stats import norm

# =============================================================================
# 1. Deterministic baseline (same reference beam as Modules 1-7 and 9)
# =============================================================================

L = 9                 # clear span [m]
bar_diam = 20          # tension-bar diameter [mm]
n_bars = 3             # number of tension bars
fck = 30               # characteristic concrete strength [MPa] (C30/37)
fyk = 500              # characteristic steel yield strength [MPa] (B500)
cover = 30             # nominal concrete cover [mm]
Es = 200000            # steel modulus of elasticity [MPa]
gk = 6                 # characteristic permanent (total) UDL [kN/m]
qk = 6                 # characteristic variable (imposed) UDL [kN/m]

As_tot = n_bars * math.pi * bar_diam**2 / 4        # tension steel area [mm^2]
d_eff = 600 - cover - bar_diam / 2                  # effective depth [mm]

print(f"As_tot = {As_tot:.1f} mm^2, d = {d_eff:.0f} mm,"
      f" fctm(mean) = {0.3*fck**(2/3):.3f} MPa")

# =============================================================================
# 2. Full candidate random-variable library (pre-screening).
# This is the SUPERSET of variables screened by this module -- note it
# includes bar_diam and L as candidate random variables, which the screening
# below determines can be fixed at their means (see Module 1's reduced sets).
# =============================================================================

RV = {
    "b": ("normal", 250, 0.019),
    "h": ("normal", 600, 0.013),
    "As": ("normal", As_tot, 0.02),
    "bar_diam": ("normal", bar_diam, 0.024),   # Adewuyi & Eric (2024)
    "L": ("normal", L, 0.006),                  # Orcesi et al. (2023)
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


RH_REF, F_E, G_E = 65.0, 5.0, 2.5


def ke_func(RH_real, RH_ref=RH_REF, f_e=F_E, g_e=G_E):
    return ((1 - (RH_real / 100.0) ** f_e) / (1 - (RH_ref / 100.0) ** f_e)) ** g_e


KE_XC3 = ke_func(65.0)
KE_XC4_DRY = ke_func(70.0)
KE_XC4_WET = ke_func(90.0)
WT_XC3, WT_XC4_DRY, WT_XC4_WET = 1.0, 1.0, 0.733

EXPOSURE = {
    "XC3": {"ke_carb": KE_XC3, "Wt_carb": WT_XC3},
    "XC4_dry": {"ke_carb": KE_XC4_DRY, "Wt_carb": WT_XC4_DRY},
    "XC4_wet": {"ke_carb": KE_XC4_WET, "Wt_carb": WT_XC4_WET},
}

BETA_TARGET_DUR = {"XC3": 0.5, "XC4": 1.5}
ICORR_BY_EXPOSURE = {"XC3": ("lognormal", 0.75, 0.70), "XC4": ("lognormal", 2.586, 0.667)}

print("k_e (XC3, RH=65%) =", KE_XC3)
print("k_e (XC4 dry, RH=70%) =", KE_XC4_DRY)
print("k_e (XC4 wet, RH=90%) =", KE_XC4_WET)

# =============================================================================
# 3. Full (unscreened) random-variable sets per limit state -- includes
# bar_diam and L, unlike Module 1's already-reduced RV_SETS.
# =============================================================================

GEOM = ["b", "h", "As", "bar_diam", "L"]

RV_SETS_FULL = {
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
    "Durability carbonation XC3": {
        k: RV[k] for k in ("RACC0", "eps_t", "thetacarb", "kt", "Cs")
    },
    "Durability carbonation XC4": {
        k: RV[k] for k in ("RACC0", "eps_t", "thetacarb", "kt", "Cs")
    },
}

FIXED = {
    "cover": cover, "bar_diam": bar_diam, "L": L, "fyk": fyk, "fck": fck,
    "psi2": 0.30, "gamma_G": 1.0, "gamma_Q": 1.0, "C_Rd_c": 0.18,
    "wk_limit": 0.30, "T_life": 50.0,
    "ke_carb": EXPOSURE["XC3"]["ke_carb"], "Wt_carb": EXPOSURE["XC3"]["Wt_carb"],
    "kc": 1.60,
}

# --- Limit-state functions (identical to Module 2) --------------------------


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
    """ULS bending: g = theta_R*M_Rd - theta_E*M_Ed.

    Lever arm from the neutral axis: z = d - x_c/2,
    with x_c = max(equilibrium depth, strain-compatibility (balanced) depth).
    Steel stress sigma_s = min(eps_cu*(d/x_c - 1)*Es, fyd) -> shows plasticity.
    Resistance M_R = As * sigma_s * z.

    Concrete-block / strain constants from Orcesi et al. (2024); alpha_cc = 1
    (EC2-1-1 3.1.6, German NA), lambda = 0.8, eps_cu = 0.0035, eps_su = 0.025.
    """
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
    """ULS shear (no stirrups): g = theta_R*V_Rd,c - theta_E*V_Ed."""
    d = _d(X)
    k = np.minimum(1 + np.sqrt(200 / d), 2)
    rho_l = np.minimum(X["As"] / (X["b"] * d), 0.02)
    V_Rd_c = X["C_Rd_c"] * k * (100 * rho_l * X["fc"]) ** (1 / 3) * X["b"] * d / 1000
    V_Ed = _load_effects(X)[1]
    return X["theta_R"] * V_Rd_c - X["theta_E"] * V_Ed


def g_deflection(X):
    """SLS deflection: g = delta_limit - theta_defl*delta (quasi-permanent)."""
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
    """SLS crack width (EC2 Eq. 7.11): g = wk_limit - theta_crack*wk (quasi-permanent)."""
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
    """SLS steel stress (EC2 7.2(5)): g = 0.8*fyk - theta_stress*sigma_s (characteristic)."""
    d = _d(X)
    ae = X["Es"] / X["Ec_eff"]
    y_II, I_cr = _state_II(X, d)
    M_char = _load_effects(X)[3]
    sig_s = ae * M_char * 1e6 * (d - y_II) / I_cr
    return 0.8 * X["fyk"] - X["theta_stress"] * sig_s


def g_conc_stress(X):
    """SLS concrete stress (EC2 7.2(2)): g = 0.45*fck - theta_stress*sigma_c (quasi-permanent)."""
    d = _d(X)
    y_II, I_cr = _state_II(X, d)
    M_qp = _load_effects(X)[2]
    sig_c = M_qp * 1e6 * y_II / I_cr
    return 0.45 * X["fck"] - X["theta_stress"] * sig_c


def g_durability_XC3(X):
    """Durability SLS - depassivation by carbonation (XC3 direct)."""
    xc = (X["thetacarb"] * np.sqrt(2 * X["ke_carb"] * X["kc"]
          * (X["kt"] * X["RACC0"] + X["eps_t"]) * X["Cs"] * X["T_life"]) * X["Wt_carb"])
    return X["cover"] - xc


def g_durability_xc4(X, duty_wet=0.5):
    """Durability SLS - depassivation by carbonation (XC4, two-phase RMS combination).

    Chen & Ho (2013) did not provide a general formula for combining the dry-
    and wet-phase carbonation coefficients at arbitrary duty cycles; they only
    measured one specific symmetric 70/90% test cycle. The root-mean-square
    combination used here extends their result to arbitrary duty cycles as an
    engineering assumption consistent with the sqrt(t) growth law already
    embedded in the fib Model Code equation, stated explicitly as such in the
    Methodology, not as a directly validated result from the paper.
    """
    xc_dry = (X["thetacarb"] * np.sqrt(2 * KE_XC4_DRY * X["kc"]
              * (X["kt"] * X["RACC0"] + X["eps_t"]) * X["Cs"] * X["T_life"]) * WT_XC4_DRY)
    xc_wet = (X["thetacarb"] * np.sqrt(2 * KE_XC4_WET * X["kc"]
              * (X["kt"] * X["RACC0"] + X["eps_t"]) * X["Cs"] * X["T_life"]) * WT_XC4_WET)
    xc_eff = np.sqrt((1.0 - duty_wet) * xc_dry**2 + duty_wet * xc_wet**2)
    return X["cover"] - xc_eff


G_FUNCS = {
    "ULS bending": g_bending, "ULS shear": g_shear,
    "SLS deflection": g_deflection, "SLS crack": g_crack,
    "SLS steel stress": g_steel_stress, "SLS concrete stress": g_conc_stress,
    "Durability carbonation XC3": g_durability_XC3,
    "Durability carbonation XC4": g_durability_xc4,
}

G_DUR_BY_EXPOSURE = {"XC3": g_durability_XC3, "XC4": g_durability_xc4}


def mean_point(rv_set, fixed):
    """X with every random variable at its mean, plus the fixed parameters."""
    X = dict(fixed)
    for nm, (dist, mean, cov) in rv_set.items():
        X[nm] = mean
    return X


print("\ng at the mean point (all random variables = mean, partial factors = 1):")
for nm, g in G_FUNCS.items():
    print(f"  {nm:<30}{g(mean_point(RV_SETS_FULL[nm], FIXED)):+12.4f}")

# =============================================================================
# FORM (HL-RF algorithm)
# =============================================================================


def form(g_func, rv_set, fixed, max_iter=200, tol=1e-5, relax=0.7, step=1e-3):
    names = list(rv_set.keys())
    n = len(names)

    def x_of_u(u):
        X = dict(fixed)
        for i, nm in enumerate(names):
            dist, mean, cov = rv_set[nm]
            X[nm] = x_from_u(u[i], dist, mean, mean * cov)
        return X

    def gU(u):
        return float(g_func(x_of_u(u)))

    u = np.zeros(n)
    g_at_mean = gU(u)
    grad = np.zeros(n)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = gU(u)
        for i in range(n):  # central-difference gradient -- tells us which
            up, um = u.copy(), u.copy()  # variables matter most
            up[i] += step
            um[i] -= step
            grad[i] = (gU(up) - gU(um)) / (2 * step)
        gn2 = grad @ grad
        u_hlrf = ((grad @ u - g) / gn2) * grad  # HL-RF update -- finding the
        u_new = np.clip(u + relax * (u_hlrf - u), -20.0, 20.0)  # closest point
        if np.linalg.norm(u_new - u) < tol:                     # on the failure surface
            u = u_new
            converged = True
            break
        u = u_new

    beta = np.linalg.norm(u)
    if g_at_mean < 0:  # mean already in failure domain
        beta = -beta
    alpha = grad / np.sqrt(grad @ grad)
    return {
        "beta": beta, "pf": float(norm.cdf(-beta)), "converged": converged,
        "iters": it, "importance": dict(zip(names, alpha**2)),
        "alpha": dict(zip(names, alpha)), "design_point": x_of_u(u),
    }


if __name__ == "__main__":
    results = {}
    print(f"\n{'Limit state':<16}{'beta':>9}{'Pf':>13}{'converged':>11}{'iters':>7}")
    print("-" * 56)
    for nm, g in G_FUNCS.items():
        res = form(g, RV_SETS_FULL[nm], FIXED)
        results[nm] = res
        print(f"{nm:<16}{res['beta']:>9.3f}{res['pf']:>13.2e}"
              f"{str(res['converged']):>11}{res['iters']:>7}")

    print("\n\nImportance factors (alpha^2, sorted high -> low, sum = 1)")
    print("=" * 56)
    for nm, res in results.items():
        print(f"\n{nm} (beta = {res['beta']:.2f})")
        for v, imp in sorted(res["importance"].items(), key=lambda kv: -kv[1]):
            bar = "#" * int(round(imp * 40))
            flag = (" <- dominant" if imp >= 0.05
                    else " (negligible)" if imp < 0.01 else "")
            print(f"  {v:<12}{imp:7.3f} {bar}{flag}")
