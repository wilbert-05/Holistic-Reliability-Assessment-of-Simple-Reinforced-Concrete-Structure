"""
Reliability Assessment of an RC Beam
=====================================
Module 2: Limit-state functions g = R - E for the seven limit states
(six structural + one durability, evaluated separately for XC3/XC4).

Corresponds to Methodology Section 3.5 and Results Section 4.1
(mean-point verification table).

Requires: 01_reference_design_and_random_variables.py
"""

import numpy as np
from _01_reference_design_and_random_variables import (
    RV_SETS, FIXED, KE_XC4_DRY, KE_XC4_WET, WT_XC4_DRY, WT_XC4_WET,
)

# =============================================================================
# Load combinations (EC2):
#   ULS bending/shear        : design combination            G + Q
#   SLS deflection           : quasi-permanent combination    G + psi2*Q
#   SLS crack                : quasi-permanent combination    G + psi2*Q
#   SLS steel stress         : characteristic combination     G + Q
#   SLS concrete stress      : quasi-permanent combination    G + psi2*Q
# =============================================================================


def _d(X):
    return X["h"] - X["cover"] - X["bar_diam"] / 2


def _load_effects(X):
    g_perm = X["G"]
    w_uls = X["gamma_G"] * g_perm + X["gamma_Q"] * X["Q"]
    w_qp = g_perm + X["psi2"] * X["Q"]
    w_char = g_perm + X["Q"]
    return (w_uls * X["L"]**2 / 8,   # [0] M_Ed,ULS
            w_uls * X["L"] / 2,      # [1] V_Ed,ULS
            w_qp * X["L"]**2 / 8,    # [2] M_qp  (quasi-permanent)
            w_char * X["L"]**2 / 8)  # [3] M_char (characteristic)


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

    Lever arm from the neutral axis: z = d - x_c/2, with
    x_c = max(equilibrium depth, strain-compatibility (balanced) depth).
    Steel stress sigma_s = min(eps_cu*(d/x_c - 1)*Es, fyd) -> shows plasticity.
    Resistance M_R = As * sigma_s * z.

    Concrete-block / strain constants from Orcesi et al. (2024);
    alpha_cc = 1 (EC2-1-1 3.1.6, German NA), lambda = 0.8,
    eps_cu = 0.0035, eps_su = 0.025.
    """
    d = _d(X)
    alpha_cc = 1.0
    lam = 0.8
    eps_cu = 0.0035
    eps_su = 0.025
    fcd = alpha_cc * X["fc"]
    fyd = X["fy"]

    x_equil = X["As"] * fyd / (lam * fcd * X["b"])   # T = C -> As*fyd = lam*fcd*b*x
    x_bal = d * eps_cu / (eps_cu + eps_su)             # balanced (strain compatibility)
    x_c = np.maximum(x_equil, x_bal)

    sig_s = np.minimum(eps_cu * (d / x_c - 1.0) * X["Es"], fyd)

    z = d - x_c / 2.0
    M_Rd = X["As"] * sig_s * z / 1e6  # [kN.m]
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
    kt = 0.4
    eps = (sig_s - kt * (X["fctm"] / rho_p_eff) * (1 + ae * rho_p_eff)) / X["Es"]
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
    """Durability SLS - depassivation by carbonation (XC3, direct single-humidity model)."""
    xc = (X["thetacarb"]
          * np.sqrt(2 * X["ke_carb"] * X["kc"]
                    * (X["kt"] * X["RACC0"] + X["eps_t"])
                    * X["Cs"] * X["T_life"])
          * X["Wt_carb"])
    return X["cover"] - xc


def g_durability_xc4(X, duty_wet=0.5):
    """Durability SLS - depassivation by carbonation (XC4, two-phase RMS combination).

    Chen & Ho (2013) did not provide a general formula for combining the dry-
    and wet-phase carbonation coefficients at arbitrary duty cycles; they only
    measured one specific symmetric 70/90% test cycle. The root-mean-square
    combination used here extends their result to arbitrary duty cycles as an
    engineering assumption consistent with the sqrt(t) growth law already
    embedded in the fib Model Code equation, and is stated explicitly as such
    (see Methodology Section 3.3), not as a directly validated relation.
    """
    xc_dry = (X["thetacarb"]
              * np.sqrt(2 * KE_XC4_DRY * X["kc"]
                        * (X["kt"] * X["RACC0"] + X["eps_t"])
                        * X["Cs"] * X["T_life"])
              * WT_XC4_DRY)
    xc_wet = (X["thetacarb"]
              * np.sqrt(2 * KE_XC4_WET * X["kc"]
                        * (X["kt"] * X["RACC0"] + X["eps_t"])
                        * X["Cs"] * X["T_life"])
              * WT_XC4_WET)
    xc_eff = np.sqrt((1.0 - duty_wet) * xc_dry**2 + duty_wet * xc_wet**2)
    return X["cover"] - xc_eff


G_FUNCS = {
    "ULS bending": g_bending,
    "ULS shear": g_shear,
    "SLS deflection": g_deflection,
    "SLS crack": g_crack,
    "SLS steel stress": g_steel_stress,
    "SLS concrete stress": g_conc_stress,
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


if __name__ == "__main__":
    print("g at the mean point (all random variables = mean, partial factors = 1):")
    for nm, g in G_FUNCS.items():
        print(f"  {nm:<30}{g(mean_point(RV_SETS[nm], FIXED)):+12.4f}")
