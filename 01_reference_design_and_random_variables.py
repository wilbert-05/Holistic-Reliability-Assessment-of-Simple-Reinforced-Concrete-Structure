"""
Reliability Assessment of an RC Beam
=====================================
Module 1: Deterministic baseline, random-variable library, and
environmental (carbonation) functions for exposure classes XC3/XC4.

Corresponds to Methodology Sections 3.2-3.4 and Results Section 4.1.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# =============================================================================
# 1. Deterministic baseline
# Inputs from the deterministic verification (Al-Mosawe et al. 2024)
# span L = 9 m, section b x h = 250 x 600 mm, 3 phi 20 tension bars,
# C30/37, B500, cover 30 mm, characteristic loads gk = 6 kN/m, qk = 6 kN/m
# =============================================================================

L = 9                 # clear span [m]
bar_diam = 20         # tension-bar diameter [mm]
n_bars = 3             # number of tension bars
fck = 30               # characteristic concrete strength [MPa] (C30/37)
fyk = 500              # characteristic steel yield strength [MPa] (B500)
cover = 30             # nominal concrete cover [mm] (design variable, kept fixed here)
Es = 200000            # steel modulus of elasticity [MPa]
gk = 6                 # characteristic permanent (total) UDL [kN/m]
qk = 6                 # characteristic variable (imposed) UDL [kN/m]

As_tot = n_bars * math.pi * bar_diam**2 / 4        # tension steel area [mm^2]
d_eff = 600 - cover - bar_diam / 2                  # effective depth [mm]

print(f"As_tot = {As_tot:.1f} mm^2, d = {d_eff:.0f} mm,"
      f" fctm(mean) = {0.3*fck**(2/3):.3f} MPa")

# =============================================================================
# 2. Random-variable library - (distribution, mean, CoV). std = mean * CoV
# =============================================================================

RV = {
    # -- geometry ----------------------------------------------------------
    "b":  ("normal", 250, 0.019),               # JCSS Pt.3 sec 3.10
    "h":  ("normal", 600, 0.013),                # JCSS Pt.3 sec 3.10
    "As": ("normal", As_tot, 0.02),              # JCSS Pt.3 sec 3.2.2
    # -- materials ---------------------------------------------------------
    "fc":     ("lognormal", 1.22 * fck, 0.15),    # Junior et al. (2023)
    "fy":     ("normal", 1.22 * fyk, 0.04),       # Junior et al. (2023)
    "fctm":   ("lognormal", 0.3 * fck**(2/3), 0.30),  # JCSS Pt.3 Table 3.1.1
    "Ec_eff": ("lognormal", Es / 15, 0.15),       # JCSS Pt.3 Table 3.1.1
    "Es":     ("normal", Es, 0.03),               # Stierschneider et al. (2025)
    # -- loads (lumped permanent G; imposed Q as 50-yr max) -----------------
    "G": ("normal", 1.06 * gk, 0.12),             # Junior et al. (2023)
    "Q": ("gumbel", 0.9 * qk, 0.24),              # Junior et al. (2023)
    # -- model uncertainties -------------------------------------------------
    "theta_R_bend":  ("lognormal", 1.2, 0.15),    # JCSS Pt.3 Table 3.9.1
    "theta_R_shear": ("lognormal", 1.00, 0.1),    # JCSS Pt.3 Table 3.9.1
    "theta_E":       ("lognormal", 1.00, 0.10),   # JCSS Pt.3 Table 3.9.1
    "theta_defl":    ("lognormal", 1.29, 0.24),   # Way et al. (2025), Table 4
    "theta_crack":   ("lognormal", 1.00, 0.30),   # Stierschneider et al. (2025)
    "theta_stress":  ("lognormal", 1.00, 0.10),   # JCSS Pt.3 Table 3.9.1
    # -- Durability - carbonation, fib Bulletin 34 Eq. B1.1-2 ----------------
    "RACC0":    ("normal", 2145.0, 0.45),          # Table B1.2, Fig B1.2-3
    "eps_t":    ("normal", 315.5, 0.152),          # Sec B1.2.5
    "thetacarb": ("lognormal", 1.0, 0.150),        # Vorechovska et al. (2010)
    "kt":       ("normal", 1.25, 0.280),           # Sec B1.2.5.4
    "Cs":       ("normal", 8.2e-4, 0.122),         # Sec B1.2.6.2
}

# quick look-up table
tab = pd.DataFrame([(k, d, m, c, m * c) for k, (d, m, c) in RV.items()],
                    columns=["variable", "distribution", "mean", "CoV", "std"])

# =============================================================================
# 3. Exposure-class environmental function k_e(RH) - fib Bulletin 34 Eq. B1.2-3
# k_e = ( (1-(RH_real/100)^f_e) / (1-(RH_ref/100)^f_e) )^g_e
# RH_ref = 65 %, f_e = 5.0, g_e = 2.5
#
# XC3 - moderate humidity, sheltered: RH_real=65% -> k_e=1.0, W(t)=1.0
# XC4 - cyclic wet-dry (Chen & Ho): two RH phases combined into ONE model
#   dry phase: RH=70% -> W(t)=1.0 ; wet phase: RH=90% -> W(t)=0.733
# =============================================================================

RH_REF, F_E, G_E = 65.0, 5.0, 2.5


def ke_func(RH_real, RH_ref=RH_REF, f_e=F_E, g_e=G_E):
    return ((1 - (RH_real / 100.0) ** f_e) / (1 - (RH_ref / 100.0) ** f_e)) ** g_e


KE_XC3 = ke_func(65.0)
KE_XC4_DRY = ke_func(70.0)
KE_XC4_WET = ke_func(90.0)

WT_XC3 = 1.0
WT_XC4_DRY = 1.0
WT_XC4_WET = 0.733

EXPOSURE = {
    "XC3":     {"ke_carb": KE_XC3, "Wt_carb": WT_XC3},
    "XC4_dry": {"ke_carb": KE_XC4_DRY, "Wt_carb": WT_XC4_DRY},
    "XC4_wet": {"ke_carb": KE_XC4_WET, "Wt_carb": WT_XC4_WET},
}

print("k_e (XC3, RH=65%) =", KE_XC3)
print("k_e (XC4 dry, RH=70%) =", KE_XC4_DRY)
print("k_e (XC4 wet, RH=90%) =", KE_XC4_WET)

BETA_TARGET_DUR = {"XC3": 0.5, "XC4": 1.5}

ICORR_BY_EXPOSURE = {
    "XC3": ("lognormal", 0.75, 0.70),
    "XC4": ("lognormal", 2.586, 0.667),
}

# =============================================================================
# 4. Transform helper: standard-normal u -> physical X
# =============================================================================


def x_from_u(u, dist, mean, std):
    """Transform a standard-normal value (or array) u into the physical X."""
    if dist == "normal":
        return mean + std * u
    if dist == "lognormal":  # real-space mean/std given
        s = np.sqrt(np.log1p((std / mean) ** 2))
        m = np.log(mean) - 0.5 * s * s
        return np.exp(m + s * u)
    if dist == "gumbel":  # Gumbel (largest)
        scale = std * np.sqrt(6) / np.pi
        loc = mean - 0.5772156649 * scale
        p = np.clip(norm.cdf(u), 1e-12, 1.0 - 1e-12)
        return loc - scale * np.log(-np.log(p))
    raise ValueError(f"unknown distribution: {dist}")


# =============================================================================
# 5. Random-variable sets per limit state (post sensitivity-screening).
# Geometry kept Random is b, h, As (bar_diam and L are fixed at their mean).
# =============================================================================

GEOM = ["b", "h", "As"]

RV_SETS = {
    "ULS bending": {
        **{k: RV[k] for k in GEOM}, "fy": RV["fy"], "fc": RV["fc"], "Es": RV["Es"],
        "G": RV["G"], "Q": RV["Q"],
        "theta_R": RV["theta_R_bend"], "theta_E": RV["theta_E"],
    },
    "ULS shear": {
        **{k: RV[k] for k in GEOM}, "fc": RV["fc"], "G": RV["G"], "Q": RV["Q"],
        "theta_R": RV["theta_R_shear"], "theta_E": RV["theta_E"],
    },
    "SLS deflection": {
        **{k: RV[k] for k in GEOM}, "fctm": RV["fctm"], "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"],
        "theta_defl": RV["theta_defl"],
    },
    "SLS crack": {
        **{k: RV[k] for k in GEOM}, "fctm": RV["fctm"], "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"],
        "theta_crack": RV["theta_crack"],
    },
    "SLS steel stress": {
        **{k: RV[k] for k in GEOM}, "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"],
        "theta_stress": RV["theta_stress"],
    },
    "SLS concrete stress": {
        **{k: RV[k] for k in GEOM}, "fc": RV["fc"], "Ec_eff": RV["Ec_eff"],
        "Es": RV["Es"], "G": RV["G"], "Q": RV["Q"],
        "theta_stress": RV["theta_stress"],
    },
    "Durability carbonation XC3": {
        k: RV[k] for k in ("RACC0", "eps_t", "thetacarb", "kt", "Cs")
    },
    "Durability carbonation XC4": {
        k: RV[k] for k in ("RACC0", "eps_t", "thetacarb", "kt", "Cs")
    },
}

# Nominal defaults; cover stays here only (never in an RV_SETS entry) -> always deterministic.
FIXED = {
    "cover": cover, "bar_diam": bar_diam, "L": L, "fyk": fyk, "fck": fck,
    "psi2": 0.30,               # quasi-permanent factor (EC2, Cat. B)
    "gamma_G": 1.0, "gamma_Q": 1.0,  # reliability analysis -> partial factors = 1
    "C_Rd_c": 0.18,             # EC2 shear constant, gamma_c removed
    "wk_limit": 0.30,           # crack-width limit [mm] (XC3, 0.3 mm)
    "T_life": 50.0,             # design service life [yr] (durability LS)
    "ke_carb": EXPOSURE["XC3"]["ke_carb"],
    "Wt_carb": EXPOSURE["XC3"]["Wt_carb"],
    "kc": 1.60,                 # curing factor (fib B34: kc=(t_c/7)^-0.567), 3-day cure
}


def fixed_for(exposure_key, base=FIXED):
    fx = dict(base)
    if exposure_key == "XC4":
        fx["ke_carb_dry"] = EXPOSURE["XC4_dry"]["ke_carb"]
        fx["Wt_carb_dry"] = EXPOSURE["XC4_dry"]["Wt_carb"]
        fx["ke_carb_wet"] = EXPOSURE["XC4_wet"]["ke_carb"]
        fx["Wt_carb_wet"] = EXPOSURE["XC4_wet"]["Wt_carb"]
    else:
        fx["ke_carb"] = EXPOSURE[exposure_key]["ke_carb"]
        fx["Wt_carb"] = EXPOSURE[exposure_key]["Wt_carb"]
    return fx
