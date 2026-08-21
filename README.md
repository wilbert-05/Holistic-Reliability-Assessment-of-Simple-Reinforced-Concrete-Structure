# Holistic Reliability Assessment of a Simple Reinforced Concrete Structure

Computational implementation of the reliability assessment and lifecycle cost optimization framework described in the dissertation *"Holistic Reliability Assessment of Simple Reinforced Concrete Structure"* (Wilbert Jonathan Yaury, NORISK Erasmus Mundus Joint Master, 2026).

This repository contains the full Python implementation used to generate every result presented in **Chapter 4 (Results and Discussion)**, following the methodology described in **Chapter 3**.

## Overview

The framework assesses the structural reliability of a simply supported reinforced concrete beam against seven limit states (two ULS, four SLS, and one exposure-differentiated durability limit state), using three complementary reliability methods (FORM, MVFOSM, and Monte Carlo Simulation), and then optimizes the concrete cover depth against a full lifecycle cost objective for two exposure classes, XC3 (moderate humidity, sheltered) and XC4 (cyclic wet-dry).

**Reference beam (used consistently across all modules):** simply supported RC beam, clear span L = 9 m, cross-section b x h = 250 x 600 mm, 3 tension bars of 20 mm diameter (As = 942.5 mm^2), effective depth d ≈ 560 mm, C30/37 concrete, B500 steel, cover 30 mm, characteristic dead load gk = 6 kN/m, characteristic live load qk = 6 kN/m (following the deterministic design reported by Al-Mosawe et al., 2024, with adjustments).

## Repository Structure

| File | Contents | Dissertation section |
|---|---|---|
| `01_reference_design_and_random_variables.py` | Deterministic reference beam, reduced (post-screening) random-variable library (`RV`), exposure-class carbonation functions (`k_e`, `W(t)`), standard-normal transform helper | Methodology 3.2, 3.3, 3.4 |
| `02_limit_state_functions.py` | The eight limit-state functions `g = R - S` (bending, shear, deflection, crack width, steel stress, concrete stress, durability XC3, durability XC4) | Methodology 3.5 |
| `03_reliability_methods.py` | FORM (HL-RF algorithm), MVFOSM, and crude Monte Carlo simulation | Methodology 3.6 |
| `04_cover_sweep_and_feasible_window.py` | Cover sweep 10-90 mm, feasible-window evaluation per exposure class, time-dependent durability / predicted service life | Methodology 3.8, Results 4.3 |
| `05_lifecycle_cost_optimization.py` | Build cost, corrosion propagation with precomputed structural response surfaces, lifecycle Monte Carlo maintenance simulation (three-tier repair rule), structural risk, exposure severity multiplier, cost-optimization driver | Methodology 3.9, Results 4.4 |
| `06_cost_sensitivity_analysis.py` | Three one-at-a-time sensitivity sweeps: discount rate (1-10%), severity exponent (0.1-1.0), and inspection interval (1-25 years) | Results 4.4.3 |
| `07_tables_and_figures_export.py` | Regenerates every table (CSV) and figure (PNG) reported in Chapter 4 | Results, Chapter 4 (all) |
| `08_sensitivity_random_variables.py` | FORM importance-factor (alpha^2) sensitivity analysis used to screen the full ~23-variable candidate library down to the reduced random-variable sets used in Module 1 | Methodology 3.7.1, Results 4.2.1 |
| `09_sensitivity_mcs_sample_size.py` | Monte Carlo sample-size convergence study (N = 1M-15M), used to justify N = 5,000,000 for the headline reliability cross-checks | Methodology 3.7.2, Results 4.2.2 |

Each module can be run standalone (`python 0N_module_name.py`) to reproduce its individual printed output, or imported into a notebook/script for interactive use.

**Recommended run order for full reproducibility:** Modules 8 and 9 (sensitivity analyses) logically precede Modules 1-7, since they determine the reduced random-variable sets and the Monte Carlo sample size that Modules 1-7 take as given. However, Modules 8 and 9 are self-contained (each carries its own full candidate variable library and duplicated limit-state functions) and can be run independently at any time to verify the screening decisions. Module 06 depends directly on Modules 04 and 05 (it imports them and reuses `run_cost_optimization` and `FEASIBLE_WINDOW`), so both must be present and runnable in the same folder before Module 06 is run.

## Setup

```bash
pip install numpy pandas scipy matplotlib
```

Requires Python 3.9+.

## Running the Full Pipeline

```bash
# Sensitivity analyses (self-contained; establish the screening used below)
python 08_sensitivity_random_variables.py     # ~1 min
python 09_sensitivity_mcs_sample_size.py       # ~5-10 min (350 MC runs)

# Main reliability + cost framework
python 01_reference_design_and_random_variables.py
python 02_limit_state_functions.py
python 03_reliability_methods.py
python 04_cover_sweep_and_feasible_window.py
python 05_lifecycle_cost_optimization.py       # computationally expensive: ~5-10 min
python 06_cost_sensitivity_analysis.py         # very expensive: ~30-50 min (26 re-runs of Module 5)
python 07_tables_and_figures_export.py         # regenerates all Chapter 4 tables/figures
```

All CSV tables and PNG figures are written to `output/`.

**Note on runtime:** Module 5's lifecycle cost optimization runs `N=6,000` Monte Carlo trajectories at every cover value in the feasible window, for both exposure classes, with structural response surfaces precomputed via FORM on a 12x7 grid per (exposure, cover) pair. Module 6 repeats this entire process ~26 times (10 discount-rate values + 10 severity-exponent values + 7 inspection-interval values, minus the shared baseline), so expect a runtime of 30-50 minutes total depending on hardware. Module 9's convergence study runs 350 independent Monte Carlo evaluations (7 limit states x 5 sample sizes x 10 replications), with the largest sample size at 15,000,000 draws per run.

## Key Results Reproduced

- **Table 4.2/4.3** (Module 9): MCS half-width convergence: five of seven limit states converge comfortably (hw <= 0.05 already at N=1M); ULS bending and SLS concrete stress do not converge even at N=15M, confirming FORM as the appropriate main engine for the full cover sweep.
- **Table 4.4** (Module 3/7): method comparison at reference cover: FORM and MCS agree within Δβ < 0.13 for every limit state; MVFOSM diverges by up to 1.9 for the more non-linear limit states (SLS concrete stress, SLS crack).
- **Table 4.6** (Module 4): feasible cover windows: **23-39 mm (XC3)**, **20-39 mm (XC4)**, both bounded above by ULS shear.
- **Table 4.9** (Module 5/7): cost-optimal cover: **34 mm (XC3)**, **26 mm (XC4)**. The Eurocode 2 nominal cover for XC3 (35 mm) sits inside the feasible window and near-optimal; the Eurocode 2 nominal cover for XC4 (40 mm) falls *outside* the feasible window entirely.
- **Module 6**: cost sensitivity: the optimal cover and minimum lifecycle cost are re-evaluated one-at-a-time against discount rate, severity exponent, and inspection interval, isolating the effect of each calibration input while holding the rest of the framework fixed. The severity-exponent sweep confirms by construction that XC3's optimum is invariant, and the power = 0.4 row reproduces the Module 5 headline result exactly as a consistency check.
- **Module 8**: FORM importance-factor screening: confirms the structural limit states and the durability limit state are governed by two almost non-overlapping groups of random variables (Results Section 4.2.1), with bar diameter and span length negligible (α² < 0.01) across every correlated limit state.

## Reproducibility Notes on Supplementary Analyses

Two supplementary analyses referenced in Chapter 4 are intentionally **not** packaged as standalone repository modules, since they apply the exact same reliability and lifecycle-cost framework as Modules 01-07 with only the input data changed, rather than introducing a new methodology:

- **City-specific XC4 case study** (Paris, Toulouse, Marseille): replaces the environmental inputs `W(t)`, `k_e`, RH, and time-of-wetness with city-specific climate-derived values, while keeping the reliability model, cost formulation, and optimization routine identical to Module 05. Supporting scripts and the climate-data workbook are retained by the author and available on request.
- **Daily k_e uncertainty investigation**: an exploratory extension examining the effect of treating `k_e` as a daily-varying, distribution-fitted quantity rather than a single deterministic value per exposure class. Retained by the author as supplementary material.

## Citation

If referencing this code, please cite the accompanying dissertation:

> Yaury, W. J. (2026). *Holistic Reliability Assessment of Simple Reinforced Concrete Structure*. MSc Dissertation, International Masters in Risk Assessment and Management of Civil Infrastructures (NORISK), Université La Rochelle & Université Gustave Eiffel.

## Contact

[wilbert.yaury@etudiant.univ-lr.fr](mailto:wilbert.yaury@etudiant.univ-lr.fr)
