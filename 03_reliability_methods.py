"""
Reliability Assessment of an RC Beam
=====================================
Module 3: The three reliability methods used throughout this study.

  FORM     - Hasofer-Lind / HL-RF iteration in standard-normal space
  MVFOSM   - Mean-Value First-Order Second-Moment
  Monte Carlo - crude sampling-based estimator

Corresponds to Methodology Section 3.6 and Results Section 4.3.1
(three-method comparison at the reference cover).
"""

import numpy as np
from scipy.stats import norm
from _01_reference_design_and_random_variables import x_from_u


def form(g_func, rv_set, fixed, max_iter=200, tol=1e-5, relax=0.7, step=1e-3):
    """First-Order Reliability Method via the HL-RF algorithm.

    Starting from the mean point (u=0), iterates the HL-RF update until the
    design point u* converges. beta = ||u*||, Pf = Phi(-beta), and the
    importance factors are alpha_i^2 (they sum to 1). The gradient is taken
    by central differences in u-space.
    """
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
        for i in range(n):  # central-difference gradient
            up, um = u.copy(), u.copy()
            up[i] += step
            um[i] -= step
            grad[i] = (gU(up) - gU(um)) / (2 * step)
        gn2 = grad @ grad
        u_hlrf = ((grad @ u - g) / gn2) * grad  # HL-RF update
        u_new = np.clip(u + relax * (u_hlrf - u), -20.0, 20.0)
        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            converged = True
            break
        u = u_new

    beta = np.linalg.norm(u)
    if g_at_mean < 0:  # mean already in failure domain
        beta = -beta
    alpha = grad / np.sqrt(grad @ grad)
    return {
        "beta": beta,
        "pf": float(norm.cdf(-beta)),
        "converged": converged,
        "iters": it,
        "importance": dict(zip(names, alpha**2)),
        "alpha": dict(zip(names, alpha)),
        "design_point": x_of_u(u),
    }


def mvfosm(g_func, rv_set, fixed):
    """Mean-Value First-Order Second-Moment.

    beta = mu_g / sigma_g, mu_g ~= g(means),
    sigma_g^2 ~= sum_i (dg/dX_i * sigma_Xi)^2

    Ignores distribution shapes, so it is least accurate when g is strongly
    non-linear or the variables are far from normal (see Results 4.3.1).
    """
    from _02_limit_state_functions import mean_point
    base = mean_point(rv_set, fixed)
    mu_g = g_func(base)
    var_g = 0.0
    for nm, (dist, mean, cov) in rv_set.items():
        sd = mean * cov
        step = 1e-3 * sd if sd > 0 else 1e-6
        xp = dict(base); xp[nm] = mean + step
        xm = dict(base); xm[nm] = mean - step
        dg = (g_func(xp) - g_func(xm)) / (2 * step)  # partial derivative at the mean
        var_g += (dg * sd) ** 2
    beta = mu_g / np.sqrt(var_g)
    return beta, float(norm.cdf(-beta))


def monte_carlo(g_func, rv_set, fixed, N=1_000_000, seed=12345):
    """Crude Monte Carlo: Pf = N_fail/N, beta = -Phi^-1(Pf).

    Needs roughly 10/Pf samples, so it cannot resolve the very small Pf of
    the ULS states - there FORM is the trustworthy estimate (see Results
    Section 4.2.2 for the sample-size convergence discussion).
    """
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


if __name__ == "__main__":
    from _01_reference_design_and_random_variables import RV_SETS, FIXED
    from _02_limit_state_functions import G_FUNCS

    print(f"{'Limit state':<28}{'beta_FORM':>12}{'beta_MC':>12}{'beta_MVFOSM':>14}")
    print("-" * 70)
    N_MC = 5_000_000
    for nm, g in G_FUNCS.items():
        res_form = form(g, RV_SETS[nm], FIXED)
        pf_mc, b_mc, nf = monte_carlo(g, RV_SETS[nm], FIXED, N=N_MC)
        b_mv, pf_mv = mvfosm(g, RV_SETS[nm], FIXED)
        print(f"{nm:<28}{res_form['beta']:>12.3f}{b_mc:>12.3f}{b_mv:>14.3f}")
