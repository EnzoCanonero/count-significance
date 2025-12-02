import math
import numpy as np
from scipy.stats import chi2, poisson, norm

# ------------------ Core utilities for simple counting model ------------------


def poisson_tail(s0: float, b: float, n: int) -> float:
    """Mid-p upward tail for N~Pois(s0+b)."""
    mu0 = s0 + b
    tail = chi2.cdf(2.0 * mu0, df=2 * n)
    return tail


def r_stat(s0: float, b: float, n: float) -> float:
    """
    Signed root likelihood ratio r(s0) for testing s = s0
    in the model n ~ Poisson(s + b), with b known.
    """
    mu0 = s0 + b
    if n == 0:
        term = 0.0
    else:
        term = n * math.log(n / mu0)

    W = 2.0 * (term - (n - mu0))
    if W < 0.0:
        W = 0.0

    sgn = 1.0 if n >= mu0 else -1.0
    return sgn * math.sqrt(W)


def q_stat(s0: float, b: float, n: float) -> float:
    """q(s0) = sqrt(n) * log(n / mu0), with mu0 = s0 + b."""
    if n <= 0:
        return float("nan")
    mu0 = s0 + b
    return math.sqrt(n) * math.log(n / mu0)


def norm_survival(x: float) -> float:
    """One-sided upper tail 1 - Phi(x)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def r_star(s0: float, b: float, n: float) -> float:
    """
    Barndorff–Nielsen / Lugannani–Rice corrected root.
    """
    n = max(n - 0.5, 0.0)

    r = r_stat(s0, b, n)
    q = q_stat(s0, b, n)

    if (not math.isfinite(q)) or abs(q) < 1e-12 or abs(r) < 1e-12:
        return r

    return r + (1.0 / r) * math.log(abs(q / r))


def compute_curves(s0: float, b: float, n_max_sigma: float = 5.0):
    """
    Build arrays of exact and asymptotic p-values over a range of n.
    """
    mu0 = s0 + b
    n_min = int(0)
    n_max = int(math.ceil(mu0 + n_max_sigma * math.sqrt(mu0)))
    n_vals = np.arange(n_min, n_max + 1, dtype=int)

    p_true_list = []
    p_r_list = []
    p_rstar_list = []

    for n in n_vals:
        p_exact = poisson_tail(s0, b, n)
        r_val = r_stat(s0, b, n)
        p_r_val = norm_survival(r_val)
        rs_val = r_star(s0, b, n)
        p_rs_val = norm_survival(rs_val)

        p_true_list.append(p_exact)
        p_r_list.append(p_r_val)
        p_rstar_list.append(p_rs_val)

    return n_vals, np.array(p_true_list), np.array(p_r_list), np.array(p_rstar_list)


def median_count(mu: float) -> int:
    """Median of Poisson(mu)."""
    return int(poisson.ppf(0.5, mu))


def median_expected_significance(s_true: float, b: float) -> float:
    """Deterministic approximation of median expected discovery significance."""
    mu1 = s_true + b
    n_med = median_count(mu1)
    p_tail = poisson_tail(s0=0.0, b=b, n=n_med)
    return norm.isf(p_tail)


__all__ = [
    "poisson_tail",
    "r_stat",
    "q_stat",
    "norm_survival",
    "r_star",
    "compute_curves",
    "median_count",
    "median_expected_significance",
]
