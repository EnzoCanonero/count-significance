import math
import numpy as np
from scipy.stats import poisson, norm

from .common import norm_survival

def poisson_tail_on(s0: float, b: float, n) -> np.ndarray:
    """
    Upward tail P[N >= n | mu0] with mu0 = s0 + b.
    Uses scipy's exact Poisson SF; supports array n.
    """
    mu0 = s0 + b
    # scipy's sf is P[X > x]; for integer counts, sf(n-1) = P[N >= n]
    return poisson.sf(np.asarray(n) - 1, mu0)


def r_stat_on(s0: float, b: float, n: float) -> float:
    """Signed root likelihood ratio r(s0) for testing s = s0."""
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


def q_stat_on(s0: float, b: float, n: float) -> float:
    """q(s0) = sqrt(n) * log(n / mu0), with mu0 = s0 + b."""
    if n <= 0:
        return float("nan")
    mu0 = s0 + b
    return math.sqrt(n) * math.log(n / mu0)

def r_star_on(s0: float, b: float, n: float) -> float:
    """
    Barndorff–Nielsen / Lugannani–Rice corrected root.
    """
    n = max(n - 0.5, 0.0)

    r = r_stat_on(s0, b, n)
    q = q_stat_on(s0, b, n)

    if (not math.isfinite(q)) or abs(q) < 1e-12 or abs(r) < 1e-12:
        return r

    return r + (1.0 / r) * math.log(abs(q / r))


def pvals_on(s0: float, b: float, n):
    """
    Vectorized p-values for observed counts n in the simple on-channel model.

    Accepts scalar or array-like n; returns arrays for exact and asymptotic p-values.
    """
    n_arr = np.asarray(n, dtype=float)
    p_true = poisson_tail_on(s0, b, n_arr)

    r_vals = np.vectorize(r_stat_on)(s0, b, n_arr)
    p_r = norm_survival(r_vals)

    rstar_vals = np.vectorize(r_star_on)(s0, b, n_arr)
    p_rstar = norm_survival(rstar_vals)

    return {
        "p_true": np.asarray(p_true, dtype=float),
        "p_r": np.asarray(p_r, dtype=float),
        "p_rstar": np.asarray(p_rstar, dtype=float),
    }


def median_expected_significance_on(
    s_true: float,
    b: float,
    n_outer: int = 200,
    seed: int = 12345,
) -> float:
    """
    MC median expected discovery Z:
      - Generate n_outer toys n_obs ~ Pois(s_true + b)
      - Compute p_tail under H0: s=0 (so mu0=b) for each toy
      - Convert each p to Z via norm.isf
      - Return median(Z)
    """
    rng = np.random.default_rng(seed)
    n_obs = rng.poisson(lam=s_true + b, size=n_outer)

    p_tail = poisson_tail_on(s0=0.0, b=b, n=n_obs)
    p_tail = np.clip(p_tail, 1e-16, 1.0 - 1e-16)
    Z = norm.isf(p_tail)
    return float(np.median(Z))


def expected_significance_on(
    s_true,
    b,
    n_outer: int = 200,
    seed: int = 12345,
) -> dict:
    """
    Vectorized expected discovery Z for the on-channel model.

    Returns Asimov Z (r, r*) and MC median Z; accepts scalars or array-like s_true and b.
    """
    s_arr, b_arr = np.broadcast_arrays(np.asarray(s_true, dtype=float), np.asarray(b, dtype=float))
    flat_s = s_arr.ravel()
    flat_b = b_arr.ravel()

    rng = np.random.default_rng(seed)
    medians = np.empty_like(flat_s, dtype=float)
    Z_A_r = np.empty_like(flat_s, dtype=float)
    Z_A_rstar = np.empty_like(flat_s, dtype=float)

    for i, (s_val, b_val) in enumerate(zip(flat_s, flat_b)):
        nA = float(s_val + b_val)
        Z_A_r[i] = r_stat_on(0.0, b_val, nA)
        Z_A_rstar[i] = r_star_on(0.0, b_val, nA)
        medians[i] = median_expected_significance_on(
            s_true=float(s_val),
            b=float(b_val),
            n_outer=n_outer,
            seed=int(rng.integers(1, 2**31 - 1)),
        )

    return {
        "Z_A_r": Z_A_r.reshape(s_arr.shape),
        "Z_A_rstar": Z_A_rstar.reshape(s_arr.shape),
        "Z_mc_median": medians.reshape(s_arr.shape),
    }


__all__ = [
    "poisson_tail_on",
    "r_stat_on",
    "q_stat_on",
    "norm_survival",
    "r_star_on",
    "pvals_on",
    "median_expected_significance_on",
    "expected_significance_on",
]
