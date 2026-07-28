import math
import numpy as np
from scipy.stats import poisson, norm

from .common import discovery_pvalue, discovery_z, norm_survival

def _correct_count(n, continuity_correction: bool):
    n_arr = np.asarray(n, dtype=float)
    if continuity_correction:
        n_arr = np.maximum(n_arr - 0.5, 0.0)
    return float(n_arr) if n_arr.shape == () else n_arr


def poisson_tail_on(s0: float, b: float, n) -> np.ndarray:
    """
    Upward tail P[N >= n | mu0] with mu0 = s0 + b.
    Uses scipy's exact Poisson SF; supports array n.
    """
    mu0 = s0 + b
    # scipy's sf is P[X > x]; for integer counts, sf(n-1) = P[N >= n]
    return poisson.sf(np.asarray(n) - 1, mu0)


def r_stat_on(s0: float, b: float, n: float, continuity_correction: bool = False) -> float:
    """Signed root likelihood ratio r(s0) for testing s = s0."""
    n = _correct_count(n, continuity_correction)
    mu0 = s0 + b
    if mu0 <= 0.0:
        return 0.0 if n <= 0.0 else float("inf")
    if n == 0:
        term = 0.0
    else:
        term = n * math.log(n / mu0)

    W = 2.0 * (term - (n - mu0))
    if W < 0.0:
        W = 0.0

    sgn = 1.0 if n >= mu0 else -1.0
    return sgn * math.sqrt(W)


def u_stat_on(s0: float, b: float, n: float, continuity_correction: bool = False) -> float:
    """u(s0) = sqrt(n) * log(n / mu0), with mu0 = s0 + b."""
    n = _correct_count(n, continuity_correction)
    if n == 0:
        return 0.0
    if n < 0:
        return float("nan")
    mu0 = s0 + b
    if mu0 <= 0.0:
        return float("inf")
    return math.sqrt(n) * math.log(n / mu0)


def r_star_on(s0: float, b: float, n: float, continuity_correction: bool = True) -> float:
    """
    Barndorff–Nielsen / Lugannani–Rice corrected root.
    """
    n_eff = _correct_count(n, continuity_correction)
    mu0 = s0 + b
    r = r_stat_on(s0, b, n_eff, continuity_correction=False)

    # At n=0 the logarithmic correction is not defined, so we use r*=r.
    if n_eff == 0:
        return r

    # At n=mu0 both r and u vanish. Use the analytic limit of r*.
    mle_residual = n_eff - mu0
    mle_scale = abs(n_eff) + abs(s0) + abs(b)
    if continuity_correction:
        mle_scale += 0.5
    machine_precision = np.finfo(float).eps
    roundoff_limit = 8.0 * machine_precision * mle_scale

    if mu0 > 0.0 and abs(mle_residual) <= roundoff_limit:
        return 1.0 / (6.0 * math.sqrt(mu0))

    u = u_stat_on(s0, b, n_eff, continuity_correction=False)

    if (not math.isfinite(u)) or u == 0.0 or r == 0.0:
        return r

    return r + (1.0 / r) * math.log(abs(u / r))


def pvals_on(
    s0: float,
    b: float,
    n,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """
    Vectorized p-values for observed counts n in the simple on-channel model.

    Accepts scalar or array-like n; returns arrays for exact and asymptotic p-values.
    """
    n_arr = np.asarray(n, dtype=float)
    p_true = np.minimum(poisson_tail_on(s0, b, n_arr), 0.5)

    r_vals = np.vectorize(r_stat_on)(s0, b, n_arr, continuity_correction_r)
    p_r = discovery_pvalue(r_vals)

    rstar_vals = np.vectorize(r_star_on)(s0, b, n_arr, continuity_correction_rstar)
    p_rstar = discovery_pvalue(rstar_vals)

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
    p_tail = np.minimum(p_tail, 0.5)
    Z = norm.isf(p_tail)
    return float(np.median(Z)), float(np.mean(Z))


def expected_significance_on(
    s_true,
    b,
    n_outer: int = 200,
    seed: int = 12345,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
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
    means = np.empty_like(flat_s, dtype=float)
    Z_A_r = np.empty_like(flat_s, dtype=float)
    Z_A_rstar = np.empty_like(flat_s, dtype=float)

    for i, (s_val, b_val) in enumerate(zip(flat_s, flat_b)):
        nA = float(s_val + b_val)
        r_asimov = r_stat_on(
            0.0,
            b_val,
            nA,
            continuity_correction=continuity_correction_r,
        )
        rstar_asimov = r_star_on(
            0.0,
            b_val,
            nA,
            continuity_correction=continuity_correction_rstar,
        )
        Z_A_r[i] = discovery_z(r_asimov)
        Z_A_rstar[i] = discovery_z(rstar_asimov)

        medians[i], means[i] = median_expected_significance_on(
            s_true=float(s_val),
            b=float(b_val),
            n_outer=n_outer,
            seed=int(rng.integers(1, 2**31 - 1)),
        )

    return {
        "Z_A_r": Z_A_r.reshape(s_arr.shape),
        "Z_A_rstar": Z_A_rstar.reshape(s_arr.shape),
        "Z_mc_median": medians.reshape(s_arr.shape),
        "Z_mc_mean": means.reshape(s_arr.shape),
    }


__all__ = [
    "poisson_tail_on",
    "r_stat_on",
    "u_stat_on",
    "norm_survival",
    "r_star_on",
    "pvals_on",
    "median_expected_significance_on",
    "expected_significance_on",
]
