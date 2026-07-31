"""Discovery statistics for a Poisson count with known background."""

import math

import numpy as np
from scipy.stats import norm, poisson

from .common import ScalarOrArray, discovery_pvalue, discovery_z

# Likelihood and signed roots

def _correct_count(n: float, continuity_correction: bool) -> float:
    """Apply n -> max(n - 1/2, 0) when continuity correction is requested."""
    n_effective = float(n)
    if continuity_correction:
        n_effective = max(n_effective - 0.5, 0.0)
    return n_effective


def poisson_tail_on(s0: float, b: float, n: ScalarOrArray) -> ScalarOrArray:
    """Return the inclusive tail P(N >= n | mu0), where mu0 = s0 + b."""
    mu0 = s0 + b
    # SciPy defines sf(x) as P(N > x), so sf(n - 1) includes the observed count.
    return poisson.sf(np.asarray(n) - 1, mu0)


def r_stat_on(
    s0: float,
    b: float,
    n: float,
    continuity_correction: bool = False,
) -> float:
    """Return the signed likelihood root for testing s = s0.

    For mu0 = s0 + b, the root is
    sign(n - mu0) sqrt(2[n log(n / mu0) - (n - mu0)]).
    """
    n = _correct_count(n, continuity_correction)
    mu0 = s0 + b
    if mu0 <= 0.0:
        return 0.0 if n <= 0.0 else float("inf")
    if n == 0:
        term = 0.0
    else:
        term = n * math.log(n / mu0)

    likelihood_ratio = 2.0 * (term - (n - mu0))
    if likelihood_ratio < 0.0:
        likelihood_ratio = 0.0

    sign = 1.0 if n >= mu0 else -1.0
    return sign * math.sqrt(likelihood_ratio)


def u_stat_on(
    s0: float,
    b: float,
    n: float,
    continuity_correction: bool = False,
) -> float:
    """Return u(s0) = sqrt(n) log(n / mu0), where mu0 = s0 + b."""
    n = _correct_count(n, continuity_correction)
    if n == 0:
        return 0.0
    if n < 0:
        return float("nan")
    mu0 = s0 + b
    if mu0 <= 0.0:
        return float("inf")
    return math.sqrt(n) * math.log(n / mu0)


def r_star_on(
    s0: float,
    b: float,
    n: float,
    continuity_correction: bool = True,
) -> float:
    """Return the higher-order root r*(s0).

    Away from the endpoints, r* = r + log|u / r| / r. At n = 0 the logarithmic
    adjustment is undefined, so r* = r. When the effective count equals
    s0 + b, its analytic limit is 1 / [6 sqrt(s0 + b)].
    """
    n_eff = _correct_count(n, continuity_correction)
    mu0 = s0 + b
    r = r_stat_on(s0, b, n_eff, continuity_correction=False)

    # At n = 0 the logarithmic correction is undefined, so use r* = r.
    if n_eff == 0:
        return r

    # At n = mu0 both r and u vanish, so use the analytic limit of r*.
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
    n: ScalarOrArray,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
) -> dict[str, np.ndarray]:
    """Return exact and asymptotic discovery p-values for observed counts.

    p_exact is the inclusive Poisson tail, while p_r and p_rstar are the
    Gaussian approximations. All three use the discovery cap p <= 0.5.
    """
    n_arr = np.asarray(n, dtype=float)
    p_exact = np.minimum(poisson_tail_on(s0, b, n_arr), 0.5)

    r_vals = np.vectorize(r_stat_on, otypes=[float])(
        s0,
        b,
        n_arr,
        continuity_correction_r,
    )
    p_r = discovery_pvalue(r_vals)

    rstar_vals = np.vectorize(r_star_on, otypes=[float])(
        s0,
        b,
        n_arr,
        continuity_correction_rstar,
    )
    p_rstar = discovery_pvalue(rstar_vals)

    return {
        "p_exact": np.asarray(p_exact, dtype=float),
        "p_r": np.asarray(p_r, dtype=float),
        "p_rstar": np.asarray(p_rstar, dtype=float),
    }


# Expected significance

def _mc_significance_summary_on(
    s_true: float,
    b: float,
    n_outer: int = 200,
    seed: int = 12345,
) -> tuple[float, float]:
    """Return the median and mean significance from Poisson data toys.

    Each toy has N ~ Pois(s_true + b), with Z obtained from the inclusive
    background-only Poisson tail.
    """
    if n_outer <= 0:
        raise ValueError("n_outer must be positive")

    rng = np.random.default_rng(seed)
    n_obs = rng.poisson(lam=s_true + b, size=n_outer)

    p_tail = poisson_tail_on(s0=0.0, b=b, n=n_obs)
    p_tail = np.minimum(p_tail, 0.5)
    z_values = norm.isf(p_tail)
    return float(np.median(z_values)), float(np.mean(z_values))


def expected_significance_on(
    s_true: ScalarOrArray,
    b: ScalarOrArray,
    n_outer: int = 200,
    seed: int = 12345,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
) -> dict[str, np.ndarray]:
    """Return Asimov and Monte Carlo expected discovery significances.

    The Asimov values use n_A = s_true + b. The Monte Carlo values are the
    median and mean of Z over Poisson data toys.
    """
    s_arr, b_arr = np.broadcast_arrays(
        np.asarray(s_true, dtype=float),
        np.asarray(b, dtype=float),
    )
    flat_s = s_arr.ravel()
    flat_b = b_arr.ravel()

    rng = np.random.default_rng(seed)
    medians = np.empty_like(flat_s, dtype=float)
    means = np.empty_like(flat_s, dtype=float)
    z_asimov_r = np.empty_like(flat_s, dtype=float)
    z_asimov_rstar = np.empty_like(flat_s, dtype=float)

    for i, (s_val, b_val) in enumerate(zip(flat_s, flat_b)):
        n_asimov = float(s_val + b_val)
        r_asimov = r_stat_on(
            0.0,
            b_val,
            n_asimov,
            continuity_correction=continuity_correction_r,
        )
        rstar_asimov = r_star_on(
            0.0,
            b_val,
            n_asimov,
            continuity_correction=continuity_correction_rstar,
        )
        z_asimov_r[i] = discovery_z(r_asimov)
        z_asimov_rstar[i] = discovery_z(rstar_asimov)

        medians[i], means[i] = _mc_significance_summary_on(
            s_true=float(s_val),
            b=float(b_val),
            n_outer=n_outer,
            seed=int(rng.integers(1, 2**31 - 1)),
        )

    return {
        "Z_A_r": z_asimov_r.reshape(s_arr.shape),
        "Z_A_rstar": z_asimov_rstar.reshape(s_arr.shape),
        "Z_mc_median": medians.reshape(s_arr.shape),
        "Z_mc_mean": means.reshape(s_arr.shape),
    }


__all__ = [
    "poisson_tail_on",
    "r_stat_on",
    "u_stat_on",
    "r_star_on",
    "pvals_on",
    "expected_significance_on",
]
