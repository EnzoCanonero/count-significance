"""Discovery statistics for the Poisson on/off counting model."""

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import norm, poisson

from .common import ScalarOrArray, discovery_pvalue, discovery_z, norm_survival

# Profile likelihood and signed roots


def _xlogy(count: ScalarOrArray, argument: ScalarOrArray) -> np.ndarray:
    """Evaluate count log(argument) using the limit 0 log(x) = 0."""
    count, argument = np.broadcast_arrays(
        np.asarray(count, dtype=float),
        np.asarray(argument, dtype=float),
    )
    result = np.zeros_like(count, dtype=float)
    positive_count = count > 0.0
    result[positive_count] = (
        count[positive_count] * np.log(argument[positive_count])
    )
    return result


def b_profiled(
    s: ScalarOrArray,
    n: ScalarOrArray,
    m: ScalarOrArray,
    tau: ScalarOrArray,
) -> ScalarOrArray:
    """Return the profiled background for a fixed signal s.

    For physical inputs s, n, m >= 0 and tau > 0, the non-negative root is

    b_tilde = [A + sqrt(A^2 + 4(1 + tau) m s)] / [2(1 + tau)],
    where A = n + m - (1 + tau) s.
    """
    s, n, m, tau = np.broadcast_arrays(
        np.asarray(s, dtype=float),
        np.asarray(n, dtype=float),
        np.asarray(m, dtype=float),
        np.asarray(tau, dtype=float),
    )
    if np.any(tau <= 0.0):
        raise ValueError("tau must be positive")

    linear_term = (n + m) - (1.0 + tau) * s
    discriminant = linear_term**2 + 4.0 * (1.0 + tau) * m * s
    discriminant = np.maximum(discriminant, 0.0)
    return (linear_term + np.sqrt(discriminant)) / (2.0 * (1.0 + tau))


def loglik_diff(
    s: ScalarOrArray,
    n: ScalarOrArray,
    m: ScalarOrArray,
    tau: ScalarOrArray,
) -> ScalarOrArray:
    """Return the profile log-likelihood difference at fixed s.

    With mu_on = s + b_tilde and mu_off = tau b_tilde,

    Delta ell = n log(n / mu_on) + m log(m / mu_off)
                - n - m + mu_on + mu_off.

    Products involving zero counts are evaluated by continuity. A positive
    count at zero mean gives inf, while a negative mean gives nan.
    """
    s = np.asarray(s, dtype=float)
    n_observed = np.asarray(n, dtype=float)
    m_observed = np.asarray(m, dtype=float)
    background_profile = b_profiled(s, n_observed, m_observed, tau)
    mean_on = s + background_profile
    mean_off = tau * background_profile

    ratio_on = np.ones_like(mean_on, dtype=float)
    ratio_off = np.ones_like(mean_off, dtype=float)
    positive_mean_on = mean_on > 0.0
    positive_mean_off = mean_off > 0.0

    ratio_on = np.divide(
        n_observed,
        mean_on,
        out=ratio_on,
        where=positive_mean_on,
    )
    ratio_off = np.divide(
        m_observed,
        mean_off,
        out=ratio_off,
        where=positive_mean_off,
    )

    # A positive count with zero mean has an infinite likelihood difference.
    ratio_on = np.where(
        (~positive_mean_on) & (n_observed > 0.0),
        np.inf,
        ratio_on,
    )
    ratio_off = np.where(
        (~positive_mean_off) & (m_observed > 0.0),
        np.inf,
        ratio_off,
    )

    delta_log_likelihood = (
        _xlogy(n_observed, ratio_on)
        + _xlogy(m_observed, ratio_off)
        - (n_observed + m_observed)
        + s
        + (1.0 + tau) * background_profile
    )

    negative_mean = (mean_on < 0.0) | (mean_off < 0.0)
    if np.any(negative_mean):
        delta_log_likelihood = np.asarray(delta_log_likelihood, dtype=float)
        delta_log_likelihood[negative_mean] = np.nan

    return delta_log_likelihood


def r_stat_onoff(
    s: ScalarOrArray,
    n: ScalarOrArray,
    m: ScalarOrArray,
    tau: ScalarOrArray,
) -> ScalarOrArray:
    """Return r(s) = sign(s_hat - s) sqrt(2 Delta ell(s)).

    The unconstrained signal estimate is s_hat = n - m / tau.
    """
    signal_hat = (
        np.asarray(n, dtype=float)
        - np.asarray(m, dtype=float) / np.asarray(tau, dtype=float)
    )
    delta_log_likelihood = np.maximum(loglik_diff(s, n, m, tau), 0.0)
    return np.sign(signal_hat - s) * np.sqrt(2.0 * delta_log_likelihood)


def u_stat_onoff(s: float, n: float, m: float, tau: float) -> float:
    """Return the higher-order auxiliary statistic u(s).

    For b_tilde = b_profiled(s), mu_on = s + b_tilde and mu_off = tau b_tilde,

    u = sqrt(n m) / sqrt(n / mu_on^2 + m / b_tilde^2)
        * [log(n / mu_on) / b_tilde - log(m / mu_off) / mu_on].

    At n = 0 or m = 0, the continuous limit is zero.
    """
    s = float(s)
    n = float(n)
    m = float(m)
    tau = float(tau)

    # At the sample-space boundaries the continuous limit is u = 0.
    if n <= 0.0 or m <= 0.0:
        return 0.0

    background_profile = float(b_profiled(s, n, m, tau))
    mean_on = s + background_profile
    mean_off = tau * background_profile

    if mean_on <= 0.0 or mean_off <= 0.0 or background_profile <= 0.0:
        return float("nan")

    numerator = math.sqrt(n * m)
    denominator = math.sqrt(n / mean_on**2 + m / background_profile**2)
    log_terms = (
        math.log(n / mean_on) / background_profile
        - math.log(m / mean_off) / mean_on
    )

    return numerator * log_terms / denominator


def r_star_onoff(s: float, n: float, m: float, tau: float) -> float:
    """Return the higher-order root r*(s) = r + log|u / r| / r.

    At n = 0 or m = 0 the logarithmic adjustment is undefined, so r* = r.
    At s = s_hat the analytic limit is
    (n tau^3 - m) / [6 (m + n tau^2)^(3/2)].
    """
    s = float(s)
    n = float(n)
    m = float(m)
    tau = float(tau)

    r = float(r_stat_onoff(s, n, m, tau))

    # At n = 0 or m = 0 the logarithmic correction is undefined, so use r* = r.
    if n <= 0.0 or m <= 0.0:
        return r

    # At s = s_hat both r and u vanish. Allow only for floating-point roundoff.
    m_over_tau = m / tau
    mle_residual = n - s - m_over_tau
    mle_scale = abs(n) + abs(s) + abs(m_over_tau)
    roundoff_limit = 8.0 * np.finfo(float).eps * mle_scale

    if tau > 0.0 and abs(mle_residual) <= roundoff_limit:
        denominator = m + n * tau**2
        return (n * tau**3 - m) / (6.0 * denominator**1.5)

    u = u_stat_onoff(s, n, m, tau)

    if not math.isfinite(r) or not math.isfinite(u) or r == 0.0 or u == 0.0:
        return r

    return r + math.log(abs(u / r)) / r


# Reference distributions and Monte Carlo

def _sample_null_toys(
    s: float,
    b: float,
    tau: float,
    n_toys: int,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw N ~ Pois(s + b) and M ~ Pois(tau b) under the tested signal."""
    rng = np.random.default_rng(seed)
    n_toys_observed = rng.poisson(lam=s + b, size=n_toys)
    m_toys_observed = rng.poisson(lam=tau * b, size=n_toys)
    return n_toys_observed, m_toys_observed


def required_toys_for_Z_precision(
    r_obs: float,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
) -> tuple[int, bool]:
    """Choose the toy count for a target relative precision on Z.

    Taking Z = max(0, r_obs) and p = 1 - Phi(Z), binomial error propagation gives

    N = p(1 - p) / [sigrel^2 Z^2 phi(Z)^2].

    Here phi is the standard-normal density. The flag is true when max_toys
    prevents reaching this precision.
    """
    min_toys = int(min_toys)
    max_toys = int(max_toys)
    sigrel = float(sigrel)

    if sigrel <= 0.0:
        raise ValueError("sigrel must be positive")

    if min_toys < 1 or max_toys < 1:
        raise ValueError("min_toys and max_toys must be positive")
    if min_toys > max_toys:
        raise ValueError("min_toys must not exceed max_toys")

    z_value = float(discovery_z(r_obs))
    # Relative precision is undefined at Z = 0, where there is no discovery.
    if z_value <= 0.0:
        return min_toys, False
    if not np.isfinite(z_value):
        return max_toys, True

    p = float(norm_survival(z_value))
    phi = float(norm.pdf(z_value))
    if (not np.isfinite(p)) or p <= 0.0 or p >= 1.0:
        return max_toys, True
    if (not np.isfinite(phi)) or phi <= 0.0:
        return max_toys, True

    numerator = p * (1.0 - p)
    denominator = sigrel**2 * z_value**2 * phi**2
    if denominator <= 0.0:
        return max_toys, True

    required_toys = numerator / denominator

    if not np.isfinite(required_toys):
        return max_toys, True

    required_toys = int(math.ceil(required_toys))
    precision_limited = required_toys > max_toys

    n_toys = max(required_toys, min_toys)
    n_toys = min(n_toys, max_toys)
    return n_toys, precision_limited


# Choose a finite Poisson support for the deterministic tail sum.
def _poisson_grid_max(mu: float, observed: int = 0, tail_mass: float = 1e-12) -> int:
    mu = float(mu)
    if mu <= 0.0:
        return max(0, int(observed))

    upper = poisson.ppf(1.0 - float(tail_mass), mu)
    if not np.isfinite(upper):
        upper = mu + 12.0 * np.sqrt(mu + 1.0) + 20.0

    return max(int(np.ceil(upper)), int(observed), 0)


def pvals_onoff_profile_sum(
    s: float,
    n: int,
    m: int,
    tau: float,
    tail_mass: float = 1e-12,
) -> dict[str, float]:
    """Return profiled-reference and asymptotic discovery p-values.

    The deterministic reference plugs the profiled background into the null
    model and sums the inclusive tail r_toy >= r_obs. Each Poisson grid omits
    at most tail_mass probability.
    """
    if not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass must lie between zero and one")

    r_observed = float(r_stat_onoff(s, n, m, tau))
    rstar_observed = float(r_star_onoff(s, n, m, tau))

    p_r = float(discovery_pvalue(r_observed))
    p_rstar = float(discovery_pvalue(rstar_observed))

    if r_observed <= 0.0:
        return {
            "p_ref": 0.5,
            "p_ref_se": 0.0,
            "p_r": p_r,
            "p_rstar": p_rstar,
        }

    background_profile = float(b_profiled(s, n, m, tau))
    mean_on = float(s + background_profile)
    mean_off = float(tau * background_profile)

    n_grid = np.arange(
        _poisson_grid_max(mean_on, observed=n, tail_mass=tail_mass) + 1
    )
    m_grid = np.arange(
        _poisson_grid_max(mean_off, observed=m, tail_mass=tail_mass) + 1
    )
    probability_n = poisson.pmf(n_grid, mean_on)
    probability_m = poisson.pmf(m_grid, mean_off)

    toy_n, toy_m = np.meshgrid(n_grid, m_grid, indexing="ij")
    r_grid = r_stat_onoff(s, toy_n, toy_m, tau)
    probabilities = probability_n[:, None] * probability_m[None, :]
    p_ref = float(np.sum(probabilities[r_grid >= r_observed]))

    return {
        "p_ref": min(p_ref, 0.5),
        "p_ref_se": 0.0,
        "p_r": p_r,
        "p_rstar": p_rstar,
    }


def pvals_onoff(
    s: float,
    n: int,
    m: int,
    tau: float,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
) -> dict[str, Union[float, int, bool]]:
    """Return Monte Carlo and asymptotic p-values for observed (n, m).

    The corrected tail estimate is (K + 1) / (N + 1). The returned p_mc applies
    the discovery cap p <= 0.5, while p_mc_raw = K / N is kept for diagnostics.
    """

    # Compute the observed roots and their Gaussian p-values.
    r_observed = float(r_stat_onoff(s, n, m, tau))
    rstar_observed = float(r_star_onoff(s, n, m, tau))

    p_r = float(discovery_pvalue(r_observed))
    p_rstar = float(discovery_pvalue(rstar_observed))

    # Profile the background under the tested signal hypothesis.
    background_profile = float(b_profiled(s, n, m, tau))

    # Choose enough toys to reach the target relative precision on Z.
    n_toys, precision_limited = required_toys_for_Z_precision(
        r_observed,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
    )

    # Generate toys under the null using the profiled background.
    toy_n, toy_m = _sample_null_toys(
        s,
        background_profile,
        tau,
        n_toys=n_toys,
        seed=seed,
    )
    r_toys = r_stat_onoff(s, toy_n, toy_m, tau)

    # Count toys in the observed-or-more-extreme discovery tail.
    n_exceedances = int(np.count_nonzero(r_toys >= r_observed))
    p_mc_raw = n_exceedances / n_toys
    p_mc_corrected = (n_exceedances + 1.0) / (n_toys + 1.0)

    if r_observed <= 0.0:
        p_mc = 0.5
    else:
        p_mc = min(p_mc_corrected, 0.5)

    p_variance = p_mc_corrected * (1.0 - p_mc_corrected) / n_toys
    p_mc_se = float(math.sqrt(p_variance))

    return {
        "p_mc": p_mc,
        "p_mc_raw": p_mc_raw,
        "p_mc_se": p_mc_se,
        "p_resolution": 1.0 / (n_toys + 1.0),
        "p_r": p_r,
        "p_rstar": p_rstar,
        "n_toys": n_toys,
        "n_exceedances": n_exceedances,
        "b_profiled": background_profile,
        "precision_limited": precision_limited,
    }


def asimov_Zs_onoff(
    s_true: float,
    b: float,
    tau: float,
) -> dict[str, float]:
    """Return Asimov discovery significances for testing s = 0.

    The representative counts are n_A = s_true + b and m_A = tau b.
    """
    n_asimov = float(s_true + b)
    m_asimov = float(tau * b)

    # Compute the signed roots before applying the discovery convention.
    r_asimov = r_stat_onoff(
        0.0,
        n_asimov,
        m_asimov,
        tau,
    )
    rstar_asimov = r_star_onoff(
        0.0,
        n_asimov,
        m_asimov,
        tau,
    )
    z_r = float(discovery_z(r_asimov))
    z_rstar = float(discovery_z(rstar_asimov))

    return {
        "Z_A_r": z_r,
        "Z_A_rstar": z_rstar,
    }


# Expected significance

def _mc_significance_summary_onoff(
    s_true: float,
    b: float,
    tau: float,
    n_outer: int = 2000,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
) -> tuple[float, float]:
    """Return the median and mean significance from nested on/off toys.

    Outer toys follow N ~ Pois(s_true + b) and M ~ Pois(tau b). Each inner
    null sample estimates the corresponding discovery-tail probability.
    """
    if n_outer <= 0:
        raise ValueError("n_outer must be positive")

    rng = np.random.default_rng(seed)

    n_observed = rng.poisson(lam=s_true + b, size=n_outer)
    m_observed = rng.poisson(lam=tau * b, size=n_outer)

    p_mc = np.empty(n_outer, dtype=float)

    for i, (n_obs, m_obs) in enumerate(zip(n_observed, m_observed)):
        inner_seed = int(rng.integers(1, 2**31 - 1))

        out = pvals_onoff(
            s=0.0,
            tau=tau,
            n=int(n_obs),
            m=int(m_obs),
            sigrel=sigrel,
            min_toys=min_toys,
            max_toys=max_toys,
            seed=inner_seed,
        )

        p_mc[i] = float(out["p_mc"])

    z_values = norm.isf(p_mc)

    return float(np.median(z_values)), float(np.mean(z_values))


def expected_significance_onoff(
    s_true: ScalarOrArray,
    b: ScalarOrArray,
    tau: ScalarOrArray,
    n_outer: int = 2000,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
) -> dict[str, np.ndarray]:
    """Return Asimov and Monte Carlo expected discovery significances.

    The Asimov values use n_A = s_true + b and m_A = tau b. The Monte Carlo
    values are the median and mean of Z over on/off data toys.
    """
    s_arr, b_arr, tau_arr = np.broadcast_arrays(
        np.asarray(s_true, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(tau, dtype=float),
    )
    flat_s = s_arr.ravel()
    flat_b = b_arr.ravel()
    flat_tau = tau_arr.ravel()

    rng = np.random.default_rng(seed)

    z_asimov_r = np.empty_like(flat_s, dtype=float)
    z_asimov_rstar = np.empty_like(flat_s, dtype=float)
    z_mc_median = np.empty_like(flat_s, dtype=float)
    z_mc_mean = np.empty_like(flat_s, dtype=float)

    for i, (s_val, b_val, tau_val) in enumerate(zip(flat_s, flat_b, flat_tau)):
        asim = asimov_Zs_onoff(
            float(s_val),
            float(b_val),
            float(tau_val),
        )
        mc_median, mc_mean = _mc_significance_summary_onoff(
            float(s_val),
            float(b_val),
            float(tau_val),
            n_outer=n_outer,
            sigrel=sigrel,
            min_toys=min_toys,
            max_toys=max_toys,
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        z_asimov_r[i] = asim["Z_A_r"]
        z_asimov_rstar[i] = asim["Z_A_rstar"]
        z_mc_median[i] = mc_median
        z_mc_mean[i] = mc_mean

    shape = s_arr.shape
    return {
        "Z_A_r": z_asimov_r.reshape(shape),
        "Z_A_rstar": z_asimov_rstar.reshape(shape),
        "Z_mc_median": z_mc_median.reshape(shape),
        "Z_mc_mean": z_mc_mean.reshape(shape),
    }


__all__ = [
    "b_profiled",
    "loglik_diff",
    "r_stat_onoff",
    "u_stat_onoff",
    "r_star_onoff",
    "required_toys_for_Z_precision",
    "pvals_onoff_profile_sum",
    "pvals_onoff",
    "asimov_Zs_onoff",
    "expected_significance_onoff",
]
