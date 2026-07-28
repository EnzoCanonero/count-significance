"""Poisson on/off counting model with an uncertain background.

The model is N ~ Pois(s + b), M ~ Pois(tau b), with ``b >= 0`` and
``tau > 0``.  The likelihood uses the unconstrained signal estimate
``s_hat = n - m/tau``.  Discovery results apply the directional convention
``Z = max(0, r)`` after evaluating either ``r`` or its higher-order form
``r*``.
"""

import math

import numpy as np
from scipy.stats import norm, poisson

from .common import discovery_pvalue, discovery_z, norm_survival


# Profile likelihood and signed roots


def _xlogy(count, argument):
    """Evaluate ``count * log(argument)`` with the limit ``0 log(x)=0``."""
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


def b_profiled(s, n, m, tau):
    """Profiled background estimate for a fixed tested signal ``s``.

    The non-negative quadratic root is

        b_tilde = [A + sqrt(A^2 + 4 (1 + tau) m s)] / [2 (1 + tau)],
        A = n + m - (1 + tau) s.

    The physical on/off model assumes ``s >= 0``, ``n,m >= 0`` and ``tau > 0``.
    Scalars and broadcastable arrays are accepted.
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


def loglik_diff(s, n, m, tau):
    """Profile log-likelihood difference at a fixed signal value.

    This is ``Delta ell(s) = ell(s_hat, b_hat) - ell_p(s)``.  Products
    ``0 log 0`` are evaluated by continuity.  With profiled means
    ``mu_on=s+b_tilde`` and ``mu_off=tau*b_tilde``, it is

        n log(n/mu_on) + m log(m/mu_off) - n - m + mu_on + mu_off.

    A zero mean with a positive observed count gives ``+inf``; a negative
    Poisson mean gives ``nan``.
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


def r_stat_onoff(s, n, m, tau):
    """Signed likelihood root ``sign(s_hat-s) sqrt(2 Delta ell(s))``."""
    signal_hat = (
        np.asarray(n, dtype=float)
        - np.asarray(m, dtype=float) / np.asarray(tau, dtype=float)
    )
    delta_log_likelihood = np.maximum(loglik_diff(s, n, m, tau), 0.0)
    return np.sign(signal_hat - s) * np.sqrt(2.0 * delta_log_likelihood)


def u_stat_onoff(s, n, m, tau):
    """Closed-form higher-order auxiliary statistic ``u(s)``.

    With ``b_tilde=b_profiled(s)`` and the two profiled means, the expression
    implemented below is

        sqrt(n m) / sqrt(n/mu_on^2 + m/b_tilde^2)
        * [log(n/mu_on)/b_tilde - log(m/mu_off)/mu_on].

    At ``n=0`` or ``m=0``, its continuous endpoint value is zero because
    ``sqrt(x) log(x) -> 0``.
    """
    s = float(s)
    n = float(n)
    m = float(m)
    tau = float(tau)

    # At the sample-space boundaries the continuous limit is u=0.
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


def r_star_onoff(s, n, m, tau):
    """Higher-order root ``r*(s) = r + log|u/r| / r``.

    At ``n=0`` or ``m=0`` the logarithmic adjustment is undefined and the
    boundary prescription is ``r*=r``.  At ``s=s_hat`` the analytic limit is

        (n tau^3 - m) / [6 (m + n tau^2)^(3/2)].
    """
    s = float(s)
    n = float(n)
    m = float(m)
    tau = float(tau)

    r = float(r_stat_onoff(s, n, m, tau))

    # At n=0 or m=0 the logarithmic correction is not defined, so use r*=r.
    if n <= 0.0 or m <= 0.0:
        return r

    # At s=s_hat both r and u vanish. Allow only for floating-point roundoff.
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


def _sample_null_toys(s, b, tau, n_toys, seed=None):
    """Generate toys under H0: N~Pois(s+b), M~Pois(tau*b)."""
    rng = np.random.default_rng(seed)
    n_toys_observed = rng.poisson(lam=s + b, size=n_toys)
    m_toys_observed = rng.poisson(lam=tau * b, size=n_toys)
    return n_toys_observed, m_toys_observed


def required_toys_for_Z_precision(
    r_obs,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
) -> tuple[int, bool]:
    """
    Estimate N so that the MC-based Z has relative uncertainty ≤ sigrel.

    Take Z ≈ r_obs and p = Φ̄(Z). With p̂ binomial, Var(p̂) = p(1−p)/N and
    |dZ/dp| = 1/φ(Z). Propagating the uncertainty from p to Z gives

        N = p(1−p) / [sigrel² Z² φ(Z)²].

    Return the number of toys after applying the requested limits and a flag
    which is true when max_toys prevents reaching the target precision.
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
    # Relative precision is not defined at Z=0, where there is no discovery.
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


def _poisson_grid_max(mu: float, observed: int = 0, tail_mass: float = 1e-12) -> int:
    """Return a finite Poisson support for a deterministic tail sum."""
    mu = float(mu)
    if mu <= 0.0:
        return max(0, int(observed))

    upper = poisson.ppf(1.0 - float(tail_mass), mu)
    if not np.isfinite(upper):
        upper = mu + 12.0 * np.sqrt(mu + 1.0) + 20.0

    return max(int(np.ceil(upper)), int(observed), 0)


def pvals_onoff_profile_sum(
    s,
    n,
    m,
    tau,
    tail_mass: float = 1e-12,
) -> dict:
    """Return the profiled-reference and asymptotic discovery p-values.

    The deterministic reference plugs the fitted background into the null
    model and sums the inclusive tail ``r_toy >= r_obs``.  Each Poisson grid
    omits at most ``tail_mass`` probability; ``p_ref_se`` is zero because no
    Monte Carlo sampling is involved.
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
    s,
    n,
    m,
    tau,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
) -> dict:
    """
    Compute Monte Carlo and asymptotic p-values for observed ``(n, m)``.

    ``p_mc=(K+1)/(N+1)`` is the corrected MC estimate, capped at 0.5 for the
    discovery test; ``p_mc_raw=K/N`` is retained for diagnostics.  The result
    also contains ``n_toys`` (N), ``n_exceedances`` (K), the profiled
    background, p-value resolution, MC standard error and precision-limit flag.
    """

    # Compute the observed test statistics and their Gaussian p-values.
    r_observed = float(r_stat_onoff(s, n, m, tau))
    rstar_observed = float(r_star_onoff(s, n, m, tau))

    p_r = float(discovery_pvalue(r_observed))
    p_rstar = float(discovery_pvalue(rstar_observed))

    # Profile the background under the signal hypothesis being tested.
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
    s_true,
    b,
    tau,
):
    """
    Asimov discovery Z-values for testing s = 0 in the on/off problem.

    The Asimov counts are ``n_A=s_true+b`` and ``m_A=tau*b``.  The returned
    dictionary contains ``Z_A_r`` and ``Z_A_rstar`` for a test of ``s=0``.
    """
    n_asimov = float(s_true + b)
    m_asimov = float(tau * b)

    # First compute the signed roots, then apply the discovery definition.
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
    s_true,
    b,
    tau,
    n_outer: int = 2000,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
):
    """Return median and mean Z from outer data toys and inner null toys."""
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
    s_true,
    b,
    tau,
    n_outer: int = 2000,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
):
    """
    Compute Asimov and MC expected significances for H0:s=0.

    ``s_true``, ``b`` and ``tau`` may be scalars or broadcastable arrays.  The
    keys ``Z_A_r``, ``Z_A_rstar``, ``Z_mc_median`` and ``Z_mc_mean`` all have
    their broadcast shape.
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
