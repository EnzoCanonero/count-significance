import math
from typing import Tuple

import numpy as np
from scipy.stats import norm, poisson

from .common import discovery_pvalue, discovery_z, norm_survival


def xlogy(a, b):
    """Compute a*log(b) with 0*log(·)=0, assumes b>0 when a>0."""
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
    )
    out = np.zeros_like(a, dtype=float)
    mask = a > 0
    out[mask] = a[mask] * np.log(b[mask])
    return out


def b_profiled(s, n, m, tau):
    s = np.asarray(s, dtype=float)
    A = (n + m) - (1.0 + tau) * s
    disc = A**2 + 4.0 * (1.0 + tau) * m * s
    disc = np.maximum(disc, 0.0)
    return (A + np.sqrt(disc)) / (2.0 * (1.0 + tau))


def loglik_diff(s, n, m, tau):
    """
    Δℓ(s) = ℓ(s_hat, b_hat) - ℓ_p(s)
    - Negative Poisson means ⇒ NaN (undefined model)
    - Zero means ⇒ use the continuous limit (finite if count is zero, +inf if count>0)
    """
    s = np.asarray(s, dtype=float)
    btilde = b_profiled(s, n, m, tau)
    mu_on = s + btilde
    mu_off = tau * btilde

    # Only negative means are invalid; zero means are allowed as limits.
    negative_mu = (mu_on < 0.0) | (mu_off < 0.0)

    # Safe ratios for log terms.
    n_arr = np.asarray(n, dtype=float)
    m_arr = np.asarray(m, dtype=float)

    ratio_on = np.ones_like(mu_on, dtype=float)
    ratio_off = np.ones_like(mu_off, dtype=float)

    pos_on = mu_on > 0
    pos_off = mu_off > 0

    ratio_on = np.divide(n_arr, mu_on, out=ratio_on, where=pos_on)
    ratio_off = np.divide(m_arr, mu_off, out=ratio_off, where=pos_off)

    # If mean is zero but count > 0, push ratio to +inf (correct limiting loglik).
    ratio_on = np.where((~pos_on) & (n_arr > 0), np.inf, ratio_on)
    ratio_off = np.where((~pos_off) & (m_arr > 0), np.inf, ratio_off)

    ll = (
        xlogy(n_arr, ratio_on)
        + xlogy(m_arr, ratio_off)
        - (n_arr + m_arr)
        + s
        + (1.0 + tau) * btilde
    )

    if np.any(negative_mu):
        ll = np.asarray(ll, dtype=float)
        ll[negative_mu] = np.nan

    return ll


def r_stat_onoff(s, n, m, tau):
    """Signed LR root: r(s) = sign(s_hat - s)*sqrt(2*Δℓ(s)), with s_hat = n - m/tau."""
    shat = np.asarray(n, dtype=float) - np.asarray(m, dtype=float) / np.asarray(tau, dtype=float)
    dll = np.maximum(loglik_diff(s, n, m, tau), 0.0)
    return np.sign(shat - s) * np.sqrt(2.0 * dll)


def u_stat_onoff(s, n, m, tau):
    """Closed-form auxiliary statistic u(s) for one on/off observation."""
    s = float(s)
    n = float(n)
    m = float(m)
    tau = float(tau)

    # At the sample-space boundaries the continuous limit is u=0.
    if n <= 0.0 or m <= 0.0:
        return 0.0

    btilde = float(b_profiled(s, n, m, tau))
    mu_on = s + btilde
    mu_off = tau * btilde

    if mu_on <= 0.0 or mu_off <= 0.0 or btilde <= 0.0:
        return float("nan")

    numerator = math.sqrt(n * m)
    denominator = math.sqrt(n / mu_on**2 + m / btilde**2)
    log_terms = math.log(n / mu_on) / btilde - math.log(m / mu_off) / mu_on

    return numerator * log_terms / denominator


def r_star_onoff(s, n, m, tau):
    """
    Modified root:
        r*(s) = r(s) + (1/r(s)) * log|u(s) / r(s)|.
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


def sample_null_toys(s, b, tau, n_toys, seed=None):
    """Generate toys under H0: N~Pois(s+b), M~Pois(tau*b)."""
    rng = np.random.default_rng(seed)
    N = rng.poisson(lam=s + b, size=n_toys)
    M = rng.poisson(lam=tau * b, size=n_toys)
    return N, M


def required_toys_for_Z_precision(
    r_obs,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
) -> Tuple[int, bool]:
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

    if max_toys < 1:
        max_toys = 1
    if min_toys < 1:
        min_toys = 1
    if min_toys > max_toys:
        min_toys = max_toys

    Z = float(discovery_z(r_obs))
    # Relative precision is not defined at Z=0, where there is no discovery.
    if Z <= 0.0:
        return min_toys, False
    if not np.isfinite(Z):
        return max_toys, True

    p = float(norm_survival(Z))
    phi = float(norm.pdf(Z))
    if (not np.isfinite(p)) or p <= 0.0 or p >= 1.0:
        return max_toys, True
    if (not np.isfinite(phi)) or phi <= 0.0:
        return max_toys, True

    numerator = p * (1.0 - p)
    denominator = sigrel**2 * Z**2 * phi**2
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
    tau,
    n,
    m,
    tail_mass: float = 1e-12,
) -> dict:
    """Compute the deterministic profiled-null reference and asymptotic p-values."""
    r_ref = float(r_stat_onoff(s, n, m, tau))
    rstar_obs = float(r_star_onoff(s, n, m, tau))

    p_r = float(discovery_pvalue(r_ref))
    p_rstar = float(discovery_pvalue(rstar_obs))

    if r_ref <= 0.0:
        return {
            "p_ref": 0.5,
            "p_ref_se": 0.0,
            "p_r": p_r,
            "p_rstar": p_rstar,
        }

    b_tilde = float(b_profiled(s, n, m, tau))
    mu_n = float(s + b_tilde)
    mu_m = float(tau * b_tilde)

    n_grid = np.arange(_poisson_grid_max(mu_n, observed=n, tail_mass=tail_mass) + 1)
    m_grid = np.arange(_poisson_grid_max(mu_m, observed=m, tail_mass=tail_mass) + 1)
    p_n = poisson.pmf(n_grid, mu_n)
    p_m = poisson.pmf(m_grid, mu_m)

    nn, mm = np.meshgrid(n_grid, m_grid, indexing="ij")
    r_grid = r_stat_onoff(s, nn, mm, tau)
    probabilities = p_n[:, None] * p_m[None, :]
    p_ref = float(np.sum(probabilities[r_grid >= r_ref]))

    return {
        "p_ref": min(p_ref, 0.5),
        "p_ref_se": 0.0,
        "p_r": p_r,
        "p_rstar": p_rstar,
    }


# Unified function to compute p-values using r, r*, and MC simulations
def pvals_onoff(
    s,
    tau,
    n,
    m,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
) -> dict:
    """
    Compute p-values for observed (n,m) testing s.

    The MC result includes the generated toy count, the number of tail
    exceedances, and the resolution of the corrected p-value estimate.
    """

    # Compute the observed test statistics and their Gaussian p-values.
    r_ref = float(r_stat_onoff(s, n, m, tau))
    rstar_obs = float(r_star_onoff(s, n, m, tau))

    p_r = float(discovery_pvalue(r_ref))
    p_rs = float(discovery_pvalue(rstar_obs))

    # Profile the background under the signal hypothesis being tested.
    b_tilde = float(b_profiled(s, n, m, tau))

    # Choose enough toys to reach the target relative precision on Z.
    n_toys, precision_limited = required_toys_for_Z_precision(
        r_ref,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
    )

    # Generate toys under the null using the profiled background.
    toys_N, toys_M = sample_null_toys(s, b_tilde, tau, n_toys=n_toys, seed=seed)
    r_toys = r_stat_onoff(s, toys_N, toys_M, tau)

    # Count toys in the observed-or-more-extreme discovery tail.
    n_exceedances = int(np.count_nonzero(r_toys >= r_ref))
    p_mc_raw = n_exceedances / n_toys
    p_mc_corrected = (n_exceedances + 1.0) / (n_toys + 1.0)

    if r_ref <= 0.0:
        p_mc = 0.5
    else:
        p_mc = min(p_mc_corrected, 0.5)

    var_p = p_mc_corrected * (1.0 - p_mc_corrected) / n_toys
    se_mc = float(math.sqrt(var_p))

    return {
        "p_mc": p_mc,
        "p_mc_raw": p_mc_raw,
        "p_mc_se": se_mc,
        "p_resolution": 1.0 / (n_toys + 1.0),
        "p_r": p_r,
        "p_rstar": p_rs,
        "n_toys": n_toys,
        "n_exceedances": n_exceedances,
        "b_profiled": b_tilde,
        "precision_limited": precision_limited,
    }


def asimov_Zs_onoff(
    s_true,
    b,
    tau,
):
    """
    Asimov discovery Z-values for testing s = 0 in the on/off problem.

    Steps:
      1. Build Asimov (expected) counts under the true model (s_true, b, τ):
           n_A = s_true + b          (on region)
           m_A = τ * b               (off region)
      2. Evaluate r(0) and r*(0) at (n_A, m_A) to get the Asimov Z-values.
    """
    nA = float(s_true + b)  # on-region expected count
    mA = float(tau * b)     # off-region expected count

    # First compute the signed roots, then apply the discovery definition.
    r_asimov = r_stat_onoff(
        0.0,
        nA,
        mA,
        tau,
    )
    rstar_asimov = r_star_onoff(
        0.0,
        nA,
        mA,
        tau,
    )
    z_r = float(discovery_z(r_asimov))
    z_rss = float(discovery_z(rstar_asimov))

    return {
        "Z_A_r": z_r,
        "Z_A_rstar": z_rss,
        "nA": nA,
        "mA": mA,
    }


def median_expected_significance_onoff(
    s_true,
    b,
    tau,
    n_outer: int = 2000,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
):
    """Two-level MC: outer toys for data, inner toys for p-value."""
    rng = np.random.default_rng(seed)

    Ns = rng.poisson(lam=s_true + b, size=n_outer)
    Ms = rng.poisson(lam=tau * b, size=n_outer)

    p_mc = np.empty(n_outer, dtype=float)

    for i, (n_obs, m_obs) in enumerate(zip(Ns, Ms)):
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

    Z = norm.isf(p_mc)

    return float(np.median(Z)), float(np.mean(Z))


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

    Accepts scalars or array-like s_true, b, tau (broadcasted); returns dict of arrays.
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

    Z_A_r = np.empty_like(flat_s, dtype=float)
    Z_A_rstar = np.empty_like(flat_s, dtype=float)
    Z_mc_median = np.empty_like(flat_s, dtype=float)
    Z_mc_mean = np.empty_like(flat_s, dtype=float)

    for i, (s_val, b_val, tau_val) in enumerate(zip(flat_s, flat_b, flat_tau)):
        asim = asimov_Zs_onoff(
            float(s_val),
            float(b_val),
            float(tau_val),
        )
        mc_median, mc_mean = median_expected_significance_onoff(
            float(s_val),
            float(b_val),
            float(tau_val),
            n_outer=n_outer,
            sigrel=sigrel,
            min_toys=min_toys,
            max_toys=max_toys,
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        Z_A_r[i] = asim["Z_A_r"]
        Z_A_rstar[i] = asim["Z_A_rstar"]
        Z_mc_median[i] = mc_median
        Z_mc_mean[i] = mc_mean

    shape = s_arr.shape
    return {
        "Z_A_r": Z_A_r.reshape(shape),
        "Z_A_rstar": Z_A_rstar.reshape(shape),
        "Z_mc_median": Z_mc_median.reshape(shape),
        "Z_mc_mean": Z_mc_mean.reshape(shape),
    }


__all__ = [
    "norm_survival",
    "xlogy",
    "b_profiled",
    "loglik_diff",
    "r_stat_onoff",
    "u_stat_onoff",
    "r_star_onoff",
    "sample_null_toys",
    "required_toys_for_Z_precision",
    "pvals_onoff_profile_sum",
    "pvals_onoff",
    "asimov_Zs_onoff",
    "median_expected_significance_onoff",
    "expected_significance_onoff",
]
