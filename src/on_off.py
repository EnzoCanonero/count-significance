import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

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
):
    """
    Estimate N so that the MC-based Z has relative uncertainty ≤ sigrel.

    Take Z ≈ r_obs and p = Φ̄(Z). With p̂ binomial, Var(p̂) = p(1−p)/N and
    σ_p = sqrt[p(1−p)/N]. Propagate to Z via

        σ_Z / Z ≈ |dZ/dp| * σ_p / Z ≤ sigrel,

    solve for N, then clip to [min_toys, max_toys].
    """
    min_toys = int(min_toys)
    max_toys = int(max_toys)
    if max_toys < 1:
        max_toys = 1
    if min_toys < 1:
        min_toys = 1
    if min_toys > max_toys:
        min_toys = max_toys

    Z = float(discovery_z(r_obs))
    if not np.isfinite(Z) or Z <= 0.0:
        return int(min_toys)

    p = norm_survival(Z)
    if (not np.isfinite(p)) or p <= 0.0 or p >= 1.0:
        return int(max_toys)

    root_two_pi = math.sqrt(2.0 * math.pi)
    u = -2.0 * math.log(root_two_pi * p)
    if (not np.isfinite(u)) or u <= 0.0:
        return int(max_toys)

    dZdp = (1.0 - u) / (u * p * Z)
    N = (dZdp * dZdp) * p * (1.0 - p) / (Z * Z * sigrel * sigrel)

    if (not np.isfinite(N)) or N <= 0.0:
        N = max_toys

    N_int = int(math.ceil(N))
    N_int = min(N_int, int(max_toys))
    N_int = max(N_int, int(min_toys))
    return N_int

# Unified function to compute p-values using r, r*, and MC simulations
def pvals_onoff(
    s,
    b,
    tau,
    n,
    m,
    sigrel: float = 0.05,
    min_toys: int = 10_000,
    max_toys: int = 2_000_000,
    seed: int = 12345,
) -> Dict[str, float]:
    """Compute p-values for observed (n,m) testing s."""

    #Compute the observed test statistics r and r* and their Gaussian-approximation p-values
    r_ref = float(r_stat_onoff(s, n, m, tau))
    r_obs = r_ref
    rs_obs = float(r_star_onoff(s, n, m, tau))

    p_r = float(discovery_pvalue(r_obs))
    p_rs = float(discovery_pvalue(rs_obs))

    #Compute the profiled background yield under the signal hypothesis s
    b_tilde = b_profiled(s, n, m, tau)

    #Estimate the required number of toys to reach the target relative precision on the p-value
    n_toys = required_toys_for_Z_precision(
        r_ref,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
    )

    #Generate toys to estimate the Monte Carlo p-value.
    #Toys are drawn using the profiled background, mimicking the realistic case where b is unknown.
    toys_N, toys_M = sample_null_toys(s, b_tilde, tau, n_toys=n_toys, seed=seed)
    r_toys = r_stat_onoff(s, toys_N, toys_M, tau)

    p_mc_raw = float(np.mean(r_toys >= r_ref))
    p_mc = 0.5 if r_ref <= 0.0 else min(p_mc_raw, 0.5)

    var_p = max(p_mc_raw * (1.0 - p_mc_raw), 0.0) / n_toys
    se_mc = float(math.sqrt(var_p))

    return {
        "p_mc": p_mc,
        "p_mc_se": se_mc,
        "p_r": p_r,
        "p_rstar": p_rs,
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
            b=b,
            tau=tau,
            n=int(n_obs),
            m=int(m_obs),
            sigrel=sigrel,
            min_toys=min_toys,
            max_toys=max_toys,
            seed=inner_seed,
        )

        p = float(out["p_mc"])
        res = 1.0 / float(max_toys)

        p = min(max(p, res), 1.0 - res)

        p_mc[i] = p

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
    "pvals_onoff",
    "asimov_Zs_onoff",
    "median_expected_significance_onoff",
    "expected_significance_onoff",
]
