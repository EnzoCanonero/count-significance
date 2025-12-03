import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

SQRT2 = np.sqrt(2.0)


def normal_sf(x: float) -> float:
    """Standard normal survival function SF(x) = 1 - Phi(x)."""
    return 0.5 * math.erfc(float(x) / SQRT2)


def normal_isf(p):
    """Inverse survival function: z with SF(z) = p (i.e. P[Z >= z] = p)."""
    p = np.asarray(p, dtype=float)
    z = norm.isf(p)
    return float(z) if z.ndim == 0 else z


def xlogy(a, b):
    """Compute a*log(b) with 0*log(·)=0, assumes b>0 when a>0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros(np.broadcast(a, b).shape, dtype=float)
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
    """Δℓ(s) = ℓ(s_hat, b_hat) - ℓ_p(s); safe for n=0 or m=0 via xlogy."""
    s = np.asarray(s, dtype=float)
    btilde = b_profiled(s, n, m, tau)
    mu_on = s + btilde
    mu_off = tau * btilde
    return (
        xlogy(n, n / mu_on)
        + xlogy(m, m / mu_off)
        - (n + m)
        + s
        + (1.0 + tau) * btilde
    )


def r_signed(s, n, m, tau):
    """Signed LR root: r(s) = sign(s_hat - s)*sqrt(2*Δℓ(s)), with s_hat = n - m/tau."""
    shat = n - m / tau
    dll = np.maximum(loglik_diff(s, n, m, tau), 0.0)
    return np.sign(shat - s) * np.sqrt(2.0 * dll)


def q_value(s, n, m, tau):
    """Closed-form q(s); returns NaN if logs are invalid (e.g. n=0 or m=0)."""
    s = np.asarray(s, dtype=float)
    btilde = b_profiled(s, n, m, tau)
    mu_on = s + btilde
    mu_off = tau * btilde
    if np.any(mu_on <= 0) or np.any(mu_off <= 0) or (n == 0) or (m == 0):
        return np.full_like(s, np.nan, dtype=float)
    num_pref = np.sqrt(n * m)
    den_pref = np.sqrt((n / (mu_on**2)) + (m / (btilde**2)))
    log_on = np.log(n / mu_on)
    log_off = np.log(m / mu_off)
    bracket = (log_on / btilde) - (log_off / mu_on)
    return num_pref / den_pref * bracket


def r_star(s, n, m, tau, eps: float = 1e-12):
    """
    Modified root:
        r*(s) = r(s) + (1/r(s)) * log( r(s) / q(s) )
    Falls back to r if |r| small or q has wrong sign/NaN.
    """
    r = r_signed(s, n, m, tau)
    out = np.array(r, copy=True, dtype=float)
    q = q_value(s, n, m, tau)

    small = np.abs(r) < eps
    valid = (~small) & np.isfinite(q) & np.isfinite(r) & (q * r > 0.0)
    out[valid] = r[valid] + (1.0 / r[valid]) * np.log(q[valid] / r[valid])
    return out


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
    """Estimate toys needed for target relative precision on Z."""
    Z = float(r_obs)
    if not np.isfinite(Z) or Z <= 0.0:
        return int(min_toys)

    p = normal_sf(Z)
    if (not np.isfinite(p)) or p <= 0.0 or p >= 1.0:
        return int(max_toys)

    root_two_pi = math.sqrt(2.0 * math.pi)
    u = -2.0 * math.log(root_two_pi * p)
    if (not np.isfinite(u)) or u <= 0.0:
        return int(max_toys)

    dZdp = (1.0 - u) / (u * p * Z)
    N = (dZdp * dZdp) * p * (1.0 - p) / (Z * Z * sigrel * sigrel)

    if (not np.isfinite(N)) or N <= 0.0:
        N = min_toys

    N_int = int(math.ceil(N))
    N_int = max(N_int, int(min_toys))
    N_int = min(N_int, int(max_toys))
    return N_int


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
    r_obs = float(r_signed(s, n, m, tau))
    rs_obs = float(r_star(s, n, m, tau))

    p_r = normal_sf(r_obs)
    p_rs = normal_sf(rs_obs)

    b_tilde = b_profiled(s, n, m, tau)

    n_toys = required_toys_for_Z_precision(
        r_obs,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
    )

    toys_N, toys_M = sample_null_toys(s, b_tilde, tau, n_toys=n_toys, seed=seed)
    r_toys = r_signed(s, toys_N, toys_M, tau)

    p_mc = float(np.mean(r_toys >= r_obs))

    var_p = max(p_mc * (1.0 - p_mc), 0.0) / n_toys
    se_mc = float(math.sqrt(var_p))

    return {
        "r_obs": r_obs,
        "rstar_obs": rs_obs,
        "p_r": p_r,
        "p_rstar": p_rs,
        "p_mc": p_mc,
        "p_mc_se": se_mc,
        "mc_resolution": 1.0 / n_toys,
        "b_tilde": float(b_tilde),
        "n_toys": int(n_toys),
    }


def asimov_counts_onoff(s_true, b, tau) -> Tuple[float, float]:
    nA = float(s_true + b)
    mA = float(tau * b)
    return nA, mA


def asimov_Zs_onoff(s_true, b, tau):
    nA, mA = asimov_counts_onoff(s_true, b, tau)
    z_r = float(r_signed(0.0, nA, mA, tau))
    z_rss = float(r_star(0.0, nA, mA, tau))
    return {"Z_A_r": z_r, "Z_A_rstar": z_rss, "nA": nA, "mA": mA}


def expected_Z_mc_onoff(
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
    mc_res = np.empty(n_outer, dtype=float)

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
        res = float(out["mc_resolution"])

        p = min(max(p, res), 1.0 - res)

        p_mc[i] = p
        mc_res[i] = res

    Z = norm.isf(p_mc)

    Z_med = float(np.median(Z))
    Z_p16 = float(np.percentile(Z, 16.0))
    Z_p84 = float(np.percentile(Z, 84.0))

    return {
        "Z_mc": Z,
        "Z_mc_median": Z_med,
        "Z_mc_p16": Z_p16,
        "Z_mc_p84": Z_p84,
        "mc_res_min": float(mc_res.min()),
        "mc_res_max": float(mc_res.max()),
        "mc_res_median": float(np.median(mc_res)),
    }


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
    """Compute both Asimov and MC expected significances for H0:s=0."""
    asim = asimov_Zs_onoff(s_true, b, tau)
    mc = expected_Z_mc_onoff(
        s_true,
        b,
        tau,
        n_outer=n_outer,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed,
    )
    return {**asim, **mc}


__all__ = [
    "SQRT2",
    "normal_sf",
    "normal_isf",
    "xlogy",
    "b_profiled",
    "loglik_diff",
    "r_signed",
    "q_value",
    "r_star",
    "sample_null_toys",
    "required_toys_for_Z_precision",
    "pvals_onoff",
    "asimov_counts_onoff",
    "asimov_Zs_onoff",
    "expected_Z_mc_onoff",
    "expected_significance_onoff",
]
