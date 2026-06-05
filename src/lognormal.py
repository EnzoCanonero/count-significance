import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar
from scipy.stats import norm, poisson

from .common import norm_survival
from .on_off import xlogy


def _as_scalar(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"{name} must be scalar")
    return float(arr)


def _validate_lognormal_inputs(b0: float, sigma: float):
    if not np.isfinite(b0) or b0 <= 0.0:
        raise ValueError("b0 must be positive and finite")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be non-negative and finite")


def _correct_count(n, continuity_correction: bool):
    n_arr = np.asarray(n, dtype=float)
    if continuity_correction:
        n_arr = np.maximum(n_arr - 0.5, 0.0)
    return float(n_arr) if n_arr.shape == () else n_arr


def background_lognormal(theta, b0, sigma):
    """Background yield b(theta) = b0 exp(sigma theta), positive by construction."""
    b0_arr = np.asarray(b0, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    theta_arr = np.asarray(theta, dtype=float)
    with np.errstate(over="ignore"):
        return b0_arr * np.exp(sigma_arr * theta_arr)


def _loglik_fixed_theta(s, n, theta, theta_tilde, b0, sigma):
    b = float(background_lognormal(theta, b0, sigma))
    mu = float(s + b)
    if (not np.isfinite(mu)) or mu <= 0.0:
        return -math.inf

    return float(xlogy(n, mu) - mu - 0.5 * (theta - theta_tilde) ** 2)


def theta_profiled_lognormal(s, n, theta_tilde, b0, sigma):
    """Profile theta for fixed s in the log-normal constrained counting model."""
    s = _as_scalar("s", s)
    n = _as_scalar("n", n)
    theta_tilde = _as_scalar("theta_tilde", theta_tilde)
    b0 = _as_scalar("b0", b0)
    sigma = _as_scalar("sigma", sigma)
    _validate_lognormal_inputs(b0, sigma)

    if n < 0.0:
        raise ValueError("n must be non-negative")

    if sigma == 0.0:
        if s + b0 <= 0.0:
            raise ValueError("s + b0 must be positive when sigma is zero")
        return theta_tilde

    def objective(theta):
        ell = _loglik_fixed_theta(s, n, float(theta), theta_tilde, b0, sigma)
        return -ell if np.isfinite(ell) else math.inf

    width = 30.0
    last_res = None
    for _ in range(5):
        lo = theta_tilde - width
        hi = theta_tilde + width
        res = minimize_scalar(
            objective,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not res.success:
            raise RuntimeError(f"theta profiling failed: {res.message}")

        last_res = res
        margin = max(1e-8, 1e-6 * width)
        if (res.x > lo + margin) and (res.x < hi - margin):
            return float(res.x)

        width *= 2.0

    if last_res is None or not np.isfinite(last_res.x):
        raise RuntimeError("theta profiling failed")
    return float(last_res.x)


def _loglik_diff_lognormal_scalar(s, n, theta_tilde, b0, sigma):
    s = float(s)
    n = float(n)
    theta_tilde = float(theta_tilde)
    b0 = float(b0)
    sigma = float(sigma)

    if n < 0.0:
        raise ValueError("n must be non-negative")

    theta_hat_hat = theta_profiled_lognormal(s, n, theta_tilde, b0, sigma)
    b_hat_hat = float(background_lognormal(theta_hat_hat, b0, sigma))
    mu_hat_hat = float(s + b_hat_hat)
    if mu_hat_hat <= 0.0:
        return math.nan

    # With unconstrained s, the global MLE has theta_hat = theta_tilde and mu_hat = n.
    dll = float(xlogy(n, n / mu_hat_hat) - (n - mu_hat_hat) + 0.5 * (theta_hat_hat - theta_tilde) ** 2)
    if dll < 0.0:
        dll = 0.0
    return dll


def loglik_diff_lognormal(s, n, theta_tilde, b0, sigma):
    """Delta log-likelihood for fixed s in the log-normal constrained model."""
    vec = np.vectorize(_loglik_diff_lognormal_scalar, otypes=[float])
    out = vec(s, n, theta_tilde, b0, sigma)
    return float(out) if np.asarray(out).shape == () else out


def _j_thetatheta_from_profile(n, mu_hat_hat, b_hat_hat, sigma):
    return (
        1.0
        - sigma * sigma * b_hat_hat * (n / mu_hat_hat - 1.0)
        + n * sigma * sigma * b_hat_hat * b_hat_hat / (mu_hat_hat * mu_hat_hat)
    )


def _j_thetatheta_lognormal_scalar(s, n, theta_tilde, b0, sigma):
    s = float(s)
    n = float(n)
    theta_tilde = float(theta_tilde)
    b0 = float(b0)
    sigma = float(sigma)

    theta_hat_hat = theta_profiled_lognormal(s, n, theta_tilde, b0, sigma)
    b_hat_hat = float(background_lognormal(theta_hat_hat, b0, sigma))
    mu_hat_hat = float(s + b_hat_hat)
    if mu_hat_hat <= 0.0:
        return math.nan

    return float(_j_thetatheta_from_profile(n, mu_hat_hat, b_hat_hat, sigma))


def j_thetatheta_lognormal(s, n, theta_tilde, b0, sigma):
    """Observed nuisance information j_theta_theta at fixed s and profiled theta."""
    vec = np.vectorize(_j_thetatheta_lognormal_scalar, otypes=[float])
    out = vec(s, n, theta_tilde, b0, sigma)
    return float(out) if np.asarray(out).shape == () else out


def _q_stat_lognormal_scalar(s, n, theta_tilde, b0, sigma, continuity_correction=False):
    s = float(s)
    n = float(_correct_count(n, continuity_correction))
    theta_tilde = float(theta_tilde)
    b0 = float(b0)
    sigma = float(sigma)

    if n <= 0.0:
        return 0.0

    theta_hat_hat = theta_profiled_lognormal(s, n, theta_tilde, b0, sigma)
    b_hat_hat = float(background_lognormal(theta_hat_hat, b0, sigma))
    mu_hat_hat = float(s + b_hat_hat)
    if mu_hat_hat <= 0.0:
        return math.nan

    j_theta_theta = float(_j_thetatheta_from_profile(n, mu_hat_hat, b_hat_hat, sigma))
    if j_theta_theta <= 0.0:
        return math.nan

    bracket = math.log(n / mu_hat_hat) + (
        sigma * sigma * b_hat_hat * b_hat_hat * (n - mu_hat_hat) / (mu_hat_hat * mu_hat_hat)
    )
    return float(math.sqrt(n / j_theta_theta) * bracket)


def q_stat_lognormal(s, n, theta_tilde, b0, sigma, continuity_correction: bool = False):
    """Auxiliary q statistic for the log-normal constrained model."""
    vec = np.vectorize(_q_stat_lognormal_scalar, otypes=[float])
    out = vec(s, n, theta_tilde, b0, sigma, continuity_correction)
    return float(out) if np.asarray(out).shape == () else out


def _r_stat_lognormal_scalar(s, n, theta_tilde, b0, sigma, continuity_correction=False):
    s = float(s)
    n = float(_correct_count(n, continuity_correction))
    theta_tilde = float(theta_tilde)
    b0 = float(b0)
    sigma = float(sigma)

    shat_minus_s = n - float(background_lognormal(theta_tilde, b0, sigma)) - s
    dll = _loglik_diff_lognormal_scalar(s, n, theta_tilde, b0, sigma)
    if not np.isfinite(dll):
        return math.nan

    return float(np.sign(shat_minus_s) * math.sqrt(max(2.0 * dll, 0.0)))


def r_stat_lognormal(s, n, theta_tilde, b0, sigma, continuity_correction: bool = False):
    """Signed LR root for testing s in the log-normal constrained model."""
    vec = np.vectorize(_r_stat_lognormal_scalar, otypes=[float])
    out = vec(s, n, theta_tilde, b0, sigma, continuity_correction)
    return float(out) if np.asarray(out).shape == () else out


def _r_star_lognormal_scalar(s, n, theta_tilde, b0, sigma, eps, continuity_correction=True):
    if continuity_correction:
        # Gradient-based lattice correction for discrete n.
        # grad_T r ∝ (log(n/mu_hat),  theta_hat - theta_tilde)  [envelope theorem].
        # Only n is discrete; theta_tilde is continuous and receives no shift.
        # Project -1/2 onto the n-axis: Delta n = -1/2 * log_ratio / norm.
        # When theta_hat ≈ theta_tilde (nominal case) the denominator collapses to
        # |log_ratio| and we recover the full n → n − 1/2 step.  When the Gaussian
        # component is large the step is proportionally reduced.
        theta_hat = theta_profiled_lognormal(s, n, theta_tilde, b0, sigma)
        b_hat     = float(background_lognormal(theta_hat, b0, sigma))
        mu_hat    = float(s + b_hat)

        log_ratio_n      = math.log(n / mu_hat) if (n > 0 and mu_hat > 0) else 0.0
        gaussian_residual = theta_hat - theta_tilde          # ≈ 0 at nominal

        cc_norm = math.sqrt(log_ratio_n ** 2 + gaussian_residual ** 2)
        if cc_norm > eps:
            n_eff = max(n - 0.5 * log_ratio_n / cc_norm, 0.0)
        else:
            n_eff = float(n)
    else:
        n_eff = float(n)

    r = float(_r_stat_lognormal_scalar(s, n_eff, theta_tilde, b0, sigma, continuity_correction=False))
    q = float(_q_stat_lognormal_scalar(s, n_eff, theta_tilde, b0, sigma, continuity_correction=False))

    if abs(r) < eps:
        return r
    if (not np.isfinite(r)) or (not np.isfinite(q)):
        return r
    if q * r <= 0.0:
        return r

    return float(r + (1.0 / r) * math.log(q / r))


def r_star_lognormal(
    s,
    n,
    theta_tilde,
    b0,
    sigma,
    eps: float = 1e-12,
    continuity_correction: bool = True,
):
    """Modified likelihood root using the q statistic."""
    vec = np.vectorize(_r_star_lognormal_scalar, otypes=[float])
    out = vec(s, n, theta_tilde, b0, sigma, eps, continuity_correction)
    return float(out) if np.asarray(out).shape == () else out


def _poisson_grid_max(mu: float, observed: int = 0, tail_mass: float = 1e-10) -> int:
    """Finite Poisson support large enough for numerical tail sums."""
    mu = float(mu)
    if mu <= 0.0:
        return max(0, int(observed))

    upper = poisson.ppf(1.0 - float(tail_mass), mu)
    if not np.isfinite(upper):
        upper = mu + 12.0 * np.sqrt(mu + 1.0) + 20.0

    return max(int(np.ceil(upper)), int(observed), 0)


def _find_n_threshold_lognormal(s0, theta_tilde_prime, b0, sigma, r_obs, n_max):
    """
    Smallest integer n' in [0, n_max] where r_stat_lognormal(..., n', theta_tilde_prime) >= r_obs.

    Returns n_max + 1 if no such n' exists within the finite grid.
    """
    if float(r_stat_lognormal(s0, n_max, theta_tilde_prime, b0, sigma)) < r_obs:
        return int(n_max) + 1

    lo, hi = 0, int(n_max)
    while lo < hi:
        mid = (lo + hi) // 2
        if float(r_stat_lognormal(s0, mid, theta_tilde_prime, b0, sigma)) >= r_obs:
            hi = mid
        else:
            lo = mid + 1
    return lo


def pvals_lognormal_numerical(
    s0,
    n,
    theta_tilde,
    b0,
    sigma,
    n_gh_pts: int = 30,
    tail_mass: float = 1e-10,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Numerical plug-in null tail using GH quadrature and the r-ordering."""
    r_ref = float(r_stat_lognormal(s0, n, theta_tilde, b0, sigma, continuity_correction=False))
    r_obs = float(
        r_stat_lognormal(
            s0,
            n,
            theta_tilde,
            b0,
            sigma,
            continuity_correction=continuity_correction_r,
        )
    )
    rs_obs = float(
        r_star_lognormal(
            s0,
            n,
            theta_tilde,
            b0,
            sigma,
            continuity_correction=continuity_correction_rstar,
        )
    )

    p_r = float(norm_survival(max(r_obs, 0.0)))
    p_rs = float(norm_survival(max(rs_obs, 0.0)))

    if r_ref <= 0.0:
        return {
            "p_numerical": 0.5,
            "p_numerical_se": 0.0,
            "p_r": p_r,
            "p_rstar": p_rs,
        }

    theta_hat_hat = float(theta_profiled_lognormal(s0, n, theta_tilde, b0, sigma))
    mu_hat_hat = float(s0 + background_lognormal(theta_hat_hat, b0, sigma))
    n_max_grid = _poisson_grid_max(mu_hat_hat, observed=int(n), tail_mass=tail_mass)

    gh_nodes, gh_weights = hermgauss(int(n_gh_pts))

    p_ref = 0.0
    for node, weight in zip(gh_nodes, gh_weights):
        theta_tilde_prime = theta_hat_hat + math.sqrt(2.0) * float(node)
        n_thresh = _find_n_threshold_lognormal(s0, theta_tilde_prime, b0, sigma, r_ref, n_max_grid)
        if n_thresh > n_max_grid:
            p_at_node = 0.0
        elif n_thresh == 0:
            p_at_node = 1.0
        else:
            p_at_node = float(poisson.sf(n_thresh - 1, mu_hat_hat))
        p_ref += float(weight) * p_at_node

    p_ref /= math.sqrt(math.pi)
    p_ref = min(max(float(p_ref), 0.0), 0.5)

    return {
        "p_numerical": p_ref,
        "p_numerical_se": 0.0,
        "p_r": p_r,
        "p_rstar": p_rs,
    }


def asimov_Z_lognormal(
    s_true,
    b0,
    sigma,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Asimov discovery Z-values for testing s = 0 with theta_tilde_A = 0."""
    s_true = _as_scalar("s_true", s_true)
    b0 = _as_scalar("b0", b0)
    sigma = _as_scalar("sigma", sigma)
    _validate_lognormal_inputs(b0, sigma)

    nA = float(s_true + b0)
    theta_tilde_A = 0.0
    z_r = max(
        float(
            r_stat_lognormal(
                0.0,
                nA,
                theta_tilde_A,
                b0,
                sigma,
                continuity_correction=continuity_correction_r,
            )
        ),
        0.0,
    )
    z_rstar = max(
        float(
            r_star_lognormal(
                0.0,
                nA,
                theta_tilde_A,
                b0,
                sigma,
                continuity_correction=continuity_correction_rstar,
            )
        ),
        0.0,
    )

    return {
        "Z_A_r": z_r,
        "Z_A_rstar": z_rstar,
        "nA": nA,
        "theta_tilde_A": theta_tilde_A,
    }


def median_expected_significance_lognormal(
    s_true,
    b0,
    sigma,
    n_outer: int,
    n_gh_pts: int = 20,
    tail_mass: float = 1e-10,
    seed: int = 12345,
):
    """Outer-toy MC median and mean discovery Z for the log-normal constrained model."""
    s_true = _as_scalar("s_true", s_true)
    b0 = _as_scalar("b0", b0)
    sigma = _as_scalar("sigma", sigma)
    _validate_lognormal_inputs(b0, sigma)

    n_outer = int(n_outer)
    if n_outer <= 0:
        raise ValueError("n_outer must be positive")

    mu_true = float(s_true + b0)
    if mu_true < 0.0:
        raise ValueError("s_true + b0 must be non-negative")

    rng = np.random.default_rng(seed)
    n_obs = rng.poisson(lam=mu_true, size=n_outer)
    theta_tilde_obs = rng.normal(loc=0.0, scale=1.0, size=n_outer)

    p_values = np.empty(n_outer, dtype=float)
    p_min = max(float(tail_mass), 1e-300)
    for i, (n_i, theta_tilde_i) in enumerate(zip(n_obs, theta_tilde_obs)):
        out = pvals_lognormal_numerical(
            s0=0.0,
            n=int(n_i),
            theta_tilde=float(theta_tilde_i),
            b0=b0,
            sigma=sigma,
            n_gh_pts=n_gh_pts,
            tail_mass=tail_mass,
        )
        p_values[i] = np.clip(float(out["p_numerical"]), p_min, 1.0 - 1e-16)

    z_values = norm.isf(p_values)
    return float(np.median(z_values)), float(np.mean(z_values))


def expected_significance_lognormal(
    s_true,
    b0,
    sigma,
    n_outer=None,
    n_gh_pts: int = 20,
    tail_mass: float = 1e-10,
    seed: int = 12345,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Vectorized expected discovery Z for the log-normal constrained model."""
    s_arr, b0_arr, sigma_arr = np.broadcast_arrays(
        np.asarray(s_true, dtype=float),
        np.asarray(b0, dtype=float),
        np.asarray(sigma, dtype=float),
    )

    flat_s = s_arr.ravel()
    flat_b0 = b0_arr.ravel()
    flat_sigma = sigma_arr.ravel()

    Z_A_r = np.empty_like(flat_s, dtype=float)
    Z_A_rstar = np.empty_like(flat_s, dtype=float)
    nA = np.empty_like(flat_s, dtype=float)
    theta_tilde_A = np.empty_like(flat_s, dtype=float)
    run_mc = n_outer is not None
    if run_mc:
        Z_mc_median = np.empty_like(flat_s, dtype=float)
        Z_mc_mean = np.empty_like(flat_s, dtype=float)
        rng = np.random.default_rng(seed)

    for i, (s_val, b0_val, sigma_val) in enumerate(zip(flat_s, flat_b0, flat_sigma)):
        asim = asimov_Z_lognormal(
            float(s_val),
            float(b0_val),
            float(sigma_val),
            continuity_correction_r=continuity_correction_r,
            continuity_correction_rstar=continuity_correction_rstar,
        )
        Z_A_r[i] = asim["Z_A_r"]
        Z_A_rstar[i] = asim["Z_A_rstar"]
        nA[i] = asim["nA"]
        theta_tilde_A[i] = asim["theta_tilde_A"]
        if run_mc:
            mc_seed = int(rng.integers(1, 2**31 - 1))
            Z_mc_median[i], Z_mc_mean[i] = median_expected_significance_lognormal(
                s_true=float(s_val),
                b0=float(b0_val),
                sigma=float(sigma_val),
                n_outer=int(n_outer),
                n_gh_pts=n_gh_pts,
                tail_mass=tail_mass,
                seed=mc_seed,
            )

    shape = s_arr.shape
    out = {
        "Z_A_r": Z_A_r.reshape(shape),
        "Z_A_rstar": Z_A_rstar.reshape(shape),
        "nA": nA.reshape(shape),
        "theta_tilde_A": theta_tilde_A.reshape(shape),
    }
    if run_mc:
        out["Z_mc_median"] = Z_mc_median.reshape(shape)
        out["Z_mc_mean"] = Z_mc_mean.reshape(shape)
    return out


__all__ = [
    "background_lognormal",
    "theta_profiled_lognormal",
    "loglik_diff_lognormal",
    "r_stat_lognormal",
    "j_thetatheta_lognormal",
    "q_stat_lognormal",
    "r_star_lognormal",
    "pvals_lognormal_numerical",
    "asimov_Z_lognormal",
    "median_expected_significance_lognormal",
    "expected_significance_lognormal",
]
