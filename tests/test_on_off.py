"""Tests for discovery statistics with an uncertain on/off background."""

import math

import numpy as np
import pytest
from scipy.stats import norm

import count_significance.on_off as on_off
from count_significance.on_off import (
    asimov_Zs_onoff,
    b_profiled,
    loglik_diff,
    pvals_onoff,
    pvals_onoff_profile_sum,
    r_star_onoff,
    r_stat_onoff,
    required_toys_for_Z_precision,
    u_stat_onoff,
)


def test_reference_calculation():
    """The profiled background follows the positive likelihood root."""
    assert b_profiled(0.0, 5.0, 3.0, 2.0) == pytest.approx(
        8.0 / 3.0,
        rel=1e-12,
    )

    signal_hat = 5.0 - 3.0 / 2.0
    assert b_profiled(signal_hat, 5.0, 3.0, 2.0) == pytest.approx(
        3.0 / 2.0,
        rel=1e-12,
    )

    # For s=2, n=5, m=3 and tau=2, A=2 and the discriminant is 76.
    expected = (2.0 + math.sqrt(76.0)) / 6.0
    assert b_profiled(2.0, 5.0, 3.0, 2.0) == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_signed_likelihood_root():
    """The signed root is obtained directly from the profile likelihood."""
    s, n, m, tau = 0.0, 5.0, 2.0, 2.0
    background = 7.0 / 3.0
    mean_on = s + background
    mean_off = tau * background

    delta_log_likelihood = (
        n * math.log(n / mean_on)
        + m * math.log(m / mean_off)
        - n
        - m
        + mean_on
        + mean_off
    )
    expected = math.sqrt(2.0 * delta_log_likelihood)

    assert expected == pytest.approx(2.0572333554850197, rel=1e-12)
    assert r_stat_onoff(s, n, m, tau) == pytest.approx(expected, rel=1e-12)


def test_auxiliary_statistic():
    """The auxiliary statistic agrees with its on/off expression."""
    s, n, m, tau = 0.0, 5.0, 2.0, 2.0
    background = 7.0 / 3.0
    mean_on = s + background
    mean_off = tau * background

    numerator = math.sqrt(n * m)
    denominator = math.sqrt(n / mean_on**2 + m / background**2)
    log_terms = (
        math.log(n / mean_on) / background
        - math.log(m / mean_off) / mean_on
    )
    expected = numerator * log_terms / denominator

    assert expected == pytest.approx(1.9236462378886596, rel=1e-12)
    assert u_stat_onoff(s, n, m, tau) == pytest.approx(expected, rel=1e-12)


def test_higher_order_root():
    """The higher-order root applies the logarithmic u/r correction."""
    r_value = 2.0572333554850197
    u_value = 1.9236462378886596
    expected = r_value + math.log(abs(u_value / r_value)) / r_value

    assert expected == pytest.approx(2.024597494641529, rel=1e-12)
    assert r_star_onoff(0.0, 5.0, 2.0, 2.0) == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_sample_space_boundary():
    """At n=0 or m=0, u vanishes and the boundary prescription is r*=r."""
    boundary_counts = [(0.0, 3.0), (3.0, 0.0), (0.0, 0.0)]

    for n, m in boundary_counts:
        r_value = r_stat_onoff(0.0, n, m, 2.0)
        assert u_stat_onoff(0.0, n, m, 2.0) == 0.0
        assert r_star_onoff(0.0, n, m, 2.0) == pytest.approx(
            r_value,
            rel=1e-12,
            abs=1e-15,
        )


def test_mle_limit():
    """At s=s_hat, the exact analytic limit replaces the 0/0 correction."""
    s, n, m, tau = 4.0, 5.0, 2.0, 2.0
    expected = 38.0 / (6.0 * 22.0**1.5)

    assert r_stat_onoff(s, n, m, tau) == pytest.approx(0.0, abs=1e-15)
    assert u_stat_onoff(s, n, m, tau) == pytest.approx(0.0, abs=1e-15)
    assert r_star_onoff(s, n, m, tau) == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_discovery_pvalues():
    """The deterministic reference uses the inclusive discovery tail."""
    result = pvals_onoff_profile_sum(
        0.0,
        5,
        1,
        1.0,
        tail_mass=1e-14,
    )

    assert set(result) == {"p_ref", "p_ref_se", "p_r", "p_rstar"}
    assert result["p_ref"] == pytest.approx(
        0.0647260263755386,
        rel=0.0,
        abs=5e-13,
    )
    assert result["p_ref_se"] == 0.0
    assert result["p_r"] == pytest.approx(
        norm.sf(max(0.0, r_stat_onoff(0.0, 5.0, 1.0, 1.0))),
        rel=1e-12,
    )
    assert result["p_rstar"] == pytest.approx(
        norm.sf(max(0.0, r_star_onoff(0.0, 5.0, 1.0, 1.0))),
        rel=1e-12,
    )

    non_discovery = pvals_onoff_profile_sum(0.0, 1, 2, 1.0)
    assert non_discovery["p_ref"] == 0.5
    assert non_discovery["p_r"] == 0.5
    assert non_discovery["p_rstar"] == 0.5


def test_asimov_significance():
    """The on/off Asimov counts are n_A=s+b and m_A=tau*b."""
    result = asimov_Zs_onoff(2.0, 1.0, 1.0)

    n_asimov = 2.0 + 1.0
    m_asimov = 1.0 * 1.0
    tau = 1.0
    expected_r = math.sqrt(
        2.0
        * (
            n_asimov
            * math.log((1.0 + tau) * n_asimov / (n_asimov + m_asimov))
            + m_asimov
            * math.log(
                (1.0 + tau)
                * m_asimov
                / (tau * (n_asimov + m_asimov))
            )
        )
    )
    expected_u = math.sqrt(
        n_asimov * m_asimov / (n_asimov + m_asimov)
    ) * math.log(n_asimov * tau / m_asimov)
    expected_rstar = expected_r + math.log(abs(expected_u / expected_r)) / expected_r

    assert result["Z_A_r"] == pytest.approx(
        expected_r,
        rel=1e-12,
    )
    assert result["Z_A_rstar"] == pytest.approx(
        expected_rstar,
        rel=1e-12,
    )
    assert expected_r == pytest.approx(1.0229840113751025, rel=1e-12)
    assert expected_rstar == pytest.approx(0.9520962306318638, rel=1e-12)


def test_array_inputs():
    """Array calculations agree with the same points evaluated one by one."""
    n_values = np.array([5.0, 2.0, 1.0])
    m_values = np.array([2.0, 3.0, 4.0])

    background_array = b_profiled(0.0, n_values, m_values, 2.0)
    background_scalar = np.array(
        [b_profiled(0.0, n, m, 2.0) for n, m in zip(n_values, m_values)]
    )
    np.testing.assert_allclose(
        background_array,
        background_scalar,
        rtol=1e-12,
        atol=0.0,
    )

    loglik_array = loglik_diff(0.0, n_values, m_values, 2.0)
    loglik_scalar = np.array(
        [loglik_diff(0.0, n, m, 2.0) for n, m in zip(n_values, m_values)]
    )
    np.testing.assert_allclose(
        loglik_array,
        loglik_scalar,
        rtol=1e-12,
        atol=0.0,
    )

    root_array = r_stat_onoff(0.0, n_values, m_values, 2.0)
    root_scalar = np.array(
        [r_stat_onoff(0.0, n, m, 2.0) for n, m in zip(n_values, m_values)]
    )
    np.testing.assert_allclose(
        root_array,
        root_scalar,
        rtol=1e-12,
        atol=1e-15,
    )


def test_required_toys_for_z_precision():
    """The toy count follows binomial error propagation in significance."""
    z_value = 2.0
    p_value = norm.sf(z_value)
    phi = norm.pdf(z_value)
    expected = math.ceil(
        p_value * (1.0 - p_value)
        / (0.05**2 * z_value**2 * phi**2)
    )

    assert expected == 763
    assert required_toys_for_Z_precision(
        z_value,
        sigrel=0.05,
        min_toys=100,
        max_toys=2_000_000,
    ) == (763, False)

    # A non-positive discovery root uses the requested minimum.
    assert required_toys_for_Z_precision(
        0.0,
        sigrel=0.05,
        min_toys=100,
        max_toys=2_000_000,
    ) == (100, False)

    # The minimum can exceed the calculated requirement without limiting precision.
    assert required_toys_for_Z_precision(
        1.0,
        sigrel=0.05,
        min_toys=1_000,
        max_toys=2_000_000,
    ) == (1_000, False)

    # The maximum reports when the requested precision cannot be reached.
    assert required_toys_for_Z_precision(
        5.0,
        sigrel=0.05,
        min_toys=100,
        max_toys=1_000,
    ) == (1_000, True)


def test_monte_carlo_pvalue_with_controlled_toys(monkeypatch):
    """The Monte Carlo result reports K, N and the finite-sample correction."""

    def one_exceedance(s, b, tau, n_toys, seed):
        assert (s, b, tau, n_toys, seed) == (0.0, 2.0, 1.0, 4, 12345)
        return np.array([3, 2, 1, 0]), np.array([1, 1, 1, 1])

    monkeypatch.setattr(on_off, "_sample_null_toys", one_exceedance)
    result = pvals_onoff(
        0.0,
        3,
        1,
        1.0,
        sigrel=0.05,
        min_toys=4,
        max_toys=4,
        seed=12345,
    )

    # With K=1 and N=4, p_raw=K/N and p=(K+1)/(N+1).
    corrected_p = (1.0 + 1.0) / (4.0 + 1.0)
    assert result["n_exceedances"] == 1
    assert result["n_toys"] == 4
    assert result["p_mc_raw"] == pytest.approx(1.0 / 4.0)
    assert result["p_mc"] == pytest.approx(corrected_p)
    assert result["p_resolution"] == pytest.approx(1.0 / 5.0)
    assert result["p_mc_se"] == pytest.approx(
        math.sqrt(corrected_p * (1.0 - corrected_p) / 4.0),
        rel=1e-12,
    )
    assert result["b_profiled"] == pytest.approx(2.0, rel=1e-12)
    assert result["precision_limited"] is True

    def no_exceedances(s, b, tau, n_toys, seed):
        return np.array([2, 1, 0, 0]), np.array([1, 1, 1, 1])

    monkeypatch.setattr(on_off, "_sample_null_toys", no_exceedances)
    zero_result = pvals_onoff(
        0.0,
        3,
        1,
        1.0,
        sigrel=0.05,
        min_toys=4,
        max_toys=4,
        seed=12345,
    )

    assert zero_result["n_exceedances"] == 0
    assert zero_result["p_mc_raw"] == 0.0
    assert zero_result["p_mc"] == pytest.approx(1.0 / 5.0)
    assert zero_result["p_resolution"] == pytest.approx(1.0 / 5.0)
