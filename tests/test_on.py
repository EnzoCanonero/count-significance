"""Core-statistics tests for a Poisson count with known background."""

import math

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import norm

from count_significance.on import (
    expected_significance_on,
    poisson_tail_on,
    pvals_on,
    r_star_on,
    r_stat_on,
    u_stat_on,
)


def test_reference_calculation():
    """The reference is the inclusive Poisson tail P(N >= n | b)."""
    p_tail = poisson_tail_on(s0=0.0, b=0.1, n=10)

    assert p_tail == pytest.approx(
        2.5163478067703154e-17,
        rel=1e-13,
        abs=0.0,
    )
    assert norm.isf(p_tail) == pytest.approx(
        8.38592841395279,
        rel=1e-12,
    )


def test_signed_likelihood_root():
    """Check r(0) directly from the signed likelihood-root equation."""
    # r(0) = sqrt(2 [n ln(n / b) - (n - b)]) for n > b.
    expected_r = math.sqrt(2.0 * (4.0 * math.log(4.0) - 3.0))

    assert r_stat_on(0.0, 1.0, 4.0) == pytest.approx(
        expected_r,
        rel=1e-12,
    )
    assert expected_r == pytest.approx(2.2561814840475765, rel=1e-12)


def test_auxiliary_statistic():
    """Check u(0)=sqrt(n) ln(n/b) directly."""
    expected_u = math.sqrt(4.0) * math.log(4.0)

    assert u_stat_on(0.0, 1.0, 4.0) == pytest.approx(
        expected_u,
        rel=1e-12,
    )
    assert expected_u == pytest.approx(2.772588722239781, rel=1e-12)


def test_higher_order_root():
    """Check r*=r+log|u/r|/r, with and without continuity correction."""
    r = math.sqrt(2.0 * (4.0 * math.log(4.0) - 3.0))
    u = math.sqrt(4.0) * math.log(4.0)
    expected_rstar = r + math.log(abs(u / r)) / r

    assert r_star_on(0.0, 1.0, 4.0, continuity_correction=False) == pytest.approx(
        expected_rstar,
        rel=1e-12,
    )
    assert expected_rstar == pytest.approx(2.347533915817222, rel=1e-12)

    # With the correction, the same equations use n_eff = n - 1/2.
    n_eff = 4.0 - 0.5
    r_corrected = math.sqrt(
        2.0 * (n_eff * math.log(n_eff / 1.0) - (n_eff - 1.0))
    )
    u_corrected = math.sqrt(n_eff) * math.log(n_eff / 1.0)
    expected_corrected = (
        r_corrected + math.log(abs(u_corrected / r_corrected)) / r_corrected
    )

    assert r_star_on(0.0, 1.0, 4.0) == pytest.approx(
        expected_corrected,
        rel=1e-12,
    )


def test_sample_space_boundary():
    """At n=0 the logarithmic adjustment is undefined, so u=0 and r*=r."""
    expected_r = -math.sqrt(2.0 * 1.0)

    assert u_stat_on(0.0, 1.0, 0.0) == 0.0
    assert r_stat_on(0.0, 1.0, 0.0) == pytest.approx(expected_r, rel=1e-12)
    assert r_star_on(0.0, 1.0, 0.0) == pytest.approx(expected_r, rel=1e-12)


def test_mle_limit():
    """At n=mu0 both r and u vanish and r*=1/(6 sqrt(mu0))."""
    mu0 = 4.0
    expected_rstar = 1.0 / (6.0 * math.sqrt(mu0))

    assert r_stat_on(0.0, mu0, mu0) == 0.0
    assert u_stat_on(0.0, mu0, mu0) == 0.0
    assert r_star_on(
        0.0,
        mu0,
        mu0,
        continuity_correction=False,
    ) == pytest.approx(expected_rstar, rel=1e-12)


def test_discovery_pvalues():
    """The returned p-values use the discovery cap without clipping rare tails."""
    counts = np.array([0.0, 1.0, 4.0])
    pvalues = pvals_on(0.0, 1.0, counts)

    assert set(pvalues) == {"p_exact", "p_r", "p_rstar"}
    for values in pvalues.values():
        assert np.all(values <= 0.5)

    assert_allclose(pvalues["p_exact"][:2], [0.5, 0.5], atol=0.0)
    assert_allclose(pvalues["p_r"][:2], [0.5, 0.5], atol=0.0)
    assert_allclose(pvalues["p_rstar"][:2], [0.5, 0.5], atol=0.0)

    rare_tail = pvals_on(0.0, 0.1, 10.0)["p_exact"]
    assert rare_tail == pytest.approx(
        2.5163478067703154e-17,
        rel=1e-13,
        abs=0.0,
    )


def test_asimov_significance():
    """Check the first-order and corrected Asimov significances."""
    result = expected_significance_on(2.0, 1.0, n_outer=1, seed=1)

    # First order uses n_A=s+b=3, while corrected r* uses n_eff=3-1/2.
    expected_r = math.sqrt(2.0 * (3.0 * math.log(3.0) - 2.0))
    n_eff = 3.0 - 0.5
    r_corrected = math.sqrt(
        2.0 * (n_eff * math.log(n_eff / 1.0) - (n_eff - 1.0))
    )
    u_corrected = math.sqrt(n_eff) * math.log(n_eff / 1.0)
    expected_rstar = (
        r_corrected + math.log(abs(u_corrected / r_corrected)) / r_corrected
    )

    assert_allclose(result["Z_A_r"], expected_r, rtol=1e-12)
    assert_allclose(result["Z_A_rstar"], expected_rstar, rtol=1e-12)
    assert expected_r == pytest.approx(1.6098676131932894, rel=1e-12)
    assert expected_rstar == pytest.approx(1.3701192199099828, rel=1e-12)


def test_array_inputs():
    """Vector p-values agree with element-by-element scalar evaluations."""
    counts = np.array([0.0, 1.0, 4.0, 10.0])
    vector_result = pvals_on(0.0, 1.0, counts)

    for name in ("p_exact", "p_r", "p_rstar"):
        scalar_result = np.array(
            [np.asarray(pvals_on(0.0, 1.0, n)[name]).item() for n in counts]
        )
        assert_allclose(vector_result[name], scalar_result, rtol=1e-12)
