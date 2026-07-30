"""Tests for the shared discovery-statistic definitions."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from count_significance.common import (
    discovery_pvalue,
    discovery_q0,
    discovery_z,
    norm_survival,
)


def test_norm_survival_matches_scipy():
    """The Gaussian upper tail agrees with scipy.stats.norm.sf."""
    z_values = np.array([-2.0, 0.0, 1.0, 5.0])

    assert_allclose(norm_survival(z_values), norm.sf(z_values), rtol=1e-12)
    assert_allclose(norm_survival(2.0), norm.sf(2.0), rtol=1e-12)


def test_discovery_definitions():
    """The one-sided discovery convention is Z=max(0,r), q0=Z^2."""
    roots = np.array([-2.0, 0.0, 3.0])
    expected_z = np.array([0.0, 0.0, 3.0])
    expected_q0 = expected_z**2
    expected_p = norm.sf(expected_z)

    assert_allclose(discovery_z(roots), expected_z, rtol=1e-12)
    assert_allclose(discovery_q0(roots), expected_q0, rtol=1e-12)
    assert_allclose(discovery_pvalue(roots), expected_p, rtol=1e-12)

    # Scalar inputs follow the same equations; their exact return type is irrelevant.
    assert_allclose(discovery_z(-2.0), 0.0, atol=0.0)
    assert_allclose(discovery_q0(3.0), 9.0, rtol=1e-12)
    assert_allclose(discovery_pvalue(3.0), norm.sf(3.0), rtol=1e-12)
