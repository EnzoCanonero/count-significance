"""Shared configuration and discovery-significance utilities."""

from typing import Any, Union

import numpy as np
from scipy.special import erfc
import yaml


ScalarOrArray = Union[float, np.ndarray]


# Load a YAML configuration file.
def load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def norm_survival(x: ScalarOrArray) -> ScalarOrArray:
    """Return the one-sided Gaussian upper tail, 1 - Phi(x)."""
    arr = np.asarray(x, dtype=float)
    return 0.5 * erfc(arr / np.sqrt(2.0))


def discovery_z(r: ScalarOrArray) -> ScalarOrArray:
    """Apply the discovery convention Z = max(0, r)."""
    return np.maximum(r, 0.0)


def discovery_q0(r: ScalarOrArray) -> ScalarOrArray:
    """Return q0 = Z^2, where Z = max(0, r)."""
    z = discovery_z(r)
    return z**2


def discovery_pvalue(r: ScalarOrArray) -> ScalarOrArray:
    """Return the one-sided discovery p-value p0 = 1 - Phi(Z)."""
    z = discovery_z(r)
    return norm_survival(z)


__all__ = [
    "load_yaml",
    "norm_survival",
    "discovery_z",
    "discovery_q0",
    "discovery_pvalue",
]
