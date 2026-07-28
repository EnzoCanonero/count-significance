import numpy as np
from scipy.special import erfc
import yaml


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def norm_survival(x):
    """
    One-sided upper tail 1 - Phi(x), accepts scalar or array-like.
    """
    arr = np.asarray(x, dtype=float)
    return 0.5 * erfc(arr / np.sqrt(2.0))


def discovery_z(r):
    """Discovery significance Z = max(0, r)."""
    return np.maximum(r, 0.0)


def discovery_q0(r):
    """Discovery test statistic q0 = Z^2, with Z = max(0, r)."""
    z = discovery_z(r)
    return z**2


def discovery_pvalue(r):
    """One-sided discovery p-value p0 = 1 - Phi(Z)."""
    z = discovery_z(r)
    return norm_survival(z)


__all__ = [
    "load_yaml",
    "norm_survival",
    "discovery_z",
    "discovery_q0",
    "discovery_pvalue",
]
