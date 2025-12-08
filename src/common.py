from pathlib import Path
import numpy as np
from scipy.special import erfc
import yaml


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def norm_survival(x):
    """
    One-sided upper tail 1 - Phi(x), accepts scalar or array-like.
    """
    arr = np.asarray(x, dtype=float)
    return 0.5 * erfc(arr / np.sqrt(2.0))


__all__ = ["load_yaml", "ensure_dir", "norm_survival"]
