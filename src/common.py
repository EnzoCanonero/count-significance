from pathlib import Path
import math
import yaml


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def norm_survival(x: float) -> float:
    """One-sided upper tail 1 - Phi(x)."""
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


__all__ = ["load_yaml", "ensure_dir", "norm_survival"]
