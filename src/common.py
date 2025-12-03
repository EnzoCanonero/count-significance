from pathlib import Path
import yaml


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["load_yaml", "ensure_dir"]
