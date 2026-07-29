#!/usr/bin/env python3
"""Run one HTCondor worker for the on/off median-significance scan."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.on_off import pvals_onoff  # noqa: E402


RESULT_FIELDS = (
    "p_mc",
    "p_mc_raw",
    "p_mc_se",
    "p_resolution",
    "n_toys",
    "n_exceedances",
    "b_profiled",
    "precision_limited",
)
SCHEMA_VERSION = 1


# Read the frozen campaign inputs and this worker's job identifiers.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one batch job for the on/off median-significance scan."
    )
    parser.add_argument("--config", required=True, help="Shared paper YAML config")
    parser.add_argument("--signal-index", required=True, type=int)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--run", required=True, help="Name of this production run")
    parser.add_argument("--commit", required=True, help="Git commit used for the run")
    parser.add_argument("--output", default="result.json", help="Output JSON path")
    return parser.parse_args()


# Read the frozen YAML configuration and hash its exact contents.
def load_config(config_path):
    raw_config = config_path.read_bytes()
    config = yaml.safe_load(raw_config.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The YAML config must contain a mapping")

    config_hash = hashlib.sha256(raw_config).hexdigest()
    return config, config_hash


# Parse and validate a positive integer setting.
def positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
        original = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number <= 0 or not math.isfinite(original) or number != original:
        raise ValueError(f"{name} must be a positive integer")
    return number


# Read a non-empty vector of finite configuration values.
def finite_vector(config, name, positive=False):
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain numbers")

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers") from error
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return array


# Parse and validate one finite numeric setting.
def finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


# Hash the frozen statistical source and this worker.
def frozen_source_sha256():
    source_files = [
        path
        for path in (ROOT / "src").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    source_files.append(Path(__file__).resolve())
    source_files.sort(key=lambda path: path.relative_to(ROOT).as_posix())

    # Include file names so that renaming a source file changes the checksum.
    digest = hashlib.sha256()
    for path in source_files:
        relative_path = path.relative_to(ROOT).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
    return digest.hexdigest()


# Validate the selected signal partition and return its worker settings.
def read_job_settings(config, signal_index, job_id):
    if "batch_mc" not in config or not isinstance(config["batch_mc"], dict):
        raise ValueError("The config must contain a batch_mc mapping")

    batch = config["batch_mc"]
    signal_jobs = batch.get("signal_jobs")
    if not isinstance(signal_jobs, list):
        raise ValueError("batch_mc.signal_jobs must be a list")

    s_vec = finite_vector(config, "s_vec")
    if np.any(s_vec < 0.0):
        raise ValueError("s_vec values must be non-negative")
    if len(signal_jobs) != len(s_vec):
        raise ValueError("batch_mc.signal_jobs must have one entry for each value in s_vec")
    if signal_index < 0 or signal_index >= len(signal_jobs):
        raise ValueError(f"signal-index must be between 0 and {len(signal_jobs) - 1}")

    settings = signal_jobs[signal_index]
    if not isinstance(settings, dict):
        raise ValueError("Each entry in batch_mc.signal_jobs must be a mapping")

    s_true = finite_number(settings.get("s_true"), "s_true")
    if s_true < 0.0:
        raise ValueError("s_true must be non-negative")
    if s_true != float(s_vec[signal_index]):
        raise ValueError("The selected signal job does not match s_vec")

    n_outer = positive_integer(batch["n_outer"], "batch_mc.n_outer")
    n_jobs = positive_integer(settings["n_jobs"], "signal job n_jobs")
    outer_per_job = positive_integer(
        settings["outer_per_job"],
        "signal job outer_per_job",
    )
    if n_jobs * outer_per_job != n_outer:
        raise ValueError(
            "For each signal, n_jobs * outer_per_job must equal batch_mc.n_outer"
        )
    if job_id < 0 or job_id >= n_jobs:
        raise ValueError(f"job-id must be between 0 and {n_jobs - 1}")

    n_bpts = positive_integer(batch["n_bpts"], "batch_mc.n_bpts")
    min_toys = positive_integer(batch["min_toys"], "batch_mc.min_toys")
    max_toys = positive_integer(settings["max_toys"], "signal job max_toys")
    mc_sigrel_z = finite_number(batch.get("mc_sigrel_Z"), "batch_mc.mc_sigrel_Z")
    if mc_sigrel_z <= 0.0:
        raise ValueError("batch_mc.mc_sigrel_Z must be positive")
    if max_toys < min_toys:
        raise ValueError("Toy limits must satisfy 0 < min_toys <= max_toys")

    return {
        "s_true": s_true,
        "outer_per_job": outer_per_job,
        "n_bpts": n_bpts,
        "mc_sigrel_z": mc_sigrel_z,
        "min_toys": min_toys,
        "max_toys": max_toys,
        "outer_seed": positive_integer(
            settings["outer_seed"],
            "signal job outer_seed",
        ),
        "inner_seed": positive_integer(
            settings["inner_seed"],
            "signal job inner_seed",
        ),
    }


def compute_points(config, settings, job_id):
    """Generate this job's pseudo-experiments on the two scan grids.

    Counts follow n ~ Pois(s + b) and m ~ Pois(tau b). For fixed relative
    uncertainty delta, tau = 1/(delta^2 b). Separate random streams generate
    the observed counts and the seeds used by the inner Monte Carlo.
    """
    tau_vec = finite_vector(config, "tau_vec", positive=True)
    rel_sig_vec = finite_vector(config, "rel_sig_vec", positive=True)
    b_min = finite_number(config.get("b_min"), "b_min")
    b_max = finite_number(config.get("b_max"), "b_max")
    if b_min <= 0.0 or b_max <= b_min:
        raise ValueError("The scan limits must satisfy 0 < b_min < b_max")

    b_values = np.logspace(
        np.log10(b_min),
        np.log10(b_max),
        settings["n_bpts"],
    )

    outer_stream_seed = settings["outer_seed"] + job_id
    inner_stream_seed = settings["inner_seed"] + job_id
    rng_outer = np.random.default_rng(outer_stream_seed)
    rng_inner = np.random.default_rng(inner_stream_seed)

    points = []
    s_true = settings["s_true"]

    for replica in range(settings["outer_per_job"]):
        # First scan the two fixed-tau curves.
        for param_idx, tau in enumerate(tau_vec):
            for b_idx, b in enumerate(b_values):
                n_obs = int(rng_outer.poisson(s_true + b))
                m_obs = int(rng_outer.poisson(tau * b))
                inner_seed = int(rng_inner.integers(1, 2**31 - 1))

                result = pvals_onoff(
                    s=0.0,
                    tau=float(tau),
                    n=n_obs,
                    m=m_obs,
                    sigrel=settings["mc_sigrel_z"],
                    min_toys=settings["min_toys"],
                    max_toys=settings["max_toys"],
                    seed=inner_seed,
                )

                point = {
                    "replica": replica,
                    "scan": "fixed_tau",
                    "param_idx": param_idx,
                    "b_idx": b_idx,
                    "b": float(b),
                    "tau": float(tau),
                    "n_obs": n_obs,
                    "m_obs": m_obs,
                    "inner_seed": inner_seed,
                }
                for name in RESULT_FIELDS:
                    point[name] = result[name]
                points.append(point)

        # Then scan the two fixed-relative-uncertainty curves.
        for param_idx, rel_sig in enumerate(rel_sig_vec):
            for b_idx, b in enumerate(b_values):
                tau = 1.0 / (float(rel_sig) ** 2 * float(b))
                n_obs = int(rng_outer.poisson(s_true + b))
                m_obs = int(rng_outer.poisson(tau * b))
                inner_seed = int(rng_inner.integers(1, 2**31 - 1))

                result = pvals_onoff(
                    s=0.0,
                    tau=tau,
                    n=n_obs,
                    m=m_obs,
                    sigrel=settings["mc_sigrel_z"],
                    min_toys=settings["min_toys"],
                    max_toys=settings["max_toys"],
                    seed=inner_seed,
                )

                point = {
                    "replica": replica,
                    "scan": "fixed_rel_sig",
                    "param_idx": param_idx,
                    "b_idx": b_idx,
                    "b": float(b),
                    "tau": tau,
                    "rel_sig": float(rel_sig),
                    "n_obs": n_obs,
                    "m_obs": m_obs,
                    "inner_seed": inner_seed,
                }
                for name in RESULT_FIELDS:
                    point[name] = result[name]
                points.append(point)

    return points


# Write the result as JSON, replacing the target only after a complete write.
def write_output(output_path, data):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2)
            output_file.write("\n")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# Run one worker and save its provenance together with all scan points.
def main():
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)

    config, config_hash = load_config(config_path)
    settings = read_job_settings(config, args.signal_index, args.job_id)
    source_hash = frozen_source_sha256()
    points = compute_points(
        config,
        settings,
        args.job_id,
    )

    output = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "run": args.run,
            "commit": args.commit,
            "config_sha256": config_hash,
            "source_sha256": source_hash,
            "signal_index": args.signal_index,
            "s_true": settings["s_true"],
            "job_id": args.job_id,
        },
        "points": points,
    }
    write_output(output_path, output)
    print(f"Saved {len(points)} points to: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
