#!/usr/bin/env python3
"""Validate an HTCondor on/off scan and write its median-significance plots."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from count_significance.on_off import (  # noqa: E402
    asimov_Zs_onoff,
    b_profiled,
    r_stat_onoff,
    required_toys_for_Z_precision,
)
from scripts.make_onoff_medsig_plots import (  # noqa: E402
    configure_plot_style,
    mask_mc_for_display,
    write_median_significance_pdfs,
)

SCHEMA_VERSION = 1
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
WORKER_NAME = "run_onoff_medsig_job.py"


# Read the campaign name to collect from the runs directory.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one complete HTCondor median-significance run."
    )
    parser.add_argument("--run", required=True, help="Run name under runs/")
    return parser.parse_args()


# Return the SHA-256 checksum of one file.
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Hash the frozen statistical source and worker stored with the campaign.
def frozen_source_sha256(input_dir: Path) -> str:
    package_dir = input_dir / "src" / "count_significance"
    source_files = [
        path
        for path in package_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    source_files.append(input_dir / "scripts" / WORKER_NAME)
    source_files.sort(key=lambda path: path.relative_to(input_dir).as_posix())

    # Include file names so that renaming a source file changes the checksum.
    digest = hashlib.sha256()
    for path in source_files:
        relative_path = path.relative_to(input_dir).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
    return digest.hexdigest()


# Require a clean repository and return its current commit.
def current_commit() -> str:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status_result.stdout.strip():
        raise ValueError("The repository must be clean before collecting a production run")
    return commit_result.stdout.strip()


# Read a JSON file and report malformed input as a validation error.
def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}: {error}") from error


# Read the frozen YAML campaign configuration.
def load_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read valid YAML from {path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError("The run config must contain a mapping")
    return config


# Convert a signal value into the directory tag used by the submitter.
def signal_tag(s_true: float) -> str:
    value = (
        f"{float(s_true):g}"
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
    )
    return f"s{value}"


# Parse and validate a positive integer setting.
def positive_integer(value: Any, name: str) -> int:
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


# Parse and validate one finite numeric setting.
def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


# Read a non-empty vector of finite configuration values.
def finite_vector(
    config: dict[str, Any],
    name: str,
    positive: bool = False,
) -> np.ndarray:
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty YAML list")
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


# Validate the shared scan definition and return its batch settings.
def read_batch_settings(config: dict[str, Any]) -> dict[str, Any]:
    batch = config.get("batch_mc")
    if not isinstance(batch, dict):
        raise ValueError("The config must contain a batch_mc mapping")

    s_vec = finite_vector(config, "s_vec")
    tau_vec = finite_vector(config, "tau_vec", positive=True)
    rel_sig_vec = finite_vector(config, "rel_sig_vec", positive=True)
    if np.any(s_vec < 0.0):
        raise ValueError("s_vec values must be non-negative")

    b_min = finite_number(config.get("b_min"), "b_min")
    b_max = finite_number(config.get("b_max"), "b_max")
    n_bpts = positive_integer(batch["n_bpts"], "batch_mc.n_bpts")
    n_outer = positive_integer(batch["n_outer"], "batch_mc.n_outer")
    min_toys = positive_integer(batch["min_toys"], "batch_mc.min_toys")
    mc_sigrel_z = finite_number(batch.get("mc_sigrel_Z"), "batch_mc.mc_sigrel_Z")
    if b_min <= 0.0 or b_max <= b_min:
        raise ValueError("The batch scan must satisfy 0 < b_min < b_max and n_bpts > 0")
    if not math.isfinite(mc_sigrel_z) or mc_sigrel_z <= 0.0:
        raise ValueError("batch_mc.mc_sigrel_Z must be positive")

    signal_jobs = batch.get("signal_jobs")
    if not isinstance(signal_jobs, list) or len(signal_jobs) != len(s_vec):
        raise ValueError("batch_mc.signal_jobs must match s_vec")

    jobs = []
    tags = set()
    for signal_index, (s_true, job) in enumerate(zip(s_vec, signal_jobs)):
        if not isinstance(job, dict):
            raise ValueError("Each signal job must be a mapping")
        job_s_true = finite_number(
            job.get("s_true"),
            f"signal_jobs[{signal_index}].s_true",
        )
        if job_s_true != float(s_true):
            raise ValueError("The order and values in signal_jobs must match s_vec")

        n_jobs = positive_integer(job["n_jobs"], f"signal_jobs[{signal_index}].n_jobs")
        outer_per_job = positive_integer(
            job["outer_per_job"],
            f"signal_jobs[{signal_index}].outer_per_job",
        )
        max_toys = positive_integer(
            job["max_toys"],
            f"signal_jobs[{signal_index}].max_toys",
        )
        outer_seed = positive_integer(
            job["outer_seed"],
            f"signal_jobs[{signal_index}].outer_seed",
        )
        inner_seed = positive_integer(
            job["inner_seed"],
            f"signal_jobs[{signal_index}].inner_seed",
        )
        if n_jobs * outer_per_job != n_outer:
            raise ValueError("n_jobs * outer_per_job must equal batch_mc.n_outer")
        if max_toys < min_toys:
            raise ValueError("Each max_toys value must be at least batch_mc.min_toys")

        tag = signal_tag(s_true)
        if tag in tags:
            raise ValueError(f"Duplicate signal tag: {tag}")
        tags.add(tag)

        jobs.append(
            {
                "signal_index": signal_index,
                "s_true": float(s_true),
                "signal_tag": tag,
                "n_jobs": n_jobs,
                "outer_per_job": outer_per_job,
                "max_toys": max_toys,
                "outer_seed": outer_seed,
                "inner_seed": inner_seed,
            }
        )

    b_values = np.logspace(np.log10(b_min), np.log10(b_max), n_bpts)
    return {
        "s_vec": s_vec,
        "tau_vec": tau_vec,
        "rel_sig_vec": rel_sig_vec,
        "b_values": b_values,
        "n_outer": n_outer,
        "min_toys": min_toys,
        "mc_sigrel_z": mc_sigrel_z,
        "jobs": jobs,
    }


# Match the manifest against the frozen inputs and current repository state.
def validate_manifest(
    manifest: Any,
    run_name: str,
    config_hash: str,
    source_hash: str,
    settings: dict[str, Any],
) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("The run manifest must contain a mapping")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The run manifest has an unsupported schema version")
    if manifest.get("run") != run_name:
        raise ValueError("The run manifest name does not match --run")
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("The run configuration does not match its manifest")
    if manifest.get("source_sha256") != source_hash:
        raise ValueError("The frozen worker source does not match its manifest")
    if manifest.get("config_path") != "config.yaml":
        raise ValueError("The run manifest must reference its frozen config.yaml")
    if manifest.get("input_path") != "input":
        raise ValueError("The run manifest must reference its frozen input snapshot")
    if manifest.get("commit") != current_commit():
        raise ValueError("The current Git commit does not match the submitted run")

    manifest_signals = manifest.get("signals")
    if not isinstance(manifest_signals, list) or len(manifest_signals) != len(
        settings["jobs"]
    ):
        raise ValueError("The run manifest has the wrong signal list")

    for expected, actual in zip(settings["jobs"], manifest_signals):
        if not isinstance(actual, dict):
            raise ValueError("Each manifest signal must be a mapping")
        for key in (
            "signal_index",
            "s_true",
            "signal_tag",
            "n_jobs",
            "outer_per_job",
        ):
            if actual.get(key) != expected[key]:
                raise ValueError(f"Manifest mismatch for signal field {key}")


# Compare stored floating-point values at validation precision.
def close_enough(actual: Any, expected: Any) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)


# Read a field that must be an integer rather than a boolean.
def require_integer(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


# Read a field that must contain a finite number.
def require_finite(record: dict[str, Any], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def validate_mc_result(
    point: dict[str, Any],
    job: dict[str, Any],
    r_obs: float,
    expected_b_profiled: float,
) -> tuple[float, int, int, bool]:
    """Validate the Monte Carlo p-value and return its significance.

    For K exceedances in N toys, the stored estimates are K/N and the corrected
    p = (K + 1)/(N + 1), with resolution 1/(N + 1). The reported discovery
    p-value is capped at 0.5, and is set to 0.5 for a non-positive signed root.
    """
    n_toys = require_integer(point, "n_toys")
    n_exceedances = require_integer(point, "n_exceedances")
    if n_toys < job["min_toys"] or n_toys > job["max_toys"]:
        raise ValueError("n_toys is outside the configured limits")
    if n_exceedances < 0 or n_exceedances > n_toys:
        raise ValueError("n_exceedances must satisfy 0 <= K <= N")

    p_mc = require_finite(point, "p_mc")
    p_mc_raw = require_finite(point, "p_mc_raw")
    p_mc_se = require_finite(point, "p_mc_se")
    p_resolution = require_finite(point, "p_resolution")
    observed_b_profiled = require_finite(point, "b_profiled")

    if not 0.0 < p_mc <= 0.5:
        raise ValueError("p_mc must be in the discovery range (0, 0.5]")
    if not 0.0 <= p_mc_raw <= 1.0:
        raise ValueError("p_mc_raw must be between zero and one")
    if p_mc_se < 0.0 or observed_b_profiled < 0.0:
        raise ValueError("MC uncertainty and profiled background must be non-negative")

    expected_n_toys, expected_precision_limited = required_toys_for_Z_precision(
        r_obs,
        sigrel=job["mc_sigrel_z"],
        min_toys=job["min_toys"],
        max_toys=job["max_toys"],
    )
    if n_toys != expected_n_toys:
        raise ValueError("n_toys does not match the configured Z precision")
    if not close_enough(observed_b_profiled, expected_b_profiled):
        raise ValueError("b_profiled is inconsistent with the observed counts")

    corrected_p = (n_exceedances + 1.0) / (n_toys + 1.0)
    expected_raw = n_exceedances / n_toys
    expected_resolution = 1.0 / (n_toys + 1.0)
    expected_se = math.sqrt(corrected_p * (1.0 - corrected_p) / n_toys)
    expected_p = 0.5 if r_obs <= 0.0 else min(corrected_p, 0.5)

    if not close_enough(p_mc_raw, expected_raw):
        raise ValueError("p_mc_raw is inconsistent with K/N")
    if not close_enough(p_resolution, expected_resolution):
        raise ValueError("p_resolution is inconsistent with N")
    if not close_enough(p_mc_se, expected_se):
        raise ValueError("p_mc_se is inconsistent with the corrected p-value")
    if not close_enough(p_mc, expected_p):
        raise ValueError("p_mc is inconsistent with K and N")

    precision_limited = point.get("precision_limited")
    if not isinstance(precision_limited, bool):
        raise ValueError("precision_limited must be a boolean")
    if precision_limited != expected_precision_limited:
        raise ValueError("precision_limited does not match the configured Z precision")

    z_value = float(norm.isf(p_mc))
    if not math.isfinite(z_value) or z_value < 0.0:
        raise ValueError("p_mc does not give a finite discovery significance")

    return z_value, n_toys, n_exceedances, precision_limited


# Validate one scan point against its grid coordinates and replayed counts.
def validate_point(
    point: dict[str, Any],
    expected: dict[tuple[int, str, int, int], dict[str, Any]],
    job: dict[str, Any],
) -> tuple[tuple[int, str, int, int], tuple[float, int, int, bool]]:
    replica = require_integer(point, "replica")
    param_idx = require_integer(point, "param_idx")
    b_idx = require_integer(point, "b_idx")
    n_obs = require_integer(point, "n_obs")
    m_obs = require_integer(point, "m_obs")
    inner_seed = require_integer(point, "inner_seed")
    if n_obs < 0 or m_obs < 0 or inner_seed <= 0:
        raise ValueError("Observed counts must be non-negative and inner_seed positive")

    identity = (replica, point.get("scan"), param_idx, b_idx)
    if identity not in expected:
        raise ValueError(f"Unexpected scan point {identity}")

    physical = expected[identity]
    if not close_enough(point["b"], physical["b"]):
        raise ValueError(f"Incorrect b value at scan point {identity}")
    if not close_enough(point["tau"], physical["tau"]):
        raise ValueError(f"Incorrect tau value at scan point {identity}")

    if point["scan"] == "fixed_tau":
        if point.get("rel_sig") is not None:
            raise ValueError("A fixed-tau point must not define rel_sig")
    elif not close_enough(point["rel_sig"], physical["rel_sig"]):
        raise ValueError(f"Incorrect rel_sig value at scan point {identity}")

    if n_obs != physical["n_obs"] or m_obs != physical["m_obs"]:
        raise ValueError(f"Observed counts do not match the job RNG at {identity}")
    if inner_seed != physical["inner_seed"]:
        raise ValueError(f"inner_seed does not match the job RNG at {identity}")

    r_obs = float(r_stat_onoff(0.0, n_obs, m_obs, point["tau"]))
    expected_b_profiled = float(b_profiled(0.0, n_obs, m_obs, point["tau"]))
    return identity, validate_mc_result(point, job, r_obs, expected_b_profiled)


def expected_points_for_job(
    settings: dict[str, Any],
    job: dict[str, Any],
    job_id: int,
) -> dict[tuple[int, str, int, int], dict[str, Any]]:
    """Replay the worker's pseudo-experiments to validate counts and scan order.

    The replay uses n ~ Pois(s + b), m ~ Pois(tau b), and
    tau = 1/(delta^2 b) for fixed relative uncertainty. Independent random
    streams reproduce the observed counts and inner Monte Carlo seeds.
    """
    expected = {}
    rng_outer = np.random.default_rng(job["outer_seed"] + job_id)
    rng_inner = np.random.default_rng(job["inner_seed"] + job_id)

    for replica in range(job["outer_per_job"]):
        for param_idx, tau in enumerate(settings["tau_vec"]):
            for b_idx, b in enumerate(settings["b_values"]):
                n_obs = int(rng_outer.poisson(job["s_true"] + b))
                m_obs = int(rng_outer.poisson(tau * b))
                inner_seed = int(rng_inner.integers(1, 2**31 - 1))
                expected[(replica, "fixed_tau", param_idx, b_idx)] = {
                    "b": float(b),
                    "tau": float(tau),
                    "n_obs": n_obs,
                    "m_obs": m_obs,
                    "inner_seed": inner_seed,
                }

        for param_idx, rel_sig in enumerate(settings["rel_sig_vec"]):
            for b_idx, b in enumerate(settings["b_values"]):
                tau = 1.0 / (float(rel_sig) ** 2 * float(b))
                n_obs = int(rng_outer.poisson(job["s_true"] + b))
                m_obs = int(rng_outer.poisson(tau * b))
                inner_seed = int(rng_inner.integers(1, 2**31 - 1))
                expected[(replica, "fixed_rel_sig", param_idx, b_idx)] = {
                    "b": float(b),
                    "tau": tau,
                    "rel_sig": float(rel_sig),
                    "n_obs": n_obs,
                    "m_obs": m_obs,
                    "inner_seed": inner_seed,
                }
    return expected


# Match a result file to the submitted code, config, signal and job.
def validate_provenance(
    provenance: dict[str, Any],
    run_name: str,
    commit: str,
    config_hash: str,
    source_hash: str,
    job: dict[str, Any],
    job_id: int,
) -> None:
    expected = {
        "run": run_name,
        "commit": commit,
        "config_sha256": config_hash,
        "source_sha256": source_hash,
        "signal_index": job["signal_index"],
        "s_true": job["s_true"],
        "job_id": job_id,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Result provenance mismatch for {key}")


# Validate one complete worker result and return its significance samples.
def validate_result_file(
    result_path: Path,
    run_name: str,
    commit: str,
    config_hash: str,
    source_hash: str,
    settings: dict[str, Any],
    job: dict[str, Any],
    job_id: int,
) -> list[tuple[str, int, int, float, int, int, bool]]:
    expected_points = expected_points_for_job(settings, job, job_id)
    result = load_json(result_path)
    if not isinstance(result, dict):
        raise ValueError(f"The result in {result_path} must contain a mapping")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema in {result_path}")

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Missing provenance in {result_path}")
    validate_provenance(
        provenance,
        run_name,
        commit,
        config_hash,
        source_hash,
        job,
        job_id,
    )

    points = result.get("points")
    if not isinstance(points, list) or len(points) != len(expected_points):
        raise ValueError(f"Incorrect point count in {result_path}")

    job_limits = dict(job)
    job_limits["min_toys"] = settings["min_toys"]
    job_limits["mc_sigrel_z"] = settings["mc_sigrel_z"]

    samples = []
    seen = set()
    for point_index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"Point {point_index} in {result_path} is not a mapping")
        try:
            identity, mc_values = validate_point(point, expected_points, job_limits)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid point {point_index} in {result_path}: {error}"
            ) from error

        if identity in seen:
            raise ValueError(f"Duplicate scan point {identity} in {result_path}")
        seen.add(identity)

        z_value, n_toys, n_exceedances, precision_limited = mc_values
        samples.append(
            (
                point["scan"],
                point["param_idx"],
                point["b_idx"],
                z_value,
                n_toys,
                n_exceedances,
                precision_limited,
            )
        )

    if seen != set(expected_points):
        raise ValueError(f"The scan grid is incomplete in {result_path}")
    return samples


# Add one validated p-value to the Monte Carlo diagnostics.
def update_summary(
    summary: dict[str, Any],
    n_toys: int,
    n_exceedances: int,
    precision_limited: bool,
) -> None:
    summary["count"] += 1
    if summary["min_toys"] is None:
        summary["min_toys"] = n_toys
        summary["max_toys"] = n_toys
    else:
        summary["min_toys"] = min(summary["min_toys"], n_toys)
        summary["max_toys"] = max(summary["max_toys"], n_toys)
    if n_exceedances == 0:
        summary["zero_exceedances"] += 1
    if precision_limited:
        summary["precision_limited"] += 1


# Validate every expected result file and group samples by physical scan point.
def collect_results(
    run_dir: Path,
    run_name: str,
    manifest: dict[str, Any],
    config_hash: str,
    source_hash: str,
    settings: dict[str, Any],
) -> tuple[
    dict[tuple[str, int, int, int], list[float]],
    dict[int, dict[str, Any]],
]:
    groups = {}
    diagnostics = {}
    commit = manifest["commit"]
    results_root = run_dir / "results"

    expected_signal_dirs = {job["signal_tag"] for job in settings["jobs"]}
    actual_signal_dirs = (
        {path.name for path in results_root.iterdir()}
        if results_root.exists()
        else set()
    )
    if actual_signal_dirs != expected_signal_dirs:
        raise ValueError(
            f"Result entries differ from the manifest: expected "
            f"{sorted(expected_signal_dirs)}, found {sorted(actual_signal_dirs)}"
        )
    if any(not (results_root / name).is_dir() for name in expected_signal_dirs):
        raise ValueError("Every result entry must be a signal directory")

    for job in settings["jobs"]:
        signal_dir = results_root / job["signal_tag"]
        expected_files = {
            signal_dir / f"job_{job_id}.json" for job_id in range(job["n_jobs"])
        }
        actual_files = set(signal_dir.iterdir())
        missing = sorted(path.name for path in expected_files - actual_files)
        extra = sorted(path.name for path in actual_files - expected_files)
        if missing or extra:
            raise ValueError(
                f"Incomplete {job['signal_tag']} results; missing={missing}, extra={extra}"
            )

        summary = {
            "count": 0,
            "min_toys": None,
            "max_toys": None,
            "zero_exceedances": 0,
            "precision_limited": 0,
        }
        for job_id in range(job["n_jobs"]):
            result_path = signal_dir / f"job_{job_id}.json"
            samples = validate_result_file(
                result_path,
                run_name,
                commit,
                config_hash,
                source_hash,
                settings,
                job,
                job_id,
            )
            for (
                scan,
                param_idx,
                b_idx,
                z_value,
                n_toys,
                n_exceedances,
                precision_limited,
            ) in samples:
                group_key = (
                    scan,
                    job["signal_index"],
                    param_idx,
                    b_idx,
                )
                groups.setdefault(group_key, []).append(z_value)
                update_summary(summary, n_toys, n_exceedances, precision_limited)

        diagnostics[job["signal_index"]] = summary

    expected_group_count = len(settings["jobs"]) * (
        len(settings["tau_vec"]) + len(settings["rel_sig_vec"])
    ) * len(settings["b_values"])
    if len(groups) != expected_group_count:
        raise ValueError(
            f"Expected {expected_group_count} populated groups, found {len(groups)}"
        )
    for key, values in groups.items():
        if len(values) != settings["n_outer"]:
            raise ValueError(
                f"Group {key} contains {len(values)} values, expected "
                f"{settings['n_outer']}"
            )

    return groups, diagnostics


def build_plot_results(
    groups: dict[tuple[str, int, int, int], list[float]],
    settings: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Build the Asimov and Monte Carlo significance grids used by the plotter.

    Each scan point stores the first-order and corrected Asimov significances,
    together with the median and mean of Z = Phi^-1(1 - p) from the validated
    pseudo-experiments.
    """
    n_s = len(settings["s_vec"])
    n_tau = len(settings["tau_vec"])
    n_rel_sig = len(settings["rel_sig_vec"])
    n_b = len(settings["b_values"])

    results = {
        "fixed_tau": {
            "Z_A_r": np.empty((n_s, n_tau, n_b)),
            "Z_A_rstar": np.empty((n_s, n_tau, n_b)),
            "Z_mc_median": np.empty((n_s, n_tau, n_b)),
            "Z_mc_mean": np.empty((n_s, n_tau, n_b)),
        },
        "fixed_rel_sig": {
            "Z_A_r": np.empty((n_s, n_rel_sig, n_b)),
            "Z_A_rstar": np.empty((n_s, n_rel_sig, n_b)),
            "Z_mc_median": np.empty((n_s, n_rel_sig, n_b)),
            "Z_mc_mean": np.empty((n_s, n_rel_sig, n_b)),
        },
    }

    for signal_index, s_true in enumerate(settings["s_vec"]):
        for param_idx, tau in enumerate(settings["tau_vec"]):
            for b_idx, b in enumerate(settings["b_values"]):
                asimov = asimov_Zs_onoff(float(s_true), float(b), float(tau))
                values = np.asarray(
                    groups[("fixed_tau", signal_index, param_idx, b_idx)],
                    dtype=float,
                )
                results["fixed_tau"]["Z_A_r"][signal_index, param_idx, b_idx] = asimov[
                    "Z_A_r"
                ]
                results["fixed_tau"]["Z_A_rstar"][signal_index, param_idx, b_idx] = (
                    asimov["Z_A_rstar"]
                )
                results["fixed_tau"]["Z_mc_median"][signal_index, param_idx, b_idx] = (
                    np.median(values)
                )
                results["fixed_tau"]["Z_mc_mean"][signal_index, param_idx, b_idx] = (
                    np.mean(values)
                )

        for param_idx, rel_sig in enumerate(settings["rel_sig_vec"]):
            for b_idx, b in enumerate(settings["b_values"]):
                tau = 1.0 / (float(rel_sig) ** 2 * float(b))
                asimov = asimov_Zs_onoff(float(s_true), float(b), tau)
                values = np.asarray(
                    groups[("fixed_rel_sig", signal_index, param_idx, b_idx)],
                    dtype=float,
                )
                results["fixed_rel_sig"]["Z_A_r"][signal_index, param_idx, b_idx] = (
                    asimov["Z_A_r"]
                )
                results["fixed_rel_sig"]["Z_A_rstar"][signal_index, param_idx, b_idx] = (
                    asimov["Z_A_rstar"]
                )
                results["fixed_rel_sig"]["Z_mc_median"][signal_index, param_idx, b_idx] = (
                    np.median(values)
                )
                results["fixed_rel_sig"]["Z_mc_mean"][signal_index, param_idx, b_idx] = (
                    np.mean(values)
                )

    for scan_results in results.values():
        for name, values in scan_results.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"The collected {name} grid contains non-finite values")
    return results


# Require a safe PDF output path below plots/.
def validate_output_path(
    value: Any,
    name: str,
    signal_template: bool = False,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    if signal_template and "{s}" not in value:
        raise ValueError(f"{name} must contain the {{s}} signal placeholder")

    try:
        rendered_value = value.format(s="2") if signal_template else value
    except (KeyError, ValueError) as error:
        raise ValueError(f"{name} is not a valid output template") from error

    path = Path(rendered_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path inside plots/")
    if not path.parts or path.parts[0] != "plots" or path.suffix.lower() != ".pdf":
        raise ValueError(f"{name} must name a PDF inside plots/")


# Validate the selected curves, display limit and output names.
def read_plot_options(
    config: dict[str, Any],
) -> tuple[list[str], list[str], Optional[float]]:
    selected_statistics = config.get("statistics", ["r", "rstar"])
    selected_mc_summaries = config.get("mc_summaries", ["median"])
    if not isinstance(selected_statistics, list):
        raise ValueError("statistics must be a YAML list")
    if not isinstance(selected_mc_summaries, list):
        raise ValueError("mc_summaries must be a YAML list")

    statistics = [str(value).lower() for value in selected_statistics]
    mc_summaries = [str(value).lower() for value in selected_mc_summaries]
    if any(value not in ("r", "rstar") for value in statistics):
        raise ValueError("statistics may contain only r and rstar")
    if any(value not in ("median", "mean") for value in mc_summaries):
        raise ValueError("mc_summaries may contain only median and mean")
    if not statistics and not mc_summaries:
        raise ValueError("Select at least one statistic or MC summary")

    z_display_max = config.get("Z_display_max")
    if z_display_max is not None:
        if isinstance(z_display_max, bool):
            raise ValueError("Z_display_max must be positive")
        z_display_max = float(z_display_max)
        if not math.isfinite(z_display_max) or z_display_max <= 0.0:
            raise ValueError("Z_display_max must be positive")

    validate_output_path(config.get("out_grid_pdf"), "out_grid_pdf")
    validate_output_path(
        config.get("out_tau_template"),
        "out_tau_template",
        signal_template=True,
    )
    validate_output_path(
        config.get("out_rel_sig_template"),
        "out_rel_sig_template",
        signal_template=True,
    )
    return statistics, mc_summaries, z_display_max


# Resolve a configured output path relative to the repository.
def rooted_output(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# Report toy-count ranges and precision limits for each signal.
def print_diagnostics(
    settings: dict[str, Any],
    diagnostics: dict[int, dict[str, Any]],
) -> None:
    print("Validated Monte Carlo diagnostics:")
    for job in settings["jobs"]:
        summary = diagnostics[job["signal_index"]]
        count = summary["count"]
        zero_fraction = summary["zero_exceedances"] / count
        limited_fraction = summary["precision_limited"] / count
        print(
            f"  s={job['s_true']:g}: {count:,} p-values, "
            f"N={summary['min_toys']:,}--{summary['max_toys']:,}, "
            f"K=0: {zero_fraction:.2%}, precision-limited: {limited_fraction:.2%}"
        )


# Validate a complete campaign before writing any final plots.
def main() -> None:
    args = parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run):
        raise ValueError("Run names may contain only letters, numbers, '.', '_' and '-'")

    run_dir = ROOT / "runs" / args.run
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.yaml"
    input_dir = run_dir / "input"
    if not manifest_path.is_file() or not config_path.is_file() or not input_dir.is_dir():
        raise ValueError(
            f"Run {args.run!r} does not contain its manifest, config and input snapshot"
        )

    manifest = load_json(manifest_path)
    config = load_config(config_path)
    config_hash = file_sha256(config_path)
    source_hash = frozen_source_sha256(input_dir)
    settings = read_batch_settings(config)
    statistics, mc_summaries, z_display_max = read_plot_options(config)
    validate_manifest(manifest, args.run, config_hash, source_hash, settings)

    groups, diagnostics = collect_results(
        run_dir,
        args.run,
        manifest,
        config_hash,
        source_hash,
        settings,
    )
    print_diagnostics(settings, diagnostics)

    results = build_plot_results(groups, settings)
    results = mask_mc_for_display(results, z_display_max)

    configure_plot_style()
    write_median_significance_pdfs(
        s_vec=settings["s_vec"],
        tau_vec=settings["tau_vec"],
        rel_sig_vec=settings["rel_sig_vec"],
        b_values_tau=settings["b_values"],
        b_values_rel_sig=settings["b_values"],
        results=results,
        out_tau_template=str(rooted_output(config["out_tau_template"])),
        out_rel_sig_template=str(rooted_output(config["out_rel_sig_template"])),
        out_grid_pdf=rooted_output(config["out_grid_pdf"]),
        statistics=statistics,
        mc_summaries=mc_summaries,
        z_display_max=z_display_max,
    )


if __name__ == "__main__":
    try:
        main()
    except (
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
