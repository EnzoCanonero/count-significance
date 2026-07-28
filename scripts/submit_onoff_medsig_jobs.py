#!/usr/bin/env python3
"""Create and submit one Condor campaign for the on/off medsig scan."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUBMIT_FILE = ROOT / "config" / "onoff_medsig_condor.sub"
WORKER = ROOT / "scripts" / "run_onoff_medsig_job.py"
WRAPPER = ROOT / "scripts" / "run_onoff_medsig_job.sh"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit the production on/off median-significance jobs."
    )
    parser.add_argument("--run", required=True, help="Unique name for this production run")
    parser.add_argument(
        "--config",
        default="config/paper_onoff_medsig.yaml",
        help="Paper YAML config containing the batch_mc settings",
    )
    return parser.parse_args()


def run_git(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def require_clean_repository():
    try:
        repository_root = Path(run_git("rev-parse", "--show-toplevel")).resolve()
        commit = run_git("rev-parse", "HEAD")
        status = run_git("status", "--porcelain", "--untracked-files=all")
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "Git could not inspect the repository"
        raise RuntimeError(message) from error

    if repository_root != ROOT.resolve():
        raise RuntimeError(f"Expected repository root {ROOT}, found {repository_root}")
    if status:
        raise RuntimeError("The repository has uncommitted or untracked changes")

    return commit


def validate_run_name(run_name):
    if not RUN_NAME_PATTERN.fullmatch(run_name) or run_name in (".", ".."):
        raise ValueError(
            "Run names must start with a letter or number and contain only "
            "letters, numbers, dots, underscores and hyphens"
        )


def validate_setup_script():
    setup_script = os.environ.get("MEDSIG_SETUP_SCRIPT")
    if not setup_script:
        return

    setup_path = Path(setup_script)
    if not setup_path.is_absolute():
        raise ValueError("MEDSIG_SETUP_SCRIPT must be an absolute path")
    if not setup_path.is_file():
        raise FileNotFoundError(f"Environment setup script not found: {setup_path}")


def signal_tag(s_true):
    text = f"{float(s_true):g}"
    text = text.replace("-", "m").replace(".", "p").replace("+", "")
    return f"s{text}"


def read_config(config_path):
    raw_config = config_path.read_bytes()
    config = yaml.safe_load(raw_config.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The YAML config must contain a mapping")

    config_sha256 = hashlib.sha256(raw_config).hexdigest()
    return config, raw_config, config_sha256


def positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
        original_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error

    if number <= 0 or not math.isfinite(original_value) or number != original_value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def finite_vector(config, name, positive=False):
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")

    numbers = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numbers") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        if positive and number <= 0.0:
            raise ValueError(f"{name} values must be positive")
        numbers.append(number)
    return numbers


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


def validate_output_path(value, name, signal_template=False):
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


def validate_plot_settings(config):
    statistics = config.get("statistics", ["r", "rstar"])
    mc_summaries = config.get("mc_summaries", ["median"])
    if not isinstance(statistics, list):
        raise ValueError("statistics must be a list")
    if not isinstance(mc_summaries, list):
        raise ValueError("mc_summaries must be a list")

    statistics = [str(value).lower() for value in statistics]
    mc_summaries = [str(value).lower() for value in mc_summaries]
    if any(value not in ("r", "rstar") for value in statistics):
        raise ValueError("statistics may contain only r and rstar")
    if any(value not in ("median", "mean") for value in mc_summaries):
        raise ValueError("mc_summaries may contain only median and mean")
    if not statistics and not mc_summaries:
        raise ValueError("Select at least one statistic or MC summary")

    z_display_max = config.get("Z_display_max")
    if z_display_max is not None:
        z_display_max = finite_number(z_display_max, "Z_display_max")
        if z_display_max <= 0.0:
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


def validate_signal_jobs(config):
    s_vec = finite_vector(config, "s_vec")
    finite_vector(config, "tau_vec", positive=True)
    finite_vector(config, "rel_sig_vec", positive=True)
    if any(s_true < 0.0 for s_true in s_vec):
        raise ValueError("s_vec values must be non-negative")

    b_min = finite_number(config.get("b_min"), "b_min")
    b_max = finite_number(config.get("b_max"), "b_max")
    if b_min <= 0.0 or b_max <= b_min:
        raise ValueError("The scan limits must satisfy 0 < b_min < b_max")

    batch = config.get("batch_mc")
    if not isinstance(batch, dict):
        raise ValueError("The config must contain a batch_mc mapping")

    signal_jobs = batch.get("signal_jobs")
    if not isinstance(signal_jobs, list):
        raise ValueError("batch_mc.signal_jobs must be a list")
    if len(signal_jobs) != len(s_vec):
        raise ValueError("batch_mc.signal_jobs must have one entry for each value in s_vec")

    n_outer = positive_integer(batch.get("n_outer"), "batch_mc.n_outer")
    positive_integer(batch.get("n_bpts"), "batch_mc.n_bpts")
    min_toys = positive_integer(batch.get("min_toys"), "batch_mc.min_toys")

    mc_sigrel_z = finite_number(batch.get("mc_sigrel_Z"), "batch_mc.mc_sigrel_Z")
    if mc_sigrel_z <= 0.0:
        raise ValueError("batch_mc.mc_sigrel_Z must be positive")

    signals = []
    seen_tags = set()
    for signal_index, job in enumerate(signal_jobs):
        if not isinstance(job, dict):
            raise ValueError("Each entry in batch_mc.signal_jobs must be a mapping")

        s_true = finite_number(
            job.get("s_true"),
            f"signal_jobs[{signal_index}].s_true",
        )
        expected_s = s_vec[signal_index]
        if s_true != expected_s:
            raise ValueError(
                f"signal_jobs[{signal_index}].s_true must match s_vec[{signal_index}]"
            )

        n_jobs = positive_integer(job.get("n_jobs"), f"signal_jobs[{signal_index}].n_jobs")
        outer_per_job = positive_integer(
            job.get("outer_per_job"),
            f"signal_jobs[{signal_index}].outer_per_job",
        )
        max_toys = positive_integer(
            job.get("max_toys"),
            f"signal_jobs[{signal_index}].max_toys",
        )
        positive_integer(job.get("outer_seed"), f"signal_jobs[{signal_index}].outer_seed")
        positive_integer(job.get("inner_seed"), f"signal_jobs[{signal_index}].inner_seed")

        if max_toys < min_toys:
            raise ValueError(
                f"signal_jobs[{signal_index}].max_toys must not be smaller than batch_mc.min_toys"
            )
        if n_jobs * outer_per_job != n_outer:
            raise ValueError(
                f"signal_jobs[{signal_index}] has n_jobs * outer_per_job = "
                f"{n_jobs * outer_per_job}, expected {n_outer}"
            )

        tag = signal_tag(s_true)
        if tag in seen_tags:
            raise ValueError(f"Signal tag {tag!r} is not unique")
        seen_tags.add(tag)

        signals.append(
            {
                "signal_index": signal_index,
                "s_true": s_true,
                "signal_tag": tag,
                "n_jobs": n_jobs,
                "outer_per_job": outer_per_job,
            }
        )

    return signals


def frozen_source_sha256(input_dir):
    source_files = [
        path
        for path in (input_dir / "src").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    source_files.append(input_dir / "scripts" / WORKER.name)
    source_files.sort(key=lambda path: path.relative_to(input_dir).as_posix())

    # Include both the relative filename and contents in the checksum.
    digest = hashlib.sha256()
    for path in source_files:
        relative_path = path.relative_to(input_dir).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
    return digest.hexdigest()


def write_campaign(run_name, commit, raw_config, config_sha256, signals):
    run_dir = ROOT / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Campaign already exists: {run_dir}")

    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    for signal in signals:
        (run_dir / "results" / signal["signal_tag"]).mkdir(parents=True)

    config_copy = run_dir / "config.yaml"
    config_copy.write_bytes(raw_config)

    # Freeze every source file used by the worker before submitting the jobs.
    input_dir = run_dir / "input"
    shutil.copytree(
        ROOT / "src",
        input_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (input_dir / "scripts").mkdir()
    shutil.copy2(WORKER, input_dir / "scripts" / WORKER.name)
    shutil.copy2(WRAPPER, input_dir / "scripts" / WRAPPER.name)
    source_sha256 = frozen_source_sha256(input_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": run_name,
        "commit": commit,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "config_path": "config.yaml",
        "input_path": "input",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    return run_dir, config_copy, input_dir


def condor_arguments(run_name, commit, config_copy, input_dir, signal):
    relative_config = config_copy.relative_to(ROOT)
    relative_source = (input_dir / "src").relative_to(ROOT)
    relative_worker = (input_dir / "scripts" / WORKER.name).relative_to(ROOT)
    relative_wrapper = (input_dir / "scripts" / WRAPPER.name).relative_to(ROOT)
    return [
        f"run_name={run_name}",
        f"signal_index={signal['signal_index']}",
        f"signal_tag={signal['signal_tag']}",
        f"n_jobs={signal['n_jobs']}",
        f"commit={commit}",
        f"config_path={relative_config}",
        f"source_path={relative_source}",
        f"worker_path={relative_worker}",
        f"wrapper_path={relative_wrapper}",
        str(SUBMIT_FILE.relative_to(ROOT)),
    ]


def check_signal(run_name, commit, config_copy, input_dir, signal):
    submit_arguments = condor_arguments(
        run_name,
        commit,
        config_copy,
        input_dir,
        signal,
    )
    dry_run_path = config_copy.parent / "logs" / f"{signal['signal_tag']}_submit.ad"
    print(f"Checking the Condor description for {signal['signal_tag']}")
    subprocess.run(
        ["condor_submit", "-dry-run", str(dry_run_path), *submit_arguments],
        cwd=ROOT,
        check=True,
    )


def submit_signal(run_name, commit, config_copy, input_dir, signal):
    submit_arguments = condor_arguments(
        run_name,
        commit,
        config_copy,
        input_dir,
        signal,
    )
    print(
        f"Submitting {signal['n_jobs']} jobs for "
        f"s={signal['s_true']:g} ({signal['signal_tag']})"
    )
    subprocess.run(["condor_submit", *submit_arguments], cwd=ROOT, check=True)


def main():
    args = parse_args()

    try:
        validate_run_name(args.run)
        validate_setup_script()
        commit = require_clean_repository()

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not SUBMIT_FILE.is_file():
            raise FileNotFoundError(f"Submit file not found: {SUBMIT_FILE}")
        if not WORKER.is_file():
            raise FileNotFoundError(f"Worker not found: {WORKER}")
        if not WRAPPER.is_file():
            raise FileNotFoundError(f"Worker wrapper not found: {WRAPPER}")
        if shutil.which("condor_submit") is None:
            raise RuntimeError("condor_submit is not available on PATH")

        config, raw_config, config_sha256 = read_config(config_path)
        signals = validate_signal_jobs(config)
        validate_plot_settings(config)
        run_dir, config_copy, input_dir = write_campaign(
            args.run,
            commit,
            raw_config,
            config_sha256,
            signals,
        )

        # Validate every cluster before submitting the first one.
        for signal in signals:
            check_signal(args.run, commit, config_copy, input_dir, signal)

        for signal in signals:
            submit_signal(args.run, commit, config_copy, input_dir, signal)

    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"Submission failed with exit status {error.returncode}. "
            "The campaign directory was preserved for inspection.",
            file=sys.stderr,
        )
        return error.returncode or 1

    print(f"Campaign submitted: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
