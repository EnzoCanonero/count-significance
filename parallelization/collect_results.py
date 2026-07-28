#!/usr/bin/env python3
"""Aggregate per-job uncertain-background medsig toys and render the paper plots.

The numerical content (MC median/mean Z over the outer pseudo-experiments) is
produced by the Condor jobs (pval.py) and stored as per-job JSONs. This script
only aggregates those toys and re-uses the shared plotting routines from
scripts/make_onoff_medsig_plots.py so the rendered figures stay consistent with
the serial paper pipeline (labels, naive curve, styling, combined per-signal
panels and the combined grid).

It supports per-signal submissions: each s_true can be produced by a separate
submission writing to its own output directory. List those directories under
`input_dirs` in the merge config; records are keyed by the physical s_true
value (not by s_idx), so single-s runs aggregate correctly into the combined
s_vec grid. All runs must share the same tau / relative-uncertainty / b grids.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.on_off import asimov_Zs_onoff  # noqa: E402
from scripts.make_onoff_medsig_plots import (  # noqa: E402
    _configure_plot_style,
    write_median_significance_pdfs,
)


def _skey(s_true):
    """Stable key for a signal strength (avoids float-repr mismatches)."""
    return round(float(s_true), 6)


def collect_results(cfg):
    """Read per-job JSONs from all input dirs, grouping single-experiment Z values.

    Keyed by (mode, s_true_key, param_idx, b_idx) so per-signal submissions merge.
    """
    outdir = cfg.get("outdir", "output")
    input_dirs = cfg.get("input_dirs", [outdir])

    files = []
    for d in input_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "pval_job_*.json"))))

    groups = {}  # key -> {"meta": rec, "Z": [...]}
    for path in files:
        with open(path, "r") as f:
            job_data = json.load(f)

        for rec in job_data["points"]:
            key = (rec["mode"], _skey(rec["s_true"]), rec["param_idx"], rec["b_idx"])
            if "Z_single" in rec:
                Z_single = float(rec["Z_single"])
            else:
                Z_single = norm.isf(rec["p_mc"])

            if key not in groups:
                groups[key] = {"meta": rec, "Z": []}
            groups[key]["Z"].append(Z_single)

    return groups, files


def groups_to_mc_grids(groups, s_vec, n_tau, n_rel_sig, n_b_tau, n_b_rel_sig):
    """Reduce grouped toys to named MC-summary grids.

    Records are matched to s_vec by physical s_true value. The returned nested
    dictionary has the same scan and result keys as the serial pipeline.
    """
    n_s = len(s_vec)
    Z_med_tau = np.full((n_s, n_tau, n_b_tau), np.nan)
    Z_mean_tau = np.full((n_s, n_tau, n_b_tau), np.nan)
    Z_med_rel_sig = np.full((n_s, n_rel_sig, n_b_rel_sig), np.nan)
    Z_mean_rel_sig = np.full((n_s, n_rel_sig, n_b_rel_sig), np.nan)

    s_index = {_skey(s): i for i, s in enumerate(s_vec)}

    for (mode, s_key, param_idx, b_idx), slot in groups.items():
        s_idx = s_index.get(s_key)
        if s_idx is None:
            continue  # signal not in the combined s_vec; skip
        Z = np.asarray(slot["Z"], dtype=float)
        if Z.size == 0:
            continue
        med = float(np.median(Z))
        mean = float(np.mean(Z))
        if mode == "tau":
            Z_med_tau[s_idx, param_idx, b_idx] = med
            Z_mean_tau[s_idx, param_idx, b_idx] = mean
        elif mode == "sig":
            Z_med_rel_sig[s_idx, param_idx, b_idx] = med
            Z_mean_rel_sig[s_idx, param_idx, b_idx] = mean

    return {
        "fixed_tau": {
            "Z_mc_median": Z_med_tau,
            "Z_mc_mean": Z_mean_tau,
        },
        "fixed_rel_sig": {
            "Z_mc_median": Z_med_rel_sig,
            "Z_mc_mean": Z_mean_rel_sig,
        },
    }


def mask_mc_at_display_max(results, z_display_max):
    """Hide MC summaries at or above the graphical display maximum."""
    if z_display_max is None:
        return
    z_display_max = float(z_display_max)
    for scan_results in results.values():
        for key in ("Z_mc_median", "Z_mc_mean"):
            values = scan_results[key]
            values[values >= z_display_max] = np.nan


def asimov_grids(
    s_vec,
    tau_vec,
    rel_sig_vec,
    b_values_tau,
    b_values_rel_sig,
):
    """Compute named Asimov q and q* grids on the same scan layout."""
    n_s, n_tau, n_rel_sig = len(s_vec), len(tau_vec), len(rel_sig_vec)
    n_b_tau, n_b_rel_sig = len(b_values_tau), len(b_values_rel_sig)

    Z_A_r_tau = np.zeros((n_s, n_tau, n_b_tau))
    Z_A_rstar_tau = np.zeros((n_s, n_tau, n_b_tau))
    Z_A_r_rel_sig = np.zeros((n_s, n_rel_sig, n_b_rel_sig))
    Z_A_rstar_rel_sig = np.zeros((n_s, n_rel_sig, n_b_rel_sig))

    for s_idx, s_true in enumerate(s_vec):
        for tau_idx, tau in enumerate(tau_vec):
            for b_idx, b in enumerate(b_values_tau):
                asim = asimov_Zs_onoff(
                    float(s_true),
                    float(b),
                    float(tau),
                )
                Z_A_r_tau[s_idx, tau_idx, b_idx] = asim["Z_A_r"]
                Z_A_rstar_tau[s_idx, tau_idx, b_idx] = asim["Z_A_rstar"]

        for rel_sig_idx, rel_sig in enumerate(rel_sig_vec):
            for b_idx, b in enumerate(b_values_rel_sig):
                tau_b = 1.0 / (float(rel_sig) ** 2 * float(b))
                asim = asimov_Zs_onoff(
                    float(s_true),
                    float(b),
                    tau_b,
                )
                Z_A_r_rel_sig[s_idx, rel_sig_idx, b_idx] = asim["Z_A_r"]
                Z_A_rstar_rel_sig[s_idx, rel_sig_idx, b_idx] = asim["Z_A_rstar"]

    return {
        "fixed_tau": {
            "Z_A_r": Z_A_r_tau,
            "Z_A_rstar": Z_A_rstar_tau,
        },
        "fixed_rel_sig": {
            "Z_A_r": Z_A_r_rel_sig,
            "Z_A_rstar": Z_A_rstar_rel_sig,
        },
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: collect_results.py CONFIG.yaml", file=sys.stderr)
        sys.exit(1)

    _configure_plot_style()

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tauVec"], dtype=float)
    rel_sig_vec = np.asarray(cfg["relSigVec"], dtype=float)

    b_values_tau = np.logspace(
        np.log10(cfg["b_min_tau"]), np.log10(cfg["b_max_tau"]), int(cfg["n_bpts_tau"])
    )
    b_values_rel_sig = np.logspace(
        np.log10(cfg["b_min_sig"]), np.log10(cfg["b_max_sig"]), int(cfg["n_bpts_sig"])
    )

    outdir = Path(cfg.get("outdir", "output"))
    selected_statistics = cfg.get("statistics", ["r", "rstar"])
    if not isinstance(selected_statistics, list):
        raise ValueError("statistics must be a YAML list")
    statistics = [str(value).lower() for value in selected_statistics]
    for statistic in statistics:
        if statistic not in ("r", "rstar"):
            raise ValueError(f"Unknown statistic={statistic!r}; choose from 'r' and 'rstar'")

    selected_mc_summaries = cfg.get("mc_summaries", ["median"])
    if not isinstance(selected_mc_summaries, list):
        raise ValueError("mc_summaries must be a YAML list")
    mc_summaries = [str(value).lower() for value in selected_mc_summaries]
    for summary in mc_summaries:
        if summary not in ("median", "mean"):
            raise ValueError(f"Unknown MC summary={summary!r}; choose from 'median' and 'mean'")
    if not statistics and not mc_summaries:
        raise ValueError("Select at least one statistic or Monte Carlo summary")

    # Aggregate the per-job toys and compute the Asimov grids.
    groups, files = collect_results(cfg)
    print(f"Read {len(files)} job files; {len(groups)} grid points populated.")
    mc_results = groups_to_mc_grids(
        groups,
        s_vec=s_vec,
        n_tau=len(tau_vec),
        n_rel_sig=len(rel_sig_vec),
        n_b_tau=len(b_values_tau),
        n_b_rel_sig=len(b_values_rel_sig),
    )
    results = asimov_grids(
        s_vec,
        tau_vec,
        rel_sig_vec,
        b_values_tau,
        b_values_rel_sig,
    )
    for scan_name in results:
        results[scan_name].update(mc_results[scan_name])

    # Z_display_max is only a graphical cutoff. It sets the y-axis maximum and
    # hides MC markers at or above it; it does not alter the stored results.
    z_display_max = cfg.get("Z_display_max")
    mask_mc_at_display_max(results, z_display_max=z_display_max)

    out_tau_template = cfg.get(
        "out_tau_template", str(outdir / "onoff_medsig_tau_s{s}.pdf")
    )
    out_rel_sig_template = cfg.get(
        "out_rel_sig_template", str(outdir / "onoff_medsig_rel_sig_s{s}.pdf")
    )
    out_grid_pdf = cfg.get("out_grid_pdf", str(outdir / "uncertain_background_asimov_grid.pdf"))
    write_median_significance_pdfs(
        s_vec=s_vec,
        tau_vec=tau_vec,
        rel_sig_vec=rel_sig_vec,
        b_values_tau=b_values_tau,
        b_values_rel_sig=b_values_rel_sig,
        results=results,
        out_tau_template=str(out_tau_template),
        out_rel_sig_template=str(out_rel_sig_template),
        out_grid_pdf=Path(out_grid_pdf),
        statistics=statistics,
        mc_summaries=mc_summaries,
        z_display_max=z_display_max,
    )
    print(f"Saved combined grid to: {Path(out_grid_pdf).resolve()}")


if __name__ == "__main__":
    main()
