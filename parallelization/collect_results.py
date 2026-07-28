#!/usr/bin/env python3
"""Aggregate per-job uncertain-background medsig toys and render the paper plots.

The numerical content (MC median/mean Z over the outer pseudo-experiments) is
produced by the Condor jobs (pval.py) and stored as per-job JSONs. This script
only aggregates those toys and re-uses the exact plotting routines from
scripts/make_onoff_medsig_plots.py so the rendered figures stay consistent with
the serial paper pipeline (labels, naive curve, styling, combined per-signal
panels and the q-only / q+q* variants).

It supports per-signal submissions: each s_true can be produced by a separate
submission writing to its own output directory. List those directories under
`input_dirs` in the merge config; records are keyed by the physical s_true
value (not by s_idx), so single-s runs aggregate correctly into the combined
s_vec grid. All runs must share the same tau / sigma_rel / b grids.
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
    make_plots_onoff,
    make_combined_grid_plot,
    make_combined_tau_plots,
    make_combined_sigrel_plots,
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


def groups_to_mc_grids(groups, s_vec, n_tau, n_sig, n_b_tau, n_b_sig):
    """Reduce grouped toys to MC-median/mean Z grids matching the serial pipeline.

    Records are matched to s_vec by physical s_true value. Returns arrays shaped
    (len(s_vec), n_tau, n_b_tau) and (len(s_vec), n_sig, n_b_sig).
    """
    n_s = len(s_vec)
    Z_med_tau = np.full((n_s, n_tau, n_b_tau), np.nan)
    Z_mean_tau = np.full((n_s, n_tau, n_b_tau), np.nan)
    Z_med_sig = np.full((n_s, n_sig, n_b_sig), np.nan)
    Z_mean_sig = np.full((n_s, n_sig, n_b_sig), np.nan)

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
            Z_med_sig[s_idx, param_idx, b_idx] = med
            Z_mean_sig[s_idx, param_idx, b_idx] = mean

    return Z_med_tau, Z_mean_tau, Z_med_sig, Z_mean_sig


def mask_mc_at_ceiling(*arrays, z_ceiling):
    """Hide MC summaries that have hit or exceeded the display ceiling."""
    if z_ceiling is None:
        return
    z_ceiling = float(z_ceiling)
    for array in arrays:
        array[array >= z_ceiling] = np.nan


def asimov_grids(
    s_vec,
    tau_vec,
    rel_sig_vec,
    b_values_tau,
    b_values_sig,
):
    """Compute the (deterministic) Asimov q / q* grids on the same scan layout."""
    n_s, n_tau, n_sig = len(s_vec), len(tau_vec), len(rel_sig_vec)
    Bt, Bs = len(b_values_tau), len(b_values_sig)

    Z_A_r_tau = np.zeros((n_s, n_tau, Bt))
    Z_A_rstar_tau = np.zeros((n_s, n_tau, Bt))
    Z_A_r_sig = np.zeros((n_s, n_sig, Bs))
    Z_A_rstar_sig = np.zeros((n_s, n_sig, Bs))

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

        for sig_idx, sigma_rel in enumerate(rel_sig_vec):
            for b_idx, b in enumerate(b_values_sig):
                tau_b = 1.0 / (float(sigma_rel) ** 2 * float(b))
                asim = asimov_Zs_onoff(
                    float(s_true),
                    float(b),
                    tau_b,
                )
                Z_A_r_sig[s_idx, sig_idx, b_idx] = asim["Z_A_r"]
                Z_A_rstar_sig[s_idx, sig_idx, b_idx] = asim["Z_A_rstar"]

    return Z_A_r_tau, Z_A_rstar_tau, Z_A_r_sig, Z_A_rstar_sig


def _render(
    s_vec,
    tau_vec,
    rel_sig_vec,
    b_values_tau,
    b_values_sig,
    grids,
    out_pdf,
    out_combined_tau_template,
    out_combined_sigrel_template,
    out_grid_pdf,
    mc_statistics,
    asimov_statistics,
    save_individual,
    outdir,
    y_ceiling=None,
):
    """One full render (summary bundle + combined per-signal panels)."""
    (
        Z_A_r_tau,
        Z_A_rstar_tau,
        Z_A_r_sig,
        Z_A_rstar_sig,
        Z_med_tau,
        Z_med_sig,
        Z_mean_tau,
        Z_mean_sig,
    ) = grids

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    make_plots_onoff(
        s_vec=s_vec,
        tau_vec=tau_vec,
        rel_sig_vec=rel_sig_vec,
        b_values_tau=b_values_tau,
        b_values_sig=b_values_sig,
        Z_A_r_tau=Z_A_r_tau,
        Z_A_rstar_tau=Z_A_rstar_tau,
        Z_A_r_sig=Z_A_r_sig,
        Z_A_rstar_sig=Z_A_rstar_sig,
        Z_med_tau=Z_med_tau,
        Z_med_sig=Z_med_sig,
        Z_mean_tau=Z_mean_tau,
        Z_mean_sig=Z_mean_sig,
        out_pdf=out_pdf,
        save_individual=save_individual,
        outdir=outdir,
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
        y_ceiling=y_ceiling,
    )
    make_combined_tau_plots(
        s_vec=s_vec,
        tau_vec=tau_vec,
        b_values=b_values_tau,
        Z_A_r_tau=Z_A_r_tau,
        Z_A_rstar_tau=Z_A_rstar_tau,
        Z_med_tau=Z_med_tau,
        out_template=str(out_combined_tau_template),
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
        y_ceiling=y_ceiling,
    )
    make_combined_sigrel_plots(
        s_vec=s_vec,
        rel_sig_vec=rel_sig_vec,
        b_values=b_values_sig,
        Z_A_r_sig=Z_A_r_sig,
        Z_A_rstar_sig=Z_A_rstar_sig,
        Z_med_sig=Z_med_sig,
        out_template=str(out_combined_sigrel_template),
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
        y_ceiling=y_ceiling,
    )
    if out_grid_pdf:
        make_combined_grid_plot(
            s_vec=s_vec,
            tau_vec=tau_vec,
            rel_sig_vec=rel_sig_vec,
            b_values_tau=b_values_tau,
            b_values_sig=b_values_sig,
            Z_A_r_tau=Z_A_r_tau,
            Z_A_rstar_tau=Z_A_rstar_tau,
            Z_A_r_sig=Z_A_r_sig,
            Z_A_rstar_sig=Z_A_rstar_sig,
            Z_med_tau=Z_med_tau,
            Z_med_sig=Z_med_sig,
            out_pdf=Path(out_grid_pdf),
            mc_statistics=mc_statistics,
            asimov_statistics=asimov_statistics,
            y_ceiling=y_ceiling,
        )


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
    b_values_sig = np.logspace(
        np.log10(cfg["b_min_sig"]), np.log10(cfg["b_max_sig"]), int(cfg["n_bpts_sig"])
    )

    outdir = Path(cfg.get("outdir", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    save_individual = bool(cfg.get("individual_plots", False))
    mc_statistics = cfg.get("mc_statistics", ["median"])
    asimov_statistics = cfg.get("asimov_statistics", ["r", "rstar"])
    # Aggregate the per-job toys and compute the Asimov grids.
    groups, files = collect_results(cfg)
    print(f"Read {len(files)} job files; {len(groups)} grid points populated.")
    Z_med_tau, Z_mean_tau, Z_med_sig, Z_mean_sig = groups_to_mc_grids(
        groups,
        s_vec=s_vec,
        n_tau=len(tau_vec),
        n_sig=len(rel_sig_vec),
        n_b_tau=len(b_values_tau),
        n_b_sig=len(b_values_sig),
    )
    Z_A_r_tau, Z_A_rstar_tau, Z_A_r_sig, Z_A_rstar_sig = asimov_grids(
        s_vec,
        tau_vec,
        rel_sig_vec,
        b_values_tau,
        b_values_sig,
    )

    # Z_ceiling sets the y-axis top and removes saturated MC markers. The
    # continuous approximations (Asimov q/q* and the naive curve) are left
    # uncapped so they run off the top margin above the ceiling.
    z_ceiling = cfg.get("Z_ceiling")
    mask_mc_at_ceiling(
        Z_med_tau,
        Z_mean_tau,
        Z_med_sig,
        Z_mean_sig,
        z_ceiling=z_ceiling,
    )

    grids = (
        Z_A_r_tau,
        Z_A_rstar_tau,
        Z_A_r_sig,
        Z_A_rstar_sig,
        Z_med_tau,
        Z_med_sig,
        Z_mean_tau,
        Z_mean_sig,
    )

    # Primary render (e.g. q + q* -> fig 6/7).
    out_pdf = Path(cfg.get("out_summary_pdf", outdir / "onoff_medsig.pdf"))
    out_combined_tau_template = cfg.get(
        "out_combined_tau_template", str(outdir / "onoff_medsig_combined_tau_s{s}.pdf")
    )
    out_combined_sigrel_template = cfg.get(
        "out_combined_sigrel_template", str(outdir / "onoff_medsig_combined_sigrel_s{s}.pdf")
    )
    out_grid_pdf = cfg.get("out_grid_pdf")
    _render(
        s_vec, tau_vec, rel_sig_vec, b_values_tau, b_values_sig, grids,
        out_pdf, out_combined_tau_template, out_combined_sigrel_template, out_grid_pdf,
        mc_statistics, asimov_statistics, save_individual, outdir,
        y_ceiling=z_ceiling,
    )
    print(f"Saved primary plots to: {Path(out_pdf).resolve()}")

    # Optional alternate render (e.g. q-only -> fig 2).
    alternate_asimov_statistics = cfg.get("alternate_asimov_statistics")
    if alternate_asimov_statistics:
        alt_pdf = cfg.get(
            "alternate_out_summary_pdf", str(out_pdf).replace(".pdf", "_alternate.pdf")
        )
        alt_tau = cfg.get("alternate_out_combined_tau_template", out_combined_tau_template)
        alt_sig = cfg.get("alternate_out_combined_sigrel_template", out_combined_sigrel_template)
        alt_grid = cfg.get("alternate_out_grid_pdf")
        _render(
            s_vec, tau_vec, rel_sig_vec, b_values_tau, b_values_sig, grids,
            alt_pdf, alt_tau, alt_sig, alt_grid,
            mc_statistics, alternate_asimov_statistics, save_individual, outdir,
            y_ceiling=z_ceiling,
        )
        print(f"Saved alternate plots to: {Path(alt_pdf).resolve()}")


if __name__ == "__main__":
    main()
