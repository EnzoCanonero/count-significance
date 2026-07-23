#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.lognormal import expected_significance_lognormal


PLOT_FIGSIZE = (6.5, 6.5)


def _configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
        }
    )


def _finish_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _medsig_ymax(*arrays) -> float:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.max(values)) * 1.04, 1.0)


def _style_medsig_axes(ax, b0_values: np.ndarray, y_max: float):
    ax.set_xscale("log")
    ax.set_xlim(float(b0_values[0]), float(b0_values[-1]))
    ax.set_xlabel(r"$b_0$")
    ax.set_ylabel(r"$\mathrm{med}[Z_0|1]$")
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    _finish_axes(ax)


def _fmt(x):
    return f"{x:g}".replace(".", "p").replace("-", "m")


def run_experiments_lognormal(
    s_vec: np.ndarray,
    sigma_vec: np.ndarray,
    b0_values: np.ndarray,
    n_outer: int,
    n_gh_pts: int,
    tail_mass: float,
    seed: int,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute Asimov and MC-median Z grids shaped (S, Sigma, B)."""
    shape = (len(s_vec), len(sigma_vec), len(b0_values))
    Z_A_r = np.empty(shape, dtype=float)
    Z_A_rstar = np.empty(shape, dtype=float)
    Z_mc_median = np.empty(shape, dtype=float)
    Z_mc_mean = np.empty(shape, dtype=float)

    total = int(np.prod(shape))
    rng = np.random.default_rng(int(seed))
    i = 0
    for s_idx, s_true in enumerate(s_vec):
        for sig_idx, sigma in enumerate(sigma_vec):
            for b_idx, b0 in enumerate(b0_values):
                i += 1
                print(f"  [{i}/{total}] s={s_true:.3g} b0={b0:.3g} sigma={sigma:.3g}", flush=True)
                res = expected_significance_lognormal(
                    s_true=float(s_true),
                    b0=float(b0),
                    sigma=float(sigma),
                    n_outer=int(n_outer),
                    n_gh_pts=int(n_gh_pts),
                    tail_mass=float(tail_mass),
                    seed=int(rng.integers(1, 2**31 - 1)),
                    continuity_correction_r=continuity_correction_r,
                    continuity_correction_rstar=continuity_correction_rstar,
                )
                Z_A_r[s_idx, sig_idx, b_idx] = float(res["Z_A_r"])
                Z_A_rstar[s_idx, sig_idx, b_idx] = float(res["Z_A_rstar"])
                Z_mc_median[s_idx, sig_idx, b_idx] = float(res["Z_mc_median"])
                Z_mc_mean[s_idx, sig_idx, b_idx] = float(res["Z_mc_mean"])

    return Z_A_r, Z_A_rstar, Z_mc_median, Z_mc_mean


def make_plots_lognormal_medsig(
    s_vec: np.ndarray,
    sigma_vec: np.ndarray,
    b0_values: np.ndarray,
    Z_A_r: np.ndarray,
    Z_A_rstar: np.ndarray,
    Z_mc_median: np.ndarray,
    Z_mc_mean: np.ndarray,
    out_pdf: Path,
    save_individual: bool,
    outdir: Path,
    mc_statistics: list,
    asimov_statistics: list,
):
    """Save log-normal medsig scans, one page per (s_true, sigma)."""
    with PdfPages(out_pdf) as pdf:
        for s_idx, s_true in enumerate(s_vec):
            for sig_idx, sigma in enumerate(sigma_vec):
                fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)

                if "r" in asimov_statistics:
                    ax.plot(b0_values, Z_A_r[s_idx, sig_idx], label=r"Asimov $q_0$")
                if "rstar" in asimov_statistics:
                    ax.plot(
                        b0_values,
                        Z_A_rstar[s_idx, sig_idx],
                        "--",
                        label=r"Asimov $q_0^\ast$",
                    )
                if "median" in mc_statistics:
                    ax.plot(
                        b0_values,
                        Z_mc_median[s_idx, sig_idx],
                        linestyle="None",
                        marker="x",
                        label=r"MC median $Z$",
                    )
                if "mean" in mc_statistics:
                    ax.plot(
                        b0_values,
                        Z_mc_mean[s_idx, sig_idx],
                        linestyle="None",
                        marker="+",
                        label=r"MC mean $Z$",
                    )

                _style_medsig_axes(ax, b0_values, _medsig_ymax(Z_A_r[s_idx, sig_idx]))
                ax.legend(frameon=False)
                plt.tight_layout()
                if save_individual:
                    fname = outdir / f"lognormal_medsig_s{_fmt(s_true)}_sigma{_fmt(sigma)}.pdf"
                    fig.savefig(fname)
                pdf.savefig(fig)
                plt.close(fig)


def main(cfg_path: str):
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    sigma_vec = np.asarray(cfg["sigma_vec"], dtype=float)
    b0_min = float(cfg["b0_min"])
    b0_max = float(cfg["b0_max"])
    n_bpts = int(cfg["n_bpts"])
    n_outer = int(cfg["n_outer"])
    n_gh_pts = int(cfg.get("n_gh_pts", 20))
    tail_mass = float(cfg.get("reference_tail_mass", 1e-10))
    seed = int(cfg.get("seed", 12345))
    outdir = Path(cfg.get("outdir", "plots"))
    out_pdf = Path(cfg.get("out_pdf", outdir / "lognormal_medsig.pdf"))
    save_individual = bool(cfg.get("individual_plots", False))
    mc_statistics = cfg.get("mc_statistics", ["median"])
    asimov_statistics = cfg.get("asimov_statistics", ["r", "rstar"])
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    outdir.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    b0_values = np.logspace(np.log10(b0_min), np.log10(b0_max), n_bpts)

    Z_A_r, Z_A_rstar, Z_mc_median, Z_mc_mean = run_experiments_lognormal(
        s_vec=s_vec,
        sigma_vec=sigma_vec,
        b0_values=b0_values,
        n_outer=n_outer,
        n_gh_pts=n_gh_pts,
        tail_mass=tail_mass,
        seed=seed,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )

    make_plots_lognormal_medsig(
        s_vec=s_vec,
        sigma_vec=sigma_vec,
        b0_values=b0_values,
        Z_A_r=Z_A_r,
        Z_A_rstar=Z_A_rstar,
        Z_mc_median=Z_mc_median,
        Z_mc_mean=Z_mc_mean,
        out_pdf=out_pdf,
        save_individual=save_individual,
        outdir=outdir,
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )

    if save_individual:
        print(f"Saved individual plots under: {outdir.resolve()}")
    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lognormal_medsig.yaml",
        help="Path to YAML config for log-normal medsig plots",
    )
    args = parser.parse_args()
    main(args.config)
