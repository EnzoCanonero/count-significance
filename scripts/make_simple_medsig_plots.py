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
from src.on import expected_significance_on

# Call graph (simple medsig)
# - main: load config, build b grid, run experiments, then plot
#   - run_experiments_on: compute Asimov + MC-median Z grids (vectorised expected_significance_on)
#   - make_plots_on: save combined/individual PDFs
#     - plot_case (inline inside make_plots_on loop): per-s_true panel with Asimov vs MC median


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def run_experiments_on(
    s_vec: np.ndarray,
    b_values: np.ndarray,
    n_outer: int,
    seed: int,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """
    Compute Asimov and MC-median Z grids for the simple-counting case via expected_significance_on.

    Returns (Z_A_r, Z_A_rstar, Z_mc_median) shaped (len(s_vec), len(b_values)).
    """
    seed = int(seed)
    n_outer = int(n_outer)

    # Vectorised grids for expected_significance_on: shape (S, B)
    s_grid = s_vec[:, None]
    b_grid = b_values[None, :]

    res = expected_significance_on(
        s_true=s_grid,
        b=b_grid,
        n_outer=n_outer,
        seed=seed,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )
    return res["Z_A_r"], res["Z_A_rstar"], res["Z_mc_median"], res["Z_mc_mean"]


def make_plots_on(
    s_vec: np.ndarray,
    b_values: np.ndarray,
    Z_A_r: np.ndarray,
    Z_A_rstar: np.ndarray,
    Z_mc_median: np.ndarray,
    Z_mc_mean: np.ndarray,
    out_pdf: Path,
    save_individual: bool,
    mc_statistics: list,
    asimov_statistics: list,
):
    """Save combined PDF (and optional per-plot PDFs) for the simple-counting medsig scans."""
    with PdfPages(out_pdf) as pdf:
        for idx, s_true in enumerate(s_vec):
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            if "r" in asimov_statistics:
                ax.plot(b_values, Z_A_r[idx], label=fr"Asimov $r$, $s_\mathrm{{true}}={s_true}$")
            if "rstar" in asimov_statistics:
                ax.plot(
                    b_values,
                    Z_A_rstar[idx],
                    linestyle="--",
                    label=fr"Asimov $r^\ast$, $s_\mathrm{{true}}={s_true}$",
                )
            if "median" in mc_statistics:
                ax.plot(
                    b_values,
                    Z_mc_median[idx],
                    linestyle="None",
                    marker="x",
                    label=fr"MC median $Z$, $s_\mathrm{{true}}={s_true}$",
                )
            if "mean" in mc_statistics:
                ax.plot(
                    b_values,
                    Z_mc_mean[idx],
                    linestyle="None",
                    marker="+",
                    label=fr"MC mean $Z$, $s_\mathrm{{true}}={s_true}$",
                )

            ax.set_xscale("log")
            ax.set_xlabel("b")
            ax.set_ylabel("Z")
            ax.set_ylim(bottom=-1)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.legend(fontsize=9)
            ax.set_title(fr"$s_{{\mathrm{{true}}}} = {s_true}$")

            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"simple_medsig_s{_fmt(s_true)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_min = float(cfg["b_min"])
    b_max = float(cfg["b_max"])
    n_bpts = int(cfg["n_bpts"])
    n_outer = int(cfg.get("n_outer", 200))
    seed = int(cfg.get("seed", 12345))
    out_pdf = Path(cfg["out_pdf"])
    save_individual = bool(cfg.get("individual_plots", False))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    mc_statistics = cfg.get("mc_statistics", ["median"])
    asimov_statistics = cfg.get("asimov_statistics", ["r", "rstar"])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    b_values = np.logspace(np.log10(b_min), np.log10(b_max), n_bpts)

    Z_A_r, Z_A_rstar, Z_mc_median, Z_mc_mean = run_experiments_on(
        s_vec=s_vec,
        b_values=b_values,
        n_outer=n_outer,
        seed=seed,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )

    make_plots_on(
        s_vec=s_vec,
        b_values=b_values,
        Z_A_r=Z_A_r,
        Z_A_rstar=Z_A_rstar,
        Z_mc_median=Z_mc_median,
        Z_mc_mean=Z_mc_mean,
        out_pdf=out_pdf,
        save_individual=save_individual,
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )

    if save_individual:
        print(f"Saved individual plots under: {out_pdf.parent.resolve()}")
    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/simple_medsig.yaml",
        help="Path to YAML config for simple medsig plots",
    )
    args = parser.parse_args()
    main(args.config)
