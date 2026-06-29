#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on import expected_significance_on


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


def _style_medsig_axes(ax, b_values: np.ndarray, y_max: float):
    ax.set_xscale("log")
    ax.set_xlim(float(b_values[0]), float(b_values[-1]))
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(r"$\mathrm{med}[Z_0|1]$")
    ax.set_ylim(bottom=-0.5, top=y_max)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    _finish_axes(ax)


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
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)
            if "r" in asimov_statistics:
                ax.plot(b_values, Z_A_r[idx], label=fr"Asimov $q$, $s_\mathrm{{true}}={s_true}$")
            if "rstar" in asimov_statistics:
                ax.plot(
                    b_values,
                    Z_A_rstar[idx],
                    linestyle="--",
                    label=fr"Asimov $q^\ast$, $s_\mathrm{{true}}={s_true}$",
                )
            ax.plot(
                b_values,
                float(s_true) / np.sqrt(b_values),
                linestyle=":",
                label=fr"$s/\sqrt{{b}}$, $s_\mathrm{{true}}={s_true}$",
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

            _style_medsig_axes(ax, b_values, _medsig_ymax(Z_A_r[idx]))
            ax.legend(frameon=False)

            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"simple_medsig_s{_fmt(s_true)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def make_combined_plot_on(
    s_vec: np.ndarray,
    b_values: np.ndarray,
    Z_A_r: np.ndarray,
    Z_A_rstar: np.ndarray,
    Z_mc_median: np.ndarray,
    out_pdf: Path,
    mc_statistics: list,
    asimov_statistics: list,
):
    """Save one panel with all requested simple-counting medsig configurations."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)

    for idx, s_true in enumerate(s_vec):
        color = colors[idx % len(colors)]
        if "r" in asimov_statistics:
            ax.plot(b_values, Z_A_r[idx], color=color, linestyle="-")
        if "rstar" in asimov_statistics:
            ax.plot(b_values, Z_A_rstar[idx], color=color, linestyle="--")
        ax.plot(b_values, float(s_true) / np.sqrt(b_values), color=color, linestyle=":")
        if "median" in mc_statistics:
            ax.plot(b_values, Z_mc_median[idx], color=color, linestyle="None", marker="x")

    _style_medsig_axes(ax, b_values, _medsig_ymax(Z_A_r))

    stat_handles = []
    if "r" in asimov_statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", linestyle="-", label=r"Asimov $q$"))
    if "rstar" in asimov_statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", linestyle="--", label=r"Asimov $q^\ast$"))
    stat_handles.append(Line2D([0], [0], color="0.15", linestyle=":", label=r"$s/\sqrt{b}$"))
    if "median" in mc_statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", marker="x", linestyle="None", label=r"MC median"))

    legend_kwargs = {
        "frameon": False,
        "fontsize": 11,
        "borderaxespad": 0.2,
        "handlelength": 1.8,
        "handletextpad": 0.7,
        "labelspacing": 0.45,
    }
    s_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], linestyle="-", label=fr"$s={s_true:g}$")
        for idx, s_true in enumerate(s_vec)
    ]
    stat_legend = ax.legend(
        handles=stat_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        **legend_kwargs,
    )
    ax.add_artist(stat_legend)
    ax.legend(
        handles=s_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.66),
        **legend_kwargs,
    )

    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main(cfg_path: str):
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_min = float(cfg["b_min"])
    b_max = float(cfg["b_max"])
    n_bpts = int(cfg["n_bpts"])
    n_outer = int(cfg.get("n_outer", 200))
    seed = int(cfg.get("seed", 12345))
    out_pdf = Path(cfg["out_pdf"])
    out_combined_pdf = Path(cfg.get("out_combined_pdf", out_pdf.with_name(f"{out_pdf.stem}_combined.pdf")))
    save_individual = bool(cfg.get("individual_plots", False))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    mc_statistics = cfg.get("mc_statistics", ["median"])
    asimov_statistics = cfg.get("asimov_statistics", ["r", "rstar"])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_combined_pdf.parent.mkdir(parents=True, exist_ok=True)
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
    make_combined_plot_on(
        s_vec=s_vec,
        b_values=b_values,
        Z_A_r=Z_A_r,
        Z_A_rstar=Z_A_rstar,
        Z_mc_median=Z_mc_median,
        out_pdf=out_combined_pdf,
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )

    if save_individual:
        print(f"Saved individual plots under: {out_pdf.parent.resolve()}")
    print(f"Saved all plots to: {out_pdf.resolve()}")
    print(f"Saved combined plot to: {out_combined_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/simple_medsig.yaml",
        help="Path to YAML config for simple medsig plots",
    )
    args = parser.parse_args()
    main(args.config)
