#!/usr/bin/env python3
"""Create the known-background median-significance plot used in the paper."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on import expected_significance_on


PLOT_FIGSIZE = (6.5, 6.5)


# Apply the common style used by the median-significance plots.
def _configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
        }
    )


# Apply the final tick and spine styling to an axis.
def _finish_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


# Choose a y-axis maximum with a small margin above finite values.
def _medsig_ymax(*arrays) -> float:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.max(values)) * 1.04, 1.0)


# Apply the common axes layout for median-significance plots.
def _style_medsig_axes(ax, b_values: np.ndarray, y_max: float, y_label: str):
    ax.set_box_aspect(1)
    ax.set_xscale("log")
    ax.set_xlim(float(b_values[0]), float(b_values[-1]))
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    _finish_axes(ax)


# Mark a legend entry when the continuity correction is applied.
def _correction_suffix(continuity_corrected: bool) -> str:
    return " (cc)" if continuity_corrected else ""


# Select the y-axis label that matches the requested Monte Carlo summaries.
def _medsig_y_label(mc_summaries: list) -> str:
    if "median" in mc_summaries and "mean" not in mc_summaries:
        return r"$\operatorname{med}[Z\mid s]$"
    return r"$Z$"


def compute_median_significance(
    s_vec: np.ndarray,
    b_values: np.ndarray,
    n_outer: int,
    seed: int,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute Asimov and Monte Carlo significance grids for known background.

    The Asimov count is n_A = s + b, while the Monte Carlo summaries use
    Poisson data toys. The calculation is broadcast over signal strengths s
    and means b.
    """
    seed = int(seed)
    n_outer = int(n_outer)

    # Build the grid with shape (signal, background).
    s_grid = s_vec[:, None]
    b_grid = b_values[None, :]

    return expected_significance_on(
        s_true=s_grid,
        b=b_grid,
        n_outer=n_outer,
        seed=seed,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )


# Write the known-background median-significance figure.
def write_median_significance_pdf(
    s_vec: np.ndarray,
    b_values: np.ndarray,
    results: dict,
    out_pdf: Path,
    statistics: list,
    mc_summaries: list,
    continuity_correction_rstar: bool,
):
    Z_A_r = results["Z_A_r"]
    Z_A_rstar = results["Z_A_rstar"]
    Z_mc_median = results["Z_mc_median"]
    Z_mc_mean = results["Z_mc_mean"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)

    for idx, s_true in enumerate(s_vec):
        color = colors[idx % len(colors)]
        if "r" in statistics:
            ax.plot(b_values, Z_A_r[idx], color=color, linestyle="-")
        if "rstar" in statistics:
            ax.plot(b_values, Z_A_rstar[idx], color=color, linestyle="--")
        ax.plot(b_values, float(s_true) / np.sqrt(b_values), color=color, linestyle=":")
        if "median" in mc_summaries:
            ax.plot(b_values, Z_mc_median[idx], color=color, linestyle="None", marker="o")
        if "mean" in mc_summaries:
            ax.plot(b_values, Z_mc_mean[idx], color=color, linestyle="None", marker="+")

    # Exclude the naive comparison so it can run above the frame.
    values_for_limits = []
    if "r" in statistics:
        values_for_limits.append(Z_A_r)
    if "rstar" in statistics:
        values_for_limits.append(Z_A_rstar)
    if "median" in mc_summaries:
        values_for_limits.append(Z_mc_median)
    if "mean" in mc_summaries:
        values_for_limits.append(Z_mc_mean)

    _style_medsig_axes(
        ax,
        b_values,
        _medsig_ymax(*values_for_limits),
        _medsig_y_label(mc_summaries),
    )

    stat_handles = []
    if "r" in statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", linestyle="-", label=r"Asimov $q_0$"))
    if "rstar" in statistics:
        stat_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                linestyle="--",
                label=rf"Asimov $q_0^\ast${_correction_suffix(continuity_correction_rstar)}",
            )
        )
    stat_handles.append(Line2D([0], [0], color="0.15", linestyle=":", label=r"$s/\sqrt{b}$"))
    if "median" in mc_summaries:
        stat_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                marker="o",
                linestyle="None",
                label=r"MC median",
            )
        )
    if "mean" in mc_summaries:
        stat_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                marker="+",
                linestyle="None",
                label=r"MC mean",
            )
        )

    legend_kwargs = {
        "frameon": False,
        "fontsize": 13,
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


# Load the configuration, calculate the grid, and write the plot.
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
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    selected_statistics = cfg.get("statistics", ["r", "rstar"])
    if not isinstance(selected_statistics, list):
        raise ValueError("statistics must be a YAML list")
    statistics = [str(statistic).lower() for statistic in selected_statistics]
    for statistic in statistics:
        if statistic not in ("r", "rstar"):
            raise ValueError(f"Unknown statistic={statistic!r}")

    selected_mc_summaries = cfg.get("mc_summaries", ["median"])
    if not isinstance(selected_mc_summaries, list):
        raise ValueError("mc_summaries must be a YAML list")
    mc_summaries = [str(summary).lower() for summary in selected_mc_summaries]
    for summary in mc_summaries:
        if summary not in ("median", "mean"):
            raise ValueError(f"Unknown MC summary={summary!r}")

    if not statistics and not mc_summaries:
        raise ValueError("Select at least one statistic or Monte Carlo summary")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    b_values = np.logspace(np.log10(b_min), np.log10(b_max), n_bpts)

    results = compute_median_significance(
        s_vec=s_vec,
        b_values=b_values,
        n_outer=n_outer,
        seed=seed,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )

    write_median_significance_pdf(
        s_vec=s_vec,
        b_values=b_values,
        results=results,
        out_pdf=out_pdf,
        statistics=statistics,
        mc_summaries=mc_summaries,
        continuity_correction_rstar=continuity_correction_rstar,
    )

    print(f"Saved plot to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/paper_simple_medsig.yaml",
        help="Path to YAML config for the known-background paper plots",
    )
    args = parser.parse_args()
    main(args.config)
