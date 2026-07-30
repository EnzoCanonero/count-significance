#!/usr/bin/env python3
"""Create the on/off median-significance plots used in the paper."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from count_significance.common import load_yaml
from count_significance.on_off import expected_significance_onoff


PLOT_FIGSIZE = (6.5, 6.5)
PLOT_ADJUST = {
    "left": 0.20,
    "right": 0.96,
    "bottom": 0.16,
    "top": 0.96,
}


# Apply the common style used by the median-significance plots.
def configure_plot_style() -> None:
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
def _finish_axes(ax) -> None:
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


# Apply the fixed margins used by single-panel figures.
def _finish_figure(fig) -> None:
    fig.subplots_adjust(**PLOT_ADJUST)


# Choose a y-axis maximum with a small margin above finite values.
def _medsig_ymax(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.max(values)) * 1.04, 1.0)


# Apply the common axes layout for median-significance plots.
def _style_medsig_axes(ax, b_values: np.ndarray, y_max: float, y_label: str) -> None:
    ax.set_box_aspect(1)
    ax.set_xscale("log")
    ax.set_xlim(float(b_values[0]), float(b_values[-1]))
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    _finish_axes(ax)


def _naive_z_fixed_tau(s_true: float, b_values: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate Z = s/sqrt(b + sigma_b^2) with sigma_b^2 = b/tau."""
    sigma_b2 = b_values / float(tau)
    return float(s_true) / np.sqrt(b_values + sigma_b2)


def _naive_z_fixed_rel_sig(s_true: float, b_values: np.ndarray, rel_sig: float) -> np.ndarray:
    """Evaluate Z = s/sqrt(b + sigma_b^2) for fixed sigma_b/b."""
    sigma_b2 = (float(rel_sig) * b_values) ** 2
    return float(s_true) / np.sqrt(b_values + sigma_b2)


# Use the configured display maximum, or infer one from the plotted values.
def _display_y_max(
    values: list[np.ndarray],
    z_display_max: Optional[float],
) -> float:
    if z_display_max is not None:
        return float(z_display_max)
    return _medsig_ymax(*values)


# Mask Monte Carlo markers above the display range without changing the inputs.
def mask_mc_for_display(
    results: dict[str, dict[str, np.ndarray]],
    z_display_max: Optional[float],
) -> dict[str, dict[str, np.ndarray]]:
    if z_display_max is None:
        return results

    display_results = {
        scan: {name: values.copy() for name, values in scan_results.items()}
        for scan, scan_results in results.items()
    }
    z_display_max = float(z_display_max)
    for scan_results in display_results.values():
        for key in ("Z_mc_median", "Z_mc_mean"):
            values = scan_results[key]
            values[values >= z_display_max] = np.nan
    return display_results


# Collect the statistical curves that determine the displayed y range.
def _selected_y_values(
    scan_results: dict[str, np.ndarray],
    s_idx: int,
    statistics: list[str],
    mc_summaries: list[str],
) -> list[np.ndarray]:
    # Exclude the naive comparison so it can run above the frame.
    values = []
    if "r" in statistics:
        values.append(scan_results["Z_A_r"][s_idx])
    if "rstar" in statistics:
        values.append(scan_results["Z_A_rstar"][s_idx])
    if "median" in mc_summaries:
        values.append(scan_results["Z_mc_median"][s_idx])
    if "mean" in mc_summaries:
        values.append(scan_results["Z_mc_mean"][s_idx])
    return values


# Select the y-axis label that matches the requested Monte Carlo summaries.
def _medsig_y_label(mc_summaries: list[str]) -> str:
    if "median" in mc_summaries and "mean" not in mc_summaries:
        return r"$\operatorname{med}[Z\mid s]$"
    return r"$Z$"


# Fill the signal-strength field in an output path template.
def _format_output_path(template: str, s_true: float) -> Path:
    return Path(str(template).format(s=_fmt(s_true)))


# Format a number for use in a file name.
def _fmt(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def compute_median_significance(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_rel_sig: np.ndarray,
    n_outer: int,
    mc_sigrel_z: float,
    min_toys: int,
    max_toys: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute Asimov and Monte Carlo significance grids for the on/off model.

    The Asimov counts are n_A = s + b and m_A = tau b. The scans keep either
    tau fixed or sigma_b/b fixed through tau = 1 / [(sigma_b/b)^2 b].
    """
    seed = int(seed)
    mc_sigrel_z = float(mc_sigrel_z)
    min_toys = int(min_toys)
    max_toys = int(max_toys)
    n_outer = int(n_outer)

    # Build the fixed-tau grid with shape (signal, tau, background).
    s_grid_tau = s_vec[:, None, None]
    b_grid_tau = b_values_tau[None, None, :]
    tau_grid = tau_vec[None, :, None]
    res_tau = expected_significance_onoff(
        s_true=s_grid_tau,
        b=b_grid_tau,
        tau=tau_grid,
        n_outer=n_outer,
        sigrel=mc_sigrel_z,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed,
    )

    # For fixed relative uncertainty, tau(b) = 1 / (rel_sig^2 b).
    s_grid_sig = s_vec[:, None, None]
    b_grid_rel_sig = b_values_rel_sig[None, None, :]
    tau_grid_rel_sig = 1.0 / (rel_sig_vec[None, :, None] ** 2 * b_grid_rel_sig)
    # Use an independent random stream for the second Monte Carlo scan.
    res_rel_sig = expected_significance_onoff(
        s_true=s_grid_sig,
        b=b_grid_rel_sig,
        tau=tau_grid_rel_sig,
        n_outer=n_outer,
        sigrel=mc_sigrel_z,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed + 1,
    )

    return {
        "fixed_tau": res_tau,
        "fixed_rel_sig": res_rel_sig,
    }


# Write one fixed-tau panel for each signal strength.
def write_combined_tau_plots(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    b_values: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    out_template: str,
    statistics: list[str],
    mc_summaries: list[str],
    z_display_max: Optional[float] = None,
) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fixed_tau = results["fixed_tau"]

    for s_idx, s_true in enumerate(s_vec):
        out_pdf = _format_output_path(out_template, float(s_true))
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)
        _plot_combined_tau_panel(
            ax,
            s_idx=s_idx,
            s_true=float(s_true),
            tau_vec=tau_vec,
            b_values=b_values,
            scan_results=fixed_tau,
            colors=colors,
            statistics=statistics,
            mc_summaries=mc_summaries,
            z_display_max=z_display_max,
        )
        _finish_figure(fig)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved combined fixed-tau plot to: {out_pdf.resolve()}")


# Write one fixed-relative-uncertainty panel for each signal strength.
def write_combined_rel_sig_plots(
    s_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    out_template: str,
    statistics: list[str],
    mc_summaries: list[str],
    z_display_max: Optional[float] = None,
) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fixed_rel_sig = results["fixed_rel_sig"]

    for s_idx, s_true in enumerate(s_vec):
        out_pdf = _format_output_path(out_template, float(s_true))
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)
        _plot_combined_rel_sig_panel(
            ax,
            s_idx=s_idx,
            s_true=float(s_true),
            rel_sig_vec=rel_sig_vec,
            b_values=b_values,
            scan_results=fixed_rel_sig,
            colors=colors,
            statistics=statistics,
            mc_summaries=mc_summaries,
            z_display_max=z_display_max,
        )
        _finish_figure(fig)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved combined fixed-relative-uncertainty plot to: {out_pdf.resolve()}")


# Write all fixed-tau and fixed-relative-uncertainty panels on one page.
def write_combined_grid_plot(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_rel_sig: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    out_pdf: Path,
    statistics: list[str],
    mc_summaries: list[str],
    z_display_max: Optional[float] = None,
) -> None:
    n_rows = len(s_vec)
    if n_rows == 0:
        raise ValueError("s_vec must contain at least one signal value")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.0, 6.17 * n_rows), dpi=150, squeeze=False)

    for s_idx, s_true in enumerate(s_vec):
        _plot_combined_tau_panel(
            axes[s_idx, 0],
            s_idx=s_idx,
            s_true=float(s_true),
            tau_vec=tau_vec,
            b_values=b_values_tau,
            scan_results=results["fixed_tau"],
            colors=colors,
            statistics=statistics,
            mc_summaries=mc_summaries,
            z_display_max=z_display_max,
        )
        _plot_combined_rel_sig_panel(
            axes[s_idx, 1],
            s_idx=s_idx,
            s_true=float(s_true),
            rel_sig_vec=rel_sig_vec,
            b_values=b_values_rel_sig,
            scan_results=results["fixed_rel_sig"],
            colors=colors,
            statistics=statistics,
            mc_summaries=mc_summaries,
            z_display_max=z_display_max,
        )

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.055, top=0.98, wspace=0.24, hspace=0.25)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved combined grid to: {out_pdf.resolve()}")


# Write the individual panels and the combined grid.
def write_median_significance_pdfs(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_rel_sig: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    out_tau_template: str,
    out_rel_sig_template: str,
    out_grid_pdf: Path,
    statistics: list[str],
    mc_summaries: list[str],
    z_display_max: Optional[float] = None,
) -> None:
    write_combined_tau_plots(
        s_vec=s_vec,
        tau_vec=tau_vec,
        b_values=b_values_tau,
        results=results,
        out_template=out_tau_template,
        statistics=statistics,
        mc_summaries=mc_summaries,
        z_display_max=z_display_max,
    )
    write_combined_rel_sig_plots(
        s_vec=s_vec,
        rel_sig_vec=rel_sig_vec,
        b_values=b_values_rel_sig,
        results=results,
        out_template=out_rel_sig_template,
        statistics=statistics,
        mc_summaries=mc_summaries,
        z_display_max=z_display_max,
    )
    write_combined_grid_plot(
        s_vec=s_vec,
        tau_vec=tau_vec,
        rel_sig_vec=rel_sig_vec,
        b_values_tau=b_values_tau,
        b_values_rel_sig=b_values_rel_sig,
        results=results,
        out_pdf=out_grid_pdf,
        statistics=statistics,
        mc_summaries=mc_summaries,
        z_display_max=z_display_max,
    )


# Draw one fixed-tau median-significance panel.
def _plot_combined_tau_panel(
    ax,
    s_idx: int,
    s_true: float,
    tau_vec: np.ndarray,
    b_values: np.ndarray,
    scan_results: dict[str, np.ndarray],
    colors: list[str],
    statistics: list[str],
    mc_summaries: list[str],
    z_display_max: Optional[float] = None,
) -> None:
    for tau_idx, tau in enumerate(tau_vec):
        color = colors[tau_idx % len(colors)]
        if "r" in statistics:
            ax.plot(b_values, scan_results["Z_A_r"][s_idx, tau_idx], color=color, linestyle="-")
        if "rstar" in statistics:
            ax.plot(
                b_values,
                scan_results["Z_A_rstar"][s_idx, tau_idx],
                color=color,
                linestyle="--",
            )
        ax.plot(
            b_values,
            _naive_z_fixed_tau(s_true, b_values, float(tau)),
            color=color,
            linestyle=":",
        )
        if "median" in mc_summaries:
            ax.plot(
                b_values,
                scan_results["Z_mc_median"][s_idx, tau_idx],
                color=color,
                linestyle="None",
                marker="o",
            )
        if "mean" in mc_summaries:
            ax.plot(
                b_values,
                scan_results["Z_mc_mean"][s_idx, tau_idx],
                color=color,
                linestyle="None",
                marker="+",
            )

    _style_medsig_axes(
        ax,
        b_values,
        _display_y_max(
            _selected_y_values(scan_results, s_idx, statistics, mc_summaries),
            z_display_max,
        ),
        _medsig_y_label(mc_summaries),
    )
    _add_combined_legends(
        ax,
        s_true=s_true,
        labels=[rf"$\tau={tau:g}$" for tau in tau_vec],
        colors=colors,
        n_configs=len(tau_vec),
        statistics=statistics,
        mc_summaries=mc_summaries,
    )


# Draw one fixed-relative-uncertainty median-significance panel.
def _plot_combined_rel_sig_panel(
    ax,
    s_idx: int,
    s_true: float,
    rel_sig_vec: np.ndarray,
    b_values: np.ndarray,
    scan_results: dict[str, np.ndarray],
    colors: list[str],
    statistics: list[str],
    mc_summaries: list[str],
    z_display_max: Optional[float] = None,
) -> None:
    # Show the larger relative uncertainty first, without changing the data order.
    display_order = np.argsort(rel_sig_vec)[::-1]
    display_colors = [
        colors[color_idx % len(colors)]
        for color_idx in range(len(display_order))
    ]

    for color_idx, rel_sig_idx in enumerate(display_order):
        rel_sig = rel_sig_vec[rel_sig_idx]
        color = display_colors[color_idx]
        if "r" in statistics:
            ax.plot(b_values, scan_results["Z_A_r"][s_idx, rel_sig_idx], color=color, linestyle="-")
        if "rstar" in statistics:
            ax.plot(
                b_values,
                scan_results["Z_A_rstar"][s_idx, rel_sig_idx],
                color=color,
                linestyle="--",
            )
        ax.plot(
            b_values,
            _naive_z_fixed_rel_sig(s_true, b_values, float(rel_sig)),
            color=color,
            linestyle=":",
        )
        if "median" in mc_summaries:
            ax.plot(
                b_values,
                scan_results["Z_mc_median"][s_idx, rel_sig_idx],
                color=color,
                linestyle="None",
                marker="o",
            )
        if "mean" in mc_summaries:
            ax.plot(
                b_values,
                scan_results["Z_mc_mean"][s_idx, rel_sig_idx],
                color=color,
                linestyle="None",
                marker="+",
            )

    _style_medsig_axes(
        ax,
        b_values,
        _display_y_max(
            _selected_y_values(scan_results, s_idx, statistics, mc_summaries),
            z_display_max,
        ),
        _medsig_y_label(mc_summaries),
    )
    _add_combined_legends(
        ax,
        s_true=s_true,
        labels=[
            rf"$\sigma_b/b={rel_sig_vec[rel_sig_idx]:g}$"
            for rel_sig_idx in display_order
        ],
        colors=display_colors,
        n_configs=len(rel_sig_vec),
        statistics=statistics,
        mc_summaries=mc_summaries,
    )


# Add the statistic and scan-configuration legends to a panel.
def _add_combined_legends(
    ax,
    s_true: float,
    labels: list[str],
    colors: list[str],
    n_configs: int,
    statistics: list[str],
    mc_summaries: list[str],
) -> None:
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
                label=r"Asimov $q_0^\ast$",
            )
        )
    stat_handles.append(
        Line2D(
            [0],
            [0],
            color="0.15",
            linestyle=":",
            label=r"$s/\sqrt{b+\sigma_b^2}$",
        )
    )
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
    stat_legend = ax.legend(
        handles=stat_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        **legend_kwargs,
    )
    ax.add_artist(stat_legend)

    if "rstar" in statistics:
        config_anchor_y = 0.69
    else:
        config_anchor_y = 0.72
    if "mean" in mc_summaries:
        config_anchor_y -= 0.07

    config_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            linestyle="None",
            label=rf"$s={float(s_true):g}$",
        )
    ]
    config_handles.extend(
        Line2D([0], [0], color=colors[idx % len(colors)], linestyle="-", label=labels[idx])
        for idx in range(n_configs)
    )
    ax.legend(
        handles=config_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, config_anchor_y),
        **legend_kwargs,
    )


# Load the configuration, calculate the grids, and write the plots.
def main(cfg_path: str) -> None:
    configure_plot_style()

    cfg = load_yaml(cfg_path)
    local_mc = cfg["local_mc"]
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tau_vec"], dtype=float)
    rel_sig_vec = np.asarray(cfg["rel_sig_vec"], dtype=float)
    b_values = np.logspace(
        np.log10(cfg["b_min"]),
        np.log10(cfg["b_max"]),
        int(local_mc["n_bpts"]),
    )

    n_outer = int(local_mc["n_outer"])
    seed = int(local_mc.get("seed", 12345))
    mc_sigrel_z = float(local_mc["mc_sigrel_Z"])
    min_toys = int(local_mc["min_toys"])
    max_toys = int(local_mc["max_toys"])

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

    z_display_max = cfg.get("Z_display_max")
    out_grid_pdf = Path(cfg["out_grid_pdf"])
    out_tau_template = str(cfg["out_tau_template"])
    out_rel_sig_template = str(cfg["out_rel_sig_template"])
    out_grid_pdf.parent.mkdir(parents=True, exist_ok=True)

    results = compute_median_significance(
        s_vec=s_vec,
        tau_vec=tau_vec,
        rel_sig_vec=rel_sig_vec,
        b_values_tau=b_values,
        b_values_rel_sig=b_values,
        n_outer=n_outer,
        mc_sigrel_z=mc_sigrel_z,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed,
    )

    # The display cap masks only Monte Carlo markers; continuous curves are unchanged.
    results = mask_mc_for_display(results, z_display_max)

    write_median_significance_pdfs(
        s_vec=s_vec,
        tau_vec=tau_vec,
        rel_sig_vec=rel_sig_vec,
        b_values_tau=b_values,
        b_values_rel_sig=b_values,
        results=results,
        out_tau_template=out_tau_template,
        out_rel_sig_template=out_rel_sig_template,
        out_grid_pdf=out_grid_pdf,
        statistics=statistics,
        mc_summaries=mc_summaries,
        z_display_max=z_display_max,
    )

    print(f"Saved plot to: {out_grid_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/paper_onoff_medsig.yaml",
        help="Path to YAML config for the uncertain-background paper plots",
    )
    args = parser.parse_args()
    main(args.config)
