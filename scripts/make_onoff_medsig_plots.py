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
from src.on_off import expected_significance_onoff


PLOT_FIGSIZE = (6.5, 6.5)
PLOT_ADJUST = {
    "left": 0.20,
    "right": 0.96,
    "bottom": 0.16,
    "top": 0.96,
}


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


def _finish_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _finish_figure(fig):
    fig.subplots_adjust(**PLOT_ADJUST)


def _medsig_ymax(*arrays) -> float:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.max(values)) * 1.04, 1.0)


def _style_medsig_axes(ax, b_values: np.ndarray, y_max: float):
    ax.set_box_aspect(1)
    ax.set_xscale("log")
    ax.set_xlim(float(b_values[0]), float(b_values[-1]))
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(r"$\operatorname{med}[Z\mid s]$")
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    _finish_axes(ax)


def _naive_z_fixed_tau(s_true: float, b_values: np.ndarray, tau: float) -> np.ndarray:
    sigma_b2 = b_values / float(tau)
    return float(s_true) / np.sqrt(b_values + sigma_b2)


def _naive_z_fixed_sigrel(s_true: float, b_values: np.ndarray, sigma_rel: float) -> np.ndarray:
    sigma_b2 = (float(sigma_rel) * b_values) ** 2
    return float(s_true) / np.sqrt(b_values + sigma_b2)


def _cap(arr, y_ceiling):
    """Cap an array at y_ceiling (no-op when y_ceiling is None)."""
    return np.minimum(arr, y_ceiling) if y_ceiling is not None else arr


def _format_output_path(template: str, s_true: float) -> Path:
    return Path(str(template).format(s=_fmt(s_true), s_value=f"{float(s_true):g}"))


# Call graph (uncertain-background medsig)
# - main: load config, prepare b grids, then run experiments and plot
#   - run_experiments_onoff: compute Asimov + MC-median Z grids (vectorised expected_significance_onoff)
#   - make_plots_onoff: save combined/individual PDFs (per-τ and per-σ_rel panels)


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def run_experiments_onoff(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_sig: np.ndarray,
    n_outer: int,
    sigrel: float,
    min_toys: int,
    max_toys: int,
    seed: int,
):
    """
    Compute Asimov and MC-median Z grids for the uncertain-background case via expected_significance_onoff.

    Returns fixed-τ arrays shaped (len(s_vec), len(tau_vec), len(b_values_tau))
    and fixed-σ_rel arrays shaped (len(s_vec), len(rel_sig_vec), len(b_values_sig)).
    """
    seed = int(seed)
    sigrel = float(sigrel)
    min_toys = int(min_toys)
    max_toys = int(max_toys)
    n_outer = int(n_outer)

    # Vectorised grids for fixed-τ scans: shape (S, T, B_tau)
    s_grid_tau = s_vec[:, None, None]
    b_grid_tau = b_values_tau[None, None, :]
    tau_grid = tau_vec[None, :, None]
    res_tau = expected_significance_onoff(
        s_true=s_grid_tau,
        b=b_grid_tau,
        tau=tau_grid,
        n_outer=n_outer,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed,
    )

    # Vectorised grids for fixed-σ_rel scans: tau(b) = 1/(sigma_rel^2 * b)
    s_grid_sig = s_vec[:, None, None]
    b_grid_sig = b_values_sig[None, None, :]
    tau_grid_sig = 1.0 / (rel_sig_vec[None, :, None] ** 2 * b_grid_sig)
    res_sig = expected_significance_onoff(
        s_true=s_grid_sig,
        b=b_grid_sig,
        tau=tau_grid_sig,
        n_outer=n_outer,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed + 1,  # decorrelate from fixed-tau call
    )

    return (
        res_tau["Z_A_r"],
        res_tau["Z_A_rstar"],
        res_sig["Z_A_r"],
        res_sig["Z_A_rstar"],
        res_tau["Z_mc_median"],
        res_sig["Z_mc_median"],
        res_tau["Z_mc_mean"],
        res_sig["Z_mc_mean"],
    )


def make_plots_onoff(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_sig: np.ndarray,
    Z_A_r_tau: np.ndarray,
    Z_A_rstar_tau: np.ndarray,
    Z_A_r_sig: np.ndarray,
    Z_A_rstar_sig: np.ndarray,
    Z_med_tau: np.ndarray,
    Z_med_sig: np.ndarray,
    Z_mean_tau: np.ndarray,
    Z_mean_sig: np.ndarray,
    out_pdf: Path,
    save_individual: bool,
    outdir: Path,
    mc_statistics: list,
    asimov_statistics: list,
    y_ceiling: float = None,
):
    """
    Save combined PDF (and optional per-plot PDFs) for the uncertain-background medsig scans.

    If y_ceiling is set, only the y-axis top is capped at it (computed from the
    capped Asimov q0). The continuous approximations (Asimov q0/q0* and the naive
    curve) are still drawn at their true values, so they run off the top margin
    above y_ceiling. MC markers are expected to be capped by the caller.
    """
    with PdfPages(out_pdf) as pdf:
        for s_idx, s_true in enumerate(s_vec):
            for tau_idx, tau in enumerate(tau_vec):
                fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)
                if "r" in asimov_statistics:
                    ax.plot(b_values_tau, Z_A_r_tau[s_idx, tau_idx], label=r"Asimov $q_0$")
                if "rstar" in asimov_statistics:
                    ax.plot(
                        b_values_tau,
                        Z_A_rstar_tau[s_idx, tau_idx],
                        "--",
                        label=r"Asimov $q_0^\ast$",
                    )
                ax.plot(
                    b_values_tau,
                    _naive_z_fixed_tau(float(s_true), b_values_tau, float(tau)),
                    ":",
                    label=r"$s/\sqrt{b+\sigma_b^2}$",
                )
                if "median" in mc_statistics:
                    ax.plot(
                        b_values_tau,
                        Z_med_tau[s_idx, tau_idx],
                        linestyle="None",
                        marker="o",
                        label=r"MC median $Z$",
                    )
                if "mean" in mc_statistics:
                    ax.plot(
                        b_values_tau,
                        Z_mean_tau[s_idx, tau_idx],
                        linestyle="None",
                        marker="+",
                        label=r"MC mean $Z$",
                    )
                _style_medsig_axes(ax, b_values_tau, _medsig_ymax(_cap(Z_A_r_tau[s_idx, tau_idx], y_ceiling)))
                ax.legend(frameon=False, loc="upper right")
                _finish_figure(fig)
                if save_individual:
                    fname = outdir / f"onoff_bscan_s{_fmt(s_true)}_tau{_fmt(tau)}.pdf"
                    fig.savefig(fname)
                pdf.savefig(fig)
                plt.close(fig)

            for sig_idx, sigma_rel in enumerate(rel_sig_vec):
                fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)
                if "r" in asimov_statistics:
                    ax.plot(b_values_sig, Z_A_r_sig[s_idx, sig_idx], label=r"Asimov $q_0$")
                if "rstar" in asimov_statistics:
                    ax.plot(
                        b_values_sig,
                        Z_A_rstar_sig[s_idx, sig_idx],
                        "--",
                        label=r"Asimov $q_0^\ast$",
                    )
                ax.plot(
                    b_values_sig,
                    _naive_z_fixed_sigrel(float(s_true), b_values_sig, float(sigma_rel)),
                    ":",
                    label=r"$s/\sqrt{b+\sigma_b^2}$",
                )
                if "median" in mc_statistics:
                    ax.plot(
                        b_values_sig,
                        Z_med_sig[s_idx, sig_idx],
                        linestyle="None",
                        marker="o",
                        label=r"MC median $Z$",
                    )
                if "mean" in mc_statistics:
                    ax.plot(
                        b_values_sig,
                        Z_mean_sig[s_idx, sig_idx],
                        linestyle="None",
                        marker="+",
                        label=r"MC mean $Z$",
                    )
                _style_medsig_axes(ax, b_values_sig, _medsig_ymax(_cap(Z_A_r_sig[s_idx, sig_idx], y_ceiling)))
                ax.legend(frameon=False, loc="upper right")
                _finish_figure(fig)
                if save_individual:
                    fname = outdir / f"onoff_bscan_s{_fmt(s_true)}_sigrel{_fmt(sigma_rel)}.pdf"
                    fig.savefig(fname)
                pdf.savefig(fig)
                plt.close(fig)


def make_combined_tau_plots(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    b_values: np.ndarray,
    Z_A_r_tau: np.ndarray,
    Z_A_rstar_tau: np.ndarray,
    Z_med_tau: np.ndarray,
    out_template: str,
    mc_statistics: list,
    asimov_statistics: list,
    y_ceiling: float = None,
):
    """Save one fixed-tau combined panel per signal strength."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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
            Z_A_r_tau=Z_A_r_tau,
            Z_A_rstar_tau=Z_A_rstar_tau,
            Z_med_tau=Z_med_tau,
            colors=colors,
            mc_statistics=mc_statistics,
            asimov_statistics=asimov_statistics,
            y_ceiling=y_ceiling,
        )
        _finish_figure(fig)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved combined fixed-tau plot to: {out_pdf.resolve()}")


def make_combined_sigrel_plots(
    s_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values: np.ndarray,
    Z_A_r_sig: np.ndarray,
    Z_A_rstar_sig: np.ndarray,
    Z_med_sig: np.ndarray,
    out_template: str,
    mc_statistics: list,
    asimov_statistics: list,
    y_ceiling: float = None,
):
    """Save one fixed-relative-uncertainty combined panel per signal strength."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for s_idx, s_true in enumerate(s_vec):
        out_pdf = _format_output_path(out_template, float(s_true))
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)
        _plot_combined_sigrel_panel(
            ax,
            s_idx=s_idx,
            s_true=float(s_true),
            rel_sig_vec=rel_sig_vec,
            b_values=b_values,
            Z_A_r_sig=Z_A_r_sig,
            Z_A_rstar_sig=Z_A_rstar_sig,
            Z_med_sig=Z_med_sig,
            colors=colors,
            mc_statistics=mc_statistics,
            asimov_statistics=asimov_statistics,
            y_ceiling=y_ceiling,
        )
        _finish_figure(fig)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved combined fixed-sigma plot to: {out_pdf.resolve()}")


def make_combined_grid_plot(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_sig: np.ndarray,
    Z_A_r_tau: np.ndarray,
    Z_A_rstar_tau: np.ndarray,
    Z_A_r_sig: np.ndarray,
    Z_A_rstar_sig: np.ndarray,
    Z_med_tau: np.ndarray,
    Z_med_sig: np.ndarray,
    out_pdf: Path,
    mc_statistics: list,
    asimov_statistics: list,
    y_ceiling: float = None,
):
    """Save all fixed-tau and fixed-sigma panels as one n_signal x 2 PDF page."""
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
            Z_A_r_tau=Z_A_r_tau,
            Z_A_rstar_tau=Z_A_rstar_tau,
            Z_med_tau=Z_med_tau,
            colors=colors,
            mc_statistics=mc_statistics,
            asimov_statistics=asimov_statistics,
            y_ceiling=y_ceiling,
        )
        _plot_combined_sigrel_panel(
            axes[s_idx, 1],
            s_idx=s_idx,
            s_true=float(s_true),
            rel_sig_vec=rel_sig_vec,
            b_values=b_values_sig,
            Z_A_r_sig=Z_A_r_sig,
            Z_A_rstar_sig=Z_A_rstar_sig,
            Z_med_sig=Z_med_sig,
            colors=colors,
            mc_statistics=mc_statistics,
            asimov_statistics=asimov_statistics,
            y_ceiling=y_ceiling,
        )

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.055, top=0.98, wspace=0.24, hspace=0.25)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved 3x2 combined grid to: {out_pdf.resolve()}")


def _plot_combined_tau_panel(
    ax,
    s_idx: int,
    s_true: float,
    tau_vec: np.ndarray,
    b_values: np.ndarray,
    Z_A_r_tau: np.ndarray,
    Z_A_rstar_tau: np.ndarray,
    Z_med_tau: np.ndarray,
    colors: list,
    mc_statistics: list,
    asimov_statistics: list,
    y_ceiling: float = None,
):
    for tau_idx, tau in enumerate(tau_vec):
        color = colors[tau_idx % len(colors)]
        if "r" in asimov_statistics:
            ax.plot(b_values, Z_A_r_tau[s_idx, tau_idx], color=color, linestyle="-")
        if "rstar" in asimov_statistics:
            ax.plot(b_values, Z_A_rstar_tau[s_idx, tau_idx], color=color, linestyle="--")
        ax.plot(b_values, _naive_z_fixed_tau(s_true, b_values, float(tau)), color=color, linestyle=":")
        if "median" in mc_statistics:
            ax.plot(b_values, Z_med_tau[s_idx, tau_idx], color=color, linestyle="None", marker="o")

    _style_medsig_axes(ax, b_values, _medsig_ymax(_cap(Z_A_r_tau[s_idx], y_ceiling)))
    _add_combined_legends(
        ax,
        s_true=s_true,
        labels=[rf"$\tau={tau:g}$" for tau in tau_vec],
        colors=colors,
        n_configs=len(tau_vec),
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )


def _plot_combined_sigrel_panel(
    ax,
    s_idx: int,
    s_true: float,
    rel_sig_vec: np.ndarray,
    b_values: np.ndarray,
    Z_A_r_sig: np.ndarray,
    Z_A_rstar_sig: np.ndarray,
    Z_med_sig: np.ndarray,
    colors: list,
    mc_statistics: list,
    asimov_statistics: list,
    y_ceiling: float = None,
):
    for sig_idx, sigma_rel in enumerate(rel_sig_vec):
        color = colors[sig_idx % len(colors)]
        if "r" in asimov_statistics:
            ax.plot(b_values, Z_A_r_sig[s_idx, sig_idx], color=color, linestyle="-")
        if "rstar" in asimov_statistics:
            ax.plot(b_values, Z_A_rstar_sig[s_idx, sig_idx], color=color, linestyle="--")
        ax.plot(
            b_values,
            _naive_z_fixed_sigrel(s_true, b_values, float(sigma_rel)),
            color=color,
            linestyle=":",
        )
        if "median" in mc_statistics:
            ax.plot(b_values, Z_med_sig[s_idx, sig_idx], color=color, linestyle="None", marker="o")

    _style_medsig_axes(ax, b_values, _medsig_ymax(_cap(Z_A_r_sig[s_idx], y_ceiling)))
    _add_combined_legends(
        ax,
        s_true=s_true,
        labels=[rf"$\sigma_b/b={sigma_rel:g}$" for sigma_rel in rel_sig_vec],
        colors=colors,
        n_configs=len(rel_sig_vec),
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )


def _add_combined_legends(
    ax,
    s_true: float,
    labels: list,
    colors: list,
    n_configs: int,
    mc_statistics: list,
    asimov_statistics: list,
):
    stat_handles = []
    if "r" in asimov_statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", linestyle="-", label=r"Asimov $q_0$"))
    if "rstar" in asimov_statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", linestyle="--", label=r"Asimov $q_0^\ast$"))
    stat_handles.append(Line2D([0], [0], color="0.15", linestyle=":", label=r"$s/\sqrt{b+\sigma_b^2}$"))
    if "median" in mc_statistics:
        stat_handles.append(Line2D([0], [0], color="0.15", marker="o", linestyle="None", label=r"MC median"))

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

    if "rstar" in asimov_statistics:
        config_anchor_y = 0.69
    else:
        config_anchor_y = 0.72

    config_handles = [Line2D([0], [0], color="none", linestyle="None", label=rf"$s={float(s_true):g}$")]
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


def main(cfg_path: str):
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    cfg["s_vec"] = np.asarray(cfg["s_vec"], dtype=float)
    cfg["tauVec"] = np.asarray(cfg["tauVec"], dtype=float)
    cfg["relSigVec"] = np.asarray(cfg["relSigVec"], dtype=float)

    b_values_tau = np.logspace(
        np.log10(cfg["b_min_tau"]), np.log10(cfg["b_max_tau"]), int(cfg["n_bpts_tau"])
    )
    b_values_sig = np.logspace(
        np.log10(cfg["b_min_sig"]), np.log10(cfg["b_max_sig"]), int(cfg["n_bpts_sig"])
    )

    n_outer = int(cfg["n_outer"])
    seed = int(cfg.get("outer_seed", cfg.get("seed", 12345)))
    sigrel = float(cfg["sigrel_Z"])
    min_toys = int(cfg["min_toys"])
    max_toys = int(cfg["max_toys"])

    outdir = Path(cfg.get("outdir", "plots"))
    save_individual = bool(cfg.get("individual_plots", False))
    mc_statistics = cfg.get("mc_statistics", ["median"])
    asimov_statistics = cfg.get("asimov_statistics", ["r", "rstar"])
    outdir.mkdir(parents=True, exist_ok=True)
    out_pdf = Path(cfg.get("out_summary_pdf", outdir / "onoff_medsig.pdf"))
    out_combined_tau_template = str(
        cfg.get("out_combined_tau_template", outdir / "onoff_medsig_combined_tau_s{s}.pdf")
    )
    out_combined_sigrel_template = str(
        cfg.get("out_combined_sigrel_template", outdir / "onoff_medsig_combined_sigrel_s{s}.pdf")
    )
    out_grid_pdf = cfg.get("out_grid_pdf")
    alternate_asimov_statistics = cfg.get("alternate_asimov_statistics")
    alternate_out_pdf = cfg.get("alternate_out_summary_pdf")
    alternate_out_combined_tau_template = cfg.get("alternate_out_combined_tau_template")
    alternate_out_combined_sigrel_template = cfg.get("alternate_out_combined_sigrel_template")
    alternate_out_grid_pdf = cfg.get("alternate_out_grid_pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    (
        Z_A_r_tau,
        Z_A_rstar_tau,
        Z_A_r_sig,
        Z_A_rstar_sig,
        Z_med_tau,
        Z_med_sig,
        Z_mean_tau,
        Z_mean_sig,
    ) = run_experiments_onoff(
        s_vec=cfg["s_vec"],
        tau_vec=cfg["tauVec"],
        rel_sig_vec=cfg["relSigVec"],
        b_values_tau=b_values_tau,
        b_values_sig=b_values_sig,
        n_outer=n_outer,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed,
    )

    make_plots_onoff(
        s_vec=cfg["s_vec"],
        tau_vec=cfg["tauVec"],
        rel_sig_vec=cfg["relSigVec"],
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
    )
    make_combined_tau_plots(
        s_vec=cfg["s_vec"],
        tau_vec=cfg["tauVec"],
        b_values=b_values_tau,
        Z_A_r_tau=Z_A_r_tau,
        Z_A_rstar_tau=Z_A_rstar_tau,
        Z_med_tau=Z_med_tau,
        out_template=out_combined_tau_template,
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )
    make_combined_sigrel_plots(
        s_vec=cfg["s_vec"],
        rel_sig_vec=cfg["relSigVec"],
        b_values=b_values_sig,
        Z_A_r_sig=Z_A_r_sig,
        Z_A_rstar_sig=Z_A_rstar_sig,
        Z_med_sig=Z_med_sig,
        out_template=out_combined_sigrel_template,
        mc_statistics=mc_statistics,
        asimov_statistics=asimov_statistics,
    )
    if out_grid_pdf:
        make_combined_grid_plot(
            s_vec=cfg["s_vec"],
            tau_vec=cfg["tauVec"],
            rel_sig_vec=cfg["relSigVec"],
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
        )

    if alternate_asimov_statistics:
        alternate_out_pdf = Path(alternate_out_pdf or out_pdf.with_name(f"{out_pdf.stem}_alternate.pdf"))
        alternate_out_pdf.parent.mkdir(parents=True, exist_ok=True)
        make_plots_onoff(
            s_vec=cfg["s_vec"],
            tau_vec=cfg["tauVec"],
            rel_sig_vec=cfg["relSigVec"],
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
            out_pdf=alternate_out_pdf,
            save_individual=False,
            outdir=outdir,
            mc_statistics=mc_statistics,
            asimov_statistics=alternate_asimov_statistics,
        )
        if alternate_out_combined_tau_template:
            make_combined_tau_plots(
                s_vec=cfg["s_vec"],
                tau_vec=cfg["tauVec"],
                b_values=b_values_tau,
                Z_A_r_tau=Z_A_r_tau,
                Z_A_rstar_tau=Z_A_rstar_tau,
                Z_med_tau=Z_med_tau,
                out_template=str(alternate_out_combined_tau_template),
                mc_statistics=mc_statistics,
                asimov_statistics=alternate_asimov_statistics,
            )
        if alternate_out_combined_sigrel_template:
            make_combined_sigrel_plots(
                s_vec=cfg["s_vec"],
                rel_sig_vec=cfg["relSigVec"],
                b_values=b_values_sig,
                Z_A_r_sig=Z_A_r_sig,
                Z_A_rstar_sig=Z_A_rstar_sig,
                Z_med_sig=Z_med_sig,
                out_template=str(alternate_out_combined_sigrel_template),
                mc_statistics=mc_statistics,
                asimov_statistics=alternate_asimov_statistics,
            )
        if alternate_out_grid_pdf:
            make_combined_grid_plot(
                s_vec=cfg["s_vec"],
                tau_vec=cfg["tauVec"],
                rel_sig_vec=cfg["relSigVec"],
                b_values_tau=b_values_tau,
                b_values_sig=b_values_sig,
                Z_A_r_tau=Z_A_r_tau,
                Z_A_rstar_tau=Z_A_rstar_tau,
                Z_A_r_sig=Z_A_r_sig,
                Z_A_rstar_sig=Z_A_rstar_sig,
                Z_med_tau=Z_med_tau,
                Z_med_sig=Z_med_sig,
                out_pdf=Path(alternate_out_grid_pdf),
                mc_statistics=mc_statistics,
                asimov_statistics=alternate_asimov_statistics,
            )

    if save_individual:
        print(f"Saved individual plots under: {outdir.resolve()}")
    print(f"Saved all plots to: {out_pdf.resolve()}")
    if alternate_asimov_statistics:
        print(f"Saved alternate plots to: {alternate_out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/onoff_medsig.yaml",
        help="Path to YAML config for uncertain-background medsig plots",
    )
    args = parser.parse_args()
    main(args.config)
