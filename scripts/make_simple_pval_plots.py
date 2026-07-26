#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on import pvals_on


PLOT_FIGSIZE = (6.5, 6.5)


def _configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "lines.markersize": 7,
        }
    )


def _finish_axes(*axes):
    for ax in axes:
        if ax is None:
            continue
        ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
        ax.tick_params(axis="both", which="minor", width=1.0, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def _z_from_p(p):
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


def _correction_suffix(continuity_corrected: bool) -> str:
    return " (cc)" if continuity_corrected else ""


def _set_count_significance_limits(ax, n_vals: np.ndarray, *z_arrays: np.ndarray):
    n_vals = np.asarray(n_vals, dtype=float)
    if n_vals.size == 0:
        return

    x_pad = max(0.5, 0.06 * max(float(n_vals[-1] - n_vals[0]), 1.0))
    ax.set_xlim(float(n_vals[0]) - x_pad, float(n_vals[-1]) + x_pad)

    z_values = np.concatenate([np.ravel(np.asarray(values, dtype=float)) for values in z_arrays])
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size == 0:
        return
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_span = max(z_max - z_min, 1.0)
    ax.set_ylim(min(0.0, z_min - 0.04 * z_span), z_max + 0.10 * z_span)


def _n_max_for_target_z_on(s0: float, b: float, start_n: int, target_z: float, max_n: int = 10_000) -> int:
    """Increase n until the exact Poisson-tail significance reaches target_z."""
    if target_z <= 0.0:
        return int(start_n)

    target_p = float(norm.sf(target_z))
    n = int(start_n)
    while n < max_n:
        p_true = float(pvals_on(float(s0), float(b), np.array([n], dtype=int))["p_true"][0])
        if p_true <= target_p:
            return n
        n += 1

    raise RuntimeError(f"Failed to reach Z={target_z:g} by n={max_n}")


def compute_pvalues_on(
    s0: float,
    b: float,
    target_z: float = 5.0,
    max_observed_count: int = None,
    trim_to_discovery_tail: bool = True,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute p-values and relative differences for all n given a single (s0, b)."""
    eps = 1e-16
    mu0 = float(s0 + b)
    # Define the scan range for counts n:
    # keep the last plateau point, then scan the one-sided discovery tail.
    n_min = 0
    tail_threshold = mu0
    first_tail_n = int(np.floor(tail_threshold)) + 1
    start_n = max(n_min, first_tail_n - 1) if trim_to_discovery_tail else n_min
    if max_observed_count is None:
        n_max_start = max(n_min, int(np.ceil(mu0 + 5.0 * np.sqrt(mu0))))
        n_max = _n_max_for_target_z_on(s0, b, n_max_start, target_z)
    else:
        n_max = int(max_observed_count)
        if n_max < start_n:
            raise ValueError(f"max_observed_count={n_max} is below the scan start n={start_n}")
    n_vals = np.arange(start_n, n_max + 1, dtype=int)

    out = pvals_on(
        float(s0),
        float(b),
        n_vals,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )
    p_true = np.asarray(out["p_true"], dtype=float)
    p_r = np.asarray(out["p_r"], dtype=float)
    p_rstar = np.asarray(out["p_rstar"], dtype=float)

    denom = np.clip(p_true, eps, None)
    rel_r = np.abs(p_r - p_true) / denom
    rel_rstar = np.abs(p_rstar - p_true) / denom

    results = []
    results.append(
        {
            "s0": float(s0),
            "b": float(b),
            "mu0": mu0,
            "n_vals": n_vals,
            "n_min": int(n_vals[0]),
            "n_max": n_max,
            "tail_threshold": tail_threshold,
            "first_tail_n": first_tail_n,
            "p_true": np.maximum(p_true, eps),
            "p_r": np.maximum(p_r, eps),
            "p_rstar": np.maximum(p_rstar, eps),
            "rel_r": np.maximum(rel_r, eps),
            "rel_rstar": np.maximum(rel_rstar, eps),
        }
    )

    return results


def make_plot_on(
    results: list,
    out_pdf: Path,
    save_individual: bool,
    include_ratio: bool,
    continuity_correction_rstar: bool,
):
    """Render the p-value and relative-diff panels for all (s0, b) configurations."""
    rstar_suffix = _correction_suffix(continuity_correction_rstar)
    with PdfPages(out_pdf) as pdf:
        for res in results:
            s0 = res["s0"]
            b = res["b"]
            mu0 = res["mu0"]
            n_vals = res["n_vals"]
            n_min = res["n_min"]
            n_max = res["n_max"]
            p_true = res["p_true"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            rel_r = res["rel_r"]
            rel_rstar = res["rel_rstar"]
            first_tail_n = res["first_tail_n"]

            if include_ratio:
                fig, (ax_top, ax_bot) = plt.subplots(
                    2,
                    1,
                    figsize=PLOT_FIGSIZE,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3.5, 1.2]},
                )
            else:
                fig, ax_top = plt.subplots(figsize=PLOT_FIGSIZE)
                ax_bot = None

            ax_top.semilogy(
                n_vals,
                p_r,
                marker="o",
                linestyle="None",
                ms=5,
                label=r"$1-\Phi(q_0)$",
                color="0.15",
            )
            ax_top.semilogy(
                n_vals,
                p_rstar,
                marker="^",
                linestyle="None",
                ms=5,
                label=rf"$1-\Phi(q_0^\ast)${rstar_suffix}",
                color="0.15",
            )
            ax_top.semilogy(
                n_vals,
                p_true,
                marker="x",
                linestyle="None",
                ms=4,
                label="Exact",
                color="0.15",
            )
            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_vals[0], n_vals[-1])
            ax_top.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
            ax_top.grid(True, which="both", alpha=0.25)
            ax_top.legend(frameon=False, loc="upper right")

            if include_ratio:
                ax_bot.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_{q_0}-p_\mathrm{Exact}|/p_\mathrm{Exact}$",
                    color="0.15",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=rf"$|p_{{q_0^\ast}}-p_\mathrm{{Exact}}|/p_\mathrm{{Exact}}${rstar_suffix}",
                    color="0.15",
                )
                ax_bot.set_xlabel("Observed count n")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend(frameon=False)
            else:
                ax_top.set_xlabel("Observed count n")

            _finish_axes(ax_top, ax_bot)
            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"simple_pval_s{_fmt(s0)}_b{_fmt(b)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def make_significance_plot_on(
    results: list,
    out_pdf: Path,
    include_ratio: bool,
    continuity_correction_rstar: bool,
):
    """Render significance panels, optionally with relative-difference panels below."""
    rstar_suffix = _correction_suffix(continuity_correction_rstar)
    with PdfPages(out_pdf) as pdf:
        for res in results:
            s0 = res["s0"]
            b = res["b"]
            mu0 = res["mu0"]
            n_vals = res["n_vals"]
            n_min = res["n_min"]
            n_max = res["n_max"]
            p_true = res["p_true"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            rel_r = res["rel_r"]
            rel_rstar = res["rel_rstar"]
            first_tail_n = res["first_tail_n"]

            z_true = _z_from_p(p_true)
            z_r = _z_from_p(p_r)
            z_rstar = _z_from_p(p_rstar)

            if include_ratio:
                fig, (ax_z, ax_bot) = plt.subplots(
                    2,
                    1,
                    figsize=PLOT_FIGSIZE,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3.5, 1.2]},
                )
            else:
                fig, ax_z = plt.subplots(figsize=PLOT_FIGSIZE)
                ax_bot = None
                ax_z.set_box_aspect(1)

            ax_z.plot(n_vals, z_r, marker="o", linestyle="None", ms=5, label=r"$q_0$", color="0.15")
            ax_z.plot(
                n_vals,
                z_rstar,
                marker="^",
                linestyle="None",
                ms=5,
                label=rf"$q_0^\ast${rstar_suffix}",
                color="0.15",
            )
            ax_z.plot(
                n_vals,
                z_true,
                marker="x",
                linestyle="None",
                ms=4,
                label="Exact",
                color="0.15",
            )
            ax_z.set_ylabel(r"$Z$")
            _set_count_significance_limits(ax_z, n_vals, z_r, z_rstar, z_true)
            ax_z.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
            ax_z.grid(True, alpha=0.25)
            ax_z.legend(frameon=False, loc="lower right")

            if include_ratio:
                ax_bot.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_{q_0}-p_\mathrm{Exact}|/p_\mathrm{Exact}$",
                    color="0.15",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=rf"$|p_{{q_0^\ast}}-p_\mathrm{{Exact}}|/p_\mathrm{{Exact}}${rstar_suffix}",
                    color="0.15",
                )
                ax_bot.set_xlabel("Observed count n")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend(frameon=False)
            else:
                ax_z.set_xlabel("Observed count n")

            _finish_axes(ax_z, ax_bot)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    target_z = float(cfg.get("target_Z", 5.0))
    max_observed_count = cfg.get("max_observed_count")
    max_observed_count = None if max_observed_count is None else int(max_observed_count)
    out_pdf = Path(cfg["out_pdf"])
    out_significance_pdf = Path(
        cfg.get("out_significance_pdf", out_pdf.with_name(f"{out_pdf.stem}_with_significance.pdf"))
    )
    save_individual = bool(cfg.get("individual_plots", False))
    include_ratio = bool(cfg.get("ratio_plots", False))
    make_significance_plots = bool(cfg.get("significance_plots", cfg.get("significance_pvalue_plots", True)))
    trim_to_discovery_tail = bool(cfg.get("trim_to_discovery_tail", True))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_significance_pdf.parent.mkdir(parents=True, exist_ok=True)

    combos = np.array(np.meshgrid(s0_vec, b_vec)).T.reshape(-1, 2)
    all_results = []
    for s0, b in combos:
        all_results.extend(
            compute_pvalues_on(
                float(s0),
                float(b),
                target_z=target_z,
                max_observed_count=max_observed_count,
                trim_to_discovery_tail=trim_to_discovery_tail,
                continuity_correction_r=continuity_correction_r,
                continuity_correction_rstar=continuity_correction_rstar,
            )
        )
    make_plot_on(
        all_results,
        out_pdf,
        save_individual,
        include_ratio,
        continuity_correction_rstar=continuity_correction_rstar,
    )
    if make_significance_plots:
        make_significance_plot_on(
            all_results,
            out_significance_pdf,
            include_ratio,
            continuity_correction_rstar=continuity_correction_rstar,
        )

    print(f"Saved all plots to: {out_pdf.resolve()}")
    if make_significance_plots:
        print(f"Saved significance plots to: {out_significance_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/simple_pval.yaml",
        help="Path to YAML config for known-background p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
