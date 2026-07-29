#!/usr/bin/env python3
"""Create the known-background observed-significance plots used in the paper."""

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


# Apply the common style used by the observed-significance plots.
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


# Apply the final tick and spine styling to an axis.
def _finish_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _z_from_p(p):
    """Convert an upper-tail p-value to Z = Phi^(-1)(1 - p)."""
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


# Mark a legend entry when the continuity correction is applied.
def _correction_suffix(continuity_corrected: bool) -> str:
    return " (cc)" if continuity_corrected else ""


# Set readable limits for an observed-count significance panel.
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


def _n_max_for_target_z_on(
    s0: float,
    b: float,
    start_n: int,
    target_z: float,
    max_n: int = 10_000,
) -> int:
    """Find the first count whose inclusive Poisson tail reaches target Z."""
    if target_z <= 0.0:
        return int(start_n)

    target_p = float(norm.sf(target_z))
    n = int(start_n)
    while n < max_n:
        p_ref = float(
            pvals_on(float(s0), float(b), np.array([n], dtype=int))["p_exact"][0]
        )
        if p_ref <= target_p:
            return n
        n += 1

    raise RuntimeError(f"Failed to reach Z={target_z:g} by n={max_n}")


def compute_pvalue_scans(
    s0: float,
    b: float,
    target_z: float = 5.0,
    max_observed_count: int = None,
    trim_to_discovery_tail: bool = True,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Scan the one-sided discovery tail for fixed signal and background.

    The inclusive Poisson tail is compared with the q0 and q0* asymptotic
    p-values returned by pvals_on.
    """
    mu0 = float(s0 + b)
    # Keep the final plateau point before scanning the discovery tail.
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
    p_ref = np.asarray(out["p_exact"], dtype=float)
    p_r = np.asarray(out["p_r"], dtype=float)
    p_rstar = np.asarray(out["p_rstar"], dtype=float)

    return {
        "n_vals": n_vals,
        "first_tail_n": first_tail_n,
        "p_ref": p_ref,
        "p_r": p_r,
        "p_rstar": p_rstar,
    }


# Write the p-value scan panels.
def write_pvalue_pdf(
    results: list,
    out_pdf: Path,
    statistics: list,
    continuity_correction_rstar: bool,
):
    rstar_suffix = _correction_suffix(continuity_correction_rstar)
    with PdfPages(out_pdf) as pdf:
        for res in results:
            n_vals = res["n_vals"]
            p_ref = res["p_ref"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            first_tail_n = res["first_tail_n"]

            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

            if "r" in statistics:
                ax.semilogy(
                    n_vals,
                    p_r,
                    marker="o",
                    linestyle="None",
                    ms=5,
                    label=r"$1-\Phi(\sqrt{q_0})$",
                    color="0.15",
                )
            if "rstar" in statistics:
                ax.semilogy(
                    n_vals,
                    p_rstar,
                    marker="^",
                    linestyle="None",
                    ms=5,
                    label=rf"$1-\Phi(\sqrt{{q_0^\ast}})${rstar_suffix}",
                    color="0.15",
                )
            ax.semilogy(
                n_vals,
                p_ref,
                marker="x",
                linestyle="None",
                ms=4,
                label="Exact",
                color="0.15",
            )
            ax.set_ylabel("p-value (upper tail)")
            ax.set_xlim(n_vals[0], n_vals[-1])
            ax.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(frameon=False, loc="lower left")
            ax.set_xlabel("Observed count n")

            _finish_axes(ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# Convert the p-values to significance and write the scan panels.
def write_significance_pdf(
    results: list,
    out_pdf: Path,
    statistics: list,
    continuity_correction_rstar: bool,
):
    rstar_suffix = _correction_suffix(continuity_correction_rstar)
    with PdfPages(out_pdf) as pdf:
        for res in results:
            n_vals = res["n_vals"]
            p_ref = res["p_ref"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]

            z_ref = _z_from_p(p_ref)
            z_r = _z_from_p(p_r)
            z_rstar = _z_from_p(p_rstar)

            fig, ax_z = plt.subplots(figsize=PLOT_FIGSIZE)
            ax_z.set_box_aspect(1)

            z_values_for_limits = [z_ref]
            if "r" in statistics:
                ax_z.plot(
                    n_vals,
                    z_r,
                    marker="o",
                    linestyle="None",
                    ms=5,
                    label=r"$q_0$",
                    color="0.15",
                )
                z_values_for_limits.append(z_r)
            if "rstar" in statistics:
                ax_z.plot(
                    n_vals,
                    z_rstar,
                    marker="^",
                    linestyle="None",
                    ms=5,
                    label=rf"$q_0^\ast${rstar_suffix}",
                    color="0.15",
                )
                z_values_for_limits.append(z_rstar)
            ax_z.plot(
                n_vals,
                z_ref,
                marker="x",
                linestyle="None",
                ms=4,
                label="Exact",
                color="0.15",
            )
            ax_z.set_ylabel(r"Significance $Z$")
            _set_count_significance_limits(ax_z, n_vals, *z_values_for_limits)
            ax_z.grid(True, alpha=0.25)
            ax_z.legend(frameon=False, loc="lower right")

            ax_z.set_xlabel("Observed count n")

            _finish_axes(ax_z)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# Load the configuration, calculate the scans, and write both plot sets.
def main(cfg_path: str):
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    target_z = float(cfg.get("target_Z", 5.0))
    max_observed_count = cfg.get("max_observed_count")
    max_observed_count = None if max_observed_count is None else int(max_observed_count)
    out_significance_pdf = Path(cfg["out_significance_pdf"])
    out_pvalue_pdf = Path(cfg["out_pvalue_pdf"])
    trim_to_discovery_tail = bool(cfg.get("trim_to_discovery_tail", True))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))
    selected_statistics = cfg.get("statistics", ["r", "rstar"])
    if not isinstance(selected_statistics, list):
        raise ValueError("statistics must be a YAML list")
    statistics = [str(statistic).lower() for statistic in selected_statistics]
    for statistic in statistics:
        if statistic not in ("r", "rstar"):
            raise ValueError(f"Unknown statistic={statistic!r}")
    if not statistics:
        raise ValueError("statistics must include r, rstar, or both")

    out_significance_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_pvalue_pdf.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for s0 in s0_vec:
        for b in b_vec:
            all_results.append(
                compute_pvalue_scans(
                    float(s0),
                    float(b),
                    target_z=target_z,
                    max_observed_count=max_observed_count,
                    trim_to_discovery_tail=trim_to_discovery_tail,
                    continuity_correction_r=continuity_correction_r,
                    continuity_correction_rstar=continuity_correction_rstar,
                )
            )

    write_pvalue_pdf(
        all_results,
        out_pvalue_pdf,
        statistics=statistics,
        continuity_correction_rstar=continuity_correction_rstar,
    )
    write_significance_pdf(
        all_results,
        out_significance_pdf,
        statistics=statistics,
        continuity_correction_rstar=continuity_correction_rstar,
    )

    print(f"Saved paper plot to: {out_significance_pdf.resolve()}")
    print(f"Saved auxiliary p-value plot to: {out_pvalue_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/paper_simple_significance.yaml",
        help="Path to YAML config for the known-background significance paper plot",
    )
    args = parser.parse_args()
    main(args.config)
