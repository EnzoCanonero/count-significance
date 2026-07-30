#!/usr/bin/env python3
"""Create the on/off observed-significance plots used in the paper."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ScalarOrArray, load_yaml
from src.on_off import pvals_onoff, pvals_onoff_profile_sum


PLOT_FIGSIZE = (6.5, 6.5)


# Apply the common style used by the observed-significance plots.
def _configure_plot_style() -> None:
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
def _finish_axes(ax) -> None:
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


# Return the shared legend styling.
def _legend_kwargs() -> dict[str, Any]:
    return {
        "frameon": False,
        "fontsize": 13,
        "borderaxespad": 0.2,
        "handlelength": 1.8,
        "handletextpad": 0.7,
        "labelspacing": 0.45,
    }


# Add separate legends for statistics, model parameters, and observed OFF counts.
def _add_plot_legends(
    ax,
    m_values: list[int],
    colors: list[str],
    b: float,
    tau: float,
    reference_label: str,
    statistics: list[str],
    r_label: str,
    rstar_label: str,
    stat_loc: str,
    stat_anchor: tuple[float, float] = (0.98, 0.98),
    parameter_anchor: tuple[float, float] = (0.98, 0.74),
    m_loc: str = "upper left",
    m_anchor: tuple[float, float] = (0.02, 0.98),
) -> None:
    stat_handles = []
    if "r" in statistics:
        stat_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                marker="o",
                ls="none",
                ms=7,
                label=r_label,
            )
        )
    if "rstar" in statistics:
        stat_handles.append(
            Line2D([0], [0], color="0.15", marker="^", ls="none", ms=7, label=rstar_label)
        )
    stat_handles.append(
        Line2D(
            [0],
            [0],
            color="0.15",
            marker="x",
            ls="none",
            ms=7,
            label=reference_label,
        )
    )

    stat_legend = ax.legend(
        handles=stat_handles,
        loc=stat_loc,
        bbox_to_anchor=stat_anchor,
        **_legend_kwargs(),
    )
    ax.add_artist(stat_legend)

    parameter_handle = Line2D(
        [],
        [],
        color="none",
        label=rf"$b={b:g},\ \tau={tau:g}$",
    )
    parameter_legend = ax.legend(
        handles=[parameter_handle],
        loc=stat_loc,
        bbox_to_anchor=parameter_anchor,
        **_legend_kwargs(),
    )
    ax.add_artist(parameter_legend)

    m_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=2.2, label=rf"$m={m0:g}$")
        for idx, m0 in enumerate(m_values)
    ]
    ax.legend(
        handles=m_handles,
        loc=m_loc,
        bbox_to_anchor=m_anchor,
        **_legend_kwargs(),
    )


# Set readable limits for an observed-count significance panel.
def _set_count_significance_limits(
    ax,
    n_lo: float,
    n_hi: float,
    *z_arrays: np.ndarray,
) -> None:
    x_span = max(float(n_hi - n_lo), 1.0)
    x_pad = max(0.5, 0.06 * x_span)
    ax.set_xlim(float(n_lo) - x_pad, float(n_hi) + x_pad)

    z_values = np.concatenate([np.ravel(np.asarray(values, dtype=float)) for values in z_arrays])
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size == 0:
        return
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_span = max(z_max - z_min, 1.0)
    ax.set_ylim(min(0.0, z_min - 0.04 * z_span), z_max + 0.10 * z_span)


def _candidate_off_counts(
    mu_b: float,
    sigma_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> np.ndarray:
    """Choose representative OFF counts at E[M] + k sqrt(E[M])."""
    sigma_m = np.sqrt(max(float(mu_b), 0.0))
    m0_raw = np.array(
        [
            max(0, int(np.floor(float(mu_b) + float(k) * sigma_m + 0.5)))
            for k in sigma_offsets
        ],
        dtype=int,
    )

    return np.unique(np.sort(m0_raw))


# Add consecutive OFF counts when too few distinct slices were selected.
def _ensure_min_off_counts(m0_list: np.ndarray, min_count: int = 3) -> np.ndarray:
    m0_list = np.unique(np.sort(np.asarray(m0_list, dtype=int)))
    min_count = max(int(min_count), 0)
    if m0_list.size >= min_count:
        return m0_list

    next_m0 = int(m0_list[-1]) + 1 if m0_list.size else 1
    extra = []
    while m0_list.size + len(extra) < min_count:
        extra.append(next_m0)
        next_m0 += 1

    return np.unique(np.sort(np.concatenate([m0_list, np.asarray(extra, dtype=int)])))


def _z_from_p(p: ScalarOrArray) -> ScalarOrArray:
    """Convert an upper-tail p-value to Z = Phi^(-1)(1 - p)."""
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


# Validate and normalise the configured reference method.
def _normalise_reference_method(reference_method: str) -> str:
    method = str(reference_method).lower()
    if method in ("profile_sum", "mc"):
        return method
    raise ValueError(f"Unknown reference_method={reference_method!r}")


def _pvals_onoff_reference(
    s: float,
    n: int,
    m: int,
    tau: float,
    sigrel: float,
    reference_method: str,
    reference_tail_mass: float,
) -> dict[str, float]:
    """Return asymptotic p-values and the selected on/off reference.

    profile_sum sums the inclusive joint-Poisson tail at the null-profiled
    background, while mc estimates the reference with simulated counts.
    """
    reference_method = _normalise_reference_method(reference_method)

    if reference_method == "profile_sum":
        return pvals_onoff_profile_sum(
            s=s,
            n=n,
            m=m,
            tau=tau,
            tail_mass=reference_tail_mass,
        )
    if reference_method == "mc":
        mc_result = pvals_onoff(
            s=s,
            n=n,
            m=m,
            tau=tau,
            sigrel=sigrel,
        )
        return {
            "p_ref": mc_result["p_mc"],
            "p_ref_se": mc_result["p_mc_se"],
            "p_r": mc_result["p_r"],
            "p_rstar": mc_result["p_rstar"],
        }


def compute_pvalue_scans(
    s0: float,
    b: float,
    tau: float,
    sigrel: float,
    target_z: float = 5.0,
    max_observed_count: Optional[int] = None,
    m_sigma_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0),
    drop_zero_m0: bool = True,
    trim_to_discovery_tail: bool = True,
    reference_method: str = "profile_sum",
    reference_tail_mass: float = 1e-12,
    max_n: int = 10_000,
    min_m0_values: int = 3,
) -> list[dict[str, Any]]:
    """Scan ON counts for representative fixed OFF counts.

    The discovery tail begins above n = s0 + m/tau. Each point compares the
    q0 and q0* approximations with the selected inclusive reference tail.
    """
    # Set the ON and OFF Poisson means under s0, b, and tau.
    mu_s = float(s0 + b)
    mu_b = float(tau * b)

    reference_method = _normalise_reference_method(reference_method)
    m0_list = _candidate_off_counts(mu_b, sigma_offsets=m_sigma_offsets)
    if drop_zero_m0:
        nonzero_m0 = m0_list[m0_list != 0]
        m0_list = nonzero_m0 if nonzero_m0.size > 0 else np.array([1], dtype=int)
    m0_list = _ensure_min_off_counts(m0_list, min_m0_values)

    # Check the stopping criterion only beyond mu_s + 5 sqrt(mu_s).
    n_min = 0
    n_max_start = max(n_min, int(np.ceil(mu_s + 5.0 * np.sqrt(mu_s))))

    results = []

    for m0 in m0_list:
        threshold = float(s0) + int(m0) / float(tau)
        first_tail_n = int(np.floor(threshold)) + 1
        if trim_to_discovery_tail:
            start_n = max(n_min, first_tail_n - 1)
        else:
            start_n = n_min

        n_vals, p_r, p_rstar, p_ref, p_ref_se = [], [], [], [], []
        n0 = int(start_n)

        stop_n = int(max_observed_count) if max_observed_count is not None else int(max_n)
        if stop_n < n0:
            raise ValueError(f"max_observed_count={stop_n} is below the scan start n={n0}")

        while n0 <= stop_n:
            out = _pvals_onoff_reference(
                s=s0,
                n=n0,
                m=int(m0),
                tau=tau,
                sigrel=sigrel,
                reference_method=reference_method,
                reference_tail_mass=reference_tail_mass,
            )
            n_vals.append(n0)
            p_r.append(out["p_r"])
            p_rstar.append(out["p_rstar"])
            p_ref.append(out["p_ref"])
            p_ref_se.append(out["p_ref_se"])

            z_ref = float(_z_from_p(out["p_ref"]))
            if max_observed_count is None and n0 >= n_max_start and z_ref >= target_z:
                break

            n0 += 1
        else:
            if max_observed_count is None:
                raise RuntimeError(f"Failed to reach Z={target_z:g} by n={max_n}")

        n_vals = np.asarray(n_vals, dtype=int)
        p_r = np.asarray(p_r, dtype=float)
        p_rstar = np.asarray(p_rstar, dtype=float)
        p_ref = np.asarray(p_ref, dtype=float)
        p_ref_se = np.asarray(p_ref_se, dtype=float)

        results.append(
            {
                "s0": float(s0),
                "b": float(b),
                "tau": float(tau),
                "m0": int(m0),
                "n_vals": n_vals,
                "p_r": p_r,
                "p_rstar": p_rstar,
                "p_ref": p_ref,
                "p_ref_se": p_ref_se,
            }
        )

    return results


# Write p-value panels using colour for m and markers for the statistic.
def write_pvalue_pdf(
    results: list[dict[str, Any]],
    out_pdf: Path,
    reference_label: str,
    statistics: list[str],
) -> None:
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["s0"], result["b"], result["tau"])].append(result)

    with PdfPages(out_pdf) as pdf:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for _, group in sorted(grouped.items()):
            group = sorted(group, key=lambda item: item["m0"])
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

            n_lo, n_hi = np.inf, -np.inf
            for idx, result in enumerate(group):
                color = colors[idx % len(colors)]
                n_vals = result["n_vals"]
                n_lo = min(n_lo, int(n_vals[0]))
                n_hi = max(n_hi, int(n_vals[-1]))

                if "r" in statistics:
                    ax.semilogy(
                        n_vals,
                        result["p_r"],
                        marker="o",
                        linestyle="None",
                        ms=5,
                        color=color,
                    )
                if "rstar" in statistics:
                    ax.semilogy(
                        n_vals,
                        result["p_rstar"],
                        marker="^",
                        linestyle="None",
                        ms=5,
                        color=color,
                    )
                ax.errorbar(
                    n_vals,
                    result["p_ref"],
                    yerr=result["p_ref_se"],
                    fmt="x",
                    ms=4,
                    lw=1,
                    capsize=2,
                    color=color,
                )
            ax.set_ylabel("p-value (upper tail)")
            ax.set_xlim(n_lo, n_hi)
            ax.set_xlabel(r"$n_0$ (observed primary count)")
            ax.grid(True, which="both", alpha=0.25)
            _add_plot_legends(
                ax,
                [result["m0"] for result in group],
                colors,
                b=group[0]["b"],
                tau=group[0]["tau"],
                reference_label=reference_label,
                statistics=statistics,
                r_label=r"$1-\Phi(\sqrt{q_0})$",
                rstar_label=r"$1-\Phi(\sqrt{q_0^\ast})$",
                stat_loc="lower left",
                stat_anchor=(0.02, 0.02),
                parameter_anchor=(0.02, 0.26),
                m_loc="upper right",
                m_anchor=(0.98, 0.98),
            )

            _finish_axes(ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# Convert p-values to significance and write the scan panels.
def write_significance_pdf(
    results: list[dict[str, Any]],
    out_pdf: Path,
    reference_label: str,
    statistics: list[str],
) -> None:
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["s0"], result["b"], result["tau"])].append(result)

    with PdfPages(out_pdf) as pdf:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for _, group in sorted(grouped.items()):
            group = sorted(group, key=lambda item: item["m0"])
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            ax.set_box_aspect(1)

            n_lo, n_hi = np.inf, -np.inf
            z_values_for_limits = []
            for idx, result in enumerate(group):
                color = colors[idx % len(colors)]
                n_vals = result["n_vals"]
                n_lo = min(n_lo, int(n_vals[0]))
                n_hi = max(n_hi, int(n_vals[-1]))

                z_r = _z_from_p(result["p_r"])
                z_rstar = _z_from_p(result["p_rstar"])
                z_ref = _z_from_p(result["p_ref"])
                z_values_for_limits.append(z_ref)

                if "r" in statistics:
                    ax.plot(n_vals, z_r, marker="o", linestyle="None", ms=5, color=color)
                    z_values_for_limits.append(z_r)
                if "rstar" in statistics:
                    ax.plot(n_vals, z_rstar, marker="^", linestyle="None", ms=5, color=color)
                    z_values_for_limits.append(z_rstar)
                ax.plot(n_vals, z_ref, marker="x", linestyle="None", ms=4, color=color)

            ax.set_ylabel(r"Significance $Z$")
            ax.set_xlabel("Observed count n")
            _set_count_significance_limits(ax, n_lo, n_hi, *z_values_for_limits)
            ax.grid(True, alpha=0.25)
            _add_plot_legends(
                ax,
                [result["m0"] for result in group],
                colors,
                b=group[0]["b"],
                tau=group[0]["tau"],
                reference_label=reference_label,
                statistics=statistics,
                r_label=r"$q_0$",
                rstar_label=r"$q_0^\ast$",
                stat_loc="lower right",
                stat_anchor=(0.98, 0.02),
                parameter_anchor=(0.98, 0.26),
                m_loc="upper left",
                m_anchor=(0.02, 0.98),
            )

            _finish_axes(ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# Load the configuration, calculate the scans, and write both plot sets.
def main(cfg_path: str) -> None:
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tau_vec"], dtype=float)
    sigrel = float(cfg.get("mc_sigrel_Z", 0.001))
    target_z = float(cfg.get("target_Z", 5.0))
    max_observed_count = cfg.get("max_observed_count")
    max_observed_count = None if max_observed_count is None else int(max_observed_count)
    m_sigma_offsets = tuple(float(x) for x in cfg.get("m_sigma_offsets", [-1.0, 0.0, 1.0]))
    reference_method = _normalise_reference_method(cfg.get("reference_method", "profile_sum"))
    reference_tail_mass = float(cfg.get("reference_tail_mass", 1e-12))
    selected_statistics = cfg.get("statistics", ["r", "rstar"])
    if not isinstance(selected_statistics, list):
        raise ValueError("statistics must be a YAML list")
    statistics = [str(statistic).lower() for statistic in selected_statistics]
    for statistic in statistics:
        if statistic not in ("r", "rstar"):
            raise ValueError(f"Unknown statistic={statistic!r}")
    if not statistics:
        raise ValueError("statistics must include r, rstar, or both")
    max_n = int(cfg.get("max_n", 10_000))
    min_m0_values = int(cfg.get("min_m0_values", 3))
    if reference_method == "profile_sum":
        reference_label = "Profile reference"
    else:
        reference_label = "MC reference"
    out_significance_pdf = Path(cfg["out_significance_pdf"])
    out_pvalue_pdf = Path(cfg["out_pvalue_pdf"])
    drop_zero_m0 = bool(cfg.get("drop_zero_m0", True))
    trim_to_discovery_tail = bool(cfg.get("trim_to_discovery_tail", True))
    out_significance_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_pvalue_pdf.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for s0 in s0_vec:
        for b in b_vec:
            for tau in tau_vec:
                scans = compute_pvalue_scans(
                    float(s0),
                    float(b),
                    float(tau),
                    sigrel,
                    target_z=target_z,
                    max_observed_count=max_observed_count,
                    m_sigma_offsets=m_sigma_offsets,
                    drop_zero_m0=drop_zero_m0,
                    trim_to_discovery_tail=trim_to_discovery_tail,
                    reference_method=reference_method,
                    reference_tail_mass=reference_tail_mass,
                    max_n=max_n,
                    min_m0_values=min_m0_values,
                )
                all_results.extend(scans)

    write_pvalue_pdf(
        all_results,
        out_pvalue_pdf,
        reference_label=reference_label,
        statistics=statistics,
    )
    write_significance_pdf(
        all_results,
        out_significance_pdf,
        reference_label,
        statistics=statistics,
    )

    print(f"Saved paper plot to: {out_significance_pdf.resolve()}")
    print(f"Saved auxiliary p-value plot to: {out_pvalue_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/paper_onoff_significance.yaml",
        help="Path to YAML config for the uncertain-background significance paper plot",
    )
    args = parser.parse_args()
    main(args.config)
