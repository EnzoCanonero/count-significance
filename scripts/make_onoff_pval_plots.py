#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import norm, poisson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on_off import b_profiled, pvals_onoff, r_star_onoff, r_stat_onoff


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


def _legend_kwargs():
    return {
        "frameon": False,
        "fontsize": 13,
        "borderaxespad": 0.2,
        "handlelength": 1.8,
        "handletextpad": 0.7,
        "labelspacing": 0.45,
    }


def _add_stat_and_m_legends(
    ax,
    m_values: list,
    colors: list,
    reference_label: str,
    include_reference: bool = True,
    loc: str = "upper right",
    stat_anchor: tuple = (0.98, 0.98),
    m_anchor: tuple = (0.98, 0.74),
):
    stat_handles = [
        Line2D([0], [0], color="0.15", marker="o", ls="none", ms=7, label=r"$q_0$"),
        Line2D([0], [0], color="0.15", marker="^", ls="none", ms=7, label=r"$q_0^\ast$"),
    ]
    if include_reference:
        stat_handles.append(Line2D([0], [0], color="0.15", marker="x", ls="none", ms=7, label=reference_label))

    stat_legend = ax.legend(
        handles=stat_handles,
        loc=loc,
        bbox_to_anchor=stat_anchor,
        **_legend_kwargs(),
    )
    ax.add_artist(stat_legend)

    m_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=2.2, label=rf"$m={m0:g}$")
        for idx, m0 in enumerate(m_values)
    ]
    ax.legend(
        handles=m_handles,
        loc=loc,
        bbox_to_anchor=m_anchor,
        **_legend_kwargs(),
    )


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def _set_count_significance_limits(ax, n_lo: float, n_hi: float, *z_arrays: np.ndarray):
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


def _candidate_off_counts(mu_b: float, sigma_offsets=(-1.0, 0.0, 1.0)) -> np.ndarray:
    """Representative control counts at E[M] + k sqrt(E[M])."""
    sigma_m = np.sqrt(max(float(mu_b), 0.0))
    m0_raw = np.array(
        [
            max(0, int(np.floor(float(mu_b) + float(k) * sigma_m + 0.5)))
            for k in sigma_offsets
        ],
        dtype=int,
    )

    return np.unique(np.sort(m0_raw))


def _ensure_min_off_counts(m0_list: np.ndarray, min_count: int = 3) -> np.ndarray:
    """Pad the control-count slices with consecutive larger integers for plotting stability."""
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


def _tail_on_counts(s0: float, m0: int, tau: float, n_min: int, n_max: int) -> np.ndarray:
    """Keep the last plateau point and the one-sided discovery tail."""
    threshold = float(s0) + float(m0) / float(tau)
    first_tail_n = int(np.floor(threshold)) + 1
    if first_tail_n > n_max:
        return np.array([], dtype=int)

    start_n = max(n_min, first_tail_n - 1)
    return np.arange(start_n, n_max + 1, dtype=int)


def _z_from_p(p):
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


def _reference_title(reference_label: str, sigrel: float) -> str:
    if reference_label == "MC":
        return rf"MC rel. $Z$ precision $={sigrel}$"
    return rf"reference $={reference_label}$"


def _poisson_grid_max(mu: float, observed: int = 0, tail_mass: float = 1e-12) -> int:
    """Finite Poisson support large enough for exact tail sums."""
    mu = float(mu)
    if mu <= 0.0:
        return max(0, int(observed))

    upper = poisson.ppf(1.0 - float(tail_mass), mu)
    if not np.isfinite(upper):
        upper = mu + 12.0 * np.sqrt(mu + 1.0) + 20.0

    return max(int(np.ceil(upper)), int(observed), 0)


def _pvals_onoff_exact(
    s,
    b,
    tau,
    n,
    m,
    tail_mass: float = 1e-12,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Exact plug-in null tail using the same r-ordering as the toy MC code."""
    r_ref = float(r_stat_onoff(s, n, m, tau, continuity_correction=False))
    r_obs = float(r_stat_onoff(s, n, m, tau, continuity_correction=continuity_correction_r))
    rs_obs = float(r_star_onoff(s, n, m, tau, continuity_correction=continuity_correction_rstar))

    p_r = float(norm.sf(max(r_obs, 0.0)))
    p_rs = float(norm.sf(max(rs_obs, 0.0)))

    if r_ref <= 0.0:
        return {
            "p_mc": 0.5,
            "p_mc_se": 0.0,
            "p_r": p_r,
            "p_rstar": p_rs,
        }

    b_tilde = float(b_profiled(s, n, m, tau))
    mu_n = float(s + b_tilde)
    mu_m = float(tau * b_tilde)

    n_grid = np.arange(_poisson_grid_max(mu_n, observed=n, tail_mass=tail_mass) + 1)
    m_grid = np.arange(_poisson_grid_max(mu_m, observed=m, tail_mass=tail_mass) + 1)
    pn = poisson.pmf(n_grid, mu_n)
    pm = poisson.pmf(m_grid, mu_m)

    nn, mm = np.meshgrid(n_grid, m_grid, indexing="ij")
    r_grid = r_stat_onoff(s, nn, mm, tau, continuity_correction=False)
    probs = pn[:, None] * pm[None, :]
    p_ref = float(np.sum(probs[r_grid >= r_ref]))
    p_ref = min(p_ref, 0.5)

    return {
        "p_mc": p_ref,
        "p_mc_se": 0.0,
        "p_r": p_r,
        "p_rstar": p_rs,
    }


def _pvals_onoff_reference(
    s,
    b,
    tau,
    n,
    m,
    sigrel: float,
    reference_method: str,
    reference_tail_mass: float,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    if reference_method == "exact":
        return _pvals_onoff_exact(
            s,
            b,
            tau,
            n,
            m,
            tail_mass=reference_tail_mass,
            continuity_correction_r=continuity_correction_r,
            continuity_correction_rstar=continuity_correction_rstar,
        )
    if reference_method == "mc":
        return pvals_onoff(
            s,
            b,
            tau,
            n,
            m,
            sigrel=sigrel,
            continuity_correction_r=continuity_correction_r,
            continuity_correction_rstar=continuity_correction_rstar,
        )
    raise ValueError(f"Unknown reference_method={reference_method!r}")


def compute_pvalues_onoff(
    s0: float,
    b: float,
    tau: float,
    sigrel: float,
    target_z: float = 5.0,
    max_observed_count: int = None,
    m_sigma_offsets=(-1.0, 0.0, 1.0),
    drop_zero_m0: bool = True,
    trim_to_discovery_tail: bool = True,
    reference_method: str = "exact",
    reference_tail_mass: float = 1e-12,
    max_n: int = 10_000,
    min_m0_values: int = 3,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute p-values and relative differences for all n0 and candidate m0 values."""
    eps = 1e-16

    # Poisson means for the signal (ON) and control (OFF) regions under s0, b, τ
    mu_s = float(s0 + b)
    mu_b = float(tau * b)

    reference_method = str(reference_method).lower()
    m0_list = _candidate_off_counts(mu_b, sigma_offsets=m_sigma_offsets)
    if drop_zero_m0:
        nonzero_m0 = m0_list[m0_list != 0]
        m0_list = nonzero_m0 if nonzero_m0.size > 0 else np.array([1], dtype=int)
    m0_list = _ensure_min_off_counts(m0_list, min_m0_values)

    # Define the scan range for primary counts n0:
    # from n = 0 up to μ_s + 5√μ_s (rounded up), ensuring at least one bin.
    n_min = 0
    n_max_start = max(n_min, int(np.ceil(mu_s + 5.0 * np.sqrt(mu_s))))

    results = []

    # Loop over the chosen control-count configurations
    for m0 in m0_list:
        threshold = float(s0) + int(m0) / float(tau)
        first_tail_n = int(np.floor(threshold)) + 1
        if trim_to_discovery_tail:
            start_n = max(n_min, first_tail_n - 1)
        else:
            start_n = n_min

        n_vals, p_r, p_rstar, p_mc, p_mc_se = [], [], [], [], []
        n0 = int(start_n)

        stop_n = int(max_observed_count) if max_observed_count is not None else int(max_n)
        if stop_n < n0:
            raise ValueError(f"max_observed_count={stop_n} is below the scan start n={n0}")

        while n0 <= stop_n:
            out = _pvals_onoff_reference(
                s0,
                b,
                tau,
                n0,
                int(m0),
                sigrel=sigrel,
                reference_method=reference_method,
                reference_tail_mass=reference_tail_mass,
                continuity_correction_r=continuity_correction_r,
                continuity_correction_rstar=continuity_correction_rstar,
            )
            n_vals.append(n0)
            p_r.append(out["p_r"])
            p_rstar.append(out["p_rstar"])
            p_mc.append(out["p_mc"])
            p_mc_se.append(out["p_mc_se"])

            z_ref = float(_z_from_p(out["p_mc"]))
            if max_observed_count is None and n0 >= n_max_start and z_ref >= target_z:
                break

            n0 += 1
        else:
            if max_observed_count is None:
                raise RuntimeError(f"Failed to reach Z={target_z:g} by n={max_n}")

        n_vals = np.asarray(n_vals, dtype=int)
        p_r = np.asarray(p_r, dtype=float)
        p_rstar = np.asarray(p_rstar, dtype=float)
        p_mc = np.asarray(p_mc, dtype=float)
        p_err = np.asarray(p_mc_se, dtype=float)

        denom = np.clip(p_mc, eps, None)
        rel_r = np.abs(p_r - p_mc) / denom
        rel_rstar = np.abs(p_rstar - p_mc) / denom

        results.append(
            {
                "s0": float(s0),
                "b": float(b),
                "tau": float(tau),
                "m0": int(m0),
                "n_vals": n_vals,
                "n_min": n_min,
                "n_max": int(n_vals[-1]),
                "tail_threshold": threshold,
                "first_tail_n": first_tail_n,
                "p_r": np.maximum(p_r, eps),
                "p_rstar": np.maximum(p_rstar, eps),
                "p_mc": np.maximum(p_mc, eps),
                "p_err": p_err,
                "rel_r": np.maximum(rel_r, eps),
                "rel_rstar": np.maximum(rel_rstar, eps),
            }
        )

    return results


def make_plots_onoff(
    results: list,
    sigrel: float,
    out_pdf: Path,
    save_individual: bool,
    include_ratio: bool,
    reference_label: str,
):
    """Render p-value panels with colour encoding m0 and marker encoding the statistic."""
    grouped = defaultdict(list)
    for res in results:
        grouped[(res["s0"], res["b"], res["tau"])].append(res)

    with PdfPages(out_pdf) as pdf:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for (s0, b, tau), group in sorted(grouped.items()):
            group = sorted(group, key=lambda item: item["m0"])
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

            n_lo, n_hi = np.inf, -np.inf
            for idx, res in enumerate(group):
                color = colors[idx % len(colors)]
                n_vals = res["n_vals"]
                n_lo = min(n_lo, int(n_vals[0]))
                n_hi = max(n_hi, int(n_vals[-1]))

                ax_top.semilogy(n_vals, res["p_r"], marker="o", linestyle="None", ms=5, color=color)
                ax_top.semilogy(n_vals, res["p_rstar"], marker="^", linestyle="None", ms=5, color=color)
                ax_top.errorbar(
                    n_vals,
                    res["p_mc"],
                    yerr=res["p_err"],
                    fmt="x",
                    ms=4,
                    lw=1,
                    capsize=2,
                    color=color,
                )
                ax_top.axvline(res["first_tail_n"] - 0.5, color=color, ls=":", lw=1, alpha=0.35)

                if include_ratio:
                    ax_bot.axvline(res["first_tail_n"] - 0.5, color=color, ls=":", lw=1, alpha=0.35)
                    ax_bot.semilogy(n_vals, res["rel_r"], marker="o", linestyle="None", ms=4, color=color)
                    ax_bot.semilogy(n_vals, res["rel_rstar"], marker="^", linestyle="None", ms=4, color=color)

            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_lo, n_hi)
            ax_top.grid(True, which="both", alpha=0.25)
            _add_stat_and_m_legends(ax_top, [res["m0"] for res in group], colors, reference_label)

            if include_ratio:
                ax_bot.set_xlabel(r"$n_0$ (observed primary count)")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.set_xlim(n_lo, n_hi)
                ax_bot.grid(True, which="both", alpha=0.25)
            else:
                ax_top.set_xlabel(r"$n_0$ (observed primary count)")

            _finish_axes(ax_top, ax_bot)
            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"onoff_pval_s{_fmt(s0)}_b{_fmt(b)}_tau{_fmt(tau)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def make_significance_plots_onoff(
    results: list,
    sigrel: float,
    out_pdf: Path,
    reference_label: str,
    include_ratio: bool,
):
    """Significance vs n0 with colour encoding m0 and marker encoding the statistic."""
    grouped = defaultdict(list)
    for res in results:
        grouped[(res["s0"], res["b"], res["tau"])].append(res)

    with PdfPages(out_pdf) as pdf:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for (s0, b, tau), group in sorted(grouped.items()):
            group = sorted(group, key=lambda item: item["m0"])

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

            n_lo, n_hi = np.inf, -np.inf
            z_values_for_limits = []
            for idx, res in enumerate(group):
                color = colors[idx % len(colors)]
                n_vals = res["n_vals"]
                n_lo = min(n_lo, int(n_vals[0]))
                n_hi = max(n_hi, int(n_vals[-1]))

                z_r = _z_from_p(res["p_r"])
                z_rstar = _z_from_p(res["p_rstar"])
                z_mc = _z_from_p(res["p_mc"])
                z_values_for_limits.extend([z_r, z_rstar, z_mc])

                ax_z.plot(n_vals, z_r, marker="o", linestyle="None", ms=5, color=color)
                ax_z.plot(n_vals, z_rstar, marker="^", linestyle="None", ms=5, color=color)
                ax_z.plot(n_vals, z_mc, marker="x", linestyle="None", ms=4, color=color)

                if include_ratio:
                    ax_bot.semilogy(n_vals, res["rel_r"], marker="o", linestyle="None", ms=4, color=color)
                    ax_bot.semilogy(n_vals, res["rel_rstar"], marker="^", linestyle="None", ms=4, color=color)

            ax_z.set_ylabel(r"$Z$")
            _set_count_significance_limits(ax_z, n_lo, n_hi, *z_values_for_limits)
            ax_z.grid(True, alpha=0.25)
            _add_stat_and_m_legends(
                ax_z,
                [res["m0"] for res in group],
                colors,
                reference_label,
                loc="lower right",
                stat_anchor=(0.98, 0.02),
                m_anchor=(0.98, 0.26),
            )

            if include_ratio:
                ax_bot.set_xlabel("Observed count n")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.set_xlim(ax_z.get_xlim())
                ax_bot.grid(True, which="both", alpha=0.25)
            else:
                ax_z.set_xlabel("Observed count n")

            _finish_axes(ax_z, ax_bot)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def make_improvement_plots_onoff(results: list, out_pdf: Path, reference_label: str):
    """Render separate pages showing where q0* is closer to MC than q0."""
    grouped = defaultdict(list)
    for res in results:
        grouped[(res["s0"], res["b"], res["tau"])].append(res)

    with PdfPages(out_pdf) as pdf:
        for (s0, b, tau), group in sorted(grouped.items()):
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)

            for res in sorted(group, key=lambda item: item["m0"]):
                n_vals = res["n_vals"]
                z_mc = _z_from_p(res["p_mc"])
                z_r = _z_from_p(res["p_r"])
                z_rstar = _z_from_p(res["p_rstar"])

                improvement = np.abs(z_r - z_mc) - np.abs(z_rstar - z_mc)
                ax.plot(
                    n_vals,
                    improvement,
                    marker="o",
                    ms=4,
                    lw=1.2,
                    label=rf"$m_0={res['m0']}$",
                )

            ax.axhline(0.0, color="0.25", lw=1)
            ax.set_xlabel(r"$n_0$ (observed primary count)")
            ax.set_ylabel(rf"$|Z_{{q_0}} - Z_\mathrm{{ref}}| - |Z_{{q_0^\ast}} - Z_\mathrm{{ref}}|$")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            _finish_axes(ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    _configure_plot_style()

    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tau_vec"], dtype=float)
    sigrel = float(cfg.get("mc_sigrel_Z", cfg.get("sigrel", 0.001)))
    target_z = float(cfg.get("target_Z", 5.0))
    max_observed_count = cfg.get("max_observed_count")
    max_observed_count = None if max_observed_count is None else int(max_observed_count)
    m_sigma_offsets = tuple(float(x) for x in cfg.get("m_sigma_offsets", [-1.0, 0.0, 1.0]))
    reference_method = str(cfg.get("reference_method", "exact")).lower()
    reference_tail_mass = float(cfg.get("reference_tail_mass", 1e-12))
    max_n = int(cfg.get("max_n", 10_000))
    min_m0_values = int(cfg.get("min_m0_values", 3))
    default_reference_label = "Exact" if reference_method == "exact" else "MC"
    reference_label = str(cfg.get("reference_label", default_reference_label))
    out_pdf = Path(cfg["out_pdf"])
    out_improvement_pdf = Path(
        cfg.get("out_improvement_pdf", out_pdf.with_name(f"{out_pdf.stem}_rstar_improvement.pdf"))
    )
    out_significance_pdf = Path(
        cfg.get("out_significance_pdf", out_pdf.with_name(f"{out_pdf.stem}_with_significance.pdf"))
    )
    save_individual = bool(cfg.get("individual_plots", False))
    include_ratio = bool(cfg.get("ratio_plots", False))
    make_improvement_plots = bool(cfg.get("improvement_plots", True))
    make_significance_plots = bool(cfg.get("significance_plots", cfg.get("significance_pvalue_plots", True)))
    drop_zero_m0 = bool(cfg.get("drop_zero_m0", cfg.get("drop_s0_zero_m0", True)))
    trim_to_discovery_tail = bool(cfg.get("trim_to_discovery_tail", True))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_improvement_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_significance_pdf.parent.mkdir(parents=True, exist_ok=True)

    combos = np.array(np.meshgrid(s0_vec, b_vec, tau_vec)).T.reshape(-1, 3)
    all_results = []
    for s0, b, tau in combos:
        all_results.extend(
            compute_pvalues_onoff(
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
                continuity_correction_r=continuity_correction_r,
                continuity_correction_rstar=continuity_correction_rstar,
            )
        )

    make_plots_onoff(all_results, sigrel, out_pdf, save_individual, include_ratio, reference_label)
    if make_significance_plots:
        make_significance_plots_onoff(
            all_results,
            sigrel,
            out_significance_pdf,
            reference_label,
            include_ratio,
        )
    if make_improvement_plots:
        make_improvement_plots_onoff(all_results, out_improvement_pdf, reference_label)

    print(f"Saved all plots to: {out_pdf.resolve()}")
    if make_significance_plots:
        print(f"Saved significance plots to: {out_significance_pdf.resolve()}")
    if make_improvement_plots:
        print(f"Saved q0* improvement plots to: {out_improvement_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/onoff_pval.yaml",
        help="Path to YAML config for uncertain-background p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
