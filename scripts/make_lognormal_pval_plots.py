#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.lognormal import (
    background_lognormal,
    pvals_lognormal_numerical,
)


PLOT_FIGSIZE = (6.5, 6.5)


def _configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
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
    return f"{x:g}".replace(".", "p").replace("-", "m")


def _z_from_p(p):
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


def _reference_title(reference_label: str) -> str:
    return rf"reference $={reference_label}$"


def compute_pvalues_lognormal(
    s0: float,
    b0: float,
    sigma: float,
    theta_tilde: float,
    target_z: float = 5.0,
    trim_to_discovery_tail: bool = True,
    n_gh_pts: int = 30,
    reference_tail_mass: float = 1e-10,
    max_n: int = 10_000,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute p-values and relative differences for a single log-normal setup."""
    eps = 1e-16
    s0 = float(s0)
    b0 = float(b0)
    sigma = float(sigma)
    theta_tilde = float(theta_tilde)

    tail_threshold = float(s0 + background_lognormal(theta_tilde, b0, sigma))
    first_tail_n = int(np.floor(tail_threshold)) + 1
    n_min = 0
    start_n = max(n_min, first_tail_n - 1) if trim_to_discovery_tail else n_min
    n_max_start = max(n_min, int(np.ceil(tail_threshold + 5.0 * np.sqrt(max(tail_threshold, 0.0)))))

    n_vals, p_r, p_rstar, p_num, p_num_se = [], [], [], [], []
    n0 = int(start_n)

    while n0 <= int(max_n):
        out = pvals_lognormal_numerical(
            s0=s0,
            n=n0,
            theta_tilde=theta_tilde,
            b0=b0,
            sigma=sigma,
            n_gh_pts=n_gh_pts,
            tail_mass=reference_tail_mass,
            continuity_correction_r=continuity_correction_r,
            continuity_correction_rstar=continuity_correction_rstar,
        )
        n_vals.append(n0)
        p_r.append(out["p_r"])
        p_rstar.append(out["p_rstar"])
        p_num.append(out["p_numerical"])
        p_num_se.append(out["p_numerical_se"])

        z_ref = float(_z_from_p(out["p_numerical"]))
        if n0 >= n_max_start and z_ref >= target_z:
            break

        n0 += 1
    else:
        raise RuntimeError(
            "Failed to reach "
            f"Z={target_z:g} by n={max_n} for s0={s0:g}, b0={b0:g}, "
            f"sigma={sigma:g}, theta_tilde={theta_tilde:g}"
        )

    n_vals = np.asarray(n_vals, dtype=int)
    p_r = np.asarray(p_r, dtype=float)
    p_rstar = np.asarray(p_rstar, dtype=float)
    p_num = np.asarray(p_num, dtype=float)
    p_num_se = np.asarray(p_num_se, dtype=float)

    denom = np.clip(p_num, eps, None)
    rel_r = np.abs(p_r - p_num) / denom
    rel_rstar = np.abs(p_rstar - p_num) / denom

    return [
        {
            "s0": s0,
            "b0": b0,
            "sigma": sigma,
            "theta_tilde": theta_tilde,
            "n_vals": n_vals,
            "n_min": int(n_vals[0]),
            "n_max": int(n_vals[-1]),
            "tail_threshold": tail_threshold,
            "first_tail_n": first_tail_n,
            "p_numerical": np.maximum(p_num, eps),
            "p_numerical_se": p_num_se,
            "p_r": np.maximum(p_r, eps),
            "p_rstar": np.maximum(p_rstar, eps),
            "rel_r": np.maximum(rel_r, eps),
            "rel_rstar": np.maximum(rel_rstar, eps),
        }
    ]


def make_plots_lognormal(
    results: list,
    out_pdf: Path,
    save_individual: bool,
    include_ratio: bool,
    reference_label: str = "Numerical",
):
    """Render p-value panels for all log-normal constrained configurations."""
    with PdfPages(out_pdf) as pdf:
        for res in results:
            s0 = res["s0"]
            b0 = res["b0"]
            sigma = res["sigma"]
            theta_tilde = res["theta_tilde"]
            n_vals = res["n_vals"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            p_num = res["p_numerical"]
            p_num_se = res["p_numerical_se"]
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
                label="1 - Phi(q)",
                color="tab:blue",
            )
            ax_top.semilogy(
                n_vals,
                p_rstar,
                marker="^",
                linestyle="None",
                ms=5,
                label="1 - Phi(q*)",
                color="tab:orange",
            )
            ax_top.errorbar(
                n_vals,
                p_num,
                yerr=p_num_se,
                fmt="x",
                ms=4,
                lw=1,
                capsize=2,
                label=reference_label,
                color="tab:green",
            )
            ax_top.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_vals[0] - 0.5, n_vals[-1] + 0.5)
            ax_top.grid(True, which="both", alpha=0.25)
            ax_top.legend(frameon=False)

            if include_ratio:
                ax_bot.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_q-p_\mathrm{ref}|/p_\mathrm{ref}$",
                    color="tab:blue",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_{q^\ast}-p_\mathrm{ref}|/p_\mathrm{ref}$",
                    color="tab:orange",
                )
                ax_bot.set_xlabel(r"$n_0$ (observed counts)")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend(frameon=False)
            else:
                ax_top.set_xlabel(r"$n_0$ (observed counts)")

            _finish_axes(ax_top, ax_bot)
            plt.tight_layout()
            if save_individual:
                fname = (
                    out_pdf.parent
                    / f"lognormal_pval_s{_fmt(s0)}_b0{_fmt(b0)}_sig{_fmt(sigma)}_th{_fmt(theta_tilde)}.pdf"
                )
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def make_significance_plots_lognormal(
    results: list,
    out_pdf: Path,
    include_ratio: bool,
    reference_label: str = "Numerical",
):
    """Render significance panels, optionally with relative-difference panels below."""
    with PdfPages(out_pdf) as pdf:
        for res in results:
            s0 = res["s0"]
            b0 = res["b0"]
            sigma = res["sigma"]
            theta_tilde = res["theta_tilde"]
            n_vals = res["n_vals"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            p_num = res["p_numerical"]
            rel_r = res["rel_r"]
            rel_rstar = res["rel_rstar"]
            first_tail_n = res["first_tail_n"]

            z_num = _z_from_p(p_num)
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

            ax_z.plot(n_vals, z_r, marker="o", linestyle="None", ms=5, label=r"$q$", color="tab:blue")
            ax_z.plot(
                n_vals,
                z_rstar,
                marker="^",
                linestyle="None",
                ms=5,
                label=r"$q^\ast$",
                color="tab:orange",
            )
            ax_z.plot(n_vals, z_num, marker="x", linestyle="None", ms=4, label=reference_label, color="tab:green")
            ax_z.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
            ax_z.set_ylabel(r"$Z$")
            ax_z.set_xlim(n_vals[0] - 0.5, n_vals[-1] + 0.5)
            ax_z.grid(True, alpha=0.25)
            ax_z.legend(frameon=False)

            if include_ratio:
                ax_bot.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_q-p_\mathrm{ref}|/p_\mathrm{ref}$",
                    color="tab:blue",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_{q^\ast}-p_\mathrm{ref}|/p_\mathrm{ref}$",
                    color="tab:orange",
                )
                ax_bot.set_xlabel(r"$n_0$ (observed counts)")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend(frameon=False)
            else:
                ax_z.set_xlabel(r"$n_0$ (observed counts)")

            _finish_axes(ax_z, ax_bot)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def make_improvement_plots_lognormal(results: list, out_pdf: Path, reference_label: str = "Numerical"):
    """Render separate pages showing where q* is closer to the numerical reference than q."""
    grouped = defaultdict(list)
    for res in results:
        grouped[(res["s0"], res["b0"], res["sigma"])].append(res)

    with PdfPages(out_pdf) as pdf:
        for (s0, b0, sigma), group in sorted(grouped.items()):
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=150)

            for res in sorted(group, key=lambda item: item["theta_tilde"]):
                n_vals = res["n_vals"]
                z_num = _z_from_p(res["p_numerical"])
                z_r = _z_from_p(res["p_r"])
                z_rstar = _z_from_p(res["p_rstar"])

                improvement = np.abs(z_r - z_num) - np.abs(z_rstar - z_num)
                ax.plot(
                    n_vals,
                    improvement,
                    marker="o",
                    ms=4,
                    lw=1.2,
                    label=rf"$\tilde{{\theta}}={res['theta_tilde']}$",
                )

            ax.axhline(0.0, color="0.25", lw=1)
            ax.set_xlabel(r"$n_0$ (observed counts)")
            ax.set_ylabel(rf"$|Z_q - Z_\mathrm{{ref}}| - |Z_{{q^\ast}} - Z_\mathrm{{ref}}|$")
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
    b0_vec = np.asarray(cfg["b0_vec"], dtype=float)
    sigma_vec = np.asarray(cfg["sigma_vec"], dtype=float)
    theta_tilde_offsets = tuple(float(x) for x in cfg.get("theta_tilde_offsets", [-1.0, 0.0, 1.0]))
    n_gh_pts = int(cfg.get("n_gh_pts", 30))
    reference_tail_mass = float(cfg.get("reference_tail_mass", 1e-10))
    target_z = float(cfg.get("target_Z", 5.0))
    max_n = int(cfg.get("max_n", 10_000))
    trim_to_discovery_tail = bool(cfg.get("trim_to_discovery_tail", True))
    out_pdf = Path(cfg["out_pdf"])
    out_significance_pdf = Path(
        cfg.get("out_significance_pdf", out_pdf.with_name(f"{out_pdf.stem}_with_significance.pdf"))
    )
    out_improvement_pdf = Path(
        cfg.get("out_improvement_pdf", out_pdf.with_name(f"{out_pdf.stem}_rstar_improvement.pdf"))
    )
    save_individual = bool(cfg.get("individual_plots", False))
    include_ratio = bool(cfg.get("ratio_plots", False))
    make_significance_plots = bool(cfg.get("significance_plots", cfg.get("significance_pvalue_plots", True)))
    make_improvement_plots = bool(cfg.get("improvement_plots", False))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))
    reference_label = "Numerical"

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_significance_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_improvement_pdf.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for s0 in s0_vec:
        for b0 in b0_vec:
            for sigma in sigma_vec:
                for theta_tilde in theta_tilde_offsets:
                    all_results.extend(
                        compute_pvalues_lognormal(
                            float(s0),
                            float(b0),
                            float(sigma),
                            float(theta_tilde),
                            target_z=target_z,
                            trim_to_discovery_tail=trim_to_discovery_tail,
                            n_gh_pts=n_gh_pts,
                            reference_tail_mass=reference_tail_mass,
                            max_n=max_n,
                            continuity_correction_r=continuity_correction_r,
                            continuity_correction_rstar=continuity_correction_rstar,
                        )
                    )

    make_plots_lognormal(all_results, out_pdf, save_individual, include_ratio, reference_label)
    if make_significance_plots:
        make_significance_plots_lognormal(all_results, out_significance_pdf, include_ratio, reference_label)
    if make_improvement_plots:
        make_improvement_plots_lognormal(all_results, out_improvement_pdf, reference_label)

    print(f"Saved all plots to: {out_pdf.resolve()}")
    if make_significance_plots:
        print(f"Saved significance plots to: {out_significance_pdf.resolve()}")
    if make_improvement_plots:
        print(f"Saved q* improvement plots to: {out_improvement_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lognormal_pval.yaml",
        help="Path to YAML config for log-normal constrained p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
