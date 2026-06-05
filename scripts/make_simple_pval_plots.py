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


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def _z_from_p(p):
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


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
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute p-values and relative differences for all n given a single (s0, b)."""
    eps = 1e-16
    mu0 = float(s0 + b)
    # Define the scan range for counts n:
    # from n = 0 up to μ₀ + 5√μ₀ (rounded up), ensuring at least one bin.
    n_min = 0
    n_max_start = max(n_min, int(np.ceil(mu0 + 5.0 * np.sqrt(mu0))))
    n_max = _n_max_for_target_z_on(s0, b, n_max_start, target_z)
    n_vals = np.arange(n_min, n_max + 1, dtype=int)

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
            "n_min": n_min,
            "n_max": n_max,
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
):
    """Render the p-value and relative-diff panels for all (s0, b) configurations."""
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

            if include_ratio:
                fig, (ax_top, ax_bot) = plt.subplots(
                    2,
                    1,
                    figsize=(12, 9),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3.5, 1.2]},
                )
            else:
                fig, ax_top = plt.subplots(figsize=(12, 6))
                ax_bot = None

            ax_top.semilogy(
                n_vals,
                p_r,
                marker="o",
                linestyle="None",
                ms=5,
                label="1 − Φ(r)",
                color="tab:blue",
            )
            ax_top.semilogy(
                n_vals,
                p_rstar,
                marker="^",
                linestyle="None",
                ms=5,
                label="1 − Φ(r*)",
                color="tab:orange",
            )
            ax_top.semilogy(
                n_vals,
                p_true,
                marker="x",
                linestyle="None",
                ms=4,
                label="MC",
                color="tab:green",
            )
            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_min - 0.5, n_max + 0.5)
            ax_top.set_title(rf"$s_0={s0}$,  $b={b}$,  $\mu_0=s_0+b={mu0}$")
            ax_top.grid(True, which="both", alpha=0.25)
            ax_top.legend()

            if include_ratio:
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=r"|r − MC| / MC",
                    color="tab:blue",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=r"|r* − MC| / MC",
                    color="tab:orange",
                )
                ax_bot.set_xlabel("Observed count n")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend()
            else:
                ax_top.set_xlabel("Observed count n")

            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"simple_pval_s{_fmt(s0)}_b{_fmt(b)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def make_significance_plot_on(results: list, out_pdf: Path, include_ratio: bool):
    """Render significance panels, optionally with relative-difference panels below."""
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

            z_true = _z_from_p(p_true)
            z_r = _z_from_p(p_r)
            z_rstar = _z_from_p(p_rstar)

            if include_ratio:
                fig, (ax_z, ax_bot) = plt.subplots(
                    2,
                    1,
                    figsize=(12, 9),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3.5, 1.2]},
                )
            else:
                fig, ax_z = plt.subplots(figsize=(12, 6))
                ax_bot = None

            ax_z.plot(n_vals, z_r, marker="o", linestyle="None", ms=5, label=r"$r$", color="tab:blue")
            ax_z.plot(
                n_vals,
                z_rstar,
                marker="^",
                linestyle="None",
                ms=5,
                label=r"$r^\ast$",
                color="tab:orange",
            )
            ax_z.plot(
                n_vals,
                z_true,
                marker="x",
                linestyle="None",
                ms=4,
                label="MC",
                color="tab:green",
            )
            ax_z.set_ylabel(r"$Z=\Phi^{-1}(1-p)$")
            ax_z.set_xlim(n_min - 0.5, n_max + 0.5)
            ax_z.set_title(rf"$s_0={s0}$,  $b={b}$,  $\mu_0=s_0+b={mu0}$")
            ax_z.grid(True, alpha=0.25)
            ax_z.legend()

            if include_ratio:
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_r-p_\mathrm{MC}|/p_\mathrm{MC}$",
                    color="tab:blue",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=r"$|p_{r^\ast}-p_\mathrm{MC}|/p_\mathrm{MC}$",
                    color="tab:orange",
                )
                ax_bot.set_xlabel("Observed count n")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend()
            else:
                ax_z.set_xlabel("Observed count n")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    target_z = float(cfg.get("target_Z", 5.0))
    out_pdf = Path(cfg["out_pdf"])
    out_significance_pdf = Path(
        cfg.get("out_significance_pdf", out_pdf.with_name(f"{out_pdf.stem}_with_significance.pdf"))
    )
    save_individual = bool(cfg.get("individual_plots", False))
    include_ratio = bool(cfg.get("ratio_plots", False))
    make_significance_plots = bool(cfg.get("significance_plots", cfg.get("significance_pvalue_plots", True)))
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
                continuity_correction_r=continuity_correction_r,
                continuity_correction_rstar=continuity_correction_rstar,
            )
        )
    make_plot_on(all_results, out_pdf, save_individual, include_ratio)
    if make_significance_plots:
        make_significance_plot_on(all_results, out_significance_pdf, include_ratio)

    print(f"Saved all plots to: {out_pdf.resolve()}")
    if make_significance_plots:
        print(f"Saved significance plots to: {out_significance_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/simple_pval.yaml",
        help="Path to YAML config for simple p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
