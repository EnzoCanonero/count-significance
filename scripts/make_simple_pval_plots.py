#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on import pvals_on


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def compute_pvalues_on(s_vec: np.ndarray, b_vec: np.ndarray):
    """Compute p-values and relative differences for all (s0, b) combinations."""
    combos = np.array(np.meshgrid(s_vec, b_vec)).T.reshape(-1, 2)
    results = []
    for s0, b in combos:
        s0f = float(s0)
        bf = float(b)
        mu0 = s0f + bf
        n_min = 0
        n_max = int(math.ceil(mu0 + 5.0 * math.sqrt(mu0)))
        n_vals = np.arange(n_min, n_max + 1, dtype=int)

        out = pvals_on(s0f, bf, n_vals)
        p_true = np.asarray(out["p_true"], dtype=float)
        p_r = np.asarray(out["p_r"], dtype=float)
        p_rstar = np.asarray(out["p_rstar"], dtype=float)

        denom = np.clip(p_true, 1e-16, None)
        rel_r = np.maximum(np.abs(p_r - p_true) / denom, 1e-16)
        rel_rstar = np.maximum(np.abs(p_rstar - p_true) / denom, 1e-16)
        p_true = np.maximum(p_true, 1e-16)
        p_r = np.maximum(p_r, 1e-16)
        p_rstar = np.maximum(p_rstar, 1e-16)

        results.append(
            {
                "s0": s0f,
                "b": bf,
                "mu0": mu0,
                "n_vals": n_vals,
                "n_min": n_min,
                "n_max": n_max,
                "p_true": p_true,
                "p_r": p_r,
                "p_rstar": p_rstar,
                "rel_r": rel_r,
                "rel_rstar": rel_rstar,
            }
        )
    return results


def make_plot_on(
    results: list,
    out_pdf: Path,
    save_individual: bool,
):
    """Render the p-value and relative-diff panels for all (s0, b) configurations."""
    with PdfPages(out_pdf) as pdf:
        for res in results:
            n_vals = res["n_vals"]
            n_min = res["n_min"]
            n_max = res["n_max"]
            s0f = res["s0"]
            bf = res["b"]
            mu0 = res["mu0"]
            p_true = res["p_true"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            rel_r = res["rel_r"]
            rel_rstar = res["rel_rstar"]

            fig, (ax_top, ax_bot) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3.5, 1.2]},
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
            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_min - 0.5, n_max + 0.5)
            ax_top.set_title(f"s₀ = {s0f},  b = {bf},  μ₀ = s₀ + b = {mu0}", fontsize=13)
            ax_top.grid(True, which="both", alpha=0.25)
            ax_top.legend()

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

            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"simple_pval_s{_fmt(s0f)}_b{_fmt(bf)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    out_pdf = Path(cfg["out_pdf"])
    save_individual = bool(cfg.get("individual_plots", False))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    results = compute_pvalues_on(s0_vec, b_vec)
    make_plot_on(results, out_pdf, save_individual)

    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/simple_pval.yaml",
        help="Path to YAML config for simple p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
