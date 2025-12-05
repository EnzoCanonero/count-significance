#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on import pvals_on


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    out_pdf = Path(cfg["out_pdf"])
    save_individual = bool(cfg.get("individual_plots", False))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        for b_fixed in b_vec:
            nrows = len(s0_vec)
            fig_height = 3.8 * nrows + 1.0
            fig, axes = plt.subplots(
                nrows=nrows, ncols=1, figsize=(14, fig_height), sharex=False
            )
            if nrows == 1:
                axes = [axes]

            for ax, s0 in zip(axes, s0_vec):
                s0f = float(s0)
                bf = float(b_fixed)
                mu0 = s0f + bf
                n_min = 0
                n_max = int(math.ceil(mu0 + 5.0 * math.sqrt(mu0)))
                n_vals = np.arange(n_min, n_max + 1, dtype=int)

                out = pvals_on(s0f, bf, n_vals)
                p_true = out["p_true"]
                p_r = out["p_r"]
                p_rstar = out["p_rstar"]

                ax.plot(
                    n_vals,
                    np.asarray(p_true, dtype=float),
                    marker="x",
                    linestyle="None",
                    label="MC",
                    color="tab:green",
                )
                ax.plot(
                    n_vals,
                    np.asarray(p_r, dtype=float),
                    marker="o",
                    linestyle="None",
                    label="1 − Φ(r)",
                    color="tab:blue",
                )
                ax.plot(
                    n_vals,
                    np.asarray(p_rstar, dtype=float),
                    marker="p",
                    linestyle="None",
                    label="1 − Φ(r*)",
                    color="tab:orange",
                )

                ax.set_yscale("log")
                ax.set_ylabel("p-value", fontsize=12)
                mu0 = s0 + b_fixed
                ax.set_title(
                    f"s₀ = {s0},  b = {b_fixed},  μ₀ = s₀ + b = {mu0}",
                    fontsize=13,
                )
                ax.grid(True, which="both", linestyle=":", alpha=0.5)
                ax.tick_params(axis="both", labelsize=11)

            axes[-1].set_xlabel("Observed count n", fontsize=12)
            axes[0].legend(fontsize=11)

            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"simple_pval_b{_fmt(b_fixed)}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)

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
