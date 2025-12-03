#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on import compute_curves


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    s0_vec = np.asarray(cfg["s0_vec"], dtype=float)
    out_pdf = Path(cfg["out_pdf"])

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
                n_vals, p_true, p_r, p_rstar = compute_curves(
                    float(s0), float(b_fixed)
                )

                ax.plot(
                    n_vals,
                    p_true,
                    marker="x",
                    linestyle="None",
                    label="MC",
                    color="tab:green",
                )
                ax.plot(
                    n_vals,
                    p_r,
                    marker="o",
                    linestyle="None",
                    label="1 − Φ(r)",
                    color="tab:blue",
                )
                ax.plot(
                    n_vals,
                    p_rstar,
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
