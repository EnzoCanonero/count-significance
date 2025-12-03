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
from src.on import r_stat, r_star, median_expected_significance


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_min = float(cfg["b_min"])
    b_max = float(cfg["b_max"])
    n_bpts = int(cfg["n_bpts"])
    n_outer = int(cfg.get("n_outer", 200))
    seed = int(cfg.get("seed", 12345))
    out_pdf = Path(cfg["out_pdf"])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    b_values = np.logspace(np.log10(b_min), np.log10(b_max), n_bpts)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        for s_true in s_vec:
            r_list, rstar_list, zmed_list = [], [], []
            for b in b_values:
                n_Asimov = s_true + b
                rA = r_stat(s0=0.0, b=b, n=n_Asimov)
                rstarA = r_star(s0=0.0, b=b, n=n_Asimov)
                Z_med = median_expected_significance(s_true, b, n_outer=n_outer, seed=seed)
                r_list.append(rA)
                rstar_list.append(rstarA)
                zmed_list.append(Z_med)

            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            ax.plot(b_values, r_list, label=fr"Asimov r, s_true={s_true}")
            ax.plot(
                b_values,
                rstar_list,
                linestyle="--",
                label=fr"Asimov r*, s_true={s_true}",
            )
            ax.plot(
                b_values,
                zmed_list,
                linestyle="None",
                marker="x",
                label=fr"MC median Z, s_true={s_true}",
            )

            ax.set_xscale("log")
            ax.set_xlabel("b")
            ax.set_ylabel("Z")
            ax.set_ylim(bottom=-1)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.legend(fontsize=9)
            ax.set_title(fr"Asimov vs MC median Z, $s_{{\mathrm{{true}}}} = {s_true}$")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/simple_medsig.yaml",
        help="Path to YAML config for simple medsig plots",
    )
    args = parser.parse_args()
    main(args.config)
