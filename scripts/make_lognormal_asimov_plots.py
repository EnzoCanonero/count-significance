#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.lognormal import expected_significance_lognormal


def _fmt(x):
    return f"{x:g}".replace(".", "p").replace("-", "m")


def run_asimov_lognormal(
    s_vec: np.ndarray,
    b0_values: np.ndarray,
    sigma_vec: np.ndarray,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute Asimov Z grids shaped (len(s_vec), len(sigma_vec), len(b0_values))."""
    s_grid = s_vec[:, None, None]
    sigma_grid = sigma_vec[None, :, None]
    b0_grid = b0_values[None, None, :]

    res = expected_significance_lognormal(
        s_true=s_grid,
        b0=b0_grid,
        sigma=sigma_grid,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )
    return res["Z_A_r"], res["Z_A_rstar"]


def make_asimov_plots_lognormal(
    s_vec: np.ndarray,
    b0_values: np.ndarray,
    sigma_vec: np.ndarray,
    Z_A_r: np.ndarray,
    Z_A_rstar: np.ndarray,
    out_pdf: Path,
    save_individual: bool,
    asimov_statistics: list,
):
    """Save Asimov discovery Z scans for the log-normal constrained model."""
    with PdfPages(out_pdf) as pdf:
        for s_idx, s_true in enumerate(s_vec):
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            for sig_idx, sigma in enumerate(sigma_vec):
                color = colors[sig_idx % len(colors)]
                if "r" in asimov_statistics:
                    ax.plot(
                        b0_values,
                        Z_A_r[s_idx, sig_idx],
                        color=color,
                        label=rf"Asimov $r$, $\sigma={sigma}$",
                    )
                if "rstar" in asimov_statistics:
                    ax.plot(
                        b0_values,
                        Z_A_rstar[s_idx, sig_idx],
                        "--",
                        color=color,
                        label=rf"Asimov $r^\ast$, $\sigma={sigma}$",
                    )

            ax.set_xscale("log")
            ax.set_xlabel(r"$b_0$")
            ax.set_ylabel(r"Asimov discovery $Z$")
            ax.set_ylim(bottom=0.0)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.legend(fontsize=9, frameon=False)
            ax.set_title(rf"log-normal constraint, $s_\mathrm{{true}}={s_true}$")

            plt.tight_layout()
            if save_individual:
                fig.savefig(out_pdf.parent / f"lognormal_asimov_s{_fmt(s_true)}.pdf")
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    sigma_vec = np.asarray(cfg["sigma_vec"], dtype=float)
    b0_min = float(cfg["b0_min"])
    b0_max = float(cfg["b0_max"])
    n_b0pts = int(cfg["n_b0pts"])
    out_pdf = Path(cfg["out_pdf"])
    save_individual = bool(cfg.get("individual_plots", False))
    asimov_statistics = cfg.get("asimov_statistics", ["r", "rstar"])
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    b0_values = np.logspace(np.log10(b0_min), np.log10(b0_max), n_b0pts)

    Z_A_r, Z_A_rstar = run_asimov_lognormal(
        s_vec=s_vec,
        b0_values=b0_values,
        sigma_vec=sigma_vec,
        continuity_correction_r=continuity_correction_r,
        continuity_correction_rstar=continuity_correction_rstar,
    )
    make_asimov_plots_lognormal(
        s_vec=s_vec,
        b0_values=b0_values,
        sigma_vec=sigma_vec,
        Z_A_r=Z_A_r,
        Z_A_rstar=Z_A_rstar,
        out_pdf=out_pdf,
        save_individual=save_individual,
        asimov_statistics=asimov_statistics,
    )

    if save_individual:
        print(f"Saved individual plots under: {out_pdf.parent.resolve()}")
    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lognormal_asimov.yaml",
        help="Path to YAML config for log-normal Asimov plots",
    )
    args = parser.parse_args()
    main(args.config)
