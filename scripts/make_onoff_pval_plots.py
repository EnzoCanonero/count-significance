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
from src.on_off import pvals_onoff


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def compute_pvalues_onoff(s0: float, b: float, tau: float, sigrel: float):
    """Compute p-values and relative differences for all n0 and candidate m0 values."""
    eps = 1e-16

    # Poisson means for the signal (ON) and control (OFF) regions under s0, b, τ
    mu_s = float(s0 + b)
    mu_b = float(tau * b)

    # Compute candidate OFF counts m0 corresponding to μ_b ± 1σ and μ_b itself,
    # with expectations floored to non-negative integers.
    m0_raw = np.array(
        [
            max(0, int(np.floor(mu_b - np.sqrt(mu_b)))),
            max(0, int(np.floor(mu_b))),
            max(0, int(np.floor(mu_b + np.sqrt(mu_b)))),
        ]
    )

    # Retain only distinct m0 values, sorted for consistency
    m0_unique = np.unique(np.sort(m0_raw))

    # If all three collapse to the same integer, add the next integer so that
    # we can still see dependence on m0 in the plots.
    if m0_unique.size == 1:
        k = m0_unique[0]
        m0_list = np.array([k, k + 1], dtype=int)
    else:
        m0_list = m0_unique

    # Define the scan range for ON counts n0:
    # from n = 0 up to μ_s + 5√μ_s (rounded up), ensuring at least one bin.
    n_min = 0
    n_max = max(n_min, int(np.ceil(mu_s + 5.0 * np.sqrt(mu_s))))
    n_vals = np.arange(n_min, n_max + 1, dtype=int)

    results = []

    # Loop over the chosen OFF-count configurations
    for m0 in m0_list:
        p_r, p_rstar, p_mc, p_mc_se = [], [], [], []

        for n0 in n_vals:
            out = pvals_onoff(s0, b, tau, n0, m0, sigrel=sigrel)
            p_r.append(out["p_r"])
            p_rstar.append(out["p_rstar"])
            p_mc.append(out["p_mc"])
            p_mc_se.append(out["p_mc_se"])

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
                "n_max": n_max,
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
):
    """Render p-value and relative-diff panels for all on/off configurations."""
    with PdfPages(out_pdf) as pdf:
        for res in results:
            s0 = res["s0"]
            b = res["b"]
            tau = res["tau"]
            m0 = res["m0"]
            n_vals = res["n_vals"]
            n_min = res["n_min"]
            n_max = res["n_max"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            p_mc = res["p_mc"]
            p_err = res["p_err"]
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
            ax_top.errorbar(
                n_vals,
                p_mc,
                yerr=p_err,
                fmt="x",
                ms=4,
                lw=1,
                capsize=2,
                label="MC",
                color="tab:green",
            )
            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_min - 0.5, n_max + 0.5)
            ax_top.set_title(
                rf"$m_0={m0}$,  $s_0={s0}$,  $b={b}$,  $\tau={tau}$,  $\sigma_{{\mathrm{{rel}}}}={sigrel}$"
            )
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
                ax_bot.set_xlabel(r"$n_0$ (observed ON counts)")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend()
            else:
                ax_top.set_xlabel(r"$n_0$ (observed ON counts)")

            plt.tight_layout()
            if save_individual:
                fname = out_pdf.parent / f"onoff_pval_s{_fmt(s0)}_b{_fmt(b)}_tau{_fmt(tau)}_m{m0}.pdf"
                fig.savefig(fname)
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tau_vec"], dtype=float)
    sigrel = float(cfg["sigrel"])
    out_pdf = Path(cfg["out_pdf"])
    save_individual = bool(cfg.get("individual_plots", False))
    include_ratio = bool(cfg.get("ratio_plots", False))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    combos = np.array(np.meshgrid(s0_vec, b_vec, tau_vec)).T.reshape(-1, 3)
    all_results = []
    for s0, b, tau in combos:
        all_results.extend(compute_pvalues_onoff(float(s0), float(b), float(tau), sigrel))

    make_plots_onoff(all_results, sigrel, out_pdf, save_individual, include_ratio)

    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/onoff_pval.yaml",
        help="Path to YAML config for on/off p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
