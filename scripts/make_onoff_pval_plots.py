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


def nonneg_int_floor(x):
    return int(max(0, np.floor(x)))


def run_case(s0: float, b: float, tau: float, sigrel: float, pdf: PdfPages):
    mu_s = s0 + b
    mu_b = tau * b

    m0_raw = np.array(
        [
            nonneg_int_floor(mu_b - np.sqrt(mu_b)),
            nonneg_int_floor(mu_b),
            nonneg_int_floor(mu_b + np.sqrt(mu_b)),
        ],
        dtype=int,
    )

    m0_unique = np.unique(np.sort(m0_raw))
    if m0_unique.size == 1:
        k = m0_unique[0]
        m0_list = np.array([k, k + 1], dtype=int)
    else:
        m0_list = m0_unique

    nmax_raw = mu_s + 5.0 * np.sqrt(mu_s) + 2
    nmin = 1
    nmax = max(nmin, nonneg_int_floor(nmax_raw))
    n0_vals = np.arange(nmin, nmax + 1, dtype=int)

    for m0 in m0_list:
        p_r, p_rs, p_mc, p_mc_se = [], [], [], []

        for n0 in n0_vals:
            out = pvals_onoff(s0, b, tau, n0, m0, sigrel=sigrel)
            p_r.append(out["p_r"])
            p_rs.append(out["p_rstar"])
            p_mc.append(out["p_mc"])
            p_mc_se.append(out["p_mc_se"])

        p_r = np.asarray(p_r, dtype=float)
        p_rs = np.asarray(p_rs, dtype=float)
        p_mc = np.asarray(p_mc, dtype=float)
        p_err = np.asarray(p_mc_se, dtype=float)

        denom = np.clip(p_mc, 1e-16, None)
        rel_r = np.abs(p_r - p_mc) / denom
        rel_rs = np.abs(p_rs - p_mc) / denom

        eps = 1e-16
        p_r = np.maximum(p_r, eps)
        p_rs = np.maximum(p_rs, eps)
        p_mc = np.maximum(p_mc, eps)
        rel_r = np.maximum(rel_r, eps)
        rel_rs = np.maximum(rel_rs, eps)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3.5, 1.2]}
        )

        ax_top.semilogy(
            n0_vals,
            p_r,
            marker="o",
            linestyle="None",
            ms=5,
            label="1 − Φ(r)",
            color="tab:blue",
        )
        ax_top.semilogy(
            n0_vals,
            p_rs,
            marker="^",
            linestyle="None",
            ms=5,
            label="1 − Φ(r*)",
            color="tab:orange",
        )
        ax_top.errorbar(
            n0_vals,
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
        ax_top.set_xlim(nmin - 0.5, nmax + 0.5)
        ax_top.set_title(
            rf"$m_0={m0}$,  $s_0={s0}$,  $b={b}$,  $\tau={tau}$,  $\sigma_{{\mathrm{{rel}}}}={sigrel}$"
        )
        ax_top.grid(True, which="both", alpha=0.25)
        ax_top.legend()

        ax_bot.semilogy(
            n0_vals,
            rel_r,
            marker="o",
            linestyle="None",
            ms=4,
            label=r"|r − MC| / MC",
            color="tab:blue",
        )
        ax_bot.semilogy(
            n0_vals,
            rel_rs,
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

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tau_vec"], dtype=float)
    sigrel = float(cfg["sigrel"])
    out_pdf = Path(cfg["out_pdf"])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        for s0, b, tau in np.array(np.meshgrid(s_vec, b_vec, tau_vec)).T.reshape(-1, 3):
            run_case(float(s0), float(b), float(tau), sigrel, pdf)

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
