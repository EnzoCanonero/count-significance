#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on_off import asimov_Zs_onoff, norm_isf, pvals_onoff


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def compute_single_Z(s_true, b, tau, rng_outer, rng_inner, sigrel_Z, min_toys, max_toys):
    """One pseudo-experiment: draw (n_obs, m_obs), compute p_mc, convert to Z."""
    n_obs = int(rng_outer.poisson(lam=s_true + b))
    m_obs = int(rng_outer.poisson(lam=tau * b))
    inner_seed_local = int(rng_inner.integers(0, 2**31 - 1))

    out = pvals_onoff(
        s=0.0,
        b=b,
        tau=tau,
        n=n_obs,
        m=m_obs,
        sigrel=sigrel_Z,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=inner_seed_local,
    )
    return norm_isf(out["p_mc"])


def run_outer_experiments(cfg, b_values_tau, b_values_sig):
    s_vec = cfg["s_vec"]
    tauVec = cfg["tauVec"]
    relSigVec = cfg["relSigVec"]
    n_outer = cfg["n_outer"]
    sigrel_Z = cfg["sigrel_Z"]
    min_toys = cfg["min_toys"]
    max_toys = cfg["max_toys"]
    outer_seed = cfg.get("outer_seed", 12345)
    inner_seed = cfg.get("inner_seed", 67890)

    Z_groups = defaultdict(list)

    for outer_idx in range(n_outer):
        rng_outer = np.random.default_rng(outer_seed + outer_idx)
        rng_inner = np.random.default_rng(inner_seed + outer_idx)

        for s_idx, s_true in enumerate(s_vec):
            # Fixed-tau scan
            for tau_idx, tau in enumerate(tauVec):
                for b_idx, b in enumerate(b_values_tau):
                    Z_single = compute_single_Z(
                        s_true, b, tau, rng_outer, rng_inner, sigrel_Z, min_toys, max_toys
                    )
                    Z_groups[("tau", s_idx, tau_idx, b_idx)].append(Z_single)

            # Fixed-sigma_rel scan (tau depends on b)
            for sig_idx, sigma_rel in enumerate(relSigVec):
                for b_idx, b in enumerate(b_values_sig):
                    tau_b = 1.0 / (sigma_rel**2 * b)
                    Z_single = compute_single_Z(
                        s_true, b, tau_b, rng_outer, rng_inner, sigrel_Z, min_toys, max_toys
                    )
                    Z_groups[("sig", s_idx, sig_idx, b_idx)].append(Z_single)

    return Z_groups


def median_grids(cfg, Z_groups, b_values_tau, b_values_sig):
    Z_med_tau = np.full((len(cfg["s_vec"]), len(cfg["tauVec"]), len(b_values_tau)), np.nan)
    Z_med_sig = np.full((len(cfg["s_vec"]), len(cfg["relSigVec"]), len(b_values_sig)), np.nan)

    for (mode, s_idx, param_idx, b_idx), values in Z_groups.items():
        if len(values) == 0:
            continue
        med = float(np.median(np.asarray(values, dtype=float)))
        if mode == "tau":
            Z_med_tau[s_idx, param_idx, b_idx] = med
        else:
            Z_med_sig[s_idx, param_idx, b_idx] = med

    return Z_med_tau, Z_med_sig


def make_plots(
    cfg,
    b_values_tau,
    b_values_sig,
    Z_med_tau,
    Z_med_sig,
    outdir,
    summary_pdf_path,
    save_individual: bool,
):
    s_vec = cfg["s_vec"]
    tauVec = cfg["tauVec"]
    relSigVec = cfg["relSigVec"]

    Z_A_r_tau = np.zeros((len(s_vec), len(tauVec), len(b_values_tau)), dtype=float)
    Z_A_rstar_tau = np.zeros_like(Z_A_r_tau)
    Z_A_r_sig = np.zeros((len(s_vec), len(relSigVec), len(b_values_sig)), dtype=float)
    Z_A_rstar_sig = np.zeros_like(Z_A_r_sig)

    for s_idx, s_true in enumerate(s_vec):
        for tau_idx, tau in enumerate(tauVec):
            for b_idx, b in enumerate(b_values_tau):
                asim = asimov_Zs_onoff(s_true, b, tau)
                Z_A_r_tau[s_idx, tau_idx, b_idx] = asim["Z_A_r"]
                Z_A_rstar_tau[s_idx, tau_idx, b_idx] = asim["Z_A_rstar"]

        for sig_idx, sigma_rel in enumerate(relSigVec):
            for b_idx, b in enumerate(b_values_sig):
                tau_b = 1.0 / (sigma_rel**2 * b)
                asim = asimov_Zs_onoff(s_true, b, tau_b)
                Z_A_r_sig[s_idx, sig_idx, b_idx] = asim["Z_A_r"]
                Z_A_rstar_sig[s_idx, sig_idx, b_idx] = asim["Z_A_rstar"]

    with PdfPages(summary_pdf_path) as pdf:
        for s_idx, s_true in enumerate(s_vec):
            # Fixed tau
            for tau_idx, tau in enumerate(tauVec):
                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.plot(b_values_tau, Z_A_r_tau[s_idx, tau_idx], label=r"Asimov $r$ (on/off)")
                ax.plot(
                    b_values_tau,
                    Z_A_rstar_tau[s_idx, tau_idx],
                    "--",
                    label=r"Asimov $r^\ast$ (on/off)",
                )
                ax.plot(
                    b_values_tau,
                    Z_med_tau[s_idx, tau_idx],
                    linestyle="None",
                    marker="x",
                    label=r"MC median $Z$",
                )
                ax.set_xscale("log")
                ax.set_xlabel(r"$b$")
                ax.set_ylabel(r"$r,\, r^\ast,\, Z$")
                ax.set_ylim(bottom=-1, top=6)
                ax.grid(True, which="both", ls="--", alpha=0.35)
                ax.set_title(rf"$s_\mathrm{{true}} = {s_true}$, fixed $\tau = {tau}$")
                ax.legend(frameon=False, loc="upper right")
                plt.tight_layout()
                fname = outdir / f"onoff_bscan_s{_fmt(s_true)}_tau{_fmt(tau)}.pdf"
                if save_individual:
                    fig.savefig(fname)
                pdf.savefig(fig)
                plt.close(fig)

        # Fixed sigma_rel
        for s_idx, s_true in enumerate(s_vec):
            for sig_idx, sigma_rel in enumerate(relSigVec):
                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.plot(b_values_sig, Z_A_r_sig[s_idx, sig_idx], label=r"Asimov $r$ (on/off)")
                ax.plot(
                    b_values_sig,
                    Z_A_rstar_sig[s_idx, sig_idx],
                    "--",
                    label=r"Asimov $r^\ast$ (on/off)",
                )
                ax.plot(
                    b_values_sig,
                    Z_med_sig[s_idx, sig_idx],
                    linestyle="None",
                    marker="x",
                    label=r"MC median $Z$",
                )
                ax.set_xscale("log")
                ax.set_xlabel(r"$b$")
                ax.set_ylabel(r"$r,\, r^\ast,\, Z$")
                ax.set_ylim(bottom=-1, top=8)
                ax.grid(True, which="both", ls="--", alpha=0.35)
                ax.set_title(
                    rf"$s_\mathrm{{true}} = {s_true}$, "
                    rf"fixed $\sigma_b/b = {sigma_rel}$ "
                    r"(i.e. $\tau(b)=1/(\sigma_\mathrm{rel}^2\,b)$)"
                )
                ax.legend(frameon=False, loc="upper right")
                plt.tight_layout()
                fname = outdir / f"onoff_bscan_s{_fmt(s_true)}_sigrel{_fmt(sigma_rel)}.pdf"
                if save_individual:
                    fig.savefig(fname)
                pdf.savefig(fig)
                plt.close(fig)

    if save_individual:
        print(f"Saved individual plots under {outdir.resolve()}")
    print(f"Saved combined PDF to {summary_pdf_path.resolve()}")


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    cfg["s_vec"] = np.asarray(cfg["s_vec"], dtype=float)
    cfg["tauVec"] = np.asarray(cfg["tauVec"], dtype=float)
    cfg["relSigVec"] = np.asarray(cfg["relSigVec"], dtype=float)

    b_values_tau = np.logspace(
        np.log10(cfg["b_min_tau"]), np.log10(cfg["b_max_tau"]), int(cfg["n_bpts_tau"])
    )
    b_values_sig = np.logspace(
        np.log10(cfg["b_min_sig"]), np.log10(cfg["b_max_sig"]), int(cfg["n_bpts_sig"])
    )

    outdir = Path(cfg.get("outdir", "plots"))
    save_individual = bool(cfg.get("individual_plots", False))
    outdir.mkdir(parents=True, exist_ok=True)
    summary_pdf_path = Path(cfg.get("out_summary_pdf", outdir / "onoff_medsig.pdf"))
    summary_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    Z_groups = run_outer_experiments(cfg, b_values_tau, b_values_sig)
    Z_med_tau, Z_med_sig = median_grids(cfg, Z_groups, b_values_tau, b_values_sig)
    make_plots(
        cfg,
        b_values_tau,
        b_values_sig,
        Z_med_tau,
        Z_med_sig,
        outdir,
        summary_pdf_path,
        save_individual,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/onoff_medsig.yaml",
        help="Path to YAML config for on/off medsig plots",
    )
    args = parser.parse_args()
    main(args.config)
