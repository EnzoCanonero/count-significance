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
from src.on_off import expected_significance_onoff

# Call graph (on/off medsig)
# - main: load config, prepare b grids, then runs experiments and plotting
#   - run_experiments_onoff: build Asimov + MC grids via expected_significance_onoff
#   - make_plots_onoff: save combined/individual PDFs
#     - plot_fixed_tau / plot_fixed_sigrel: per-τ or per-σ_rel panels


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def run_experiments_onoff(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau: np.ndarray,
    b_values_sig: np.ndarray,
    cfg: dict,
):
    """
    Compute Asimov and MC-median Z grids for the on/off case using expected_significance_onoff.
    """
    seed = int(cfg.get("seed", 12345))
    sigrel = float(cfg["sigrel_Z"])
    min_toys = int(cfg["min_toys"])
    max_toys = int(cfg["max_toys"])
    n_outer = int(cfg["n_outer"])

    # Fixed tau grids: shape (S, T, B_tau)
    s_grid_tau = s_vec[:, None, None]
    b_grid_tau = b_values_tau[None, None, :]
    tau_grid = tau_vec[None, :, None]
    res_tau = expected_significance_onoff(
        s_true=s_grid_tau,
        b=b_grid_tau,
        tau=tau_grid,
        n_outer=n_outer,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed,
    )

    # Fixed sigma_rel grids: tau(b) = 1/(sigma_rel^2 * b)
    s_grid_sig = s_vec[:, None, None]
    b_grid_sig = b_values_sig[None, None, :]
    tau_grid_sig = 1.0 / (rel_sig_vec[None, :, None] ** 2 * b_grid_sig)
    res_sig = expected_significance_onoff(
        s_true=s_grid_sig,
        b=b_grid_sig,
        tau=tau_grid_sig,
        n_outer=n_outer,
        sigrel=sigrel,
        min_toys=min_toys,
        max_toys=max_toys,
        seed=seed + 1,  # decorrelate from fixed-tau call
    )

    return (
        res_tau["Z_A_r"],
        res_tau["Z_A_rstar"],
        res_sig["Z_A_r"],
        res_sig["Z_A_rstar"],
        res_tau["Z_mc_median"],
        res_sig["Z_mc_median"],
    )


def plot_fixed_tau(
    s_true: float,
    tau: float,
    b_values: np.ndarray,
    Z_A_r: np.ndarray,
    Z_A_rstar: np.ndarray,
    Z_med: np.ndarray,
    pdf: PdfPages,
    save_individual: bool,
    outdir: Path,
):
    """Plot Asimov/MC median Z for a fixed τ scan over b."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(b_values, Z_A_r, label=r"Asimov $r$ (on/off)")
    ax.plot(b_values, Z_A_rstar, "--", label=r"Asimov $r^\ast$ (on/off)")
    ax.plot(b_values, Z_med, linestyle="None", marker="x", label=r"MC median $Z$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(r"$r,\, r^\ast,\, Z$")
    ax.set_ylim(bottom=-1, top=6)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.set_title(rf"$s_\mathrm{{true}} = {s_true}$, fixed $\tau = {tau}$")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    if save_individual:
        fname = outdir / f"onoff_bscan_s{_fmt(s_true)}_tau{_fmt(tau)}.pdf"
        fig.savefig(fname)
    pdf.savefig(fig)
    plt.close(fig)


def plot_fixed_sigrel(
    s_true: float,
    sigma_rel: float,
    b_values: np.ndarray,
    Z_A_r: np.ndarray,
    Z_A_rstar: np.ndarray,
    Z_med: np.ndarray,
    pdf: PdfPages,
    save_individual: bool,
    outdir: Path,
):
    """Plot Asimov/MC median Z for a fixed σ_rel scan over b."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(b_values, Z_A_r, label=r"Asimov $r$ (on/off)")
    ax.plot(b_values, Z_A_rstar, "--", label=r"Asimov $r^\ast$ (on/off)")
    ax.plot(b_values, Z_med, linestyle="None", marker="x", label=r"MC median $Z$")
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
    if save_individual:
        fname = outdir / f"onoff_bscan_s{_fmt(s_true)}_sigrel{_fmt(sigma_rel)}.pdf"
        fig.savefig(fname)
    pdf.savefig(fig)
    plt.close(fig)


def make_plots_onoff(
    s_vec: np.ndarray,
    tau_vec: np.ndarray,
    rel_sig_vec: np.ndarray,
    b_values_tau,
    b_values_sig,
    Z_A_r_tau: np.ndarray,
    Z_A_rstar_tau: np.ndarray,
    Z_A_r_sig: np.ndarray,
    Z_A_rstar_sig: np.ndarray,
    Z_med_tau: np.ndarray,
    Z_med_sig: np.ndarray,
    outdir,
    summary_pdf_path,
    save_individual: bool,
):
    """
    Save combined PDF (and optional per-plot PDFs) for the on/off medsig scans.
    """
    with PdfPages(summary_pdf_path) as pdf:
        for s_idx, s_true in enumerate(s_vec):
            # Fixed tau
            for tau_idx, tau in enumerate(tau_vec):
                plot_fixed_tau(
                    s_true=float(s_true),
                    tau=float(tau),
                    b_values=b_values_tau,
                    Z_A_r=Z_A_r_tau[s_idx, tau_idx],
                    Z_A_rstar=Z_A_rstar_tau[s_idx, tau_idx],
                    Z_med=Z_med_tau[s_idx, tau_idx],
                    pdf=pdf,
                    save_individual=save_individual,
                    outdir=outdir,
                )

        # Fixed sigma_rel
        for s_idx, s_true in enumerate(s_vec):
            for sig_idx, sigma_rel in enumerate(rel_sig_vec):
                plot_fixed_sigrel(
                    s_true=float(s_true),
                    sigma_rel=float(sigma_rel),
                    b_values=b_values_sig,
                    Z_A_r=Z_A_r_sig[s_idx, sig_idx],
                    Z_A_rstar=Z_A_rstar_sig[s_idx, sig_idx],
                    Z_med=Z_med_sig[s_idx, sig_idx],
                    pdf=pdf,
                    save_individual=save_individual,
                    outdir=outdir,
                )

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

    (
        Z_A_r_tau,
        Z_A_rstar_tau,
        Z_A_r_sig,
        Z_A_rstar_sig,
        Z_med_tau,
        Z_med_sig,
    ) = run_experiments_onoff(
        s_vec=cfg["s_vec"],
        tau_vec=cfg["tauVec"],
        rel_sig_vec=cfg["relSigVec"],
        b_values_tau=b_values_tau,
        b_values_sig=b_values_sig,
        cfg=cfg,
    )
    make_plots_onoff(
        s_vec=cfg["s_vec"],
        tau_vec=cfg["tauVec"],
        rel_sig_vec=cfg["relSigVec"],
        b_values_tau=b_values_tau,
        b_values_sig=b_values_sig,
        Z_A_r_tau=Z_A_r_tau,
        Z_A_rstar_tau=Z_A_rstar_tau,
        Z_A_r_sig=Z_A_r_sig,
        Z_A_rstar_sig=Z_A_rstar_sig,
        Z_med_tau=Z_med_tau,
        Z_med_sig=Z_med_sig,
        outdir=outdir,
        summary_pdf_path=summary_pdf_path,
        save_individual=save_individual,
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
