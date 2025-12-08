#!/usr/bin/env python3
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.on_off import asimov_Zs_onoff  # noqa: E402


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def collect_results(cfg):
    outdir = cfg.get("outdir", "output")
    files = sorted(glob.glob(os.path.join(outdir, "pval_job_*.json")))
    groups = {}  # key -> {"meta": rec, "Z": []}

    for path in files:
        with open(path, "r") as f:
            job_data = json.load(f)

        for rec in job_data["points"]:
            key = (rec["mode"], rec["s_idx"], rec["param_idx"], rec["b_idx"])
            Z_single = norm.isf(rec["p_mc"])

            if key not in groups:
                groups[key] = {"meta": rec, "Z": []}
            groups[key]["Z"].append(Z_single)

    return groups


def make_plots(
    s_vec,
    tau_vec,
    rel_sig_vec,
    b_values_tau,
    b_values_sig,
    groups,
    out_pdf: Path,
    save_individual: bool,
    outdir: Path,
):
    """Render combined (and optional individual) PDFs, mirroring make_plots_onoff."""
    with PdfPages(out_pdf) as pdf:
        for s_idx, s_true in enumerate(s_vec):
            for tau_idx, tau in enumerate(tau_vec):
                Z_A_r = np.zeros_like(b_values_tau, dtype=float)
                Z_A_rstar = np.zeros_like(b_values_tau, dtype=float)
                Z_med = np.zeros_like(b_values_tau, dtype=float)

                for b_idx, b in enumerate(b_values_tau):
                    key = ("tau", s_idx, tau_idx, b_idx)
                    slot = groups.get(key)
                    Z_med[b_idx] = np.nan if slot is None or len(slot["Z"]) == 0 else np.median(np.asarray(slot["Z"]))

                    asim = asimov_Zs_onoff(s_true, b, tau)
                    Z_A_r[b_idx] = asim["Z_A_r"]
                    Z_A_rstar[b_idx] = asim["Z_A_rstar"]

                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.plot(b_values_tau, Z_A_r, label=r"Asimov $r$ (on/off)")
                ax.plot(b_values_tau, Z_A_rstar, "--", label=r"Asimov $r^\ast$ (on/off)")
                ax.plot(b_values_tau, Z_med, linestyle="None", marker="x", label=r"MC median $Z$")

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

            for sig_idx, sigma_rel in enumerate(rel_sig_vec):
                Z_A_r = np.zeros_like(b_values_sig, dtype=float)
                Z_A_rstar = np.zeros_like(b_values_sig, dtype=float)
                Z_med = np.zeros_like(b_values_sig, dtype=float)

                for b_idx, b in enumerate(b_values_sig):
                    key = ("sig", s_idx, sig_idx, b_idx)
                    slot = groups.get(key)
                    Z_med[b_idx] = np.nan if slot is None or len(slot["Z"]) == 0 else np.median(np.asarray(slot["Z"]))

                    tau_b = 1.0 / (sigma_rel**2 * b)
                    asim = asimov_Zs_onoff(s_true, b, tau_b)
                    Z_A_r[b_idx] = asim["Z_A_r"]
                    Z_A_rstar[b_idx] = asim["Z_A_rstar"]

                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.plot(b_values_sig, Z_A_r, label=r"Asimov $r$ (on/off)")
                ax.plot(b_values_sig, Z_A_rstar, "--", label=r"Asimov $r^\ast$ (on/off)")
                ax.plot(b_values_sig, Z_med, linestyle="None", marker="x", label=r"MC median $Z$")

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


def main():
    if len(sys.argv) != 2:
        print("Usage: collect_results.py CONFIG.yaml", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    outdir = Path(cfg.get("outdir", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    save_individual = bool(cfg.get("individual_plots", False))

    groups = collect_results(cfg)

    s_vec = cfg["s_vec"]
    tau_vec = cfg["tauVec"]
    rel_sig_vec = cfg["relSigVec"]

    b_values_tau = np.logspace(
        np.log10(cfg["b_min_tau"]), np.log10(cfg["b_max_tau"]), int(cfg["n_bpts_tau"])
    )
    b_values_sig = np.logspace(
        np.log10(cfg["b_min_sig"]), np.log10(cfg["b_max_sig"]), int(cfg["n_bpts_sig"])
    )

    out_pdf = Path(cfg.get("out_summary_pdf", outdir / "onoff_medsig.pdf"))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    make_plots(
        s_vec=s_vec,
        tau_vec=tau_vec,
        rel_sig_vec=rel_sig_vec,
        b_values_tau=b_values_tau,
        b_values_sig=b_values_sig,
        groups=groups,
        out_pdf=out_pdf,
        save_individual=save_individual,
        outdir=outdir,
    )

    if save_individual:
        print(f"Saved individual plots under: {outdir.resolve()}")
    print(f"Saved all plots to: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
