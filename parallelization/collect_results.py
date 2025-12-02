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


def make_plots(cfg, groups):
    outdir = cfg.get("outdir", "output")
    summary_pdf_path = os.path.join(outdir, "summary_plots.pdf")

    s_vec = cfg["s_vec"]
    tauVec = cfg["tauVec"]
    relSigVec = cfg["relSigVec"]

    b_min_tau = cfg["b_min_tau"]
    b_max_tau = cfg["b_max_tau"]
    n_bpts_tau = cfg["n_bpts_tau"]

    b_min_sig = cfg["b_min_sig"]
    b_max_sig = cfg["b_max_sig"]
    n_bpts_sig = cfg["n_bpts_sig"]

    b_values_tau = np.logspace(np.log10(b_min_tau), np.log10(b_max_tau), n_bpts_tau)
    b_values_sig = np.logspace(np.log10(b_min_sig), np.log10(b_max_sig), n_bpts_sig)

    with PdfPages(summary_pdf_path) as pdf:
        # ----- 1) Fixed tau -----
        for s_idx, s_true in enumerate(s_vec):
            for tau_idx, tau in enumerate(tauVec):
                Z_A_r = np.zeros(n_bpts_tau, dtype=float)
                Z_A_rstar = np.zeros(n_bpts_tau, dtype=float)
                Z_med = np.zeros(n_bpts_tau, dtype=float)

                for b_idx, b in enumerate(b_values_tau):
                    key = ("tau", s_idx, tau_idx, b_idx)
                    slot = groups.get(key)

                    if slot is None or len(slot["Z"]) == 0:
                        Z_med[b_idx] = np.nan
                    else:
                        Z_arr = np.asarray(slot["Z"])
                        Z_med[b_idx] = np.median(Z_arr)

                    asim = asimov_Zs_onoff(s_true, b, tau)
                    Z_A_r[b_idx] = asim["Z_A_r"]
                    Z_A_rstar[b_idx] = asim["Z_A_rstar"]

                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.plot(b_values_tau, Z_A_r, label=r"Asimov $r$ (on/off)")
                ax.plot(
                    b_values_tau, Z_A_rstar, "--", label=r"Asimov $r^\\ast$ (on/off)"
                )
                ax.plot(
                    b_values_tau,
                    Z_med,
                    linestyle="None",
                    marker="x",
                    label=r"MC median $Z$",
                )

                ax.set_xscale("log")
                ax.set_xlabel(r"$b$")
                ax.set_ylabel(r"$r,\, r^\ast,\, Z$")
                ax.set_ylim(bottom=-1, top=6)
                ax.grid(True, which="both", ls="--", alpha=0.35)
                ax.set_title(rf"$s_\\mathrm{{true}} = {s_true}$, fixed $\\tau = {tau}$")
                ax.legend(frameon=False, loc="upper right")

                plt.tight_layout()
                fname = f"onoff_bscan_s{_fmt(s_true)}_tau{_fmt(tau)}.pdf"
                fig.savefig(os.path.join(outdir, fname))
                pdf.savefig(fig)
                plt.close(fig)

        # ----- 2) Fixed sigma_rel -----
        for s_idx, s_true in enumerate(s_vec):
            for sig_idx, sigma_rel in enumerate(relSigVec):
                Z_A_r = np.zeros(n_bpts_sig, dtype=float)
                Z_A_rstar = np.zeros(n_bpts_sig, dtype=float)
                Z_med = np.zeros(n_bpts_sig, dtype=float)

                for b_idx, b in enumerate(b_values_sig):
                    key = ("sig", s_idx, sig_idx, b_idx)
                    slot = groups.get(key)

                    if slot is None or len(slot["Z"]) == 0:
                        Z_med[b_idx] = np.nan
                    else:
                        Z_arr = np.asarray(slot["Z"])
                        Z_med[b_idx] = np.median(Z_arr)

                    tau_b = 1.0 / (sigma_rel**2 * b)
                    asim = asimov_Zs_onoff(s_true, b, tau_b)
                    Z_A_r[b_idx] = asim["Z_A_r"]
                    Z_A_rstar[b_idx] = asim["Z_A_rstar"]

                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.plot(b_values_sig, Z_A_r, label=r"Asimov $r$ (on/off)")
                ax.plot(
                    b_values_sig, Z_A_rstar, "--", label=r"Asimov $r^\\ast$ (on/off)"
                )
                ax.plot(
                    b_values_sig,
                    Z_med,
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
                    rf"$s_\\mathrm{{true}} = {s_true}$, "
                    rf"fixed $\\sigma_b/b = {sigma_rel}$ "
                    r"(i.e. $\\tau(b)=1/(\\sigma_\\mathrm{rel}^2\\,b)$)"
                )
                ax.legend(frameon=False, loc="upper right")

                plt.tight_layout()
                fname = f"onoff_bscan_s{_fmt(s_true)}_sigrel{_fmt(sigma_rel)}.pdf"
                fig.savefig(os.path.join(outdir, fname))
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved individual plots and summary PDF in {outdir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: collect_results.py CONFIG.yaml", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg.get("outdir", "output"), exist_ok=True)
    groups = collect_results(cfg)
    make_plots(cfg, groups)


if __name__ == "__main__":
    main()
