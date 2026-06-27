#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import norm, poisson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.on_off import b_profiled, pvals_onoff, r_star_onoff, r_stat_onoff


def _fmt(x):
    return f"{x:g}".replace(".", "p")


def _candidate_off_counts(mu_b: float, sigma_offsets=(-1.0, 0.0, 1.0)) -> np.ndarray:
    """Representative OFF counts at E[M] + k sqrt(E[M])."""
    sigma_m = np.sqrt(max(float(mu_b), 0.0))
    m0_raw = np.array(
        [
            max(0, int(np.floor(float(mu_b) + float(k) * sigma_m + 0.5)))
            for k in sigma_offsets
        ],
        dtype=int,
    )

    return np.unique(np.sort(m0_raw))


def _tail_on_counts(s0: float, m0: int, tau: float, n_min: int, n_max: int) -> np.ndarray:
    """Keep the last plateau point and the one-sided discovery tail."""
    threshold = float(s0) + float(m0) / float(tau)
    first_tail_n = int(np.floor(threshold)) + 1
    if first_tail_n > n_max:
        return np.array([], dtype=int)

    start_n = max(n_min, first_tail_n - 1)
    return np.arange(start_n, n_max + 1, dtype=int)


def _z_from_p(p):
    return norm.isf(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16))


def _reference_title(reference_label: str, sigrel: float) -> str:
    if reference_label == "MC":
        return rf"MC rel. $Z$ precision $={sigrel}$"
    return rf"reference $={reference_label}$"


def _poisson_grid_max(mu: float, observed: int = 0, tail_mass: float = 1e-12) -> int:
    """Finite Poisson support large enough for exact tail sums."""
    mu = float(mu)
    if mu <= 0.0:
        return max(0, int(observed))

    upper = poisson.ppf(1.0 - float(tail_mass), mu)
    if not np.isfinite(upper):
        upper = mu + 12.0 * np.sqrt(mu + 1.0) + 20.0

    return max(int(np.ceil(upper)), int(observed), 0)


def _pvals_onoff_exact(
    s,
    b,
    tau,
    n,
    m,
    tail_mass: float = 1e-12,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Exact plug-in null tail using the same r-ordering as the toy MC code."""
    r_ref = float(r_stat_onoff(s, n, m, tau, continuity_correction=False))
    r_obs = float(r_stat_onoff(s, n, m, tau, continuity_correction=continuity_correction_r))
    rs_obs = float(r_star_onoff(s, n, m, tau, continuity_correction=continuity_correction_rstar))

    p_r = float(norm.sf(max(r_obs, 0.0)))
    p_rs = float(norm.sf(max(rs_obs, 0.0)))

    if r_ref <= 0.0:
        return {
            "p_mc": 0.5,
            "p_mc_se": 0.0,
            "p_r": p_r,
            "p_rstar": p_rs,
        }

    b_tilde = float(b_profiled(s, n, m, tau))
    mu_n = float(s + b_tilde)
    mu_m = float(tau * b_tilde)

    n_grid = np.arange(_poisson_grid_max(mu_n, observed=n, tail_mass=tail_mass) + 1)
    m_grid = np.arange(_poisson_grid_max(mu_m, observed=m, tail_mass=tail_mass) + 1)
    pn = poisson.pmf(n_grid, mu_n)
    pm = poisson.pmf(m_grid, mu_m)

    nn, mm = np.meshgrid(n_grid, m_grid, indexing="ij")
    r_grid = r_stat_onoff(s, nn, mm, tau, continuity_correction=False)
    probs = pn[:, None] * pm[None, :]
    p_ref = float(np.sum(probs[r_grid >= r_ref]))
    p_ref = min(p_ref, 0.5)

    return {
        "p_mc": p_ref,
        "p_mc_se": 0.0,
        "p_r": p_r,
        "p_rstar": p_rs,
    }


def _pvals_onoff_reference(
    s,
    b,
    tau,
    n,
    m,
    sigrel: float,
    reference_method: str,
    reference_tail_mass: float,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    if reference_method == "exact":
        return _pvals_onoff_exact(
            s,
            b,
            tau,
            n,
            m,
            tail_mass=reference_tail_mass,
            continuity_correction_r=continuity_correction_r,
            continuity_correction_rstar=continuity_correction_rstar,
        )
    if reference_method == "mc":
        return pvals_onoff(
            s,
            b,
            tau,
            n,
            m,
            sigrel=sigrel,
            continuity_correction_r=continuity_correction_r,
            continuity_correction_rstar=continuity_correction_rstar,
        )
    raise ValueError(f"Unknown reference_method={reference_method!r}")


def compute_pvalues_onoff(
    s0: float,
    b: float,
    tau: float,
    sigrel: float,
    target_z: float = 5.0,
    m_sigma_offsets=(-1.0, 0.0, 1.0),
    drop_zero_m0: bool = True,
    trim_to_discovery_tail: bool = True,
    reference_method: str = "exact",
    reference_tail_mass: float = 1e-12,
    max_n: int = 10_000,
    continuity_correction_r: bool = False,
    continuity_correction_rstar: bool = True,
):
    """Compute p-values and relative differences for all n0 and candidate m0 values."""
    eps = 1e-16

    # Poisson means for the signal (ON) and control (OFF) regions under s0, b, τ
    mu_s = float(s0 + b)
    mu_b = float(tau * b)

    reference_method = str(reference_method).lower()
    m0_list = _candidate_off_counts(mu_b, sigma_offsets=m_sigma_offsets)
    if drop_zero_m0:
        nonzero_m0 = m0_list[m0_list != 0]
        m0_list = nonzero_m0 if nonzero_m0.size > 0 else np.array([1], dtype=int)

    # Define the scan range for ON counts n0:
    # from n = 0 up to μ_s + 5√μ_s (rounded up), ensuring at least one bin.
    n_min = 0
    n_max_start = max(n_min, int(np.ceil(mu_s + 5.0 * np.sqrt(mu_s))))

    results = []

    # Loop over the chosen OFF-count configurations
    for m0 in m0_list:
        threshold = float(s0) + int(m0) / float(tau)
        first_tail_n = int(np.floor(threshold)) + 1
        if trim_to_discovery_tail:
            start_n = max(n_min, first_tail_n - 1)
        else:
            start_n = n_min

        n_vals, p_r, p_rstar, p_mc, p_mc_se = [], [], [], [], []
        n0 = int(start_n)

        while n0 <= int(max_n):
            out = _pvals_onoff_reference(
                s0,
                b,
                tau,
                n0,
                int(m0),
                sigrel=sigrel,
                reference_method=reference_method,
                reference_tail_mass=reference_tail_mass,
                continuity_correction_r=continuity_correction_r,
                continuity_correction_rstar=continuity_correction_rstar,
            )
            n_vals.append(n0)
            p_r.append(out["p_r"])
            p_rstar.append(out["p_rstar"])
            p_mc.append(out["p_mc"])
            p_mc_se.append(out["p_mc_se"])

            z_ref = float(_z_from_p(out["p_mc"]))
            if n0 >= n_max_start and z_ref >= target_z:
                break

            n0 += 1
        else:
            raise RuntimeError(f"Failed to reach Z={target_z:g} by n={max_n}")

        n_vals = np.asarray(n_vals, dtype=int)
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
                "n_max": int(n_vals[-1]),
                "tail_threshold": threshold,
                "first_tail_n": first_tail_n,
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
    reference_label: str,
):
    """Render p-value and relative-diff panels for all on/off configurations."""
    with PdfPages(out_pdf) as pdf:
        for res in results:
            s0 = res["s0"]
            b = res["b"]
            tau = res["tau"]
            m0 = res["m0"]
            n_vals = res["n_vals"]
            p_r = res["p_r"]
            p_rstar = res["p_rstar"]
            p_mc = res["p_mc"]
            p_err = res["p_err"]
            rel_r = res["rel_r"]
            rel_rstar = res["rel_rstar"]
            first_tail_n = res["first_tail_n"]

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
                label=reference_label,
                color="tab:green",
            )
            ax_top.set_ylabel("p-value (upper tail)")
            ax_top.set_xlim(n_vals[0] - 0.5, n_vals[-1] + 0.5)
            ax_top.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
            ax_top.set_title(
                rf"$m_0={m0}$,  $s_0={s0}$,  $b={b}$,  $\tau={tau}$,  "
                + _reference_title(reference_label, sigrel)
            )
            ax_top.grid(True, which="both", alpha=0.25)
            ax_top.legend()

            if include_ratio:
                ax_bot.axvline(first_tail_n - 0.5, color="0.55", ls=":", lw=1)
                ax_bot.semilogy(
                    n_vals,
                    rel_r,
                    marker="o",
                    linestyle="None",
                    ms=4,
                    label=rf"$|r - {reference_label}|/{reference_label}$",
                    color="tab:blue",
                )
                ax_bot.semilogy(
                    n_vals,
                    rel_rstar,
                    marker="^",
                    linestyle="None",
                    ms=4,
                    label=rf"$|r^\ast - {reference_label}|/{reference_label}$",
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


def make_significance_plots_onoff(
    results: list,
    sigrel: float,
    out_pdf: Path,
    reference_label: str,
    include_ratio: bool,
):
    """Significance vs n0 with all OFF-count (m0) slices overlaid per (s0, b, tau).

    Marker and colour encode the STATISTIC (fixed across m0); the m0 slices are
    drawn in the same style and overlaid, so their spread is read off directly.
    """
    grouped = defaultdict(list)
    for res in results:
        grouped[(res["s0"], res["b"], res["tau"])].append(res)

    with PdfPages(out_pdf) as pdf:
        for (s0, b, tau), group in sorted(grouped.items()):
            group = sorted(group, key=lambda item: item["m0"])

            if include_ratio:
                fig, (ax_z, ax_bot) = plt.subplots(
                    2,
                    1,
                    figsize=(12, 9),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3.5, 1.2]},
                )
            else:
                fig, ax_z = plt.subplots(figsize=(12, 6))
                ax_bot = None

            n_lo, n_hi = np.inf, -np.inf
            for res in group:
                n_vals = res["n_vals"]
                n_lo = min(n_lo, int(n_vals[0]))
                n_hi = max(n_hi, int(n_vals[-1]))

                z_r = _z_from_p(res["p_r"])
                z_rstar = _z_from_p(res["p_rstar"])
                z_mc = _z_from_p(res["p_mc"])

                ax_z.plot(n_vals, z_r, marker="o", linestyle="None", ms=5, color="tab:blue")
                ax_z.plot(n_vals, z_rstar, marker="^", linestyle="None", ms=5, color="tab:orange")
                ax_z.plot(n_vals, z_mc, marker="x", linestyle="None", ms=4, color="tab:green")

                if include_ratio:
                    ax_bot.semilogy(n_vals, res["rel_r"], marker="o", linestyle="None", ms=4, color="tab:blue")
                    ax_bot.semilogy(n_vals, res["rel_rstar"], marker="^", linestyle="None", ms=4, color="tab:orange")

            # legend encodes only the statistic (identical style across m0)
            stat_handles = [
                Line2D([0], [0], color="tab:blue", marker="o", ls="none", ms=7, label=r"$r$"),
                Line2D([0], [0], color="tab:orange", marker="^", ls="none", ms=7, label=r"$r^\ast$"),
                Line2D([0], [0], color="tab:green", marker="x", ls="none", ms=7, label=reference_label),
            ]

            m_list = ", ".join(str(res["m0"]) for res in group)
            ax_z.set_ylabel(r"$Z=\Phi^{-1}(1-p)$")
            ax_z.set_xlim(n_lo - 0.5, n_hi + 0.5)
            ax_z.set_title(
                rf"$m_0\in\{{{m_list}\}}$,  $s_0={s0}$,  $b={b}$,  $\tau={tau}$,  "
                + _reference_title(reference_label, sigrel)
            )
            ax_z.grid(True, alpha=0.25)
            ax_z.legend(handles=stat_handles)

            if include_ratio:
                ax_bot.set_xlabel(r"$n_0$ (observed ON counts)")
                ax_bot.set_ylabel("rel. abs. diff")
                ax_bot.grid(True, which="both", alpha=0.25)
                ax_bot.legend(handles=stat_handles[:2])
            else:
                ax_z.set_xlabel(r"$n_0$ (observed ON counts)")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def make_improvement_plots_onoff(results: list, out_pdf: Path, reference_label: str):
    """Render separate pages showing where r* is closer to MC than r."""
    grouped = defaultdict(list)
    for res in results:
        grouped[(res["s0"], res["b"], res["tau"])].append(res)

    with PdfPages(out_pdf) as pdf:
        for (s0, b, tau), group in sorted(grouped.items()):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            for res in sorted(group, key=lambda item: item["m0"]):
                n_vals = res["n_vals"]
                z_mc = _z_from_p(res["p_mc"])
                z_r = _z_from_p(res["p_r"])
                z_rstar = _z_from_p(res["p_rstar"])

                improvement = np.abs(z_r - z_mc) - np.abs(z_rstar - z_mc)
                ax.plot(
                    n_vals,
                    improvement,
                    marker="o",
                    ms=4,
                    lw=1.2,
                    label=rf"$m_0={res['m0']}$",
                )

            ax.axhline(0.0, color="0.25", lw=1)
            ax.set_xlabel(r"$n_0$ (observed ON counts)")
            ax.set_ylabel(rf"$|Z_r - Z_\mathrm{{ref}}| - |Z_{{r^\ast}} - Z_\mathrm{{ref}}|$")
            ax.set_title(
                rf"$r^\ast$ improvement,  $s_0={s0}$,  $b={b}$,  $\tau={tau}$"
                f"\npositive values mean $r^\\ast$ is closer to {reference_label}"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    s0_vec = np.asarray(cfg["s_vec"], dtype=float)
    b_vec = np.asarray(cfg["b_vec"], dtype=float)
    tau_vec = np.asarray(cfg["tau_vec"], dtype=float)
    sigrel = float(cfg.get("mc_sigrel_Z", cfg.get("sigrel", 0.001)))
    target_z = float(cfg.get("target_Z", 5.0))
    m_sigma_offsets = tuple(float(x) for x in cfg.get("m_sigma_offsets", [-1.0, 0.0, 1.0]))
    reference_method = str(cfg.get("reference_method", "exact")).lower()
    reference_tail_mass = float(cfg.get("reference_tail_mass", 1e-12))
    max_n = int(cfg.get("max_n", 10_000))
    reference_label = "Exact" if reference_method == "exact" else "MC"
    out_pdf = Path(cfg["out_pdf"])
    out_improvement_pdf = Path(
        cfg.get("out_improvement_pdf", out_pdf.with_name(f"{out_pdf.stem}_rstar_improvement.pdf"))
    )
    out_significance_pdf = Path(
        cfg.get("out_significance_pdf", out_pdf.with_name(f"{out_pdf.stem}_with_significance.pdf"))
    )
    save_individual = bool(cfg.get("individual_plots", False))
    include_ratio = bool(cfg.get("ratio_plots", False))
    make_improvement_plots = bool(cfg.get("improvement_plots", True))
    make_significance_plots = bool(cfg.get("significance_plots", cfg.get("significance_pvalue_plots", True)))
    drop_zero_m0 = bool(cfg.get("drop_zero_m0", cfg.get("drop_s0_zero_m0", True)))
    trim_to_discovery_tail = bool(cfg.get("trim_to_discovery_tail", True))
    continuity_correction_r = bool(cfg.get("continuity_correction_r", False))
    continuity_correction_rstar = bool(cfg.get("continuity_correction_rstar", True))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_improvement_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_significance_pdf.parent.mkdir(parents=True, exist_ok=True)

    combos = np.array(np.meshgrid(s0_vec, b_vec, tau_vec)).T.reshape(-1, 3)
    all_results = []
    for s0, b, tau in combos:
        all_results.extend(
            compute_pvalues_onoff(
                float(s0),
                float(b),
                float(tau),
                sigrel,
                target_z=target_z,
                m_sigma_offsets=m_sigma_offsets,
                drop_zero_m0=drop_zero_m0,
                trim_to_discovery_tail=trim_to_discovery_tail,
                reference_method=reference_method,
                reference_tail_mass=reference_tail_mass,
                max_n=max_n,
                continuity_correction_r=continuity_correction_r,
                continuity_correction_rstar=continuity_correction_rstar,
            )
        )

    make_plots_onoff(all_results, sigrel, out_pdf, save_individual, include_ratio, reference_label)
    if make_significance_plots:
        make_significance_plots_onoff(
            all_results,
            sigrel,
            out_significance_pdf,
            reference_label,
            include_ratio,
        )
    if make_improvement_plots:
        make_improvement_plots_onoff(all_results, out_improvement_pdf, reference_label)

    print(f"Saved all plots to: {out_pdf.resolve()}")
    if make_significance_plots:
        print(f"Saved significance plots to: {out_significance_pdf.resolve()}")
    if make_improvement_plots:
        print(f"Saved r* improvement plots to: {out_improvement_pdf.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/onoff_pval.yaml",
        help="Path to YAML config for on/off p-value plots",
    )
    args = parser.parse_args()
    main(args.config)
