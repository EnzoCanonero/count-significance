#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.on_off import pvals_onoff  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("Usage: pval.py CONFIG.yaml JOB_ID", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    job_id = int(sys.argv[2])

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    s_vec = cfg["s_vec"]
    tauVec = cfg["tauVec"]
    relSigVec = cfg["relSigVec"]

    # b-ranges
    b_min_tau = cfg["b_min_tau"]
    b_max_tau = cfg["b_max_tau"]
    n_bpts_tau = cfg["n_bpts_tau"]

    b_min_sig = cfg["b_min_sig"]
    b_max_sig = cfg["b_max_sig"]
    n_bpts_sig = cfg["n_bpts_sig"]

    b_values_tau = np.logspace(np.log10(b_min_tau), np.log10(b_max_tau), n_bpts_tau)
    b_values_sig = np.logspace(np.log10(b_min_sig), np.log10(b_max_sig), n_bpts_sig)

    # Number of outer pseudo-experiments drawn *per grid point* by this job.
    # Total outer toys per point = (number of jobs) * outer_per_job.
    outer_per_job = int(cfg.get("outer_per_job", 1))

    # RNGs for this job. Each job gets an independent stream; the outer_per_job
    # replicas are consecutive draws from the same stream (still independent).
    outer_seed_base = cfg.get("outer_seed", 12345)
    inner_seed_base = cfg.get("inner_seed", 67890)

    rng_outer = np.random.default_rng(outer_seed_base + job_id)
    rng_inner = np.random.default_rng(inner_seed_base + job_id)

    sigrel_Z = cfg["sigrel_Z"]
    min_toys = cfg.get("min_toys", 10_000)
    max_toys = cfg.get("max_toys", 2_000_000)
    p_res = 1.0 / float(max_toys)

    points = []

    for replica in range(outer_per_job):
        # --------- 1) Fixed tau ---------
        for s_idx, s_true in enumerate(s_vec):
            for tau_idx, tau in enumerate(tauVec):
                for b_idx, b in enumerate(b_values_tau):
                    n_obs = int(rng_outer.poisson(lam=s_true + b))
                    m_obs = int(rng_outer.poisson(lam=tau * b))
                    inner_seed = int(rng_inner.integers(1, 2**31 - 1))

                    out_p = pvals_onoff(
                        s=0.0,
                        b=b,
                        tau=tau,
                        n=n_obs,
                        m=m_obs,
                        sigrel=sigrel_Z,
                        min_toys=min_toys,
                        max_toys=max_toys,
                        seed=inner_seed,
                    )

                    p_mc = min(max(out_p["p_mc"], p_res), 1.0 - p_res)
                    Z_single = norm.isf(p_mc)

                    rec = {
                        "mode": "tau",
                        "s_idx": s_idx,
                        "param_idx": tau_idx,
                        "b_idx": b_idx,
                        "s_true": float(s_true),
                        "b": float(b),
                        "tau": float(tau),
                        "sigma_rel": None,
                        "n_obs": int(n_obs),
                        "m_obs": int(m_obs),
                        "Z_single": float(Z_single),
                    }
                    rec.update(out_p)
                    rec["p_mc"] = p_mc
                    points.append(rec)

            # --------- 2) Fixed sigma_rel ---------
            for sig_idx, sigma_rel in enumerate(relSigVec):
                for b_idx, b in enumerate(b_values_sig):
                    tau_b = 1.0 / (sigma_rel**2 * b)

                    n_obs = int(rng_outer.poisson(lam=s_true + b))
                    m_obs = int(rng_outer.poisson(lam=tau_b * b))
                    inner_seed = int(rng_inner.integers(1, 2**31 - 1))

                    out_p = pvals_onoff(
                        s=0.0,
                        b=b,
                        tau=tau_b,
                        n=n_obs,
                        m=m_obs,
                        sigrel=sigrel_Z,
                        min_toys=min_toys,
                        max_toys=max_toys,
                        seed=inner_seed,
                    )

                    p_mc = min(max(out_p["p_mc"], p_res), 1.0 - p_res)
                    Z_single = norm.isf(p_mc)

                    rec = {
                        "mode": "sig",
                        "s_idx": s_idx,
                        "param_idx": sig_idx,
                        "b_idx": b_idx,
                        "s_true": float(s_true),
                        "b": float(b),
                        "tau": float(tau_b),
                        "sigma_rel": float(sigma_rel),
                        "n_obs": int(n_obs),
                        "m_obs": int(m_obs),
                        "Z_single": float(Z_single),
                    }
                    rec.update(out_p)
                    rec["p_mc"] = p_mc
                    points.append(rec)

    outdir = cfg.get("outdir", "output")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    out_data = {"job_id": job_id, "outer_per_job": outer_per_job, "points": points}
    out_path = Path(outdir) / f"pval_job_{job_id:06d}.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)


if __name__ == "__main__":
    main()
