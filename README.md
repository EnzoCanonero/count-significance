# medsig

`medsig` contains the statistical calculations and plotting workflows used for
the accompanying paper on higher-order discovery significance in Poisson
counting experiments.

The two models are

- known background: `N ~ Pois(s + b)`;
- on/off measurement: `N ~ Pois(s + b)` and `M ~ Pois(tau b)`.

The code evaluates the signed likelihood root `r` and its higher-order form
`r*`. The discovery statistics are `q0 = max(0, r)^2` and
`q0* = max(0, r*)^2`; their significances are the corresponding square roots.
The one-sided p-value is `1 - Phi(Z)`.

## Repository layout

- `src/common.py`: the shared discovery convention and Gaussian tail.
- `src/on.py`: known-background likelihood, p-values and expected significance.
- `src/on_off.py`: profiled on/off likelihood, higher-order statistic and MC.
- `config/`: one paper configuration for each plotting workflow.
- `scripts/`: local plotting commands and the HTCondor production workflow.
- `notebooks/playground.ipynb`: a short interactive introduction to both models.
- `plots/`: the PDFs produced for the paper.
- `runs/`: ignored HTCondor campaign data, created at submission time.

## Installation

From the repository root, install the package in editable mode:

```bash
python3 -m pip install -e .
```

## Paper plots

Run the four plotting commands from the repository root:

```bash
python3 scripts/make_simple_pval_plots.py
python3 scripts/make_simple_medsig_plots.py
python3 scripts/make_onoff_pval_plots.py
python3 scripts/make_onoff_medsig_plots.py
```

The known-background reference is the inclusive Poisson tail. For the on/off
observed-significance plot, `profile_sum` is a deterministic plug-in reference:
the background is profiled under the tested signal and the joint Poisson tail
is summed on a truncated grid. It is a reference calculation, not an exact
elimination of the nuisance parameter.

Plot configurations use `statistics: [r, rstar]` to select asymptotic curves.
Median-significance configurations use `mc_summaries: [median]`; `mean` is also
available. The local on/off MC settings are intentionally modest enough for a
terminal run and do not reproduce the manuscript's production precision.

## HTCondor production

The local and batch calculations share `config/paper_onoff_medsig.yaml`. The
serial plotter reads `local_mc`; the cluster workflow reads `batch_mc`.

From a clean checkout on the cluster, optionally identify a script that
activates the Python environment, then submit a named campaign:

```bash
export MEDSIG_SETUP_SCRIPT=/path/to/setup_medsig_env.sh
python3 scripts/submit_onoff_medsig_jobs.py --run paper-production
```

The submitter records the Git commit and freezes copies of the configuration,
worker and statistical source under `runs/paper-production/`. It checks the
Condor description with `condor_submit -dry-run` before submitting any jobs.

```text
runs/paper-production/
├── config.yaml
├── manifest.json
├── input/
├── results/
│   ├── s2/
│   ├── s5/
│   └── s10/
└── logs/
```

After all jobs finish, collect the campaign from the same clean Git commit:

```bash
python3 scripts/collect_onoff_medsig_results.py --run paper-production
```

The collector rejects missing, duplicated, corrupt or inconsistent results.
Only after the complete campaign passes validation does it write the standard
PDFs to `plots/`.
