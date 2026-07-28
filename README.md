This repo contains code to:
- Compute p-values and Asimov/MC median significances for simple counting (“on”) and on/off experiments.
- Compare asymptotic test statistics r, r* to MC.

Structure (flattened modules, scripts, and configs):
- `src/on.py`, `src/on_off.py`: model-specific math/stats helpers; `src/common.py` for small utilities.
- `config/`: one paper YAML config for each plotting script.
- `scripts/`: local plotting CLIs and the HTCondor batch workflow.
- `notebooks/playground.ipynb`: lightweight demo for single p-value/medsig plots (interactive, not batch).
- `plots/`: consolidated all-statistics plots and the auxiliary PDFs produced by the same runs.
- `runs/`: ignored HTCondor campaign data, created only when batch jobs are submitted.

Paper plot generation (from the repository root):
- Known-background significance: `python3 scripts/make_simple_pval_plots.py`
- Known-background median significance: `python3 scripts/make_simple_medsig_plots.py`
- On/off significance: `python3 scripts/make_onoff_pval_plots.py`
- On/off median significance (serial): `python3 scripts/make_onoff_medsig_plots.py`

The plotting configs use `statistics: [r, rstar]` to select the asymptotic
curves. Median-significance configs additionally use
`mc_summaries: [median]`, with `mean` available when needed.

The serial on/off configuration is deliberately limited to a practical local
runtime. It is suitable for regenerating and checking the figures, but it does
not reproduce the higher-precision batch calculation used for the manuscript.

## HTCondor production run

The local and batch calculations share `config/paper_onoff_medsig.yaml`. The
serial script reads `local_mc`; the HTCondor scripts read `batch_mc`. Physics
parameters, plot selection and output names are therefore defined only once.

From a clean repository checkout on the cluster, optionally identify the
script that activates the Python environment and submit a named campaign:

```bash
export MEDSIG_SETUP_SCRIPT=/path/to/setup_medsig_env.sh
python3 scripts/submit_onoff_medsig_jobs.py --run paper-production
```

The submitter freezes the Git commit, configuration and worker source under
`runs/paper-production/`. The manifest records checksums for both the YAML and
the Python source. Each Condor job receives that snapshot and writes one JSON
result back to the campaign. If
`MEDSIG_SETUP_SCRIPT` is set, it must be an absolute path visible from every
execute node. Monitor the jobs with `condor_q`.

```text
runs/paper-production/
├── config.yaml
├── manifest.json
├── input/          # frozen worker and src/ snapshot
├── results/
│   ├── s2/
│   ├── s5/
│   └── s10/
└── logs/
```

After every job has finished, return to the same clean Git commit and Python
environment, validate the complete campaign and render the standard PDFs under
`plots/`:

```bash
python3 scripts/collect_onoff_medsig_results.py --run paper-production
```

The collector refuses to render incomplete or mixed campaigns. Before each
cluster is submitted, the submission helper automatically runs
`condor_submit -dry-run` with the complete set of required macros.

## Installation

From the repository root, install in editable mode so notebooks and scripts can import `src`:

```bash
python3 -m pip install -e .
```
