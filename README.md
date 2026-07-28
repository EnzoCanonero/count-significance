This repo contains code to:
- Compute p-values and Asimov/MC median significances for simple counting (“on”) and on/off experiments.
- Compare asymptotic test statistics r, r* to MC.

Structure (flattened modules, scripts, and configs):
- `src/on.py`, `src/on_off.py`: model-specific math/stats helpers; `src/common.py` for small utilities.
- `config/`: one paper YAML config for each plotting script.
- `scripts/`: CLIs to generate plots (see usage below).
- `notebooks/playground.ipynb`: lightweight demo for single p-value/medsig plots (interactive, not batch).
- `parallelization/`: Condor-friendly on/off medsig sweep (uses `src.on_off`).
- `plots/`: consolidated all-statistics plots and the auxiliary PDFs produced by the same runs.

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

## Installation

From the repository root, install in editable mode so notebooks and scripts can import `src`:

```bash
python3 -m pip install -e .
```
