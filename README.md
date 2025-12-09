This repo contains code to:
- Compute p-values and Asimov/MC median significances for simple counting (“on”) and on/off experiments.
- Compare asymptotic test statistics r, r* to MC.

Structure (flattened modules, scripts, and configs):
- `src/on.py`, `src/on_off.py`: model-specific math/stats helpers; `src/common.py` for small utilities.
- `config/`: YAML configs for each plotting script (`simple_pval.yaml`, `simple_medsig.yaml`, `onoff_pval.yaml`, `onoff_medsig.yaml`).
- `scripts/`: CLIs to generate plots (see usage below).
- `notebooks/on_playgroung.ipynb`, `notebooks/onoff_playground.ipynb`: two lightweight tutorial notebooks.
- `parallelization/`: Condor-friendly on/off medsig sweep (uses `src.on_off`).
- `plots/`: output PDFs.

Script usage (from repo root):
- Simple counting p-values: `python scripts/make_simple_pval_plots.py --config config/simple_pval.yaml`
- Simple counting medsig:   `python scripts/make_simple_medsig_plots.py --config config/simple_medsig.yaml`
- On/off p-values:          `python scripts/make_onoff_pval_plots.py --config config/onoff_pval.yaml`
- On/off medsig (serial):   `python scripts/make_onoff_medsig_plots.py --config config/onoff_medsig.yaml`

Parallel on/off medsig:
- Submit jobs: `condor_submit parallelization/pval_condor.sub`
- Aggregate:   `python parallelization/collect_results.py parallelization/config.yaml`

## Installation

From the repo root, install in editable mode so notebooks/scripts can import `src`:

```bash
python -m pip install -e .
```
