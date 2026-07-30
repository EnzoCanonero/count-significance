# Count Significance

Accurate significance and p-values for count data.

Count Significance contains the statistical calculations and reproducible
plotting workflows for the manuscript *Discovery Sensitivity for a
Counting Experiment with Background Uncertainty*. It compares first-order
profile-likelihood results with the Barndorff–Nielsen $r^\ast$ correction, exact or
profiled reference calculations, and Monte Carlo estimates of the median
discovery significance.

The repository treats two models:

- a single Poisson count with known background;
- an on/off experiment in which a Poisson control count constrains the
  background.

The code provides the numerical routines, the YAML configurations used for the
paper figures, practical serial workflows, and a reproducible HTCondor workflow
for the high-precision on/off calculation.

## Contents

- [Statistical overview](#statistical-overview)
- [Installation](#installation)
- [Python interface](#python-interface)
- [Reproducing the plots](#reproducing-the-plots)
- [HTCondor production workflow](#htcondor-production-workflow)
- [Monte Carlo precision](#monte-carlo-precision)
- [Repository layout](#repository-layout)
- [Companion manuscript](#companion-manuscript)
- [Citation](#citation)
- [License](#license)

## Statistical overview

For a tested signal value $s_0$, define the profile-likelihood ratio

$$
\lambda(s_0)=
\frac{L\left(s_0,\hat{\hat\theta}(s_0)\right)}
     {L\left(\hat s,\hat\theta\right)}.
$$

The signed likelihood root $r(s_0)$ and its higher-order correction
$r^\ast(s_0)$ are constructed from this ratio. Their expressions for a
general tested signal value are given in the manuscript; here we
focus on the discovery test, for which $s_0=0$.

The discovery test uses the one-sided convention

$$
q_0=\left[\max(0,r(0))\right]^2,
\qquad
\sqrt{q_0}=\max(0,r(0)),
\qquad
p_{\mathrm{asym}}=1-\Phi\left(\sqrt{q_0}\right).
$$

The corrected discovery statistic implemented here is

$$
q_0^\ast=\left[\max(0,r^\ast(0))\right]^2,
\qquad
\sqrt{q_0^\ast}=\max(0,r^\ast(0)).
$$

The maximum implements the one-sided discovery convention. The non-negative
square root of whichever statistic is under discussion is reported as $Z$,
with the corresponding Gaussian-tail approximation $1-\Phi(Z)$; $Z_{\mathrm A}$
is reserved for an Asimov evaluation.

### Poisson case with known background

The observed count follows

$$
N\sim\mathrm{Pois}(s+b),
\qquad
L(s)=\frac{(s+b)^n}{n!}e^{-(s+b)},
$$

with $b$ known. The general expressions for $r(s_0)$, its auxiliary quantity
$u(s_0)$, and $r^\ast(s_0)$ are given in the manuscript. For the
discovery test, $s_0=0$ and

$$
r(0)=\mathrm{sgn}(n-b)
\sqrt{2\left[n\ln\frac{n}{b}+b-n\right]},
\qquad
u(0)=\sqrt n\ln\frac{n}{b},
$$

$$
r^\ast(0)=r(0)+\frac{1}{r(0)}
\ln\left|\frac{u(0)}{r(0)}\right|.
$$

Evaluating $q_0$ on the Asimov count $n_{\mathrm A}=s+b$ gives

$$
Z_{\mathrm A}=
\sqrt{2\left[(s+b)\ln\left(1+\frac{s}{b}\right)-s\right]}.
$$

The numerical reference is the inclusive Poisson
tail. The paper configuration applies a half-count continuity correction to
$r^\ast$ so that its continuous approximation targets the same inclusive tail.
The higher-order Asimov significance $Z_{\mathrm A}^\ast$ is obtained by
evaluating the corrected discovery statistic on the same Asimov count, with
this continuity correction, and taking its non-negative square root.

### Poisson case with uncertain background

The primary and control counts follow

$$
N\sim\mathrm{Pois}(s+b),
\qquad
M\sim\mathrm{Pois}(\tau b),
$$

with likelihood

$$
L(s,b)=
\frac{(s+b)^n}{n!}e^{-(s+b)}
\frac{(\tau b)^m}{m!}e^{-\tau b}.
$$

Here $\tau$ is known and controls the precision of the background measurement.
For a general tested signal value, the nuisance background is profiled before
constructing $r(s_0)$, $u(s_0)$, and $r^\ast(s_0)$; the full expressions are
given in the manuscript. For discovery, $s_0=0$ and
$\tilde b(0)=(n+m)/(1+\tau)$, giving

$$
r(0)=\mathrm{sgn}\left(n-\frac{m}{\tau}\right)
\sqrt{2\left[
n\ln\frac{(1+\tau)n}{n+m}
+m\ln\frac{(1+\tau)m}{\tau(n+m)}
\right]},
$$

$$
u(0)=\sqrt{\frac{nm}{n+m}}\ln\frac{n\tau}{m},
\qquad
r^\ast(0)=r(0)+\frac{1}{r(0)}
\ln\left|\frac{u(0)}{r(0)}\right|.
$$

Evaluating $q_0$ on the Asimov data $n_{\mathrm A}=s+b$ and
$m_{\mathrm A}=\tau b$ gives

$$
Z_{\mathrm A}=\sqrt{-2\left[
(s+b)\ln\frac{s+(1+\tau)b}{(1+\tau)(s+b)}
+\tau b\ln\left(1+\frac{s}{(1+\tau)b}\right)
\right]}.
$$

No half-count continuity correction is used for the
two-dimensional observation $(n,m)$.
The higher-order Asimov significance $Z_{\mathrm A}^\ast$ is found by
evaluating the corrected discovery statistic directly on the same Asimov pair
and taking its non-negative square root.

## Installation

The package supports Python 3.9 and newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

The runtime dependencies are NumPy, SciPy, Matplotlib, and PyYAML and are
installed from `pyproject.toml`. Jupyter is optional and is not installed by
the package; install it separately to use
[`notebooks/playground.ipynb`](notebooks/playground.ipynb).

`pyproject.toml` specifies compatible minimum versions rather than a lock file.
For an archival production run, record the resolved environment separately if
exact dependency-level reproducibility is required.

## Python interface

The current import namespace follows the source layout:

```python
from src.on import expected_significance_on, pvals_on
from src.on_off import asimov_Zs_onoff, pvals_onoff_profile_sum

# Known background: exact, first-order and higher-order observed p-values.
known = pvals_on(s0=0.0, b=1.0, n=4)

# On/off model: deterministic profiled reference and asymptotic p-values.
onoff = pvals_onoff_profile_sum(s=0.0, n=5, m=1, tau=1.0)

# First-order and higher-order Asimov significances.
known_expected = expected_significance_on(s_true=5.0, b=10.0)
onoff_expected = asimov_Zs_onoff(s_true=5.0, b=10.0, tau=2.0)
```

The principal modules are:

- `src.common`: the shared one-sided discovery convention and Gaussian tail;
- `src.on`: exact and asymptotic known-background calculations;
- `src.on_off`: profiling, on/off reference calculations, nested Monte Carlo,
  and expected significance.

For the on/off model, `pvals_onoff_profile_sum` is a deterministic plug-in
reference: it profiles the background under the tested signal and sums the
joint-Poisson discovery tail on a truncated grid. It is not an exact
elimination of the nuisance parameter. The Monte Carlo alternative
`pvals_onoff` returns the discovery-capped `p_mc`, the raw fraction $K/N$, its
standard error, the asymptotic $r$ and $r^\ast$ p-values, the generated toy
count, the number of tail exceedances, the profiled background, the effective
resolution, and a precision-limit flag. The uncapped corrected estimate
$(K+1)/(N+1)$ can be reconstructed from the returned $K$ and $N$.

## Reproducing the plots

Run plotting scripts from the repository root. Each command accepts an
optional `--config PATH`; omitting it selects the paper configuration shown
below.

| Calculation | Configuration | Command | Main output |
|---|---|---|---|
| Known background, observed | `config/paper_simple_significance.yaml` | `python3 scripts/make_simple_pval_plots.py` | `plots/simple_significance_s0_b1.pdf` |
| Known background, expected | `config/paper_simple_medsig.yaml` | `python3 scripts/make_simple_medsig_plots.py` | `plots/simple_asimov.pdf` |
| On/off, observed | `config/paper_onoff_significance.yaml` | `python3 scripts/make_onoff_pval_plots.py` | `plots/onoff_significance_s0_b1_tau1.pdf` |
| On/off, expected | `config/paper_onoff_medsig.yaml` | `python3 scripts/make_onoff_medsig_plots.py` | `plots/uncertain_background_asimov_grid.pdf` |

The complete local sequence is therefore

```bash
python3 scripts/make_simple_pval_plots.py
python3 scripts/make_simple_medsig_plots.py
python3 scripts/make_onoff_pval_plots.py
python3 scripts/make_onoff_medsig_plots.py
```

The observed-significance scripts also write the auxiliary PDFs
`plots/simple_pval_s0_b1.pdf` and `plots/onoff_pval_s0_b1_tau1.pdf`. The on/off
median-significance workflow additionally writes
`plots/onoff_asimov_tau_s{s}.pdf` and
`plots/onoff_asimov_sigrel_s{s}.pdf` for $s=2,5,10$.

Plot contents are selected with

```yaml
statistics: [r, rstar]
mc_summaries: [median]
```

`statistics` may contain `r`, `rstar`, or both; `mc_summaries` may contain
`median`, `mean`, or both. In the on/off configuration, `Z_display_max` sets
the displayed y-axis maximum and hides Monte Carlo markers at or above that
value. It does not alter the calculated values, although continuous curves
outside the displayed range are naturally outside the frame.

The serial on/off script reads `local_mc`, whose settings are deliberately
small enough for a workstation run. They do **not** reproduce the production
precision of the manuscript. Both the local script and the batch collector
write the same seven configured on/off PDF paths, so a local run can overwrite
production plots.

## HTCondor production workflow

The production on/off calculation reads `batch_mc` from
`config/paper_onoff_medsig.yaml`. It distributes the outer Monte Carlo replicas
while preserving the same physical grids, statistical routines, and plot
writer as the serial workflow.

The cluster requires `condor_submit`, Bash, and a Python environment containing
the dependencies above. If worker nodes need an activation script, provide an
absolute path that is visible on those nodes:

```bash
export COUNT_SIGNIFICANCE_SETUP_SCRIPT=/absolute/path/to/setup_count_significance_env.sh
```

From a clean Git checkout, submit a uniquely named campaign:

```bash
python3 scripts/submit_onoff_medsig_jobs.py --run paper-production
```

Use `--config PATH` to select another validated campaign configuration. Run
names must start with a letter or number and may contain letters, numbers,
dots, underscores, and hyphens.

The submitter validates the shared configuration and job partition, records the
current commit, freezes the source and configuration, runs
`condor_submit -dry-run` for every signal cluster, and only then submits the
jobs. It requests one CPU and 4 GB of memory per job. A run name is never
overwritten; a failed dry run is preserved for inspection and a later attempt
must use a new name or deliberately remove that failed campaign.

Campaign data are ignored by Git and have the following layout:

```text
runs/<run-name>/
├── config.yaml
├── manifest.json
├── input/
│   ├── src/
│   └── scripts/
├── results/
│   ├── s2/
│   ├── s5/
│   └── s10/
└── logs/
```

Monitor the campaign with the usual HTCondor tools, for example `condor_q`.
After every job has completed, collect the results from the same clean Git
commit:

```bash
python3 scripts/collect_onoff_medsig_results.py --run paper-production
```

The collector fails without writing PDFs unless the manifest, commit,
configuration and frozen-source hashes, job IDs, grids, seeds, observed counts,
and Monte Carlo diagnostics are complete and mutually consistent. It reports
the toy-count range, zero-exceedance fraction, and precision-limited fraction
for every signal before writing the seven standard on/off PDFs to `plots/`.

Both submission and collection require a clean working tree, including no
non-ignored untracked files. Ignored campaign data under `runs/` may remain in
place. Because the generated PDFs are tracked, local plot generation normally
makes the tree dirty and should not be mixed with a production campaign.

## Monte Carlo precision

For the observed first-order significance $Z=\max(0,r_{\mathrm{obs}})$, the on/off
routine estimates the number of null toys required for a target relative
uncertainty $\epsilon_Z$ as

$$
N=\left\lceil
\frac{p(1-p)}{\epsilon_Z^2 Z^2\phi(Z)^2}
\right\rceil,
\qquad p=1-\Phi(Z),
$$

subject to the configured minimum and maximum. At $Z=0$, relative precision is
undefined; the routine generates `min_toys` and sets `precision_limited=False`.
Otherwise, if the maximum prevents the requested precision from being reached,
`precision_limited` is set. With $K$ toys at least as discovery-like as the
observation, the finite-sample corrected estimate is

$$
\hat p=\frac{K+1}{N+1}.
$$

The returned `p_mc` is $0.5$ when $r_{\mathrm{obs}}\le0$ and otherwise the smaller
of this estimate and $0.5$. The uncapped `p_mc_raw = K/N` is retained as a
diagnostic. The effective resolution is $1/(N+1)$ and therefore depends on the
number of toys actually generated, not on the configured maximum.

## Repository layout

```text
count-significance/
├── src/          statistical definitions and numerical routines
├── scripts/      plotting, submission, worker and collection commands
├── config/       paper YAML files and the generic HTCondor description
├── notebooks/    interactive introduction to the two models
├── plots/        generated paper and auxiliary PDFs
└── runs/         ignored HTCondor campaign data, created at submission
```

Scheduler-specific logic is kept out of `src/`; both serial and batch workflows
call the same mathematical implementation in `src/on_off.py`.

## Reproducibility conventions

- Random-number seeds are explicit in the paper configurations.
- The known-background reference uses the inclusive Poisson tail.
- The on/off `profile_sum` result is labelled as a profile reference, not as an
  exact nuisance-parameter treatment.
- Production workers receive a frozen source/configuration sandbox.
- Batch results record their schema, commit, configuration and source hashes,
  physical grid coordinates, seeds, observed counts, and Monte Carlo
  diagnostics.
- The collector validates a complete campaign before producing any final PDF.

## Companion manuscript

This repository accompanies:

> Enzo Canonero and Glen Cowan, *Discovery Sensitivity for a Counting
> Experiment with Background Uncertainty*.

- [Enzo Canonero](https://orcid.org/0000-0002-7180-4562) —
  [Enzo.Canonero@rhul.ac.uk](mailto:Enzo.Canonero@rhul.ac.uk)
- [Glen Cowan](https://orcid.org/0000-0001-8363-9827) —
  [G.Cowan@rhul.ac.uk](mailto:G.Cowan@rhul.ac.uk)

Publication metadata for the manuscript will be added when available.

## Citation

If you use Count Significance, please cite the software using the metadata in
[`CITATION.cff`](CITATION.cff) and cite the accompanying manuscript.

## License

Count Significance is distributed under the [MIT License](LICENSE).
