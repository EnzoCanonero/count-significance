# Count Significance

`count-significance` provides numerical tools for discovery significances and
p-values in Poisson counting experiments.

It supports:

- a single Poisson count with known background;
- an on/off experiment with a Poisson control measurement;
- exact or profiled reference calculations;
- first-order profile-likelihood approximations;
- higher-order Barndorff-Nielsen corrections;
- Asimov and Monte Carlo estimates of median discovery significance.

## Installation

The package requires Python 3.9 or newer.

```bash
python -m pip install count-significance
```

## Example

```python
from count_significance import (
    asimov_Zs_onoff,
    expected_significance_on,
    pvals_on,
    pvals_onoff_profile_sum,
)

known = pvals_on(s0=0.0, b=1.0, n=4)
onoff = pvals_onoff_profile_sum(s=0.0, n=5, m=1, tau=1.0)

known_expected = expected_significance_on(s_true=5.0, b=10.0)
onoff_expected = asimov_Zs_onoff(s_true=5.0, b=10.0, tau=2.0)
```

The public namespace is `count_significance`. The package reports one-sided
discovery significances and distinguishes numerical reference calculations
from asymptotic approximations.

## Documentation and reproducibility

The [GitHub repository](https://github.com/EnzoCanonero/count-significance)
contains the full mathematical overview, API notes, tests, plotting scripts,
paper configurations, notebook, generated figures, and HTCondor production
workflow.

This software accompanies the manuscript *Discovery Sensitivity for a Counting
Experiment with Background Uncertainty* by Enzo Canonero and Glen Cowan.

## Citation

Citation metadata are provided in
[`CITATION.cff`](https://github.com/EnzoCanonero/count-significance/blob/main/CITATION.cff).

## License

Count Significance is distributed under the
[MIT License](https://github.com/EnzoCanonero/count-significance/blob/main/LICENSE).
