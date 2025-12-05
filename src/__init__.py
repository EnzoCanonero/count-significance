# Shared statistical utilities for counting experiments.

from .on import (
    poisson_tail_on,
    r_stat_on,
    q_stat_on,
    norm_survival,
    norm_survival as norm_survival_on,
    r_star_on,
    pvals_on,
    median_expected_significance_on,
    expected_significance_on,
)
from .on_off import (
    SQRT2,
    norm_survival as norm_survival_onoff,
    xlogy,
    b_profiled,
    loglik_diff,
    r_stat_onoff,
    q_stat_onoff,
    r_star_onoff,
    sample_null_toys,
    required_toys_for_Z_precision,
    pvals_onoff,
    asimov_Zs_onoff,
    median_expected_significance_onoff,
    expected_significance_onoff,
)

__all__ = [
    # simple counting
    "poisson_tail_on",
    "r_stat_on",
    "q_stat_on",
    "norm_survival",
    "norm_survival_on",
    "r_star_on",
    "pvals_on",
    "median_expected_significance_on",
    "expected_significance_on",
    # on/off
    "SQRT2",
    "norm_survival_onoff",
    "xlogy",
    "b_profiled",
    "loglik_diff",
    "r_stat_onoff",
    "q_stat_onoff",
    "r_star_onoff",
    "sample_null_toys",
    "required_toys_for_Z_precision",
    "pvals_onoff",
    "asimov_Zs_onoff",
    "median_expected_significance_onoff",
    "expected_significance_onoff",
]
