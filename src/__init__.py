# Shared statistical utilities for counting experiments.

from .on import (
    poisson_tail,
    r_stat,
    q_stat,
    norm_survival,
    r_star as r_star_on,
    compute_curves,
    median_count,
    median_expected_significance,
)
from .on_off import (
    SQRT2,
    normal_sf,
    normal_isf,
    xlogy,
    b_profiled,
    loglik_diff,
    r_signed,
    q_value,
    r_star,
    sample_null_toys,
    required_toys_for_Z_precision,
    pvals_onoff,
    asimov_counts_onoff,
    asimov_Zs_onoff,
    expected_Z_mc_onoff,
    expected_significance_onoff,
)

__all__ = [
    # simple counting
    "poisson_tail",
    "r_stat",
    "q_stat",
    "norm_survival",
    "r_star_on",
    "compute_curves",
    "median_count",
    "median_expected_significance",
    # on/off
    "SQRT2",
    "normal_sf",
    "normal_isf",
    "xlogy",
    "b_profiled",
    "loglik_diff",
    "r_signed",
    "q_value",
    "r_star",
    "sample_null_toys",
    "required_toys_for_Z_precision",
    "pvals_onoff",
    "asimov_counts_onoff",
    "asimov_Zs_onoff",
    "expected_Z_mc_onoff",
    "expected_significance_onoff",
]
