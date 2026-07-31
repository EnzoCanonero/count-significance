"""Statistical tools for known-background and on/off counting experiments."""

from .common import discovery_pvalue, discovery_q0, discovery_z, norm_survival
from .on import (
    expected_significance_on,
    poisson_tail_on,
    pvals_on,
    r_star_on,
    r_stat_on,
    u_stat_on,
)
from .on_off import (
    asimov_Zs_onoff,
    b_profiled,
    expected_significance_onoff,
    loglik_diff,
    pvals_onoff,
    pvals_onoff_profile_sum,
    r_star_onoff,
    r_stat_onoff,
    required_toys_for_Z_precision,
    u_stat_onoff,
)

__all__ = [
    # Shared discovery convention
    "norm_survival",
    "discovery_z",
    "discovery_q0",
    "discovery_pvalue",
    # Known-background model
    "poisson_tail_on",
    "r_stat_on",
    "u_stat_on",
    "r_star_on",
    "pvals_on",
    "expected_significance_on",
    # On/off model
    "b_profiled",
    "loglik_diff",
    "r_stat_onoff",
    "u_stat_onoff",
    "r_star_onoff",
    "required_toys_for_Z_precision",
    "pvals_onoff_profile_sum",
    "pvals_onoff",
    "asimov_Zs_onoff",
    "expected_significance_onoff",
]
