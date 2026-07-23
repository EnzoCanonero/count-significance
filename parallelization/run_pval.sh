#!/bin/bash

# Override these to customize where to run and which environment to source.
WORKDIR=${WORKDIR:-/nfs/scratch2/ecanoner/medsig_paper_code/parallelization}
SETUP_SCRIPT=${SETUP_SCRIPT:-/nfs/scratch2/ecanoner/setup_medsig_env.sh}

cd "$WORKDIR" || exit 1
source "$SETUP_SCRIPT"

# Usage: run_pval.sh CONFIG.yaml JOB_ID
python3 pval.py "$@"
