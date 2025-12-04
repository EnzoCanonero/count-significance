#!/bin/bash

# Override these to customize where to run and which environment to source.
WORKDIR=${WORKDIR:-/nfs/scratch2/ecanoner/counting-exp/python_code}
SETUP_SCRIPT=${SETUP_SCRIPT:-/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh}

cd "$WORKDIR" || exit 1
source "$SETUP_SCRIPT"

python3 pval.py "$@"
