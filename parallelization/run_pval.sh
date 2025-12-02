#!/bin/bash

cd /nfs/scratch2/ecanoner/counting-exp/python_code
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

python3 pval.py $1 $2
