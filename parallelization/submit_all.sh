#!/bin/bash
# Submit the three per-signal medsig scans (Fig 2 / 6 / 7).
set -e
cd /nfs/scratch2/ecanoner/medsig_paper_code/parallelization
mkdir -p logs output_s2 output_s5 output_s10 plots_parallel

condor_submit pval_s2.sub
condor_submit pval_s5.sub
condor_submit pval_s10.sub

echo "Submitted. Watch with: condor_q"
echo "When all jobs are done, render the figures with:"
echo "  source /nfs/scratch2/ecanoner/setup_medsig_env.sh"
echo "  python3 collect_results.py config_merge.yaml"
