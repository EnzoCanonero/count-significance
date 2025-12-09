Parallel on/off medsig sweep

This folder contains HTCondor-friendly scripts to run many toy studies for the on/off model and aggregate them into plots.

Contents
- config.yaml: scan ranges, toy statistics, RNG seeds, and output locations.
- pval.py: worker script; loops over all scan points and uses the Condor job id to shift RNG seeds.
- run_pval.sh: wrapper sourced by Condor; override WORKDIR/SETUP_SCRIPT to point to the repo and your environment setup.
- pval_condor.sub: Condor submit file (queues 1000 jobs by default) and stages pval.py/run_pval.sh/config.yaml.
- collect_results.py: combines per-job JSONs into medians and plots; respects outdir/out_summary_pdf/individual_plots in the config.
- logs/, output/: stdout/err/logs from Condor and collected JSON/plot outputs.

Typical usage
1) Edit config.yaml to set the scan ranges and precision settings.
2) Ensure the repo path and environment on the worker nodes match WORKDIR and SETUP_SCRIPT, then submit:
   WORKDIR=/path/to/repo SETUP_SCRIPT=/path/to/env.sh condor_submit parallelization/pval_condor.sub
3) Each job writes output/pval_job_XXXXXX.json under the chosen outdir.
4) After jobs finish, aggregate and plot:
   python parallelization/collect_results.py parallelization/config.yaml

Local smoke test (single job)
python parallelization/pval.py parallelization/config.yaml 0
