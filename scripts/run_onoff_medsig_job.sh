#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${COUNT_SIGNIFICANCE_SETUP_SCRIPT:-}" ]]; then
    if [[ ! -f "$COUNT_SIGNIFICANCE_SETUP_SCRIPT" ]]; then
        echo "COUNT_SIGNIFICANCE_SETUP_SCRIPT does not exist: $COUNT_SIGNIFICANCE_SETUP_SCRIPT" >&2
        exit 1
    fi
    job_directory=$PWD
    source "$COUNT_SIGNIFICANCE_SETUP_SCRIPT"
    cd "$job_directory"
fi

exec python3 "$@"
