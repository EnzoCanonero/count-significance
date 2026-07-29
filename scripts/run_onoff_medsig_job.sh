#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MEDSIG_SETUP_SCRIPT:-}" ]]; then
    if [[ ! -f "$MEDSIG_SETUP_SCRIPT" ]]; then
        echo "MEDSIG_SETUP_SCRIPT does not exist: $MEDSIG_SETUP_SCRIPT" >&2
        exit 1
    fi
    job_directory=$PWD
    source "$MEDSIG_SETUP_SCRIPT"
    cd "$job_directory"
fi

exec python3 "$@"
