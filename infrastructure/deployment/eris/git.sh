#!/usr/bin/env bash
# Load the host's Git dependencies only in this subprocess, not in Torch workers.
set -Eeuo pipefail
source /apps/lmod/lmod/init/bash
module load git/2.45.1-GCCcore-13.3.0
exec /apps/software/git/2.45.1-GCCcore-13.3.0/bin/git "$@"
