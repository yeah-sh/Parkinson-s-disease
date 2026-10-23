#!/bin/sh
# Make the directories that /app/reports and /app/mlruns link to in the mounted data volume,
# then hand over to pdeeg with the container's arguments (default: all).
set -eu
mkdir -p data/reports data/mlruns
exec pdeeg "$@"
