#!/bin/sh
# Make the directories that /app/reports, /app/mlruns, /app/models and /app/predictions link
# to in the mounted data volume, then hand over to pdeeg with the container's arguments
# (default: all). Any pdeeg command works, among them:
#   train-final                     the final models, into data/models
#   predict data/recording.bdf [--paired data/other.bdf]
#                                   a research demo, not a diagnostic tool; the recordings
#                                   must be inside the mounted data volume
set -eu
mkdir -p data/reports data/mlruns data/models data/predictions
exec pdeeg "$@"
