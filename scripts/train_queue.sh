#!/usr/bin/env bash
# Train every dataset config sitting in queue/train/, one run per GPU at a time.
#
# The queue itself, and what happens to a config that succeeds or fails, is scripts/lib/queue.sh.
set -uo pipefail

source "$(dirname "$0")/lib/queue.sh"

readonly QUEUE_DIR='queue/train'
readonly LOGS='outputs/queue/train'
readonly SUFFIX='.yaml'
gpus=(0 1)
hardware=''
model=''

usage() {
  echo "usage: $0 -m config/models/<architecture>.yaml -w config/hardware/<profile>.yaml [-g 0,1]" >&2
  exit 2
}

# One architecture per sweep: the queue holds dataset configs, so the model is named once here
# rather than copied into every one of them.
while getopts ':m:w:g:' opt; do
  case "$opt" in
    m) model="$OPTARG" ;;
    w) hardware="$OPTARG" ;;
    g) IFS=',' read -r -a gpus <<<"$OPTARG" ;;
    *) usage ;;
  esac
done
[[ -f "$model" && -f "$hardware" ]] || usage

mkdir -p "$QUEUE_DIR"

# A training job is the dataset config it is run from, so the config's own name is its name.
job_name() {
  basename "${1%"$SUFFIX".running}"
}

run_job() {
  uv run python -m pipelines.train -m "$model" -c "$1" -w "$hardware"
}

queue_run "${gpus[@]}"
