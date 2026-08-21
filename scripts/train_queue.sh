#!/usr/bin/env bash
# Train every dataset config sitting in config/queue/, one run per GPU at a time.
#
# A config is claimed by renaming it `.yaml.running`, deleted once its run exits 0, and renamed
# `.yaml.failed` when it does not. Only a plain `.yaml` is ever claimed, so a failure stays in the
# folder to be looked at rather than being picked straight back up by the next free GPU.
set -uo pipefail

readonly QUEUE='config/queue'
readonly LOGS='outputs/queue'
gpus=(0 1)
hardware=''

usage() {
  echo "usage: $0 -w config/hardware/<profile>.yaml [-g 0,1]" >&2
  exit 2
}

while getopts ':w:g:' opt; do
  case "$opt" in
    w) hardware="$OPTARG" ;;
    g) IFS=',' read -r -a gpus <<<"$OPTARG" ;;
    *) usage ;;
  esac
done
[[ -f "$hardware" ]] || usage

shopt -s nullglob
queued=("$QUEUE"/*.yaml)
(( ${#queued[@]} )) || { echo "nothing queued in $QUEUE/" >&2; exit 1; }

mkdir -p "$LOGS"
results="$(mktemp)"

# Interrupting stops the queue rather than the script. Ctrl-C reaches the runs in flight through
# the process group and ends them; this is what stops a freed GPU from starting the next config,
# so the summary is still printed and what never ran is still queued.
stop="$results.stop"
trap 'touch "$stop"' INT TERM

# Take the next config off the queue. `mv` is a rename, so exactly one worker wins a given file.
claim() {
  local candidate
  [[ -e "$stop" ]] && return 1
  for candidate in "$QUEUE"/*.yaml; do
    if mv "$candidate" "$candidate.running" 2>/dev/null; then
      printf '%s' "$candidate.running"
      return 0
    fi
  done
  return 1
}

# Run every config this GPU manages to claim, until the queue is empty. `CUDA_VISIBLE_DEVICES`
# masks in one GPU, so the profile's `cuda:0` is this worker's GPU.
worker() {
  local gpu="$1" claimed name log started elapsed status
  while claimed="$(claim)"; do
    name="$(basename "${claimed%.yaml.running}")"
    log="$LOGS/$name-$(date +%Y%m%d-%H%M%S).log"
    printf '[gpu %s] start  %-16s -> %s\n' "$gpu" "$name" "$log"
    started="$SECONDS"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python -m pipelines.train \
      -c "$claimed" -w "$hardware" >"$log" 2>&1
    status=$?
    elapsed=$(( SECONDS - started ))
    if (( status == 0 )); then
      rm -f "$claimed"
      printf '[gpu %s] ok     %-16s (%dh %02dm)\n' \
        "$gpu" "$name" $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 ))
      echo "ok $name" >>"$results"
    else
      mv "$claimed" "${claimed%.running}.failed"
      printf '[gpu %s] FAILED %-16s (%dh %02dm, exit %d) see %s\n' \
        "$gpu" "$name" $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) "$status" "$log"
      echo "failed $name $log" >>"$results"
    fi
  done
  return 0
}

for gpu in "${gpus[@]}"; do
  worker "$gpu" &
done
# A caught signal interrupts `wait`, so keep waiting until the workers are actually done.
until wait; do :; done

ok_count=$(grep -c '^ok ' "$results" || true)
failed_count=$(grep -c '^failed ' "$results" || true)
printf '\nSummary: %d ok, %d failed\n' "$ok_count" "$failed_count"
grep '^failed ' "$results" | awk '{ printf "  %-16s %s\n", $2, $3 }'
(( failed_count )) && echo "Re-queue a failure by dropping the .failed suffix in $QUEUE/."
rm -f "$results" "$stop"
(( failed_count == 0 ))
