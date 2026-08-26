#!/usr/bin/env bash
# Evaluate every generations file under outputs/generations/ that has no report yet, one at a
# time. Evaluation is CPU-bound and already parallelizes internally via pipelines.evaluate's own
# -j/--workers, so this is a plain sequential loop rather than scripts/lib/queue.sh's per-GPU
# worker queue.
set -uo pipefail

readonly GENERATIONS_DIR='outputs/generations'
readonly EVAL_DIR='outputs/eval'
readonly LOGS='outputs/queue/evaluate'
force=0
workers=()

usage() {
  echo "usage: $0 [-f] [-j <workers>]" >&2
  exit 2
}

while getopts ':fj:' opt; do
  case "$opt" in
    f) force=1 ;;
    j) workers=(-j "$OPTARG") ;;
    *) usage ;;
  esac
done

mkdir -p "$LOGS"

shopt -s nullglob globstar
generations=("$GENERATIONS_DIR"/**/*.parquet)
(( ${#generations[@]} )) || { echo "nothing under $GENERATIONS_DIR/" >&2; exit 1; }

ok=0
failed=()

for file in "${generations[@]}"; do
  rel="${file#"$GENERATIONS_DIR"/}"
  name="${rel%.parquet}"
  report="$EVAL_DIR/$name.json"

  if (( ! force )) && [[ -e "$report" ]]; then
    echo "skip   $name (already evaluated, see $report)"
    continue
  fi

  log="$LOGS/${name//\//_}-$(date +%Y%m%d-%H%M%S).log"
  echo "start  $name -> $log"
  if uv run python -m pipelines.evaluate -g "$file" "${workers[@]+"${workers[@]}"}" 2>&1 | tee "$log"; then
    echo "ok     $name"
    ok=$((ok + 1))
  else
    echo "FAILED $name (see $log)"
    failed+=("$name")
  fi
done

printf '\nSummary: %d ok, %d failed\n' "$ok" "${#failed[@]}"
for name in "${failed[@]+"${failed[@]}"}"; do
  echo "  $name"
done
(( ${#failed[@]} == 0 ))
