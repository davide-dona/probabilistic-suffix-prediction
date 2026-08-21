#!/usr/bin/env bash
# Mirror a training VM's outputs/ back onto this machine.
#
# Every path under outputs/ is deterministic (dataset/model/tag, from RunIdentity), so there is
# nothing to decide about where a file goes: this is a plain one-way rsync, pulling whatever is
# new or changed and never deleting anything on this side. checkpoints/last and checkpoints/best
# are the only files a run overwrites in place, and that is by design (resume-from-last,
# promote-on-improvement), so letting rsync update them here too is correct rather than something
# to guard against.
set -euo pipefail

usage() {
  echo "usage: $0 [-n] <ssh-host> [<remote-repo-path>]" >&2
  echo "  -n   dry run: show what would be pulled without copying anything" >&2
  exit 2
}

dry_run=()
while getopts ':n' opt; do
  case "$opt" in
    n) dry_run=(--dry-run) ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

[[ $# -ge 1 ]] || usage
host="$1"
remote_path="${2:-probabilistic-suffix-prediction}"

rsync -avz --progress "${dry_run[@]}" \
  "$host:$remote_path/outputs/" outputs/
