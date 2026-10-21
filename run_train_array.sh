#!/bin/bash
# run_train_array.sh
# SLURM array task: read line ${SLURM_ARRAY_TASK_ID+1} from CONFIG_LIST_FILE
# and run `main.py fit -c <config>` on it.
# Driven by run_train_configs.sh — not meant to be submitted by hand.
#
# Required positional:  $1 = path to newline-separated config list
# Remaining positionals are forwarded verbatim to main.py.
#
# GPUs, CPUs and memory are set by run_train_configs.sh; add the partition,
# account or container options your cluster needs below or via --partition.

#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --job-name="luma_train"
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG_LIST_FILE [main.py args...]" >&2
  exit 2
fi

CONFIG_LIST_FILE="$1"
shift
if [[ ! -f "$CONFIG_LIST_FILE" ]]; then
  echo "error: config list file not found: $CONFIG_LIST_FILE" >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "error: SLURM_ARRAY_TASK_ID not set (submit with --array)" >&2
  exit 2
fi

line_no=$((SLURM_ARRAY_TASK_ID + 1))
config="$(sed -n "${line_no}p" "$CONFIG_LIST_FILE")"
if [[ -z "$config" ]]; then
  echo "error: empty line ${line_no} in $CONFIG_LIST_FILE" >&2
  exit 2
fi

echo "[array task ${SLURM_ARRAY_TASK_ID}] config: $config"
echo "[array task ${SLURM_ARRAY_TASK_ID}] extra args: $*"

srun uv run --no-sync python main.py fit -c "$config" "$@"
