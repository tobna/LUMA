#!/bin/bash
# run_infer_array.sh
# SLURM array task: read line ${SLURM_ARRAY_TASK_ID+1} of CKPT_LIST_FILE and
# run inference.py on it. Driven by run_infer_configs.sh — not meant to be
# submitted by hand.
#
# Required positional:  $1 = path to a newline-separated checkpoint list
# Remaining positionals are forwarded verbatim to inference.py.
#
# Add the partition, account or container options your cluster needs below.
# Throughput numbers are only comparable when all tasks run on the same GPU type.

#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --job-name="luma_infer"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-1

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CKPT_LIST_FILE [inference.py args...]" >&2
  exit 2
fi

CKPT_LIST_FILE="$1"
shift
if [[ ! -f "$CKPT_LIST_FILE" ]]; then
  echo "error: ckpt list file not found: $CKPT_LIST_FILE" >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "error: SLURM_ARRAY_TASK_ID not set (this script must be sbatch'd with --array)" >&2
  exit 2
fi

# Array indices are 0-based; sed lines are 1-based.
line_no=$((SLURM_ARRAY_TASK_ID + 1))
ckpt="$(sed -n "${line_no}p" "$CKPT_LIST_FILE")"
if [[ -z "$ckpt" ]]; then
  echo "error: empty line ${line_no} in $CKPT_LIST_FILE" >&2
  exit 2
fi

echo "[array task ${SLURM_ARRAY_TASK_ID}] ckpt: $ckpt"
echo "[array task ${SLURM_ARRAY_TASK_ID}] extra args: $*"

srun uv run --no-sync python inference.py "$ckpt" "$@"
