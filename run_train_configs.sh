#!/bin/bash
# run_train_configs.sh
# Queue segmentation training for many configs as a single SLURM job array.
# Each task runs `main.py fit -c <config>` on one config via run_train_array.sh.
#
# Inputs may be:
#   - explicit .yaml config files
#   - directories — every *.yaml found recursively is collected
#
# Usage:
#   bash run_train_configs.sh [options] CONFIG_OR_DIR [CONFIG_OR_DIR ...] \
#                             [-- extra main.py args]
#
# Options:
#   --config_glob GLOB     Glob used when an input is a directory.
#                          (default: *.yaml)
#   --concurrency N        Max simultaneously-running array tasks (0 = no limit).
#                          (default: 0)
#   --gpus N               Number of GPUs per job. Must match trainer.devices in
#                          the configs. Resources scale as: N GPUs, N*2 cpus/task,
#                          N*48 GB mem. (default: 2)
#   --partition LIST       Comma-separated SLURM partition(s) to run on.
#                          (default: the cluster default)
#   --output_root PATH     Root to check for "already done" training ckpts.
#                          Must match main.py --output_dir. (default: outputs)
#   --list_file PATH       Where to write the config list.
#                          (default: slurm/lists/train-<stamp>.txt)
#   --force                Don't skip configs that already have a trained ckpt.
#   --dry_run              Print everything without submitting.
#   -h, --help             Show this help.
#
# "Already done" means any .ckpt file exists under:
#   <output_root>/<config_subdir>/
# where config_subdir mirrors the config path (strip "configs/" prefix + .yaml).
# This matches main.py's `_resolve_run_info()` output layout.
#
# Any args after a literal "--" are forwarded verbatim to main.py inside each
# array task. SLURM logs go to slurm/.
#
# Examples:
#   # Preview, then train every backbone config for ADE20K semantic
#   bash run_train_configs.sh --dry_run --gpus 2 configs/ade20k/semantic/backbones
#   bash run_train_configs.sh --gpus 2 configs/ade20k/semantic/backbones
#
#   # A few specific configs, at most 4 jobs at a time
#   bash run_train_configs.sh --gpus 4 --concurrency 4 \
#       configs/coco/panoptic/pretrained_l/dinov2_large.yaml \
#       configs/coco/panoptic/pretrained_l/dinov3_large.yaml
#
#   # Only some files of a folder, on a specific partition
#   bash run_train_configs.sh --gpus 8 --partition gpu-large \
#       --config_glob '*_1280.yaml' configs/coco/panoptic/scaling
#
#   # Re-run everything ignoring previous ckpts, passing an extra main.py flag
#   bash run_train_configs.sh --force configs/ade20k/semantic/pretrained_l -- --data.path /my/data

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
CONFIG_GLOB="*.yaml"
CONCURRENCY=0
GPUS=2
PARTITION=""
OUTPUT_ROOT="outputs"
LIST_FILE=""
FORCE=false
DRY_RUN=false
INPUTS=()
PASSTHROUGH=()

usage() {
  sed -n '2,52p' "$0"
  exit "${1:-0}"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
  --config_glob)
    CONFIG_GLOB="$2"
    shift 2
    ;;
  --concurrency)
    CONCURRENCY="$2"
    shift 2
    ;;
  --gpus)
    GPUS="$2"
    shift 2
    ;;
  --partition)
    PARTITION="$2"
    shift 2
    ;;
  --output_root)
    OUTPUT_ROOT="$2"
    shift 2
    ;;
  --list_file)
    LIST_FILE="$2"
    shift 2
    ;;
  --force)
    FORCE=true
    shift 1
    ;;
  --dry_run)
    DRY_RUN=true
    shift 1
    ;;
  -h | --help) usage 0 ;;
  --)
    shift
    PASSTHROUGH=("$@")
    break
    ;;
  -*)
    echo "Unknown option: $1" >&2
    usage 1
    ;;
  *)
    INPUTS+=("$1")
    shift 1
    ;;
  esac
done

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "error: no config paths or directories given" >&2
  usage 1
fi
if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]]; then
  echo "error: --concurrency must be a non-negative integer" >&2
  exit 1
fi
if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --gpus must be a positive integer" >&2
  exit 1
fi

# ── GPU-count-dependent resource settings (scale linearly with N) ─────────────
CPUS=$((GPUS * 2))
MEM=$((GPUS * 48))
GPU_SBATCH_ARGS=(--gpus="$GPUS" --ntasks-per-node="$GPUS" --cpus-per-task="$CPUS" --mem="${MEM}G")

# ── Helper: derive the training output dir for a given config path ───────────
# configs/ade20k/semantic/pretrained/beit3_large_384.yaml
#   → ade20k/semantic/pretrained/beit3_large_384
#   → <output_root>/ade20k/semantic/pretrained/beit3_large_384
_config_to_out_dir() {
  local cfg="$1"
  local sub="${cfg#configs/}" # strip leading "configs/"
  sub="${sub%.yaml}"          # strip .yaml
  sub="${sub%.yml}"           # strip .yml
  echo "${OUTPUT_ROOT}/${sub}"
}

_train_done() {
  local out_dir
  out_dir="$(_config_to_out_dir "$1")"
  [[ -n "$(find "$out_dir" -maxdepth 3 -name "*.ckpt" 2>/dev/null | head -1)" ]]
}

# ── Collect configs ───────────────────────────────────────────────────────────
CONFIGS=()
for inp in "${INPUTS[@]}"; do
  if [[ -f "$inp" ]]; then
    CONFIGS+=("$inp")
  elif [[ -d "$inp" ]]; then
    while IFS= read -r -d '' f; do
      CONFIGS+=("$f")
    done < <(find "$inp" -type f -name "$CONFIG_GLOB" -print0 | sort -z)
  else
    echo "[skip] not a file or directory: $inp" >&2
  fi
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "error: no configs matched" >&2
  exit 1
fi

# ── Skip configs that already have a trained checkpoint ──────────────────────
if ! $FORCE; then
  PENDING=()
  SKIPPED=0
  for cfg in "${CONFIGS[@]}"; do
    if _train_done "$cfg"; then
      echo "[skip done] $(_config_to_out_dir "$cfg") (has *.ckpt)"
      SKIPPED=$((SKIPPED + 1))
    else
      PENDING+=("$cfg")
    fi
  done
  CONFIGS=("${PENDING[@]+"${PENDING[@]}"}")
  [[ "$SKIPPED" -gt 0 ]] &&
    echo "[info] skipped $SKIPPED config(s) with existing ckpt (--force to re-run)"
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "nothing to do — all configs already have a checkpoint"
  exit 0
fi

N=${#CONFIGS[@]}

# ── Write config list ─────────────────────────────────────────────────────────
if [[ -z "$LIST_FILE" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  LIST_FILE="slurm/lists/train-${stamp}.txt"
fi
mkdir -p "$(dirname "$LIST_FILE")" slurm
printf '%s\n' "${CONFIGS[@]}" >"$LIST_FILE"

# ── Build sbatch spec ─────────────────────────────────────────────────────────
ARRAY_SPEC="0-$((N - 1))"
[[ "$N" -eq 1 ]] && ARRAY_SPEC="0"
[[ "$CONCURRENCY" -gt 0 ]] && ARRAY_SPEC="${ARRAY_SPEC}%${CONCURRENCY}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$REPO_ROOT/run_train_array.sh"
[[ -f "$WRAPPER" ]] || {
  echo "error: $WRAPPER not found" >&2
  exit 1
}

cmd=(sbatch --array="$ARRAY_SPEC" "${GPU_SBATCH_ARGS[@]}")
[[ -n "$PARTITION" ]] && cmd+=(--partition="$PARTITION")
cmd+=("$WRAPPER" "$LIST_FILE")
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && cmd+=("${PASSTHROUGH[@]}")

# ── Print summary ─────────────────────────────────────────────────────────────
echo "configs:      $N"
echo "list file:    $LIST_FILE"
echo "array spec:   $ARRAY_SPEC"
echo "gpus/job:     $GPUS  (${GPU_SBATCH_ARGS[*]})"
echo "partition:    ${PARTITION:-<cluster default>}"
echo "wrapper:      $WRAPPER"
echo "passthrough:  ${PASSTHROUGH[*]:-<none>}"
echo

if $DRY_RUN; then
  echo "─── config list ───────────────────────────────────────"
  cat "$LIST_FILE"
  echo "───────────────────────────────────────────────────────"
  echo
  printf '[dry] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
