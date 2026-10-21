#!/bin/bash
# run_infer_configs.sh
# Queue inference for many checkpoints as a single SLURM job array.
#
# `inference.py` auto-infers --config from the checkpoint path
# (configs/<X>.yaml ↔ <output_root>/<X>/.../<name>.ckpt mirror), so this
# script just needs a list of ckpts. It collects them, writes them to a
# list file, and submits one `sbatch --array=0-(N-1)[%C] run_infer_array.sh`.
#
# Inputs may be:
#   - explicit .ckpt files
#   - directories — every ckpt matching --ckpt_glob (recursively) is collected
#
# Usage:
#   bash run_infer_configs.sh [options] CKPT_OR_DIR [CKPT_OR_DIR ...] \
#                              [-- extra inference.py args]
#
# Options:
#   --ckpt_glob GLOB   Glob used when an input is a directory (default: *.ckpt)
#   --pick MODE        When a leaf folder contains multiple ckpts: all | newest
#                      (default: newest). 'newest' is usually what you want
#                      because ModelCheckpoint may keep last.ckpt next to the
#                      best-by-val_iou ckpt.
#   --concurrency N    Max simultaneously-running array tasks (sbatch %N).
#                      0 = no limit (default: 0).
#   --list_file PATH   Where to write the ckpt list. Default:
#                      slurm/lists/infer-<UTC-stamp>.txt
#   --metrics_name F   Filename to test for "already done" (default: metrics.json).
#                      Must match inference.py's --output filename.
#   --no-flash         Forward --no-flash to inference.py (disables the fused
#                      SDPA path) and look for metrics-no-flash-attn.json
#                      instead of metrics.json.
#   --force            Don't skip ckpts whose folder already contains
#                      <metrics_name>.
#   --dry_run          Print everything (incl. list file contents) without
#                      submitting.
#   -h, --help         Show this help.
#
# By default, any ckpt whose containing folder already has <metrics_name> is
# skipped — re-running an inference is opt-in via --force.
#
# Any args after a literal `--` are forwarded verbatim to inference.py inside
# each array task (e.g. --batch_size 1 --bf16 --output /path/m.json). NOTE
# that --output is *shared* across array tasks, so if you set it explicitly
# all tasks will overwrite the same file — usually you want to omit --output
# and let inference.py default to <checkpoint_dir>/metrics.json per ckpt.
# SLURM logs go to slurm/.
#
# Examples:
#   # One ckpt (still uses the array machinery; array size = 1)
#   bash run_infer_configs.sh \
#       outputs/ade20k/semantic/backbones/cvt_384/<timestamp>/ade20k_semantic_bb_384_cvt.ckpt
#
#   # Preview, then sweep every newest ckpt under a tree, at most 8 jobs at a time
#   bash run_infer_configs.sh --dry_run outputs/ade20k/semantic/backbones
#   bash run_infer_configs.sh --concurrency 8 outputs/ade20k/semantic/backbones \
#       -- --batch_size 1

set -euo pipefail

CKPT_GLOB="*.ckpt"
PICK="newest"
CONCURRENCY=0
LIST_FILE=""
DRY_RUN=false
FORCE=false
METRICS_NAME="metrics.json"
NO_FLASH=false
METRICS_NAME_SET=false
INPUTS=()
PASSTHROUGH=()

usage() { sed -n '2,58p' "$0"; exit "${1:-0}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_glob)   CKPT_GLOB="$2";   shift 2 ;;
    --pick)        PICK="$2";        shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --list_file)   LIST_FILE="$2";   shift 2 ;;
    --metrics_name) METRICS_NAME="$2"; METRICS_NAME_SET=true; shift 2 ;;
    --no-flash)    NO_FLASH=true;    shift 1 ;;
    --force)       FORCE=true;       shift 1 ;;
    --dry_run)     DRY_RUN=true;     shift 1 ;;
    -h|--help)     usage 0 ;;
    --)            shift; PASSTHROUGH=("$@"); break ;;
    -*)            echo "Unknown option: $1" >&2; usage 1 ;;
    *)             INPUTS+=("$1");   shift 1 ;;
  esac
done

if $NO_FLASH; then
  PASSTHROUGH+=(--disable-flash-attn)
  $METRICS_NAME_SET || METRICS_NAME="metrics-no-flash-attn.json"
fi

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "error: no ckpt paths or directories given" >&2
  usage 1
fi
case "$PICK" in
  all|newest) ;;
  *) echo "error: --pick must be 'all' or 'newest' (got: $PICK)" >&2; exit 1 ;;
esac
if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]]; then
  echo "error: --concurrency must be a non-negative integer (got: $CONCURRENCY)" >&2
  exit 1
fi

# ── Expand inputs into a flat ckpt list ───────────────────────────────────────
CKPTS=()
for inp in "${INPUTS[@]}"; do
  if [[ -f "$inp" ]]; then
    CKPTS+=("$inp")
  elif [[ -d "$inp" ]]; then
    while IFS= read -r -d '' f; do
      CKPTS+=("$f")
    done < <(find "$inp" -type f -name "$CKPT_GLOB" -print0 | sort -z)
  else
    echo "[skip] not a file or directory: $inp" >&2
  fi
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "error: no checkpoints matched" >&2
  exit 1
fi

# ── Collapse to one ckpt per leaf folder if requested ─────────────────────────
if [[ "$PICK" == "newest" ]]; then
  declare -A SEEN_DIR
  FILTERED=()
  for c in "${CKPTS[@]}"; do
    d="$(dirname "$c")"
    if [[ -z "${SEEN_DIR[$d]:-}" ]]; then
      newest="$(ls -1t "$d"/$CKPT_GLOB 2>/dev/null | head -n1 || true)"
      [[ -n "$newest" ]] && FILTERED+=("$newest")
      SEEN_DIR[$d]=1
    fi
  done
  CKPTS=("${FILTERED[@]}")
fi

# ── Skip ckpts whose folder already has metrics.json ──────────────────────────
# Mirrors inference.py's default --output (<ckpt_dir>/metrics.json), so a
# previously-completed run is detected reliably as long as the passthrough
# args didn't override --output to somewhere else.
if ! $FORCE; then
  FILTERED=()
  SKIPPED_DONE=0
  for c in "${CKPTS[@]}"; do
    if [[ -f "$(dirname "$c")/$METRICS_NAME" ]]; then
      echo "[skip done] $(dirname "$c")/$METRICS_NAME exists"
      SKIPPED_DONE=$((SKIPPED_DONE + 1))
      continue
    fi
    FILTERED+=("$c")
  done
  CKPTS=("${FILTERED[@]}")
  if [[ "$SKIPPED_DONE" -gt 0 ]]; then
    echo "[info] skipped $SKIPPED_DONE ckpt(s) that already have $METRICS_NAME (use --force to re-run)"
  fi
fi

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "nothing to do — all matched ckpts already have $METRICS_NAME"
  exit 0
fi

N=${#CKPTS[@]}

# ── Write the list file ───────────────────────────────────────────────────────
if [[ -z "$LIST_FILE" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  LIST_FILE="slurm/lists/infer-${stamp}.txt"
fi
mkdir -p "$(dirname "$LIST_FILE")" slurm

printf '%s\n' "${CKPTS[@]}" > "$LIST_FILE"

# ── Build sbatch array spec ───────────────────────────────────────────────────
if [[ "$N" -eq 1 ]]; then
  ARRAY_SPEC="0"
else
  ARRAY_SPEC="0-$((N - 1))"
fi
if [[ "$CONCURRENCY" -gt 0 ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${CONCURRENCY}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$REPO_ROOT/run_infer_array.sh"
if [[ ! -f "$WRAPPER" ]]; then
  echo "error: $WRAPPER not found (expected sibling of $0)" >&2
  exit 1
fi

cmd=(sbatch --array="$ARRAY_SPEC" "$WRAPPER" "$LIST_FILE")
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
  cmd+=("${PASSTHROUGH[@]}")
fi

# ── Submit / dry-run ──────────────────────────────────────────────────────────
echo "ckpts:        $N"
echo "list file:    $LIST_FILE"
echo "array spec:   $ARRAY_SPEC"
echo "wrapper:      $WRAPPER"
echo "passthrough:  ${PASSTHROUGH[*]:-<none>}"
echo

if $DRY_RUN; then
  echo "─── list file contents ───"
  cat "$LIST_FILE"
  echo "──────────────────────────"
  echo
  printf '[dry] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
