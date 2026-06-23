#!/usr/bin/env bash
# Run Stage A natively with thread limits (keeps Apple MPS / CUDA).
# Leaves RESERVED_CPUS and RESERVED_MEM_GB headroom for macOS.
#
# Pause:  Ctrl+C  (saves stage_a_checkpoints/latest.pt)
# Resume: RESUME=stage_a_checkpoints/latest.pt ./scripts/run_stage_a_native.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESERVED_CPUS="${RESERVED_CPUS:-2}"
RESERVED_MEM_GB="${RESERVED_MEM_GB:-2}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv python at $PYTHON" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  TOTAL_CPUS="$(sysctl -n hw.ncpu)"
  TOTAL_MEM_GB="$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
else
  TOTAL_CPUS="$(nproc)"
  TOTAL_MEM_GB="$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))"
fi

TRAIN_CPUS=$(( TOTAL_CPUS - RESERVED_CPUS ))
TRAIN_MEM_GB=$(( TOTAL_MEM_GB - RESERVED_MEM_GB ))

if (( TRAIN_CPUS < 1 )); then
  echo "Not enough CPUs after reserving ${RESERVED_CPUS} for the system." >&2
  exit 1
fi

export TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-$TRAIN_CPUS}"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"

echo "Host: ${TOTAL_CPUS} CPUs, ${TOTAL_MEM_GB}GB RAM"
echo "Training threads: ${TRAIN_NUM_THREADS} (reserved ${RESERVED_CPUS} CPUs for system)"
echo "Tip: keep ~${RESERVED_MEM_GB}GB free for macOS; reduce --batch-size if memory spikes."

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
  echo "Resuming from: ${RESUME}"
fi

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

mkdir -p stage_a_checkpoints

CMD=("$PYTHON" train_stage_a.py)
if ((${#RESUME_ARGS[@]})); then
  CMD+=("${RESUME_ARGS[@]}")
fi
if ((${#EXTRA_ARGS[@]})); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

# Lower CPU scheduling priority so the desktop stays responsive.
nice -n 10 "${CMD[@]}" \
  2>&1 | tee -a stage_a_training.log
