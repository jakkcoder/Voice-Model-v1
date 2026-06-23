#!/usr/bin/env bash
# Run Stage A (custom_encoder_v3.py cells 6–12) in Docker with CPU/RAM caps.
#
# Equivalent to the notebook block: SmolLM2+LoRA, adapter, forward pass,
# optimizer, training loop → stage_a_checkpoints/
#
# NOTE: Docker on Mac cannot use Apple MPS. Training runs on CPU inside the
# container and is much slower than native MPS. For Apple Silicon GPU, use:
#   ./scripts/run_stage_a_native.sh
#
# Pause:  Ctrl+C  (saves stage_a_checkpoints/latest.pt)
# Resume: RESUME=stage_a_checkpoints/ckpt_step1500.pt ./scripts/run_stage_a_docker.sh
# Detached: DETACH=1 ./scripts/run_stage_a_docker.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESERVED_CPUS="${RESERVED_CPUS:-2}"
RESERVED_MEM_GB="${RESERVED_MEM_GB:-2}"
IMAGE="${IMAGE:-voice-model-stage-a:latest}"
BUILD="${BUILD:-1}"
DETACH="${DETACH:-0}"

# CPU Docker: effective batch=8 (1×8); --low-mem fits better in constrained RAM
DEFAULT_ARGS=(--low-mem --batch-size 1 --grad-accum 8 --max-steps 15000 --save-every 1500)
MIN_DOCKER_MEM_GB=14

if [[ "$(uname -s)" == "Darwin" ]]; then
  TOTAL_CPUS="$(sysctl -n hw.ncpu)"
  TOTAL_MEM_GB="$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
else
  TOTAL_CPUS="$(nproc)"
  TOTAL_MEM_GB="$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))"
fi

TRAIN_CPUS=$(( TOTAL_CPUS - RESERVED_CPUS ))
TRAIN_MEM_GB=$(( TOTAL_MEM_GB - RESERVED_MEM_GB ))

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_CPUS="$(docker info --format '{{.NCPU}}')"
  DOCKER_MEM_GB="$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 ))"
  DOCKER_TRAIN_CPUS=$(( DOCKER_CPUS - RESERVED_CPUS ))
  DOCKER_TRAIN_MEM_GB=$(( DOCKER_MEM_GB - RESERVED_MEM_GB ))
  if (( DOCKER_TRAIN_CPUS < TRAIN_CPUS )); then
    echo "Docker VM: ${DOCKER_CPUS} CPUs, ${DOCKER_MEM_GB}GB RAM (capping container limits)"
    TRAIN_CPUS=$DOCKER_TRAIN_CPUS
    TRAIN_MEM_GB=$DOCKER_TRAIN_MEM_GB
  fi
fi

if (( TRAIN_CPUS < 1 )); then
  echo "Not enough CPUs after reserving ${RESERVED_CPUS} for the system." >&2
  exit 1
fi
if (( TRAIN_MEM_GB < 4 )); then
  echo "Not enough RAM after reserving ${RESERVED_MEM_GB}GB for the system." >&2
  exit 1
fi
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_MEM_GB="$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 ))"
  if (( DOCKER_MEM_GB < MIN_DOCKER_MEM_GB )); then
    echo "WARNING: Docker Desktop has only ${DOCKER_MEM_GB}GB RAM allocated." >&2
    echo "SmolLM2-1.7B Stage A needs ~${MIN_DOCKER_MEM_GB}GB+ inside Docker." >&2
    echo "Increase it: Docker Desktop → Settings → Resources → Memory → 16–18GB" >&2
    echo "Continuing with --low-mem; training may still OOM." >&2
  fi
fi

echo "Host: ${TOTAL_CPUS} CPUs, ${TOTAL_MEM_GB}GB RAM"
echo "Container limits: ${TRAIN_CPUS} CPUs, ${TRAIN_MEM_GB}GB RAM"
if [[ "$BUILD" == "1" ]]; then
  echo "Building image ${IMAGE}..."
  docker build -f Dockerfile.train -t "$IMAGE" .
else
  echo "Skipping build (BUILD=0). Using existing image ${IMAGE}."
fi

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "/workspace/${RESUME#./}")
  echo "Resuming from: ${RESUME}"
fi

EXTRA_ARGS=("${DEFAULT_ARGS[@]}")
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

mkdir -p stage_a_checkpoints

DOCKER_RUN=(
  --name voice-stage-a-train
  --cpus="${TRAIN_CPUS}"
  --memory="${TRAIN_MEM_GB}g"
  --memory-swap="${TRAIN_MEM_GB}g"
  -e "TRAIN_NUM_THREADS=${TRAIN_CPUS}"
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -v "$ROOT:/workspace"
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
  -w /workspace
  "$IMAGE"
)
if ((${#RESUME_ARGS[@]})); then
  DOCKER_RUN+=("${RESUME_ARGS[@]}")
fi
DOCKER_RUN+=("${EXTRA_ARGS[@]}")

if [[ "$DETACH" == "1" ]]; then
  docker run --rm -d "${DOCKER_RUN[@]}"
  echo "Container started. Follow training with:"
  echo "  docker logs -f voice-stage-a-train"
else
  docker run --rm -it "${DOCKER_RUN[@]}" 2>&1 | tee -a stage_a_training.log
fi
