#!/usr/bin/env bash
# Sync Voice-Model-v1 artifacts with Google Cloud Storage.
#
# First-time setup (creates bucket if missing):
#   ./fleet/sync_gcs.sh create-bucket
#
# Upload best weights for finetuning on another machine (~7 GB):
#   ./fleet/sync_gcs.sh push-best
#
# Download on a new machine:
#   ./fleet/sync_gcs.sh pull
#
# List remote files:
#   ./fleet/sync_gcs.sh status

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-vertex-ai-learning-487906}"
BUCKET="${GCS_BUCKET:-voice-model-v1-487906}"
LOCATION="${GCS_LOCATION:-asia-south1}"
PREFIX="${GCS_PREFIX:-voice-model-v1}"
GS="gs://${BUCKET}/${PREFIX}"

mkdir -p encoder_v2_checkpoints stage_a_checkpoints data fleet

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

cmd="${1:-status}"

case "$cmd" in
  create-bucket)
    if gcloud storage buckets describe "gs://${BUCKET}" --project="${PROJECT}" &>/dev/null; then
      echo "Bucket already exists: gs://${BUCKET}"
    else
      echo "Creating bucket gs://${BUCKET} in ${LOCATION} (project ${PROJECT}) ..."
      gcloud storage buckets create "gs://${BUCKET}" \
        --project="${PROJECT}" \
        --location="${LOCATION}" \
        --uniform-bucket-level-access
      echo "Bucket created."
    fi
    echo "Console: https://console.cloud.google.com/storage/browser/${BUCKET}?project=${PROJECT}"
    ;;

  push-best)
    require_file encoder_v2_checkpoints/custom_encoder_v2.pt
    require_file stage_a_checkpoints/best.pt
    require_file train.csv
    require_file val.csv

    echo "Uploading finetuning bundle to ${GS} ..."
    echo "(encoder ~182 MB + best.pt ~6.6 GB + CSVs — may take 10-30 min on home upload)"
    echo

    gcloud storage cp encoder_v2_checkpoints/custom_encoder_v2.pt \
      "${GS}/encoder_v2_checkpoints/custom_encoder_v2.pt"
    gcloud storage cp stage_a_checkpoints/best.pt \
      "${GS}/stage_a_checkpoints/best.pt"
    gcloud storage cp train.csv "${GS}/data/train.csv"
    gcloud storage cp val.csv "${GS}/data/val.csv"
    gcloud storage cp fleet/gcs_manifest.json "${GS}/fleet/gcs_manifest.json"
    gcloud storage cp fleet/job_state.json "${GS}/fleet/job_state.json"

    echo
    echo "Done. On another machine:"
    echo "  git clone <repo> && cd Voice-Model-v1"
    echo "  ./fleet/sync_gcs.sh pull"
    echo "  # download LJSpeech locally, then:"
    echo "  RESUME=stage_a_checkpoints/best.pt ./scripts/run_stage_a_native.sh"
    ;;

  push-checkpoints)
    echo "Uploading all stage_a_checkpoints (excludes tb_logs) ..."
    gcloud storage rsync -r stage_a_checkpoints/ "${GS}/stage_a_checkpoints/" \
      --exclude "tb_logs/**"
    echo "Done."
    ;;

  pull)
    echo "Pulling finetuning bundle from ${GS} ..."
    gcloud storage cp "${GS}/data/train.csv" train.csv
    gcloud storage cp "${GS}/data/val.csv" val.csv
    gcloud storage cp "${GS}/encoder_v2_checkpoints/custom_encoder_v2.pt" \
      encoder_v2_checkpoints/custom_encoder_v2.pt
    gcloud storage cp "${GS}/stage_a_checkpoints/best.pt" \
      stage_a_checkpoints/best.pt
    gcloud storage cp "${GS}/fleet/gcs_manifest.json" fleet/gcs_manifest.json
    gcloud storage cp "${GS}/fleet/job_state.json" fleet/job_state.json
    echo "Done. Download LJSpeech locally:"
    echo "  https://keithito.com/LJ-Speech-Dataset/"
    ;;

  status)
    gcloud storage ls -r "gs://${BUCKET}/${PREFIX}/" --readable-sizes 2>/dev/null || \
      echo "Bucket empty or not found. Run: ./fleet/sync_gcs.sh create-bucket"
    ;;

  *)
    echo "Usage: $0 {create-bucket|push-best|push-checkpoints|pull|status}" >&2
    exit 1
    ;;
esac
