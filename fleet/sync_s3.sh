#!/usr/bin/env bash
# Sync Voice-Model-v1 fleet artifacts with S3.
#
# Pull (new Mac setup):
#   ./fleet/sync_s3.sh pull
#
# Push checkpoints after training:
#   ./fleet/sync_s3.sh push-checkpoints
#
# Push job state only:
#   ./fleet/sync_s3.sh push-state

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUCKET="${S3_BUCKET:-voice-model-v1-fleet-079984577428}"
PREFIX="${S3_PREFIX:-voice-model-v1}"
S3="s3://${BUCKET}/${PREFIX}"

mkdir -p encoder_v2_checkpoints stage_a_checkpoints data fleet

cmd="${1:-pull}"

case "$cmd" in
  pull)
    echo "Pulling from ${S3} ..."
    aws s3 cp "${S3}/data/train.csv" train.csv
    aws s3 cp "${S3}/data/val.csv" val.csv
    aws s3 cp "${S3}/encoder_v2_checkpoints/custom_encoder_v2.pt" \
      encoder_v2_checkpoints/custom_encoder_v2.pt
    aws s3 cp "${S3}/fleet/s3_manifest.json" fleet/s3_manifest.json
    aws s3 cp "${S3}/fleet/job_state.json" fleet/job_state.json
    aws s3 sync "${S3}/stage_a_checkpoints/" stage_a_checkpoints/ \
      --exclude "tb_logs/*"
    echo "Done. Download LJSpeech locally if needed:"
    echo "  https://keithito.com/LJ-Speech-Dataset/"
    ;;
  push-checkpoints)
    echo "Pushing checkpoints to ${S3}/stage_a_checkpoints/ ..."
    aws s3 sync stage_a_checkpoints/ "${S3}/stage_a_checkpoints/" \
      --exclude "tb_logs/*" \
      --exclude "*.log"
    echo "Done."
    ;;
  push-state)
    aws s3 cp fleet/job_state.json "${S3}/fleet/job_state.json"
    echo "job_state.json uploaded."
    ;;
  status)
    aws s3 ls "${S3}/" --recursive --human-readable --summarize
    ;;
  *)
    echo "Usage: $0 {pull|push-checkpoints|push-state|status}" >&2
    exit 1
    ;;
esac
