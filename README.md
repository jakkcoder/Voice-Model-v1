# Voice-Model-v1

Speech-to-LLM voice agent pipeline: custom speech encoder → ModalityAdapter → SmolLM2-1.7B (LoRA).

**Repository:** https://github.com/jakkcoder/Voice-Model-v1

---

## Project layout

```
Voice-Model-v1/
├── train_encoder.py          # Stage 0: train custom speech encoder
├── train_stage_a.py          # Stage A: adapter + SmolLM2 LoRA (main training script)
├── custom_encoder_v2.py        # Encoder notebook/script (v2)
├── custom_encoder_v3.py        # Full pipeline notebook export (Colab-style)
├── api.py                      # FastAPI inference server
├── frontend/index.html         # Web UI
│
├── scripts/
│   ├── run_stage_a_native.sh   # Train on Mac MPS (recommended)
│   └── run_stage_a_docker.sh   # Train in Docker (CPU only on Mac)
│
├── fleet/
│   ├── sync_s3.sh              # Pull/push artifacts from shared S3 bucket
│   ├── s3_manifest.json        # S3 path reference
│   └── job_state.json          # Distributed training progress tracker
│
├── Dockerfile.train            # Docker image for Stage A
├── pyproject.toml              # Python dependencies
├── train.csv                   # Training manifest (paths + transcripts)
├── val.csv                     # Validation manifest
│
├── LJSpeech-1.1/               # Audio dataset (local only, not in git)
│   └── wavs/                   # .wav files referenced by train.csv
│
├── encoder_v2_checkpoints/     # Stage 0 encoder weights (local / S3)
│   └── custom_encoder_v2.pt
│
└── stage_a_checkpoints/        # Stage A checkpoints (local / S3)
    ├── latest.pt               # Saved on Ctrl+C / docker stop
    ├── ckpt_step1500.pt        # Periodic checkpoint (every 1500 steps)
    ├── best.pt                 # Best validation loss
    └── tb_logs/                # TensorBoard logs
```

---

## What lives where

| Item | Git | S3 | Local only |
|------|-----|----|----|
| Source code (`train_*.py`, `scripts/`) | Yes | — | — |
| `train.csv`, `val.csv` | Yes | Yes | — |
| `custom_encoder_v2.pt` (encoder) | No | Yes | Yes |
| Stage A checkpoints (`*.pt`) | No | Yes | Yes |
| `LJSpeech-1.1/` audio (~3.6 GB) | No | No | Yes (download) |
| `.env`, `.venv/`, logs | No | No | Yes |
| TensorBoard logs | No | No | Yes |

---

## Shared S3 bucket (team artifacts)

| | |
|---|---|
| **Bucket** | `voice-model-v1-fleet-079984577428` |
| **Region** | `us-east-1` |
| **Prefix** | `voice-model-v1/` |

### S3 paths

| File | S3 key |
|------|--------|
| Encoder checkpoint | `voice-model-v1/encoder_v2_checkpoints/custom_encoder_v2.pt` |
| Train manifest | `voice-model-v1/data/train.csv` |
| Val manifest | `voice-model-v1/data/val.csv` |
| Stage A checkpoints | `voice-model-v1/stage_a_checkpoints/` |
| Fleet job state | `voice-model-v1/fleet/job_state.json` |
| Path manifest | `voice-model-v1/fleet/s3_manifest.json` |

**Console:** https://s3.console.aws.amazon.com/s3/buckets/voice-model-v1-fleet-079984577428

---

## Setup

### 1. Clone and install

```bash
git clone git@github.com:jakkcoder/Voice-Model-v1.git
cd Voice-Model-v1
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download LJSpeech (local, required for training)

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
```

Dataset info: https://keithito.com/LJ-Speech-Dataset/

### 3. Pull shared artifacts from S3

Requires AWS credentials with read access to the bucket.

```bash
./fleet/sync_s3.sh pull
```

This downloads encoder weights, CSVs, and Stage A checkpoints into the local paths listed above.

### 4. Verify Apple GPU (Mac)

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Training

### Stage 0 — Speech encoder

```bash
python train_encoder.py
# Output: encoder_v2_checkpoints/custom_encoder_v2.pt
```

### Stage A — Adapter + SmolLM2 LoRA (recommended: native MPS)

```bash
./scripts/run_stage_a_native.sh
```

| Setting | Value |
|---------|-------|
| Target steps | 15,000 |
| Checkpoint every | 1,500 steps |
| Effective batch size | 8 (batch 2 × grad accum 4 on Mac) |
| Output dir | `stage_a_checkpoints/` |

**Pause:** Ctrl+C → saves `stage_a_checkpoints/latest.pt`

**Resume:**

```bash
RESUME=stage_a_checkpoints/ckpt_step1500.pt ./scripts/run_stage_a_native.sh
```

**Upload new checkpoints to S3 after a session:**

```bash
./fleet/sync_s3.sh push-checkpoints
./fleet/sync_s3.sh push-state
```

### Stage A — Docker (optional, CPU only on Mac)

Docker on Mac cannot use Apple MPS. Use only if you need hard RAM/CPU caps.

```bash
./scripts/run_stage_a_docker.sh
# Detached: BUILD=0 DETACH=1 ./scripts/run_stage_a_docker.sh
```

---

## Distributed training (Mac fleet over VPN)

Training uses **checkpoint handoff**, not synced multi-GPU DDP, so teammates can join and leave anytime.

1. Each MacBook: VPN + SSH + clone repo + `./fleet/sync_s3.sh pull` + LJSpeech locally
2. When available: run `./scripts/run_stage_a_native.sh` with `--resume` from latest S3 checkpoint
3. After session: `./fleet/sync_s3.sh push-checkpoints`
4. Coordinator tracks progress in `fleet/job_state.json`

See `fleet/s3_manifest.json` for all S3 paths.

---

## Inference API

```bash
uvicorn api:app --reload
# Frontend: open frontend/index.html
```

---

## Loss interpretation (Stage A)

| Val loss | Meaning |
|----------|---------|
| > 4.0 | Adapter not converging |
| 2.5–4.0 | Learning, may need more steps |
| 1.5–2.5 | Good alignment, ready for Stage B |
| < 1.5 | Strong audio conditioning |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Faster HuggingFace downloads |
| `RESUME` | Checkpoint path for `run_stage_a_native.sh` / docker script |
| `S3_BUCKET` | Override bucket in `fleet/sync_s3.sh` (default: `voice-model-v1-fleet-079984577428`) |
| `S3_PREFIX` | Override S3 prefix (default: `voice-model-v1`) |
| `TRAIN_NUM_THREADS` | CPU thread limit for training |
| `RESERVED_CPUS` / `RESERVED_MEM_GB` | Headroom left for macOS in run scripts |

---

## Logs

| File | Description |
|------|-------------|
| `stage_a_training.log` | Stage A training output (appended) |
| `encoder_training.log` | Encoder training output |
| `stage_a_checkpoints/tb_logs/` | TensorBoard metrics |

```bash
tensorboard --logdir stage_a_checkpoints/tb_logs
```
