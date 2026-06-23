"""
Voice Encoder Inference API
----------------------------
POST /encode  — accepts a WAV/MP3/OGG audio file, runs it through
                CustomSpeechEncoder, returns the frame-level embeddings
                and a mean-pooled speaker vector.

Run:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Audio / model hyperparameters (must match training) ──────────────────────
MEL_SR      = 22050
N_MELS      = 80
N_FFT       = 1024
WIN_LENGTH  = 1024
HOP_LENGTH  = 256
FMIN        = 0.0
FMAX        = 8000.0
MAX_DURATION = 10.0
MAX_MEL_LEN = int(MAX_DURATION * MEL_SR / HOP_LENGTH)   # 862

HIDDEN_DIM    = 768
ENCODER_HEADS = 12
ENCODER_LAYERS= 6
FFN_DIM       = 3072
DROPOUT       = 0.1

# ── Model definitions (copied verbatim from custom_encoder_v2.py) ─────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConvStem(nn.Module):
    def __init__(self, n_mels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.downsample = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.net[0](mel)
        x = self.net[1](x)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.downsample(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x.permute(0, 2, 1)


class CustomSpeechEncoder(nn.Module):
    def __init__(
        self,
        n_mels      : int   = N_MELS,
        hidden_dim  : int   = HIDDEN_DIM,
        num_layers  : int   = ENCODER_LAYERS,
        nhead       : int   = ENCODER_HEADS,
        ffn_dim     : int   = FFN_DIM,
        dropout     : float = DROPOUT,
    ):
        super().__init__()
        self.conv_stem = ConvStem(n_mels, hidden_dim)
        self.pos_enc   = SinusoidalPE(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm    = nn.LayerNorm(hidden_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(mel)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.out_norm(x)


# ── Device + model loading ────────────────────────────────────────────────────

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()

CHECKPOINT_CANDIDATES = [
    Path("encoder_v2_checkpoints/custom_encoder_v2.pt"),
    Path("saved_models/encoder_v2_paused_step12000/custom_encoder_v2.pt"),
    Path("encoder_v2_checkpoints/best.pt"),
    Path("saved_models/encoder_v2_paused_step12000/best.pt"),
]


def _load_encoder() -> CustomSpeechEncoder:
    """Load encoder weights from the first available checkpoint."""
    encoder = CustomSpeechEncoder().to(DEVICE)

    for ckpt_path in CHECKPOINT_CANDIDATES:
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

            # Handle both raw state_dict and full training checkpoints
            if isinstance(state, dict) and "encoder" in state:
                # full checkpoint: {"encoder": state_dict, "decoder": ..., ...}
                state_dict = state["encoder"]
            elif isinstance(state, dict) and any(
                k.startswith("encoder.") for k in state
            ):
                # full training model checkpoint: keys like "encoder.conv_stem...."
                state_dict = {
                    k[len("encoder."):]: v
                    for k, v in state.items()
                    if k.startswith("encoder.")
                }
            elif isinstance(state, dict) and "model_state_dict" in state:
                raw = state["model_state_dict"]
                # strip encoder. prefix if present
                if any(k.startswith("encoder.") for k in raw):
                    state_dict = {
                        k[len("encoder."):]: v
                        for k, v in raw.items()
                        if k.startswith("encoder.")
                    }
                else:
                    state_dict = raw
            else:
                state_dict = state

            encoder.load_state_dict(state_dict, strict=False)
            encoder.eval()
            print(f"[api] Loaded encoder from {ckpt_path}  (device={DEVICE})")
            return encoder

    raise RuntimeError(
        "No checkpoint found. Searched:\n  "
        + "\n  ".join(str(p) for p in CHECKPOINT_CANDIDATES)
    )


# Mel transform (CPU — we move tensor to DEVICE inside wav_to_mel)
_mel_transform = T.MelSpectrogram(
    sample_rate=MEL_SR,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=1.0,
)


def _load_audio(path: str, suffix: str) -> tuple[torch.Tensor, int]:
    """
    Decode audio to a (C, T) float32 waveform tensor, trying several strategies
    so the server works without torchcodec / without ffmpeg being installed.

    Priority:
      1. soundfile  — zero-copy, works for WAV / FLAC / OGG-Vorbis
      2. torchaudio with explicit 'soundfile' backend (same formats)
      3. torchaudio with explicit 'ffmpeg' backend   (MP3, M4A, WebM, …)
      4. torchaudio default  (lets torchaudio pick whatever is available)
    """
    import soundfile as sf

    errors: list[str] = []

    # ── Strategy 1: soundfile directly (best for WAV / FLAC / OGG) ──────────
    if suffix in {".wav", ".flac", ".ogg"}:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)   # (C, T)
            return wav, int(sr)
        except Exception as exc:
            errors.append(f"soundfile: {exc}")

    # ── Strategy 2: torchaudio soundfile backend ─────────────────────────────
    try:
        wav, sr = torchaudio.load(path, backend="soundfile")
        return wav, int(sr)
    except Exception as exc:
        errors.append(f"torchaudio[soundfile]: {exc}")

    # ── Strategy 3: torchaudio ffmpeg backend (needs ffmpeg installed) ───────
    try:
        wav, sr = torchaudio.load(path, backend="ffmpeg")
        return wav, int(sr)
    except Exception as exc:
        errors.append(f"torchaudio[ffmpeg]: {exc}")

    # ── Strategy 4: torchaudio default ───────────────────────────────────────
    try:
        wav, sr = torchaudio.load(path)
        return wav, int(sr)
    except Exception as exc:
        errors.append(f"torchaudio[default]: {exc}")

    tip = ""
    if suffix in {".webm", ".mp3", ".m4a"}:
        tip = (
            f" Note: '{suffix}' requires ffmpeg. "
            "Install it with: brew install ffmpeg"
        )
    raise RuntimeError("; ".join(errors) + tip)


def wav_to_mel(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """(C, T) waveform → (1, N_MELS, MAX_MEL_LEN) batched mel tensor on DEVICE."""
    if orig_sr != MEL_SR:
        wav = T.Resample(orig_sr, MEL_SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    mel = _mel_transform(wav).squeeze(0)          # (N_MELS, T)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    T_frames = mel.shape[1]
    if T_frames > MAX_MEL_LEN:
        mel = mel[:, :MAX_MEL_LEN]
    elif T_frames < MAX_MEL_LEN:
        mel = torch.nn.functional.pad(mel, (0, MAX_MEL_LEN - T_frames), value=-11.5)
    return mel.unsqueeze(0).to(DEVICE)            # (1, N_MELS, MAX_MEL_LEN)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice Encoder API",
    description="Feed your voice → get encoder embeddings from the custom speech encoder.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.on_event("startup")
async def startup() -> None:
    app.state.encoder = _load_encoder()
    print("[api] Encoder ready ✓")


@app.get("/", include_in_schema=False)
async def root():
    index = Path(__file__).parent / "frontend" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Voice Encoder API — POST audio to /encode"}


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


ACCEPTED_FORMATS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}


@app.post("/encode")
async def encode_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV, MP3, OGG, FLAC, WebM, M4A).

    Returns:
    - `embeddings`: frame-level encoder output — shape (T, 768), each frame ~11.6 ms
    - `speaker_vector`: mean-pooled 768-dim speaker representation (L2-normalised)
    - `shape`: [T, 768]
    - `duration_s`: approximate audio duration in seconds
    - `device`: which device inference ran on
    """
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ACCEPTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Accepted: {sorted(ACCEPTED_FORMATS)}",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # Write to temp file so audio decoders can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        wav, sr = _load_audio(tmp_path, suffix)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    duration_s = wav.shape[-1] / sr

    mel = wav_to_mel(wav, sr)   # (1, N_MELS, MAX_MEL_LEN)

    encoder: CustomSpeechEncoder = app.state.encoder
    with torch.no_grad():
        z = encoder(mel)        # (1, T_enc, 768)

    z_np = z.squeeze(0).cpu().float().numpy()              # (T_enc, 768)
    speaker_vec = z_np.mean(axis=0)                        # (768,)
    speaker_vec = speaker_vec / (np.linalg.norm(speaker_vec) + 1e-8)

    return {
        "embeddings"    : z_np.tolist(),
        "speaker_vector": speaker_vec.tolist(),
        "shape"         : list(z_np.shape),
        "duration_s"    : round(float(duration_s), 3),
        "device"        : str(DEVICE),
    }
