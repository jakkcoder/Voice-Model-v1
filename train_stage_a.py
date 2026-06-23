#!/usr/bin/env python3
"""Stage A: train ModalityAdapter + SmolLM2 LoRA with frozen speech encoder."""

from __future__ import annotations

import argparse
import math
import os
import signal
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Audio ────────────────────────────────────────────────────────────────────
MEL_SR = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0
MAX_DURATION = 10.0
MAX_MEL_LEN = int(MAX_DURATION * MEL_SR / HOP_LENGTH)

# ── Encoder architecture ─────────────────────────────────────────────────────
HIDDEN_DIM = 768
ENCODER_HEADS = 12
ENCODER_LAYERS = 6
FFN_DIM = 3072
DROPOUT = 0.1

# ── Stage A ──────────────────────────────────────────────────────────────────
LLM_ID = "HuggingFaceTB/SmolLM2-1.7B"
LLM_DIM = 2048
NUM_AUDIO_TOKENS = 32
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]

BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 2e-4
WARMUP_STEPS = 300
MAX_STEPS = 15000
SAVE_EVERY = 1500
LOG_EVERY = 50
MAX_TEXT_LEN = 128
OUTPUT_DIR = "stage_a_checkpoints"

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"

ENCODER_CHECKPOINTS = [
    Path("encoder_v2_checkpoints/custom_encoder_v2.pt"),
    Path("saved_models/encoder_v2_paused_step12000/custom_encoder_v2.pt"),
]

_shutdown_requested = False


def _request_shutdown(signum: int, frame) -> None:
    del signum, frame
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        print(
            "\nShutdown requested — finishing current optimizer step, "
            "then saving latest checkpoint..."
        )


def configure_thread_limits() -> int | None:
    raw = os.environ.get("TRAIN_NUM_THREADS")
    if not raw:
        return None
    n = max(1, int(raw))
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    print(f"Thread limit: {n}")
    return n


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_fp16(device: torch.device) -> bool:
    return device.type == "cuda"


def resolve_encoder_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
        return path
    for candidate in ENCODER_CHECKPOINTS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No encoder checkpoint found. Searched:\n  "
        + "\n  ".join(str(p) for p in ENCODER_CHECKPOINTS)
    )


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
        n_mels: int = N_MELS,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = ENCODER_LAYERS,
        nhead: int = ENCODER_HEADS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.conv_stem = ConvStem(n_mels, hidden_dim)
        self.pos_enc = SinusoidalPE(hidden_dim)
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
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(mel)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.out_norm(x)


class ModalityAdapter(nn.Module):
    def __init__(
        self,
        encoder_dim: int = HIDDEN_DIM,
        llm_dim: int = LLM_DIM,
        num_query_tokens: int = NUM_AUDIO_TOKENS,
        num_qformer_layers: int = 2,
        nhead: int = 8,
        ffn_dim: int = 512,
    ):
        super().__init__()
        self.conv_downsample = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, encoder_dim)
        )
        qformer_layer = nn.TransformerDecoderLayer(
            d_model=encoder_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            batch_first=True,
            activation="gelu",
        )
        self.qformer = nn.TransformerDecoder(qformer_layer, num_layers=num_qformer_layers)
        self.norm = nn.LayerNorm(encoder_dim)
        self.linear_proj = nn.Linear(encoder_dim, llm_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        x = encoder_output.permute(0, 2, 1)
        x = self.conv_downsample(x)
        x = x.permute(0, 2, 1)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        out = self.qformer(queries, memory=x)
        out = self.norm(out)
        return self.linear_proj(out)


mel_transform_cpu = T.MelSpectrogram(
    sample_rate=MEL_SR,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=1.0,
)


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    return wav, sr


def wav_to_mel_cpu(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    if orig_sr != MEL_SR:
        wav = T.Resample(orig_sr, MEL_SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    mel = mel_transform_cpu(wav).squeeze(0)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    t_frames = mel.shape[1]
    if t_frames > MAX_MEL_LEN:
        mel = mel[:, :MAX_MEL_LEN]
    elif t_frames < MAX_MEL_LEN:
        mel = torch.nn.functional.pad(mel, (0, MAX_MEL_LEN - t_frames), value=-11.5)
    return mel


class StageADataset(Dataset):
    def __init__(self, csv_path: str, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, sr = load_wav(row["wav_path"])
        mel = wav_to_mel_cpu(wav, sr)
        tokens = self.tokenizer(
            str(row["text"]),
            max_length=MAX_TEXT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return mel, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)


def collate_fn(batch):
    mels, ids, masks = zip(*batch)
    return torch.stack(mels), torch.stack(ids), torch.stack(masks)


def load_encoder(encoder: CustomSpeechEncoder, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "encoder_state" in state:
        state = state["encoder_state"]
    encoder.load_state_dict(state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False


def load_llm(device: torch.device, fp16: bool, low_mem: bool = False):
    print(f"Loading {LLM_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if fp16 else torch.float32
    load_kwargs = {"torch_dtype": dtype}
    if low_mem:
        load_kwargs["low_cpu_mem_usage"] = True

    if device.type == "cuda":
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            device_map="auto",
            **load_kwargs,
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(LLM_ID, **load_kwargs)
        llm = llm.to(device)

    for param in llm.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    llm = get_peft_model(llm, lora_cfg)
    if low_mem:
        llm.gradient_checkpointing_enable()
        print("Low-memory mode: fp16 weights + gradient checkpointing enabled")
    llm.print_trainable_parameters()
    return tokenizer, llm


def use_autocast(device: torch.device, fp16: bool) -> bool:
    return fp16 and device.type == "cuda"


def build_forward_pass(encoder, adapter, llm, tokenizer):
    def forward_pass(mel_batch, input_ids, attention_mask):
        batch_size = mel_batch.size(0)

        with torch.no_grad():
            z = encoder(mel_batch)
        audio_tokens = adapter(z)
        text_embeds = llm.get_input_embeddings()(input_ids)
        audio_tokens = audio_tokens.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)

        audio_mask = torch.ones(
            batch_size,
            NUM_AUDIO_TOKENS,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat([audio_mask, attention_mask], dim=1)

        ignore = torch.full(
            (batch_size, NUM_AUDIO_TOKENS),
            -100,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        labels = torch.cat([ignore, input_ids], dim=1)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )
        return outputs.loss

    return forward_pass


def get_lr(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


def validate(forward_pass, val_loader, device, autocast) -> float:
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            mel, ids, mask = [x.to(device) for x in batch]
            with torch.autocast(device_type=device.type, enabled=autocast):
                total += forward_pass(mel, ids, mask).item()
            n += 1
    return total / max(1, n)


def save_checkpoint(
    path: str,
    step: int,
    adapter: ModalityAdapter,
    llm,
    optimizer,
    val_loss: float,
    best_val: float,
) -> None:
    torch.save(
        {
            "step": step,
            "adapter_state": adapter.state_dict(),
            "lora_state": llm.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val,
        },
        path,
    )


def load_checkpoint(
    path: str,
    adapter: ModalityAdapter,
    llm,
    optimizer,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state"])
    llm.load_state_dict(ckpt["lora_state"])
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step = int(ckpt["step"])
    best_val = float(ckpt.get("best_val_loss", ckpt["val_loss"]))
    print(f"Resumed from {path} at step {step} (val_loss={ckpt['val_loss']:.4f})")
    return step, best_val


def train(args: argparse.Namespace) -> None:
    global _shutdown_requested
    _shutdown_requested = False
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    configure_thread_limits()
    device = get_device()
    fp16 = (use_fp16(device) or args.low_mem) and not args.no_fp16
    autocast = use_autocast(device, fp16)
    num_workers = 0 if device.type in {"mps", "cpu"} else args.num_workers

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"FP16 weights: {fp16} | autocast: {autocast} | low_mem: {args.low_mem} | "
        f"batch_size={args.batch_size} | grad_accum={args.grad_accum} | "
        f"max_steps={args.max_steps}"
    )

    encoder_path = resolve_encoder_path(args.encoder_path)
    print(f"Encoder checkpoint: {encoder_path}")

    tokenizer, llm = load_llm(device, fp16, low_mem=args.low_mem)
    adapter = ModalityAdapter().to(device)

    encoder = CustomSpeechEncoder().to(device)
    load_encoder(encoder, encoder_path, device)

    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad) / 1e6
    print(f"Encoder params (frozen): {enc_params:.1f}M")
    print(f"Adapter params (train):  {adapter_params:.1f}M")

    train_loader = DataLoader(
        StageADataset(args.train_csv, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        StageADataset(args.val_csv, tokenizer),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    print(f"train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}")

    forward_pass = build_forward_pass(encoder, adapter, llm, tokenizer)

    trainable_params = list(adapter.parameters()) + [
        p for p in llm.parameters() if p.requires_grad
    ]
    total_trainable = sum(p.numel() for p in trainable_params) / 1e6
    print(f"Total trainable params: {total_trainable:.1f}M")

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr(step, args.warmup_steps, args.max_steps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=autocast)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.output_dir}/tb_logs")

    step = 0
    best_val = float("inf")
    if args.resume:
        step, best_val = load_checkpoint(
            args.resume, adapter, llm, optimizer, device
        )
        for _ in range(step):
            scheduler.step()

    mel_t, ids_t, mask_t = [x.to(device) for x in next(iter(train_loader))]
    with torch.autocast(device_type=device.type, enabled=autocast):
        test_loss = forward_pass(mel_t, ids_t, mask_t)
    print(f"Test loss (untrained): {test_loss.item():.4f}")

    adapter.train()
    llm.train()
    optimizer.zero_grad()

    accum_step = 0
    running_loss = 0.0
    log_steps = 0

    print(f"Starting Stage A — target {args.max_steps} steps")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    pbar = tqdm(total=args.max_steps, initial=step, desc="Stage A", unit="step")

    def persist_latest(val_loss: float | None = None) -> None:
        if step <= 0:
            return
        if val_loss is None:
            adapter.eval()
            llm.eval()
            val_loss = validate(forward_pass, val_loader, device, autocast)
            adapter.train()
            llm.train()
        latest_path = f"{args.output_dir}/latest.pt"
        save_checkpoint(
            latest_path,
            step,
            adapter,
            llm,
            optimizer,
            val_loss,
            best_val,
        )
        pbar.write(f">>> Saved resume checkpoint: {latest_path} (step {step})")

    interrupted = False
    while step < args.max_steps and not _shutdown_requested:
        for batch in train_loader:
            if step >= args.max_steps or _shutdown_requested:
                break

            mel, ids, mask = [x.to(device) for x in batch]

            with torch.autocast(device_type=device.type, enabled=autocast):
                loss = forward_pass(mel, ids, mask) / args.grad_accum

            scaler.scale(loss).backward()
            accum_step += 1
            running_loss += loss.item() * args.grad_accum
            log_steps += 1

            if accum_step == args.grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_step = 0
                step += 1
                pbar.update(1)

                if step % args.log_every == 0:
                    avg = running_loss / log_steps
                    lr_now = scheduler.get_last_lr()[0] * args.lr
                    writer.add_scalar("train/loss", avg, step)
                    writer.add_scalar("train/lr", lr_now, step)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
                    running_loss = 0.0
                    log_steps = 0

                if step % args.save_every == 0:
                    adapter.eval()
                    llm.eval()
                    val_loss = validate(forward_pass, val_loader, device, autocast)
                    writer.add_scalar("val/loss", val_loss, step)
                    pbar.write(f">>> VAL step {step} | loss={val_loss:.4f}")

                    ckpt_path = f"{args.output_dir}/ckpt_step{step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        step,
                        adapter,
                        llm,
                        optimizer,
                        val_loss,
                        best_val,
                    )
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            f"{args.output_dir}/best.pt",
                            step,
                            adapter,
                            llm,
                            optimizer,
                            val_loss,
                            best_val,
                        )
                        pbar.write(f"  New best: {best_val:.4f}")

                    adapter.train()
                    llm.train()

    if _shutdown_requested and step > 0:
        interrupted = True
        persist_latest()

    pbar.close()
    writer.close()
    if interrupted:
        print(
            f"\nTraining paused at step {step}. Resume with:\n"
            f"  --resume {args.output_dir}/latest.pt"
        )
        return

    print(f"\nStage A complete. Best val loss: {best_val:.4f}")
    print("Loss interpretation:")
    print("  > 4.0   → adapter not converging")
    print("  2.5-4.0 → learning, may need more steps")
    print("  1.5-2.5 → good alignment, ready for Stage B")
    print("  < 1.5   → excellent conditioning on audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage A (adapter + LoRA)")
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--val-csv", default=VAL_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="Use fp16 + gradient checkpointing (recommended for Docker/CPU)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to Stage A checkpoint (.pt) to resume from",
    )
    args = parser.parse_args()

    if args.batch_size is None:
        device = get_device()
        args.batch_size = BATCH_SIZE if device.type == "cuda" else 2

    return args


if __name__ == "__main__":
    train(parse_args())
