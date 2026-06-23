#!/usr/bin/env python3
"""Train Custom Speech Encoder v2 on LJSpeech."""

from __future__ import annotations

import argparse
import math
import os
import tarfile
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Audio ──────────────────────────────────────────────────────────────────
MEL_SR = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0
MAX_DURATION = 10.0
MAX_MEL_LEN = int(MAX_DURATION * MEL_SR / HOP_LENGTH)

# ── Encoder architecture ───────────────────────────────────────────────────
HIDDEN_DIM = 768
ENCODER_HEADS = 12
ENCODER_LAYERS = 6
FFN_DIM = 3072
DROPOUT = 0.1

# ── CTC head ───────────────────────────────────────────────────────────────
CHARS = list(" abcdefghijklmnopqrstuvwxyz',-.")
VOCAB = {c: i + 1 for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 1

# ── Default training ───────────────────────────────────────────────────────
BATCH_SIZE = 16
LR = 3e-4
WARMUP_STEPS = 500
MAX_STEPS = 20000
GRAD_ACCUM = 2
SAVE_EVERY = 2000
LOG_EVERY = 100

W_RECON = 0.4
W_CTC = 0.4
W_CONTRAST = 0.2
TEMP = 0.07

DATA_ROOT = "LJSpeech-1.1"
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
OUTPUT_DIR = "encoder_v2_checkpoints"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_fp16(device: torch.device) -> bool:
    return device.type == "cuda"


def ensure_dataset(data_root: str, tar_path: str = "LJSpeech-1.1.tar.bz2") -> None:
    root = Path(data_root)
    if root.exists():
        print(f"LJSpeech already present at {data_root}")
        return

    archive = Path(tar_path)
    if not archive.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected {data_root}/ or {tar_path}"
        )

    print(f"Extracting {tar_path} ...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall()
    print("Extraction complete.")


def prepare_manifests(data_root: str, train_csv: str, val_csv: str) -> tuple[int, int]:
    if Path(train_csv).exists() and Path(val_csv).exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        print(f"Loaded existing splits: train={len(train_df)}, val={len(val_df)}")
        return len(train_df), len(val_df)

    df = pd.read_csv(
        f"{data_root}/metadata.csv",
        sep="|",
        header=None,
        names=["id", "transcript", "normalized"],
    )
    df["wav_path"] = f"{data_root}/wavs/" + df["id"] + ".wav"
    df["text"] = df["normalized"].str.lower().str.strip()
    df["text_len"] = df["text"].str.len()
    df = df[(df["text_len"] >= 5) & (df["text_len"] <= 190)]

    train_val, _test = train_test_split(df, test_size=0.10, random_state=42)
    train, val = train_test_split(train_val, test_size=0.111, random_state=42)

    train[["id", "wav_path", "text"]].to_csv(train_csv, index=False)
    val[["id", "wav_path", "text"]].to_csv(val_csv, index=False)
    print(f"Created splits: train={len(train)}, val={len(val)}")
    return len(train), len(val)


mel_transform = T.MelSpectrogram(
    sample_rate=MEL_SR,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=1.0,
)


def wav_to_mel(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    if orig_sr != MEL_SR:
        wav = T.Resample(orig_sr, MEL_SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    mel = mel_transform(wav).squeeze(0)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    t_frames = mel.shape[1]
    if t_frames > MAX_MEL_LEN:
        mel = mel[:, :MAX_MEL_LEN]
    elif t_frames < MAX_MEL_LEN:
        pad = MAX_MEL_LEN - t_frames
        mel = nn.functional.pad(mel, (0, pad), value=-11.5)
    return mel


def augment_mel(mel: torch.Tensor) -> torch.Tensor:
    mel = mel.clone()
    t_len = mel.shape[-1]
    t_mask = max(1, int(0.10 * t_len))
    t0 = torch.randint(0, max(1, t_len - t_mask), (1,)).item()
    mel[:, t0 : t0 + t_mask] = -11.5

    f_mask = torch.randint(1, 16, (1,)).item()
    f0 = torch.randint(0, max(1, N_MELS - f_mask), (1,)).item()
    mel[f0 : f0 + f_mask, :] = -11.5
    return mel


def text_to_ids(text: str) -> list[int]:
    return [VOCAB[c] for c in text.lower() if c in VOCAB]


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    return wav, sr


class LJSpeechEncoderDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav, sr = load_wav(row["wav_path"])
        mel = wav_to_mel(wav, sr)
        mel_aug = augment_mel(mel)
        label_ids = text_to_ids(str(row["text"]))
        return {
            "mel": mel,
            "mel_aug": mel_aug,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "label_len": len(label_ids),
        }


def collate_fn(batch: list[dict]):
    mels = torch.stack([b["mel"] for b in batch])
    mels_aug = torch.stack([b["mel_aug"] for b in batch])
    label_ids = nn.utils.rnn.pad_sequence(
        [b["label_ids"] for b in batch], batch_first=True, padding_value=0
    )
    label_lens = torch.tensor([b["label_len"] for b in batch], dtype=torch.long)
    return mels, mels_aug, label_ids, label_lens


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


class ReconDecoder(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.detach().permute(0, 2, 1)
        x = self.upsample(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.proj(x)


class CTCHead(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class SpeechEncoderTraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomSpeechEncoder()
        self.decoder = ReconDecoder()
        self.ctc_head = CTCHead()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.recon_loss = nn.MSELoss()

    def forward(self, mel, mel_aug, label_ids, label_lens):
        batch_size, _, t_orig = mel.shape
        z = self.encoder(mel)
        z_aug = self.encoder(mel_aug)
        t_enc = z.size(1)

        mel_hat = self.decoder(z)
        min_t = min(mel_hat.shape[-1], t_orig)
        loss_recon = self.recon_loss(mel_hat[:, :, :min_t], mel[:, :, :min_t])

        log_probs = self.ctc_head(z).log_softmax(-1)
        log_probs_t = log_probs.permute(1, 0, 2)
        input_lens = torch.full((batch_size,), t_enc, dtype=torch.long, device=mel.device)
        # CTC is not implemented on MPS; run on CPU and move loss back.
        loss_ctc = self.ctc_loss(
            log_probs_t.cpu(),
            label_ids.cpu(),
            input_lens.cpu(),
            label_lens.cpu(),
        ).to(z.device)

        z_pool = z.mean(dim=1)
        z_aug_pool = z_aug.mean(dim=1)
        z_n = nn.functional.normalize(z_pool, dim=-1)
        z_aug_n = nn.functional.normalize(z_aug_pool, dim=-1)
        sim = torch.matmul(z_n, z_aug_n.T) / TEMP
        labels_c = torch.arange(batch_size, device=mel.device)
        loss_contrast = (
            nn.functional.cross_entropy(sim, labels_c)
            + nn.functional.cross_entropy(sim.T, labels_c)
        ) / 2.0

        loss = W_RECON * loss_recon + W_CTC * loss_ctc + W_CONTRAST * loss_contrast
        return loss, {
            "total": loss.item(),
            "recon": loss_recon.item(),
            "ctc": loss_ctc.item(),
            "contrast": loss_contrast.item(),
        }


def get_lr(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


def validate(model, val_loader, device, fp16: bool) -> dict[str, float]:
    model.eval()
    val_losses = {k: 0.0 for k in ["total", "recon", "ctc", "contrast"]}
    n_val = 0
    with torch.no_grad():
        for vbatch in val_loader:
            vm, vma, vli, vll = [x.to(device) for x in vbatch]
            with torch.autocast(device_type=device.type, enabled=fp16):
                _, vd = model(vm, vma, vli, vll)
            for k in val_losses:
                val_losses[k] += vd[k]
            n_val += 1
    for k in val_losses:
        val_losses[k] /= max(1, n_val)
    model.train()
    return val_losses


def save_checkpoint(
    path: str,
    step: int,
    model: SpeechEncoderTraining,
    optimizer,
    scheduler,
    val_loss: float,
    best_val_loss: float,
) -> None:
    ckpt = {
        "step": step,
        "best_val_loss": best_val_loss,
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "ctc_head_state": model.ctc_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: SpeechEncoderTraining,
    optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.decoder.load_state_dict(ckpt["decoder_state"])
    model.ctc_head.load_state_dict(ckpt["ctc_head_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    step = int(ckpt["step"])
    best_val_loss = float(ckpt.get("best_val_loss", ckpt["val_loss"]))
    print(f"Resumed from {path} at step {step} (val_loss={ckpt['val_loss']:.4f})")
    return step, best_val_loss


def export_encoder(checkpoint_path: str, output_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = CustomSpeechEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    torch.save(encoder.state_dict(), output_path)
    print(f"Encoder exported to {output_path}")


def train(args: argparse.Namespace) -> None:
    device = get_device()
    fp16 = use_fp16(device) and not args.no_fp16
    num_workers = 0 if device.type in {"mps", "cpu"} else args.num_workers

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FP16: {fp16} | batch_size={args.batch_size} | max_steps={args.max_steps}")

    ensure_dataset(args.data_root)
    prepare_manifests(args.data_root, args.train_csv, args.val_csv)

    train_loader = DataLoader(
        LJSpeechEncoderDataset(args.train_csv),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        LJSpeechEncoderDataset(args.val_csv),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = SpeechEncoderTraining().to(device)
    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Encoder params: {enc_params:.1f}M | total trainable: {total_params:.1f}M")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr(step, args.warmup_steps, args.max_steps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{args.output_dir}/tb_logs")

    step = 0
    accum_step = 0
    best_val_loss = float("inf")
    running = {k: 0.0 for k in ["total", "recon", "ctc", "contrast"]}

    if args.resume:
        step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        pbar = tqdm(total=args.max_steps, initial=step, desc="train", unit="step")
    else:
        pbar = tqdm(total=args.max_steps, desc="train", unit="step")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"Starting training for {args.max_steps} steps...")
    if args.resume:
        print(f"Resuming from step {step}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            mel, mel_aug, label_ids, label_lens = [x.to(device) for x in batch]

            with torch.autocast(device_type=device.type, enabled=fp16):
                loss, loss_dict = model(mel, mel_aug, label_ids, label_lens)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_step += 1

            for k in running:
                running[k] += loss_dict[k] / args.grad_accum

            if accum_step == args.grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                step += 1
                pbar.update(1)

                if step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    for k, v in running.items():
                        writer.add_scalar(f"train/{k}", v / args.log_every, step)
                    pbar.set_postfix(
                        loss=f"{running['total']/args.log_every:.4f}",
                        ctc=f"{running['ctc']/args.log_every:.4f}",
                        lr=f"{lr_now:.2e}",
                    )
                    running = {k: 0.0 for k in running}

                if step % args.save_every == 0 or step == args.max_steps:
                    val_losses = validate(model, val_loader, device, fp16)
                    for k, v in val_losses.items():
                        writer.add_scalar(f"val/{k}", v, step)
                    print(
                        f"\n>>> VAL step {step} | total={val_losses['total']:.4f} "
                        f"recon={val_losses['recon']:.4f} ctc={val_losses['ctc']:.4f} "
                        f"contrast={val_losses['contrast']:.4f}"
                    )

                    ckpt_path = f"{args.output_dir}/ckpt_step{step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        step,
                        model,
                        optimizer,
                        scheduler,
                        val_losses["total"],
                        best_val_loss,
                    )

                    if val_losses["total"] < best_val_loss:
                        best_val_loss = val_losses["total"]
                        save_checkpoint(
                            f"{args.output_dir}/best.pt",
                            step,
                            model,
                            optimizer,
                            scheduler,
                            val_losses["total"],
                            best_val_loss,
                        )
                        print(f"  New best val loss: {best_val_loss:.4f}")

    pbar.close()
    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    best_path = f"{args.output_dir}/best.pt"
    if Path(best_path).exists():
        export_encoder(
            best_path,
            f"{args.output_dir}/custom_encoder_v2.pt",
            device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Custom Speech Encoder v2")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--val-csv", default=VAL_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint (.pt) to resume training from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
