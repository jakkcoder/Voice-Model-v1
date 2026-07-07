#!/usr/bin/env python3
"""Stage A — interpret best.pt and run teacher-forced forward pass on a real clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from train_stage_a import (
    MAX_STEPS,
    MAX_TEXT_LEN,
    CustomSpeechEncoder,
    ModalityAdapter,
    TRAIN_CSV,
    build_forward_pass,
    get_device,
    load_encoder,
    load_llm,
    load_wav,
    resolve_encoder_path,
    use_autocast,
    wav_to_mel_cpu,
)

DEFAULT_BASELINE = 5.7490  # notebook reference (untrained single-sample loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage A best.pt")
    parser.add_argument(
        "--checkpoint",
        default="stage_a_checkpoints/best.pt",
        help="Stage A checkpoint path",
    )
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Untrained baseline loss (computed live if omitted)",
    )
    parser.add_argument("--encoder-path", default=None)
    return parser.parse_args()


@torch.no_grad()
def single_sample_loss(
    forward_pass,
    mel,
    text: str,
    tokenizer,
    device: torch.device,
    autocast: bool,
) -> float:
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_TEXT_LEN,
        truncation=True,
        padding="max_length",
    )
    ids = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device)
    with torch.autocast(device_type=device.type, enabled=autocast):
        return forward_pass(mel, ids, mask).item()


def main() -> None:
    args = parse_args()
    device = get_device()
    autocast = use_autocast(device, fp16=False)
    best_pt = Path(args.checkpoint)

    if not best_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_pt}")

    print(f"Device: {device}\n")

    tokenizer, llm = load_llm(device, fp16=False, low_mem=False)
    adapter = ModalityAdapter().to(device)
    encoder = CustomSpeechEncoder().to(device)
    load_encoder(encoder, resolve_encoder_path(args.encoder_path), device)

    forward_pass = build_forward_pass(encoder, adapter, llm, tokenizer)

    df = pd.read_csv(args.train_csv)
    sample = df.iloc[args.sample_idx]
    wav, sr = load_wav(sample["wav_path"])
    mel = wav_to_mel_cpu(wav, sr).unsqueeze(0).to(device)

    baseline = args.baseline
    if baseline is None:
        adapter_untrained = ModalityAdapter().to(device)
        fp_untrained = build_forward_pass(encoder, adapter_untrained, llm, tokenizer)
        baseline = single_sample_loss(
            fp_untrained,
            mel,
            str(sample["text"]),
            tokenizer,
            device,
            autocast,
        )

    best = torch.load(best_pt, map_location=device, weights_only=False)

    print("STAGE A TRAINING SUMMARY")
    print(f'  Best val loss  : {best["val_loss"]:.4f}')
    print(f'  Reached at step: {best["step"]} / {MAX_STEPS}')
    print()
    print("  Val loss interpretation:")
    print("  > 4.0   → adapter not converging")
    print("  2.5-4.0 → learning, needs more steps")
    print("  1.5-2.5 → good alignment, ready for Stage B")
    print("  < 1.5   → excellent")
    print()

    adapter.load_state_dict(best["adapter_state"])
    llm.load_state_dict(best["lora_state"])
    adapter.eval()
    llm.eval()
    print("Weights loaded into adapter and LLM ✓")
    print()
    print(f"  Sample: {sample['id']}")
    print(f"  WAV   : {sample['wav_path']}")
    print(f"  Text  : {sample['text'][:80]}{'...' if len(str(sample['text'])) > 80 else ''}")

    trained_loss = single_sample_loss(
        forward_pass,
        mel,
        str(sample["text"]),
        tokenizer,
        device,
        autocast,
    )

    reduction = (baseline - trained_loss) / baseline * 100 if baseline else 0.0

    print(f"\n  Single sample loss (trained)  : {trained_loss:.4f}")
    print(f"  Baseline loss  (untrained)    : {baseline:.4f}")
    print(f"  Reduction                     : {reduction:.1f}%")


if __name__ == "__main__":
    main()
