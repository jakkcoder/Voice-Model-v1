#!/usr/bin/env python3
"""Stage A inference: transcribe WAV files with encoder + adapter + LoRA."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from train_stage_a import (
    MAX_TEXT_LEN,
    CustomSpeechEncoder,
    ModalityAdapter,
    get_device,
    load_encoder,
    load_llm,
    load_wav,
    resolve_encoder_path,
    wav_to_mel_cpu,
)


def load_stage_a_checkpoint(
    path: str | Path,
    adapter: ModalityAdapter,
    llm,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state"])
    llm.load_state_dict(ckpt["lora_state"])
    step = int(ckpt.get("step", 0))
    val_loss = float(ckpt.get("val_loss", float("nan")))
    print(f"Loaded Stage A checkpoint: {path}")
    print(f"  step={step}  val_loss={val_loss:.4f}")
    return step, val_loss


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\",.\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1] / len(ref)


@torch.no_grad()
def transcribe(
    encoder: CustomSpeechEncoder,
    adapter: ModalityAdapter,
    llm,
    tokenizer,
    mel: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = MAX_TEXT_LEN,
) -> str:
    encoder.eval()
    adapter.eval()
    llm.eval()

    mel_batch = mel.unsqueeze(0).to(device)
    z = encoder(mel_batch)
    audio_tokens = adapter(z)
    text_dtype = next(llm.parameters()).dtype
    audio_tokens = audio_tokens.to(dtype=text_dtype)
    audio_mask = torch.ones(
        audio_tokens.shape[:2],
        dtype=torch.long,
        device=device,
    )

    generated = llm.generate(
        inputs_embeds=audio_tokens,
        attention_mask=audio_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def run_samples(
    encoder,
    adapter,
    llm,
    tokenizer,
    device: torch.device,
    csv_path: str,
    num_samples: int,
) -> list[dict]:
    df = pd.read_csv(csv_path).head(num_samples)
    rows: list[dict] = []
    for _, row in df.iterrows():
        wav_path = row["wav_path"]
        reference = str(row["text"])
        wav, sr = load_wav(wav_path)
        mel = wav_to_mel_cpu(wav, sr)
        hypothesis = transcribe(encoder, adapter, llm, tokenizer, mel, device)
        wer = word_error_rate(reference, hypothesis)
        rows.append(
            {
                "wav_path": wav_path,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
            }
        )
        print("-" * 72)
        print(f"WAV: {wav_path}")
        print(f"REF: {reference}")
        print(f"HYP: {hypothesis}")
        print(f"WER: {wer:.1%}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with Stage A model")
    parser.add_argument(
        "--checkpoint",
        default="stage_a_checkpoints/best.pt",
        help="Stage A checkpoint (adapter + LoRA)",
    )
    parser.add_argument("--wav", default=None, help="Single WAV path to transcribe")
    parser.add_argument("--val-csv", default="val.csv")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_TEXT_LEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    encoder_path = resolve_encoder_path(args.encoder_path)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tokenizer, llm = load_llm(device, fp16=False, low_mem=False)
    adapter = ModalityAdapter().to(device)
    encoder = CustomSpeechEncoder().to(device)
    load_encoder(encoder, encoder_path, device)
    load_stage_a_checkpoint(ckpt_path, adapter, llm, device)

    if args.wav:
        wav, sr = load_wav(args.wav)
        mel = wav_to_mel_cpu(wav, sr)
        text = transcribe(
            encoder,
            adapter,
            llm,
            tokenizer,
            mel,
            device,
            max_new_tokens=args.max_new_tokens,
        )
        print(text)
        return

    rows = run_samples(
        encoder,
        adapter,
        llm,
        tokenizer,
        device,
        args.val_csv,
        args.num_samples,
    )
    if rows:
        avg_wer = sum(r["wer"] for r in rows) / len(rows)
        print("=" * 72)
        print(f"Average WER over {len(rows)} samples: {avg_wer:.1%}")


if __name__ == "__main__":
    main()
