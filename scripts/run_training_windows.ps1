# Voice-Model-v1 - Windows CUDA training (all artifacts on E: drive)
$ErrorActionPreference = "Continue"
$Root = "E:\Voice-Model-v1"
Set-Location $Root

$Python = "$Root\.conda-env\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Missing venv at $Python - run setup first."
}

# Store models, HF cache, and checkpoints on E: drive
$env:HF_HOME = "$Root\hf_cache"
$env:TRANSFORMERS_CACHE = "$Root\hf_cache"
$env:TORCH_HOME = "$Root\torch_cache"
$env:HF_HUB_CACHE = "$Root\hf_cache\hub"

New-Item -ItemType Directory -Force -Path `
    "$Root\encoder_v2_checkpoints", `
    "$Root\stage_a_checkpoints", `
    "$Root\hf_cache", `
    "$Root\torch_cache" | Out-Null

Write-Host "Device check:"
& $Python -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else 'N/A')"

$Stage = $args[0]
if (-not $Stage) { $Stage = "stage_a" }

switch ($Stage) {
    "encoder" {
        Write-Host "`n=== Stage 0: Speech Encoder ==="
        $encoderArgs = @()
        if ($args.Length -gt 1) { $encoderArgs = $args[1..($args.Length - 1)] }
        & $Python train_encoder.py @encoderArgs 2>&1 | Tee-Object -FilePath "$Root\encoder_training.log" -Append
    }
    "stage_a" {
        Write-Host "`n=== Stage A: Adapter + SmolLM2 LoRA ==="
        $Extra = @()
        if ($env:RESUME) {
            $Extra += "--resume", $env:RESUME
        } elseif (Test-Path "$Root\stage_a_checkpoints\best.pt") {
            $Extra += "--resume", "$Root\stage_a_checkpoints\best.pt"
            $Extra += "--reset-lr-schedule"
            $Extra += "--lr", "1e-4"
            $Extra += "--warmup-steps", "100"
            $Extra += "--max-steps", "12000"
            $Extra += "--early-stop-patience", "2"
            $Extra += "--target-val-loss", "2.5"
            Write-Host "Resuming finetune from best.pt (lr=1e-4, early-stop patience=2, target val 2.5 or below)"
        }
        $stageArgs = @()
        if ($args.Length -gt 1) { $stageArgs = $args[1..($args.Length - 1)] }
        & $Python train_stage_a.py `
            --output-dir "$Root\stage_a_checkpoints" `
            --batch-size 2 `
            --grad-accum 4 `
            --low-mem `
            @Extra @stageArgs 2>&1 | Tee-Object -FilePath "$Root\stage_a_training.log" -Append
    }
    default {
        Write-Host "Usage: .\scripts\run_training_windows.ps1 encoder|stage_a"
    }
}
