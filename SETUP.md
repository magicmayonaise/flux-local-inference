# Hybrid local → cloud workflow

The repo's default config works on 8 GB. Anything past that line — bigger models, training, batch inference at scale — runs cleaner on a one-off rented GPU. This document is the decision tree and the exact commands.

## When 8 GB stops being enough

Three signals that point to "rent a GPU":

1. **Model weights > 12 B.** FLUX.1 [schnell] is ~12 B and *barely* fits with sequential offload. Qwen-Image-2512 (~20 B FP8) doesn't fit at all without paging through CPU on every step, which inflates wall-clock to the point that local stops being faster than waking a remote box.
2. **Any training.** Even a rank-16 LoRA needs activation memory roughly equal to the inference footprint *plus* the gradient + optimizer state. 8 GB is already saturated by inference alone.
3. **Batch inference at scale.** 60–90 s per image on this hardware × 100 images = an evening. The same 100 images on a 4090 is ~10 minutes for ~$0.06.

## RunPod pricing (verified May 2026)

| GPU | VRAM | Community | Secure |
| --- | ---- | --------- | ------ |
| RTX 4090 | 24 GB | $0.34/hr | $0.59/hr |
| A100 | 80 GB | $0.89/hr | $1.89/hr |
| H100 | 80 GB | $2.49/hr | $2.69/hr |

Community pods are bid-priced and can be reclaimed; for an interactive iteration session of an hour or two, the savings are worth the small reliability hit. For an overnight training run, use Secure.

## Pick by workload

| Workload | Recommendation | Reason |
| -------- | -------------- | ------ |
| FLUX-schnell inference (the work in this repo) | local 2070 | works as-is, no cost |
| FLUX-dev inference (50 steps) | RTX 4090 24 GB Community | dev weights fit at 1024² without offload; ~4 s/image |
| Qwen-Image-2512 inference (~20 GB FP8) | RTX 4090 24 GB Community | fits the FP8 weights without offload, no Turing emulation tax |
| LoRA fine-tune on FLUX | A100 80 GB Community | gradients + optimizer state push past 24 GB |
| Anything multi-GPU or > 80 GB activations | H100 80 GB | only when actually needed |

## Cost math

Concrete example for the Qwen-Image-2512 path:

```
target:          10 portfolio images on Qwen-Image-2512
RTX 4090:        ~90 s/image cold, ~30 s/image warm
wall-clock:      90 + 9*30 = ~6 min generation
+ pod overhead:  ~5 min boot, ~2 min model load = ~13 min total
billed time:     ceil(13 min) = ~15 min @ $0.34/hr = $0.085
```

A 24 GB rental for ten Qwen images costs ~$0.08. The threshold at which this stops being a no-brainer is, roughly, "you're doing it less than once a month and value your time at less than $5/hr".

## Template launch (RunPod, 4090)

The diffusers pipeline in `src/pipeline.py` is the same on cloud as it is locally, with two differences:

1. **Override `model_repo`.** Override `InferenceConfig.model_repo` to point at the target (e.g. `"Qwen/Qwen-Image-2512"`).
2. **Drop the offload calls.** On 24 GB, `pipe.enable_sequential_cpu_offload()` is wasted work — comment it out and `pipe.to("cuda")` instead. Keep VAE tiling on; it's nearly free.

Bring-up sequence on a fresh pod:

```sh
# 1. clone
git clone <your-fork-url> ~/projects/flux-local-inference
cd ~/projects/flux-local-inference

# 2. install (uv is preinstalled on most RunPod PyTorch templates;
#    if not, curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync

# 3. HF login (paste your Read token)
huggingface-cli login

# 4. confirm CUDA + VRAM
uv run python scripts/check_environment.py
# FLUX_EXPECTED_GPU=4090 uv run python scripts/check_environment.py  # to silence the WARN

# 5. download whichever model
uv run python scripts/download_model.py

# 6. smoke + generate
uv run python scripts/smoke_test.py
uv run python -m src.generate --prompt "..." --height 1024 --width 1024
```

## Cleanup checklist

The single most important habit when renting GPUs: **stop the pod the moment you're done.**

- [ ] **Stop the pod** from the RunPod dashboard (not just "disconnect" — that keeps billing).
- [ ] **Detach or delete the network volume** if you used one. Volumes are billed separately ($0.07/GB/month for persistent storage); 100 GB attached but idle is ~$7/month bleeding silently.
- [ ] **Verify $0 active billing** on the RunPod billing page. Active spend should drop to zero within ~60 s of stopping the pod; if it doesn't, something is still running.
- [ ] **Pull artifacts down before stopping.** A stopped pod's local disk is wiped; anything not on a network volume or rsync'd off is gone. I use:

  ```sh
  rsync -avz --progress runpod:~/projects/flux-local-inference/outputs/ ./outputs-cloud/
  ```

  ...as the last command on the pod before stopping it.

The cleanup discipline here matters more than the per-hour price. The horror stories I've heard from people I know are all "I forgot the pod was running for three days" — which on an H100 is more than my entire 4090 budget for the year.
