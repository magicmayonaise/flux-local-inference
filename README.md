# flux-local-inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5+cu121](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested on RTX 2070](https://img.shields.io/badge/Tested%20on-RTX%202070-76B900.svg?logo=nvidia&logoColor=white)](#hardware-tested)

A reproducible local inference pipeline for [FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) on consumer hardware, built as part of a transition from a neuroscience PhD into ML engineering. The repo is small on purpose — it is meant to demonstrate clean engineering practice (configuration discipline, pre-flight checks, mock-tested pipeline init, atomic commits) on a problem with real constraints (an 8 GB Turing GPU, no native bf16, no FP8), rather than to be a feature-rich product.

## Why FLUX.1 [schnell]

- **Apache 2.0 licence.** The dev variant ships under a non-commercial research licence; schnell is the only FLUX model I can use in a public portfolio without friction.
- **Timestep distilled to 4 steps.** A full FLUX-dev generation at this resolution is 50+ steps. The schnell distillation is what makes 8 GB Turing viable at all.
- **Consumer-hardware viable.** 8 GB VRAM is below the comfortable threshold for FLUX (the canonical examples assume 24 GB). The pipeline therefore leans on sequential CPU offload + VAE slicing/tiling, and the wall-clock numbers in the table below reflect that.
- **Turing dtype constraint is real and load-bearing.** Compute capability 7.5 has no native bf16 tensor cores. The pipeline loads weights in bf16 (matches the on-disk checkpoint) but casts to fp16 *after* enabling offload, so compute uses fp16 tensor cores. Reversing that order is the most common silent failure mode on this hardware.

## Hardware tested

| field | value |
| ----- | ----- |
| GPU | NVIDIA GeForce RTX 2070 |
| VRAM | 8192 MiB |
| Driver | 591.86 (Windows host) |
| CUDA runtime | 13.1 (driver) / 12.1 (torch wheel via forward compat) |
| OS | Windows 11 + WSL2 Ubuntu 24.04 |
| Torch | 2.5.1+cu121 |

Measured on this hardware (numbers from a `scripts/benchmark.py` run; the file itself is gitignored, the values are committed):

| metric | value |
| ------ | ----- |
| Resolution | 1360 × 768 |
| Inference steps | 4 |
| Offload strategy | sequential CPU offload |
| Cold start (load + first gen) | 46.4 s |
| Cold gen only | 45.2 s |
| Warm gen (avg of 2) | 45.5 s |
| Peak VRAM | 1.80 GB |
| Per-step time | ~10.3 s (transformer forward, dominated by PCIe offload traffic) |
| Output size | ~717 KB per PNG |

A note on these numbers: cold and warm are nearly identical (46.4 s vs 45.5 s) because after the first model load the OS page cache holds the weights in RAM — `from_pretrained` returns in under a second on subsequent invocations within a session. The "cold tax" you actually feel is only on the very first generation after a fresh boot. Peak VRAM at 1.80 GB on an 8 GB card means there is real headroom for higher resolutions; the limiting factor on this hardware is wall-clock from the PCIe-bound offload, not memory.

## Quick start

1. **Pre-flight environment check** — fails fast if anything (Python, CUDA, VRAM, disk, imports) is misconfigured:

   ```sh
   uv run python scripts/check_environment.py
   ```

2. **Hugging Face login** — FLUX.1-schnell is a gated repo. Open <https://huggingface.co/black-forest-labs/FLUX.1-schnell>, click *Agree and access repository*, then mint a Read token at <https://huggingface.co/settings/tokens> and:

   ```sh
   huggingface-cli login
   ```

3. **Download weights** (~24 GB, one time):

   ```sh
   uv run python scripts/download_model.py
   ```

4. **Smoke test** — minimal 512×512 / 1-step generation that verifies the pipeline plumbing:

   ```sh
   uv run python scripts/smoke_test.py
   ```

5. **Generate** — the CLI, optionally with a LoRA via `--lora REPO --lora-scale FLOAT`:

   ```sh
   uv run python -m src.generate --prompt "a coronal section of mouse hippocampus, brightfield microscopy"

   # with a LoRA (compatibility with schnell varies; see Limitations):
   uv run python -m src.generate \
     --prompt "a futuristic city, anime style" \
     --lora alimama-creative/FLUX.1-Turbo-Alpha --lora-scale 0.8
   ```

6. **Benchmark** — three generations at default config; writes `outputs/benchmark.md`:

   ```sh
   uv run python scripts/benchmark.py
   ```

7. **Web UI (optional)** — Gradio app at <http://127.0.0.1:7860/> with a prompt textbox, starting-seed input, batch-size slider (1–8), LoRA textbox + scale slider, and a streaming gallery. Hit Generate and images appear one-by-one as each finishes (seeds increment from your starting seed). Every generation is auto-saved to `outputs/ui_<timestamp>_seed<N>.png` with the prompt and seed embedded directly in the PNG (tEXt chunks) plus an append-only `outputs/history.jsonl` manifest line. Pipeline loads once at startup; the cold-load cost is paid on app boot, not per click:

   ```sh
   uv run python -m src.ui
   ```

## Architecture notes

I chose [diffusers](https://github.com/huggingface/diffusers) over [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for this repo for one specific reason: diffusers is a *library*, ComfyUI is an *application*. For a portfolio piece intended to demonstrate ML-engineering literacy, the library route lets me show that I understand the pipeline mechanics (load order, dtype boundaries, offload hooks) rather than how to wire up a node graph. ComfyUI is the right tool if you want a fast iteration loop on prompts and LoRA stacks; it's the wrong tool for showing you can read the diffusers source.

The single non-obvious design decision is in `src/pipeline.py`: the order of `from_pretrained → enable_sequential_cpu_offload → vae slicing → cast to fp16`. Each step is commented in-line with the reason it has to happen when it does. The mock-based test in `tests/test_pipeline.py` exists specifically to lock this order in — if someone refactors the constructor and reorders the dtype cast above the offload hook, that test fails before the pipeline ever sees real weights.

## Hybrid local → cloud workflow

8 GB caps what I can do locally. Anything bigger (Qwen-Image-2512, FLUX-dev at higher resolutions, anything involving training) gets a one-off RunPod box. The decision criteria, current pricing, and exact commands are in [`SETUP.md`](SETUP.md).

## Limitations and honest trade-offs

- **Throughput is bad and that's fine.** Warm generation on this hardware is ~60–90 s/image. A 4090 would be 8–10×. For a portfolio piece this is *the point* — the engineering is in making 8 GB work at all, not in beating the wall-clock of a real workstation.
- **bf16 emulation is wasted compute.** Loading in bf16 then casting to fp16 means we briefly hold both copies in CPU RAM during the cast. On a 64 GB host that's invisible; on a 16 GB laptop it would matter. I could load directly in fp16 to skip the cast, at the cost of a non-bit-identical checkpoint — I chose the bit-identical path for reproducibility.
- **LoRA support is off-the-shelf, not custom.** The pipeline supports loading a single LoRA via the CLI (`--lora REPO --lora-scale FLOAT`) or the UI textbox. Most public FLUX LoRAs target FLUX-dev and have variable compatibility with schnell at 4 steps — quality ranges from clean to garbage. Training a custom LoRA on neuro imagery (Future work, below) is the right answer if I want guaranteed-quality behaviour.
- **VAE tiling visible at low resolutions.** At 512×512 with both slicing and tiling on, you can occasionally see seams in flat-coloured regions. The default 768×1360 hides this; the smoke test runs at 512 anyway, where the goal is "valid PNG" not "good image".
- **CPU offload latency is PCIe-bound.** On a Gen3 x16 slot, ~16 GB/s effective. A Gen4 board would not change the VRAM footprint but would cut the per-step overhead roughly in half. I can't test this; if you're reviewing this on a Gen4 system, expect numbers to improve over what's in the table.

## Future work

- LoRA fine-tune (rank 16) on a small curated set of confocal calcium-imaging stills, validating the hypothesis that the FLUX prior already encodes "fluorescent microscopy" semantically and only needs style adjustment.
- CLIP-score evaluation harness using SigLIP (not in FLUX training) for quantitative prompt-following measurement across seed sweeps.
- Compute-budget Pareto sweep: 1/2/4/8 steps × {512, 768, 1024} resolution → wall-clock × CLIP-score chart. Validates the schnell-paper claim that 4 steps is near-optimal.
- Port to ComfyUI as a comparison piece, with measured throughput against the library implementation here.

## Project layout

```
flux-local-inference/
├── notebooks/01_exploration.ipynb   exploratory notebook with neuro-flavoured prompts and seed sweep
├── scripts/                         standalone entry points
│   ├── check_environment.py         pre-flight: Python/CUDA/VRAM/disk/HF auth
│   ├── download_model.py            HF snapshot download with auth-error mapping
│   ├── smoke_test.py                cheapest valid end-to-end generation (gate)
│   └── benchmark.py                 cold/warm timings; writes outputs/benchmark.md
├── src/
│   ├── config.py                    InferenceConfig — Turing-aware defaults with rationale
│   ├── pipeline.py                  FluxGenerator — load order documented in comments
│   ├── generate.py                  CLI entry point (python -m src.generate)
│   └── ui.py                        Gradio web UI (python -m src.ui)
└── tests/                           mock-based; no real model loads
```

## Licence

This repository itself is MIT. The FLUX.1 [schnell] weights are Apache 2.0 (Black Forest Labs).
