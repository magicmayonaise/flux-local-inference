"""Minimal Gradio web UI for FLUX.1 [schnell] inference.

Run with:
    uv run python -m src.ui

Default URL: http://127.0.0.1:7860/

The pipeline is loaded once at module import; each "Generate" click reuses
the same in-memory pipeline (cold-load cost paid only at app boot). All
defaults match `python -m src.generate` so the UI and CLI give bit-identical
outputs for the same prompt + seed.

Intentionally minimal surface: prompt textbox, seed, Generate. For non-default
resolution or step count, use the CLI - keeping the UI lean is more honest about
what's been exercised end-to-end on this hardware.
"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

# Must run before any HF/diffusers import so HF_TOKEN is picked up.
load_dotenv()

from src.config import InferenceConfig  # noqa: E402
from src.pipeline import FluxGenerator  # noqa: E402

# Loaded once at module import. Gradio's queue serialises requests so we only
# need one pipeline instance for the single-GPU case.
_config = InferenceConfig()
_generator = FluxGenerator(_config)


def infer(prompt: str, seed: float) -> tuple:
    """Run one generation. Returns (PIL.Image | None, metrics_markdown)."""
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "Enter a prompt above and click **Generate**."

    image, metrics = _generator.generate_with_metrics(prompt, seed=int(seed))
    info = (
        f"**Elapsed:** {metrics['elapsed_s']:.1f} s  "
        f"·  **Peak VRAM:** {metrics['peak_vram_gb']:.2f} GB  "
        f"·  **Seed:** {int(seed)}  "
        f"·  **Resolution:** {_config.width}×{_config.height}  "
        f"·  **Steps:** {_config.num_inference_steps}"
    )
    return image, info


with gr.Blocks(title="flux-local-inference") as demo:
    gr.Markdown(
        "# flux-local-inference\n"
        "FLUX.1 [schnell] running locally on an 8 GB RTX 2070. "
        "Each generation takes ~45 seconds on this hardware "
        "(sequential CPU offload + fp16 compute, 4 distilled steps at "
        f"{_config.width}×{_config.height})."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder=(
                    "a fluorescence confocal microscopy image of cortical "
                    "pyramidal neurons expressing GFP, dendritic spines visible"
                ),
                lines=4,
            )
            seed_box = gr.Number(
                label="Seed",
                value=_config.seed,
                precision=0,
                info="Same seed + same prompt = same image (CPU generator, cross-machine reproducible).",
            )
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output", type="pil", height=512)
            info_md = gr.Markdown()

    go.click(
        infer,
        inputs=[prompt_box, seed_box],
        outputs=[output_image, info_md],
    )
    # Pressing Enter inside the textbox should also submit.
    prompt_box.submit(
        infer,
        inputs=[prompt_box, seed_box],
        outputs=[output_image, info_md],
    )


if __name__ == "__main__":
    # 127.0.0.1 (not 0.0.0.0) so the UI is only reachable from this machine.
    # inbrowser=True opens the system default browser at the right URL.
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
