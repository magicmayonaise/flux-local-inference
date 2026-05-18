"""Minimal Gradio web UI for FLUX.1 [schnell] with streaming batch generation.

Run with:
    uv run python -m src.ui

Default URL: http://127.0.0.1:7860/

UX shape (Grok-style): one prompt, N seeds, images appear in a gallery as
each generation completes. The inference function is a Python generator;
Gradio wires successive `yield`s into the gallery + status outputs so the
user sees progress instead of staring at a spinner for ~45s per image.

The pipeline is loaded once at module import; every click reuses the same
in-memory pipeline. Defaults match `python -m src.generate` so the UI and
CLI produce identical images for the same prompt+seed.
"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

# Must run before any HF/diffusers import so HF_TOKEN propagates.
load_dotenv()

from src.config import InferenceConfig  # noqa: E402
from src.history import save_image_with_metadata, ui_filename  # noqa: E402
from src.pipeline import FluxGenerator  # noqa: E402

# Loaded once at module import. Gradio's queue serialises requests so we
# only need one pipeline instance for the single-GPU case.
_config = InferenceConfig()
_generator = FluxGenerator(_config)


def infer_stream(prompt: str, base_seed: float, n_images: float):
    """Generator: yields (gallery_items, status_md) as each image finishes.

    Gradio recognises generator functions and pipes each yield into the
    bound outputs - this is what gives the streaming UI feel without any
    explicit websockets / async on our side.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        yield [], "Enter a prompt above and click **Generate**."
        return

    n = max(1, int(n_images))
    base = int(base_seed)
    images: list[tuple] = []
    total_elapsed = 0.0
    peak_vram_max = 0.0

    yield images, f"Starting **{n}** generation{'s' if n > 1 else ''}…"

    for i in range(n):
        seed = base + i
        img, m = _generator.generate_with_metrics(prompt, seed=seed)
        # Persist each generation: PNG with embedded metadata + JSONL manifest line.
        save_image_with_metadata(
            img,
            _config.output_dir / ui_filename(seed),
            prompt=prompt,
            seed=seed,
            metrics=m,
            config=_config,
        )
        # Gallery accepts (image, caption) tuples - caption shows under each tile.
        images.append((img, f"seed {seed}  ·  {m['elapsed_s']:.1f}s"))
        total_elapsed += m["elapsed_s"]
        peak_vram_max = max(peak_vram_max, m["peak_vram_gb"])
        status = (
            f"**{i + 1}/{n}**  ·  "
            f"**Elapsed:** {total_elapsed:.1f} s  ·  "
            f"**Peak VRAM:** {peak_vram_max:.2f} GB  ·  "
            f"**Seeds:** {base}..{base + i}  ·  "
            f"**Resolution:** {_config.width}×{_config.height}  ·  "
            f"**Steps:** {_config.num_inference_steps}"
        )
        yield images, status


with gr.Blocks(title="flux-local-inference") as demo:
    gr.Markdown(
        "# flux-local-inference\n"
        "FLUX.1 [schnell] running locally on an 8 GB RTX 2070. "
        "Pick a batch size, hit **Generate**; images appear in the gallery "
        "as each one finishes. Each takes ~45 s "
        f"({_config.width}×{_config.height}, {_config.num_inference_steps} "
        "distilled steps, sequential CPU offload)."
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
            with gr.Row():
                seed_box = gr.Number(
                    label="Starting seed",
                    value=_config.seed,
                    precision=0,
                )
                n_slider = gr.Slider(
                    label="Number of images",
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                )
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Outputs",
                columns=2,
                height=600,
                show_label=True,
            )
            info_md = gr.Markdown()

    go.click(
        infer_stream,
        inputs=[prompt_box, seed_box, n_slider],
        outputs=[gallery, info_md],
    )
    # Enter inside the textbox also submits.
    prompt_box.submit(
        infer_stream,
        inputs=[prompt_box, seed_box, n_slider],
        outputs=[gallery, info_md],
    )


if __name__ == "__main__":
    # 0.0.0.0 binds to every interface inside the WSL2 VM. That's what lets the
    # Windows host reach http://127.0.0.1:7860/ through WSL's automatic localhost
    # forwarding - binding to 127.0.0.1 inside WSL means "WSL-internal only" and
    # Windows-localhost can't reach it reliably. Windows Defender Firewall still
    # blocks inbound LAN traffic by default, so this stays a localhost-only UI
    # from the user's perspective.
    # inbrowser=True opens the system default browser at the right URL.
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
