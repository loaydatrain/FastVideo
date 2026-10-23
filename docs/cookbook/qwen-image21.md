# Qwen-Image-2.1 recipes

<RecipeFamily family="qwen_image21" />

Choose the text-to-image or references-and-editing recipe on the model page.
The text-to-image YAML keeps prefix keys and values on CUDA; the Python
reference/editing example uses the default CPU cache. Both use one GPU with
CPU offload enabled. Replace `source.png` with your own image for editing.

The native FastVideo implementation includes text-to-image generation, image
editing, reference-to-image generation with 1–10 ordered images, and RGBA
generation and editing. Annotations and separate mask images use the same
ordered-reference path. Describe their meaning in the prompt.

Released-weight parity has passed on one NVIDIA GB200 for 1024×1024,
40-step text-to-image, plain image-to-image and instruction editing. With the
same SDPA policy in FastVideo and the pinned upstream implementation, all
40 denoiser predictions, scheduler updates, final latents and RGBA pixels
matched exactly. Dedicated transparency checks, 2K output and ten-reference
requests remain unverified. See the
[validation runbook](https://github.com/hao-ai-lab/FastVideo/blob/main/tests/local_tests/qwen_image21/README.md)
for the test commands and validation scope.

## Run on a GPU

Use one CUDA GPU, with `tp_size=sp_size=1`. By default, text layers and DiT
blocks offload to CPU and the VAE tiles. The T2I YAML uses CUDA prefix caches;
the Python reference/editing example uses CPU caches.

The following peaks were measured on one **NVIDIA GB200 per request**, using
BF16, 1024×1024 output, seed 42 and two full 40-step requests per process.
Measurements include initialization, deferred model loading, a cold request,
a warm request, PNG export and shutdown. Image cases use one original Qwen
reference image. Compilation is disabled. The offloaded rows explicitly select
`FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` and `FASTVIDEO_FA4=0`; the resident row
selects FA4 as described below. All figures are **GiB**.

| Configuration | KV cache | GPU allocated | GPU reserved | Device VRAM including driver | Host process-tree RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| T2I YAML, SDPA, offloaded, tiled | CUDA | 13.28 | 22.54 | 24.19 | 43.87 |
| Plain I2I Python, SDPA, offloaded, tiled | CPU | 13.29 | 16.11 | 17.76 | 48.29 |
| Editing Python, SDPA, offloaded, tiled | CPU | 13.29 | 16.11 | 17.76 | 49.25 |
| T2I, FA4, resident models, untiled VAE | CUDA | 32.77 | 35.50 | 37.09 | 13.12 |

GPU allocated/reserved peaks use PyTorch counters across all stages. Device
VRAM is sampled with NVML and includes the GB200's 0.72 GiB driver reservation;
the corresponding peak used-memory readings are 23.47 / 17.03 / 17.03 / 36.36 GiB.
Host RAM is the sampled simultaneous RSS of the process tree, including model
loading and output validation. Device and RSS sampling targets 20 ms; brief
transients can fall between samples. These process measurements do not include
all operating-system memory or establish a minimum-capacity requirement.
Minimum hardware VRAM remains **unknown**; operation on an actual 24 GB GPU
has not been verified. Allow host RAM headroom beyond the observed process RSS.

The resident FA4 row uses a separate performance configuration with offload and
VAE tiling disabled; those settings increase GPU residency. Its memory run is
not a controlled latency comparison. Use a PyTorch CUDA build compatible with
your GPU. CPU transfers can limit throughput.

Provide a complete local Diffusers-format `Qwen/Qwen-Image-2.1` checkpoint, or
let the Hugging Face loader fetch it on the GPU machine. Model weights are
subject to Qwen's checkpoint license. The base checkpoint contains the processor,
Qwen3-VL encoder, DiT, VAE and scheduler. No conversion is needed.

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --model-path /path/to/Qwen-Image-2.1 \
  --prompt 'A red ceramic teapot, isolated on a transparent background.' \
  --output outputs/teapot.png
```

For a single-image edit:

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --model-path /path/to/Qwen-Image-2.1 \
  --reference source.png \
  --prompt 'Change the shirt to blue; retain the subject and composition.'
```

For reference-to-image or an annotation-guided edit, repeat `--reference` in the
order referred to by the prompt:

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --reference subject.png --reference style.png \
  --prompt 'Paint the subject from image 1 in the style of image 2.'
```

```bash
fastvideo generate --config examples/inference/basic/qwen_image21_edit.yaml \
  --request.inputs.references '["source.png", "mask.png"]' \
  --request.prompt 'In image 1, replace the area highlighted by image 2 with flowers.'
```

Guided mask edits follow the released model workflow. Pixels outside the mask
are not guaranteed to remain identical. The mask is supplied as a reference
image, alongside the source, with instructions in the prompt.

## Python API

The example above uses `VideoGenerator.from_config` and `GenerationRequest`.
`InputConfig.references` accepts local paths, PIL images, or uint8 image arrays.
The legacy API also supports `generate_video(prompt=..., references=[...])`.
For one reference, `image_path` or `pil_image` is a shortcut. Supply only one
of these input sources.

Outputs are four-channel PNGs. Reference alpha remains intact for VAE encoding;
only the vision encoder's copy is composited over white. Ask for a transparent
background in the prompt when you want the model to generate transparency.

Defaults are 1024×1024, 40 steps, one image and guidance 1. Output dimensions and
`QwenImage21PipelineConfig.reference_resolution` must be positive multiples of 32.
References are resized to approximately `reference_resolution²` pixels while retaining their aspect
ratio. A 2K output can keep the reference resolution at 1024.

Set `true_cfg_scale > 1` with an explicit `negative_prompt` for two-branch CFG.
An empty string is an explicit negative prompt. `guidance_scale` stays at 1;
this checkpoint does not use embedded guidance.

Set `QwenImage21PipelineConfig(use_kv_cache=False)` to recompute prefixes each
step. In YAML, set `generator.pipeline.experimental.use_kv_cache: false`.
The default CPU cache avoids holding every reference prefix on the GPU; it
falls back to uncached execution if a host cache allocation fails. Request
caches are cleared after success or failure. GPU cache placement is available
through `QwenImage21PipelineConfig(kv_cache_device="cuda")` in a pipeline config
file, for hardware with enough VRAM.

## Optional FlashAttention 4

With the repository's pinned FA4 dependencies installed on supported hardware,
select the backend before starting Python:

```bash
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export FASTVIDEO_FA4=1
export FASTVIDEO_NVFP4_FA4=0
```

These flags select FA4 for dense reference-image and generated-image
attention; masked text attention uses SDPA. Compare numerical parity with the
same backend in both implementations: switching kernels can change generated
pixels. Automatic backend selection follows FastVideo's shared policy. Select
`FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` and `FASTVIDEO_FA4=0` to force SDPA.

## Scope

The first port is single GPU, batch one, single frame and unquantized. SP/TP,
custom attention processors, LoRA, prompt rewriting and HTTP serving are not
included. The prompt rewriter's separate 9B checkpoints can be integrated later.
The confirmed full-pipeline modes are the three listed above. Transparency,
2K output and ten-reference requests require additional validation.

The numerical source is [Diffusers at revision
578c9b2](https://github.com/huggingface/diffusers/tree/578c9b2c6636ab2424a0e56186268b83623656b2).
Runtime numerical models are native FastVideo classes. Transformers supplies
the processor/tokenizer; reference Diffusers model classes are used only by
parity tests.
