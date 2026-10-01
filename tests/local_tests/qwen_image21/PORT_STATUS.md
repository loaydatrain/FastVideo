# Qwen-Image-2.1 port status

## Implemented

| Surface | Implementation | Validation so far |
| --- | --- | --- |
| DiT | Native 32-block single-stream transformer, segmented block-causal SDPA, RoPE, CPU/GPU prefix cache and safe host-allocation fallback | CPU contracts; 297 keys match HF metadata |
| RGBA VAE | Native single-frame encode/decode, posterior mode, latent statistics, slicing and overlap tiling | CPU contracts; 238 keys/shapes match reference metadata |
| Qwen3-VL | Native shared H3 graph, all 36 text layers, pre-final-norm tap, per-image vision execution | CPU contracts; 749 keys match HF metadata, excluding only `lm_head.weight` |
| Pipeline | Separate input/text/latent/schedule/denoise/decode stages, strict loading, registry and presets | CPU request-flow and scheduler checks; typed CLI examples parse |
| Inputs/outputs | Ordered 0–10 refs, edit/ref shortcuts, RGBA preservation, white vision compositing, four-channel PNG allocation | CPU alpha/layout checks; real PNG export gate written |
| API/CLI | `GenerationRequest`, legacy `generate_video`, two YAML presets and a Python example | CPU parser checks; production runtime smoke pending |

The CPU suite contains native component tests and pipeline tests with substituted
components. It does not prove production imports, released-weight parity,
offload memory use or rendered-image quality. Reference/GPU skips remain pending.

Final local verification: **62 passed, 22 skipped** in the documented pytest
command. All 35 changed/new Python files compile; TOML/YAML parsing and
`git diff --check` pass. The skipped cases require the installed tokenizer,
pinned reference checkout, CUDA or released weights.

Changed-file formatting, lint, spelling and Markdown hooks were run through
pre-commit with the repository's exclusions. The mypy environment could not be
installed because the Mac ran out of disk space; its check remains pending.

No model weights have been downloaded on this Mac. No GPU was used. The source
reference is Diffusers commit `578c9b2c6636ab2424a0e56186268b83623656b2`;
small official source/config/index files were inspected. Temporary source-isolated
reference experiments also found exact tiny VAE results and matching DiT control
flow, with independent normalization dtype checks. They are supplemental evidence,
not the production parity gates.

## Remaining GPU gates

1. Install the resolved dependency environment; run the CLIP tokenizer regression
   and the pending pre-commit mypy check.
2. Strictly load all released components through FastVideo and pass each
   official-reference component parity test without skips.
3. Run the 1024 pipeline cases, including true CFG, caching on/off and ten refs.
4. Verify public Python and CLI PNG output, including transparent generation
   and RGBA edits. Inspect 40-step output quality and calibrate test tolerances.
5. Measure peak memory with text/DiT/VAE offload and CPU KV caches on the planned
   24 GB GPU with >=96 GB host RAM. Validate interrupted-request cleanup there.
6. Run 2K T2I separately. Seed approved image/latent regression artifacts after
   review; none exist yet for this new model family.

Commands and environment variables are in [README.md](README.md).

## Explicit first-release limits

One GPU, one prompt/image per request, one frame, unquantized weights. SP/TP,
LoRA/custom attention processors, prompt rewriting and HTTP serving are outside
this implementation. The separate 9B rewrite checkpoints are deferred.

Annotations and masks are ordered reference images interpreted from the prompt.
The pipeline has no hard inpainting preservation constraint. The initial memory
gate does not include 2K generation with ten references.
