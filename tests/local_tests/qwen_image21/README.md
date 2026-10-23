# Qwen-Image-2.1 validation

Released-weight CUDA validation completed on an RTX 5090 using the local
fork installed editable in the existing environment. The production encoder,
DiT and VAE component gates pass without skips. Plain I2I and editing both
match upstream exactly at 1024 x 1024 and 40 steps and have been visually
confirmed. T2I and two-reference generation match exactly at four steps.
The requested non-layerwise-offload benchmark also passes, with a 42.44 s
generation request and 16.85 GiB sampled peak GPU usage for 40-step plain I2I.
With preloaded models, no DiT offload and CUDA prefix KV storage, two warm
40-step T2I requests averaged 15.38 s for encoder + DiT + decoder and 15.64 s
including PNG saving. Their four-step warmup exactly matched the validated
upstream baseline; the forty-step outputs matched each other.
Local-only `GPU_VALIDATION.md` and `PORT_STATUS.md` records contain confirmed
runs, commands, numerical metrics, log paths and remaining gates. These notes
and generated run artifacts are retained in developer workspaces and are not
part of the source checkout.

Local text evidence is stored under `outputs/qwen_image21/`. Images,
tensor dumps, weights and the upstream checkout are not included in this
pod-deletion archive. Reproduction requires downloading the pinned assets,
updating historical absolute paths and regenerating baseline tensors.

## CI regression tests

The maintained Qwen CI suite contains 12 cases in four files:

| File | Cases | CI lane |
|---|---:|---|
| `fastvideo/tests/transformers/test_qwen_image21_attention.py` | 4 | Transformer |
| `fastvideo/tests/stages/test_qwen_image21_stages.py` | 5 | Unit, through the existing stages directory |
| `fastvideo/tests/golden_gate/test_qwen_image21.py` | 1 | Golden gate |
| `fastvideo/tests/ssim/test_qwen_image21_similarity.py` | 2 | SSIM |

The first nine cases use native package imports, real inference batches and
small deterministic components. The transformer cases require a CUDA runtime.
Run these before the checkpoint-backed tests:

```bash
python -m pytest fastvideo/tests/stages/test_qwen_image21_stages.py \
  fastvideo/tests/transformers/test_qwen_image21_attention.py -xq
```

The golden loads one real BF16 transformer block and uses the shared device and
runtime fingerprint machinery. The SSIM file covers T2I and one reference-image
edit, comparing all RGBA channels through the existing PNG similarity helper.
Default SSIM requests use 512 x 512 and four steps; `--ssim-full-quality` selects
1024 x 1024 and 40 steps. Both use FlashAttention and the pinned Qwen checkpoint.
Select FA4 before Python starts, as below, so the runtime fingerprint matches.
Provide matching reviewed references before claiming either gate passes:

```bash
FASTVIDEO_FA4=1 python -m pytest \
  fastvideo/tests/golden_gate/test_qwen_image21.py -xq
FASTVIDEO_FA4=1 python -m pytest \
  fastvideo/tests/ssim/test_qwen_image21_similarity.py -xq
```

Independent component and full-pipeline upstream parity remain local and
opt-in. `tests/local_tests/` is excluded from CI. Earlier CPU porting scaffolds
have been consolidated into the package regressions; they are no longer part
of the PR. Setup and local parity commands follow below.

## Prepare an allocated CUDA machine

The initial target is one 24 GB CUDA GPU and at least 96 GB host RAM. Use
`tp_size=sp_size=1`, BF16 components, CPU text and layerwise DiT offload, VAE
tiling, deferred module loading and CPU prefix caches. The 24 GB bound is a
target until measured by the GPU tests.

Install FastVideo following the repository's CUDA installation guide and
existing `pyproject.toml` constraints: Transformers 5.15+, tokenizers below
0.23 and Accelerate 1.0.1. Main-compatible versions have passed 1024-square,
40-step native T2I, plain I2I and editing comparisons against fresh pinned
upstream outputs. A separate Transformers 5.18 / tokenizers 0.23.2 /
Accelerate 1.15 environment also passed; that validation environment explicitly
overrides the tokenizers/Accelerate constraints and does not change the package
requirements. Run the offline CLIP tokenizer regression after changing the
resolved environment.

Prepare the reference checkout and a complete local checkpoint on that machine:

```bash
git clone https://github.com/huggingface/diffusers.git /path/to/diffusers-reference
git -C /path/to/diffusers-reference checkout 578c9b2c6636ab2424a0e56186268b83623656b2
export QWEN_IMAGE21_DIFFUSERS_DIR=/path/to/diffusers-reference
export QWEN_IMAGE21_MODEL_ROOT=/path/to/Qwen-Image-2.1
export PYTHONPATH="$QWEN_IMAGE21_DIFFUSERS_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
```

The local model must contain `model_index.json`, `processor`, `text_encoder`,
`transformer`, `vae` and `scheduler`, including all weight shards. Acquire
the checkpoint under Qwen's license on the GPU machine. There is no conversion
script because native keys preserve the released checkpoint layout; the encoder
excludes only the unused language-model head.

## Component parity gates

Run these individually to keep failure diagnosis clear:

```bash
QWEN_IMAGE21_RUN_DIT_PARITY=1 python -m pytest \
  tests/local_tests/transformers/test_qwen_image21_transformer_parity.py -q

QWEN_IMAGE21_RUN_VAE_PARITY=1 python -m pytest \
  tests/local_tests/vaes/test_qwen_image21_vae_parity.py -q

QWEN_IMAGE21_RUN_ENCODER_PARITY=1 python -m pytest \
  tests/local_tests/encoders/test_qwen_image21_conditioner_parity.py -q
```

These use the production FastVideo component loaders and the official reference.
The DiT test checks every block, the final prediction and cached decode. The VAE
test checks posterior statistics and RGBA decode; random-weight reference tests
cover tiling. The encoder test checks text, long text, 1/2/10 images and padding,
with the reference final norm neutralized to expose the required pre-norm output.
The existing component tolerances have passed on the recorded RTX 5090 runs;
no tolerance was widened. Set `QWEN_IMAGE21_ENCODER_PARITY_REPORT` and
`QWEN_IMAGE21_DIT_PARITY_REPORT` to save numerical JSON diagnostics.

## Pipeline and output gates

```bash
QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 python -m pytest \
  tests/local_tests/qwen_image21/test_pipeline_parity.py -q

QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 QWEN_IMAGE21_RUN_2K_PARITY=1 \
  python -m pytest tests/local_tests/qwen_image21/test_pipeline_parity.py \
  -k t2i_2k -q
```

Each test runs native and reference pipelines in separate processes using
identical explicit BF16 initial latents, sigmas and cache settings. Synthetic
references are generated by default; `QWEN_IMAGE21_PARITY_CASES_JSON` can
override prompts and ordered paths with original-source images. Native
execution uses CPU KV caches; the reference uses its canonical cache behavior
with the same `use_kv_cache` value. Tests compare four-channel pixel
tensors, verify the exported PNG retains alpha, and check the native reported
memory peak when available. The 1024 cases cover T2I, plain I2I, edit, Ref2Img, ten refs,
RGBA generation/edit, annotations, separate mask images, true CFG and no cache.
The separate 2K case covers T2I only. `QWEN_IMAGE21_PARITY_DIAGNOSTICS=1`
additionally saves observational conditioning, scheduler and denoising tensors
to local run artifacts; it does not change model arithmetic or assertions.

Pipeline tests default to four steps for debugging. After these pass, repeat
with `QWEN_IMAGE21_PARITY_STEPS=40` and inspect generated images for text
rendering, subject identity, edit instructions and alpha edges. Capture peak
GPU memory and host RSS on the actual 24 GB machine. Do not claim mask exterior
pixel preservation: masks are semantic references in the released workflow.

For a CLI smoke run:

```bash
fastvideo generate --config examples/inference/basic/qwen_image21_t2i.yaml \
  --generator.model_path "$QWEN_IMAGE21_MODEL_ROOT"
```

Before release, run the changed-file pre-commit checks in the installed
development environment and calibrate parity tolerances from actual GPU results.
Add approved image/latent quality references only after inspecting real outputs.
