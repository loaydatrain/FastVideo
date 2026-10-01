# SPDX-License-Identifier: Apache-2.0
"""Opt-in, released-weight pipeline parity through the public FastVideo API.

Each implementation runs in its own process so their weights never overlap in
VRAM. Synthetic references need no access to a user's photo or media library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "578c9b2c6636ab2424a0e56186268b83623656b2"
PARITY_SCOPE = "pipeline"
CASES = ("t2i", "edit", "ref2img", "ten_refs", "rgba", "rgba_edit", "annotation", "mask_image", "true_cfg", "uncached")


def _worker(kind: str, spec_path: str, output_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    sys.path.insert(0, str(ROOT))
    noise = torch.load(spec["latents"], map_location="cpu", weights_only=True)
    if kind == "native":
        from fastvideo import VideoGenerator
        from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig,
                                   OutputConfig, ParallelismConfig, PipelineSelection, SamplingConfig)

        generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=spec["model"],
                engine=EngineConfig(num_gpus=1, parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                                    offload=OffloadConfig(dit=True, dit_layerwise=True, text_encoder=True,
                                                          vae=True, lazy_module_load=True)),
                pipeline=PipelineSelection(workload_type="i2i" if spec["refs"] else "t2i",
                                           preset="qwen_image21_edit" if spec["refs"] else "qwen_image21"),
            ))
        try:
            result = generator.generate(
                GenerationRequest(
                    prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
                    inputs=InputConfig(references=spec["refs"] or None, latents=noise),
                    sampling=SamplingConfig(height=spec["resolution"], width=spec["resolution"], num_frames=1,
                                            fps=1, num_inference_steps=spec["steps"], guidance_scale=1,
                                            true_cfg_scale=spec["true_cfg_scale"], seed=21, reference_resolution=1024,
                                            use_kv_cache=spec["cache"]),
                    output=OutputConfig(output_path=str(Path(output_path).with_suffix(".png")),
                                        return_frames=True, save_video=True),
                ))
            torch.save({"pixels": result.samples, "peak_memory_mb": result.peak_memory_mb}, output_path)
        finally:
            generator.shutdown()
    else:
        sys.path.insert(0, str(Path(spec["reference"]) / "src"))
        from diffusers.pipelines.qwenimage import pipeline_qwenimage21

        expected = Path(spec["reference"]) / "src/diffusers/pipelines/qwenimage/pipeline_qwenimage21.py"
        assert Path(pipeline_qwenimage21.__file__).resolve() == expected.resolve()
        pipe = pipeline_qwenimage21.QwenImage21Pipeline.from_pretrained(
            spec["model"], torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.vae.enable_tiling()
        pipe.enable_sequential_cpu_offload()
        references = [Image.open(path).convert("RGBA") for path in spec["refs"]] or None
        with torch.no_grad():
            output = pipe(prompt=spec["prompt"], image=references, height=spec["resolution"], width=spec["resolution"],
                          output_resolution=1024, num_inference_steps=spec["steps"],
                          negative_prompt=spec["negative_prompt"], true_cfg_scale=spec["true_cfg_scale"],
                          latents=noise.to(torch.bfloat16), output_type="pt", use_kv_cache=False)
        torch.save({"pixels": output.images.unsqueeze(2).float().cpu()}, output_path)


def _assets(tmp_path: Path) -> list[str]:
    files = []
    for index in range(10):
        image = Image.new("RGBA", (256, 256), (230, 235, 240, 0 if index == 0 else 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((48 + index, 48, 208, 208), fill=(30 + index * 20, 80, 140, 200))
        if index == 1:
            draw.rectangle((24, 24, 232, 232), outline="red", width=8)
        path = tmp_path / f"reference_{index + 1}.png"
        image.save(path)
        files.append(str(path))
    mask = Image.new("RGBA", (256, 256), "black")
    ImageDraw.Draw(mask).ellipse((48, 48, 208, 208), fill="white")
    path = tmp_path / "mask.png"
    mask.save(path)
    return files + [str(path)]


def _run_parity(case, resolution, tmp_path):
    if os.environ.get("QWEN_IMAGE21_RUN_PIPELINE_PARITY") != "1":
        pytest.skip("Set QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 on an allocated CUDA node")
    if not torch.cuda.is_available():
        pytest.fail("Qwen-Image-2.1 pipeline parity requires CUDA", pytrace=False)
    model = Path(os.environ.get("QWEN_IMAGE21_MODEL_ROOT", ROOT / "official_weights/Qwen-Image-2.1"))
    reference = Path(os.environ.get("QWEN_IMAGE21_DIFFUSERS_DIR", ROOT / "official_reference/diffusers"))
    if not (model / "model_index.json").is_file() or not (reference / "src").is_dir():
        pytest.fail("Provide QWEN_IMAGE21_MODEL_ROOT and the pinned QWEN_IMAGE21_DIFFUSERS_DIR", pytrace=False)
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION
    files = _assets(tmp_path)
    counts = dict(edit=1, ref2img=2, ten_refs=10, rgba_edit=1, annotation=2, mask_image=1, true_cfg=1)
    refs = files[:counts.get(case, 0)]
    prompt = "An elegant blue ceramic vase in a studio, with the word Qwen written on it."
    if refs:
        prompt = "Make a blue sculpture inspired by the shapes in the reference images."
    if case in ("rgba", "rgba_edit"):
        prompt = "A blue sphere isolated on a transparent background."
    if case == "annotation":
        prompt = "Change the object enclosed by the red annotation in image 2 into a blue sphere in image 1."
    if case == "mask_image":
        refs.append(files[-1])
        prompt = "Replace the white masked area of image 2 in image 1 with flowers."
    noise_path = tmp_path / "initial_latents.pt"
    torch.save(torch.randn(1, (resolution // 16)**2, 64, generator=torch.Generator().manual_seed(21),
                           dtype=torch.bfloat16), noise_path)
    spec = dict(model=str(model.resolve()), reference=str(reference.resolve()), prompt=prompt, refs=refs,
                resolution=resolution, steps=int(os.environ.get("QWEN_IMAGE21_PARITY_STEPS", "4")),
                negative_prompt="" if case == "true_cfg" else None,
                true_cfg_scale=2 if case == "true_cfg" else 1, cache=case != "uncached", latents=str(noise_path))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    outputs = []
    for kind in ("native", "reference"):
        output = tmp_path / f"{kind}.pt"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", kind, str(spec_path), str(output)],
                       check=True, cwd=ROOT)
        outputs.append(torch.load(output, map_location="cpu", weights_only=True))
    actual, expected = outputs[0]["pixels"], outputs[1]["pixels"]
    assert actual.shape == expected.shape == (1, 4, 1, resolution, resolution)
    assert torch.isfinite(actual).all() and bool(((actual >= 0) & (actual <= 1)).all())
    # Provisional tolerances: calibrate from measured BF16 CUDA evidence before release.
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
    native_png = Image.open(tmp_path / "native.png")
    assert native_png.mode == "RGBA"
    expected_u8 = (actual[0, :, 0].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).numpy()
    np.testing.assert_array_equal(np.array(native_png), expected_u8)
    if outputs[0]["peak_memory_mb"] is not None:
        assert outputs[0]["peak_memory_mb"] <= 24 * 1024


@pytest.mark.parametrize("case", CASES)
def test_qwen_image21_modes_1024_pipeline_parity(case, tmp_path):
    _run_parity(case, 1024, tmp_path)


def test_qwen_image21_t2i_2k_pipeline_parity(tmp_path):
    if os.environ.get("QWEN_IMAGE21_RUN_2K_PARITY") != "1":
        pytest.skip("Set QWEN_IMAGE21_RUN_2K_PARITY=1 for the separate 2K gate")
    if os.environ.get("QWEN_IMAGE21_RUN_PIPELINE_PARITY") != "1":
        pytest.fail("The 2K gate also requires QWEN_IMAGE21_RUN_PIPELINE_PARITY=1", pytrace=False)
    _run_parity("t2i", 2048, tmp_path)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("This helper is launched by pytest")
    _worker(sys.argv[2], sys.argv[3], sys.argv[4])
