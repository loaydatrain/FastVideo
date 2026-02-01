#!/bin/bash
# Profile Wan2.1 inference with PyTorch profiler and per-stage timing
#
# Usage:
#   bash scripts/benchmark/profile_inference_wan.sh
#
# For nsys profiling, wrap with:
#   nsys profile --trace=cuda,nvtx --output=wan21_profile.nsys-rep \
#       bash scripts/benchmark/profile_inference_wan.sh

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
NUM_FRAMES="${NUM_FRAMES:-77}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
PROMPT="${PROMPT:-A cat walking through snow}"
OUTPUT_DIR="${OUTPUT_DIR:-./profiler_outputs}"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/traces"
mkdir -p "$OUTPUT_DIR/videos"

# Enable profiling
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export FASTVIDEO_STAGE_LOGGING=1  # Per-stage timing (text encoder, VAE, denoising, etc.)
export FASTVIDEO_TORCH_PROFILER_DIR="$OUTPUT_DIR/traces"
export FASTVIDEO_TORCH_PROFILE_REGIONS="profiler_region_inference_denoising"

# Optional: Additional profiler settings
# export FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES=1
# export FASTVIDEO_TORCH_PROFILER_WITH_STACK=1
# export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=1
# export FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY=1

echo "=================================================="
echo "FastVideo Wan2.1 Inference Profiling"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "Resolution: ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "Inference steps: $NUM_INFERENCE_STEPS"
echo "Attention backend: $FASTVIDEO_ATTENTION_BACKEND"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="

# Run inference with profiling
fastvideo generate \
    --model-path "$MODEL_PATH" \
    --num-frames "$NUM_FRAMES" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT_DIR/videos/"

echo ""
echo "=================================================="
echo "Profiling complete!"
echo "=================================================="
echo "Per-stage timing: Check console output above"
echo "PyTorch traces: $OUTPUT_DIR/traces/"
echo "  - Open traces in https://ui.perfetto.dev/"
echo "Generated video: $OUTPUT_DIR/videos/"
echo "=================================================="
