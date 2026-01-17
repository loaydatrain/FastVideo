#!/bin/bash
# TurboDiffusion SLA Distillation Training for Wan2.1-T2V-1.3B
#
# This script trains a SLA-enabled student model to match the predictions
# of a full-attention teacher model (white-box distillation).
#
# Uses DistillationPipeline infrastructure with:
# - real_score_transformer: Teacher (FlashAttention, frozen)
# - fake_score_transformer: Student (SLA, trainable)
#
# Prerequisites:
# - Set FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN for student to use SLA attention
# - Prepare your training data in parquet format
#
# Usage:
#   bash train_sla.sh

set -e

# Environment setup
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

# SLA attention backend for student model
# Teacher will use FlashAttention (forced in code)
export FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN
export MASTER_PORT=29600

# Model and data paths
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"

# Distributed training settings
NUM_GPUS=4

# Training arguments
training_args=(
  --inference_mode False
  --tracker_project_name "turbodiffusion_sla_training"
  --output_dir "checkpoints/turbodiffusion_sla_1.3B"
  --max_train_steps 100000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 4
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size $NUM_GPUS
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments (for distillation: teacher and student paths)
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $MODEL_PATH
  --fake_score_model_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path $DATA_DIR
  --dataloader_num_workers 1
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --weight_decay 0.01
  --max_grad_norm 1.0
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --enable_gradient_checkpointing_type "full"
  --training_cfg_rate 0.0
)

# Distillation-specific arguments
distillation_args=(
  --generator_update_interval 1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --training_state_checkpointing_steps 1000
  --log_validation
  --validation_steps 100
  --validation_sampling_steps "50"
  --validation_dataset_file $VALIDATION_DATASET_FILE
)

# Run training
echo "Starting TurboDiffusion SLA distillation training..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "GPUs: $NUM_GPUS"
echo "Teacher: FlashAttention (frozen)"
echo "Student: SLA attention (trainable)"

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/turbodiffusion_sla_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${distillation_args[@]}"

echo "Training completed!"
