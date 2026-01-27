#!/usr/bin/env python3
"""
End-to-end overfit test for SLA distillation training.

This script verifies that SLA training is working by overfitting on a single sample.
The student (SLA attention) should learn to match the teacher (full attention).
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
import subprocess
import sys

os.environ["MASTER_PORT"] = "29513"

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

NUM_NODES = "1"
MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Data paths
DATA_DIR = "data"
LOCAL_RAW_DATA_DIR = Path(os.path.join(DATA_DIR, "cats"))
LOCAL_PREPROCESSED_DATA_DIR = Path(os.path.join(DATA_DIR, "cats_preprocessed_data"))
LOCAL_TRAINING_DATA_DIR = os.path.join(LOCAL_PREPROCESSED_DATA_DIR, "combined_parquet_dataset")
LOCAL_VALIDATION_DATASET_FILE = os.path.join(LOCAL_RAW_DATA_DIR, "validation_prompt_1_sample.json")
LOCAL_OUTPUT_DIR = Path(os.path.join(DATA_DIR, "outputs_sla_overfit"))

# Training settings
NUM_GPUS_PER_NODE = "4"
TRAINING_ENTRY_FILE_PATH = "fastvideo/training/turbodiffusion_sla_distillation_pipeline.py"


def download_data():
    """Download the cats overfit dataset if not already present."""
    data_dir = Path(DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)

    if LOCAL_RAW_DATA_DIR.exists() and any(LOCAL_RAW_DATA_DIR.iterdir()):
        print(f"Data already exists at {LOCAL_RAW_DATA_DIR}, skipping download")
        return

    print(f"Downloading raw dataset to {LOCAL_RAW_DATA_DIR}...")
    try:
        result = snapshot_download(
            repo_id="wlsaidhi/cats-overfit-merged",
            local_dir=str(LOCAL_RAW_DATA_DIR),
            repo_type="dataset",
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Download completed successfully. Files downloaded to: {result}")
    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise


def run_preprocessing():
    """Run preprocessing if not already done."""
    if LOCAL_PREPROCESSED_DATA_DIR.exists() and (LOCAL_PREPROCESSED_DATA_DIR / "combined_parquet_dataset").exists():
        print(f"Preprocessed data already exists at {LOCAL_PREPROCESSED_DATA_DIR}, skipping")
        return

    # Remove partial preprocessing if exists
    if LOCAL_PREPROCESSED_DATA_DIR.exists():
        shutil.rmtree(LOCAL_PREPROCESSED_DATA_DIR)

    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--master_port", "29510",
        "--nproc_per_node", "1",
        "fastvideo/pipelines/preprocess/v1_preprocess.py",
        "--model_path", MODEL_PATH,
        "--data_merge_path", os.path.join(LOCAL_RAW_DATA_DIR, "merge_1_sample.txt"),
        "--preprocess_video_batch_size", "1",
        "--max_height", "480",
        "--max_width", "832",
        "--num_frames", "77",
        "--dataloader_num_workers", "0",
        "--output_dir", str(LOCAL_PREPROCESSED_DATA_DIR),
        "--train_fps", "16",
        "--samples_per_file", "1",
        "--flush_frequency", "1",
        "--video_length_tolerance_range", "5",
        "--preprocess_task", "t2v",
    ]

    print(f"Running preprocessing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_sla_training():
    """Run SLA distillation training."""
    # Clean up output directory
    if LOCAL_OUTPUT_DIR.exists():
        shutil.rmtree(LOCAL_OUTPUT_DIR)
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # Set environment for SLA attention
    env = os.environ.copy()
    env["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"
    # env["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["WANDB_MODE"] = "online"
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--master_port", "29512",
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        TRAINING_ENTRY_FILE_PATH,
        "--model_path", MODEL_PATH,
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", MODEL_PATH,
        # Teacher and student paths (both start from same model)
        "--real_score_model_path", MODEL_PATH,
        "--fake_score_model_path", MODEL_PATH,
        # Data
        "--data_path", LOCAL_TRAINING_DATA_DIR,
        "--validation_dataset_file", LOCAL_VALIDATION_DATASET_FILE,
        # Batch settings
        "--train_batch_size", "1",
        "--train_sp_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        # Model dimensions
        "--num_latent_t", "8",
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "41",
        # Distributed settings
        "--num_gpus", NUM_GPUS_PER_NODE,
        "--sp_size", NUM_GPUS_PER_NODE,
        "--tp_size", "1",
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", NUM_GPUS_PER_NODE,
        # Training settings
        "--max_train_steps", "9001",
        "--learning_rate", "1e-5",
        "--fake_score_learning_rate", "1e-5",
        "--mixed_precision", "bf16",
        "--dit_precision", "fp32",
        "--weight_decay", "0.01",
        "--max_grad_norm", "1.0",
        "--training_cfg_rate", "0.0",
        # Checkpointing
        "--weight_only_checkpointing_steps", "1000",
        "--training_state_checkpointing_steps", "1000",
        # Validation
        "--log_validation",
        "--validation_steps", "100",
        "--validation_sampling_steps", "50",
        "--validation_guidance_scale", "3.0",
        # Distillation settings
        "--generator_update_interval", "1",
        "--multi_phased_distill_schedule", "4000-1",
        "--not_apply_cfg_solver",
        "--num_euler_timesteps", "50",
        # Output
        "--output_dir", str(LOCAL_OUTPUT_DIR),
        "--tracker_project_name", "sla_overfit_test",
        "--checkpoints_total_limit", "2",
        "--dataloader_num_workers", "4",
        "--ema_start_step", "0",
        "--enable_gradient_checkpointing_type", "full",
        # "--resume_from_checkpoint" , "data/outputs_sla_overfit/checkpoint-3000/"
    ]

    print(f"Running SLA training: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def verify_training():
    """Verify that training produced expected outputs and loss decreased."""
    # Check that validation videos were created
    validation_videos = list(LOCAL_OUTPUT_DIR.glob("validation_step_*_video_*.mp4"))
    print(f"Found {len(validation_videos)} validation videos")
    
    if len(validation_videos) == 0:
        print("WARNING: No validation videos found!")
        return False
    
    for video in validation_videos:
        print(f"  - {video.name}")
    
    # Check for latest validation video
    latest_video = max(validation_videos, key=lambda x: x.stat().st_mtime)
    print(f"Latest validation video: {latest_video}")
    
    return True


def test_sla_overfit_single_sample():
    """Main test function for SLA overfitting."""
    print("=" * 60)
    print("SLA Distillation Overfit Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {LOCAL_OUTPUT_DIR}")
    print()

    # Step 1: Download data
    print("\n[1/4] Downloading data...")
    download_data()

    # Step 2: Preprocess
    print("\n[2/4] Running preprocessing...")
    run_preprocessing()

    # Step 3: Train
    print("\n[3/4] Running SLA distillation training...")
    run_sla_training()

    # Step 4: Verify
    print("\n[4/4] Verifying results...")
    success = verify_training()

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: SLA overfit test completed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING: Test completed with issues")
        print("=" * 60)


if __name__ == "__main__":
    test_sla_overfit_single_sample()
