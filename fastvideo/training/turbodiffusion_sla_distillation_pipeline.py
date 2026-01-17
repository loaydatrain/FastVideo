# SPDX-License-Identifier: Apache-2.0
"""
TurboDiffusion SLA (Sparse-Linear Attention) Distillation Pipeline.

This pipeline trains a student model with SLA attention to match the predictions
of a teacher model with full attention (white-box distillation).

Extends DistillationPipeline to leverage FastVideo's existing distillation infrastructure.

Based on: https://github.com/thu-ml/TurboDiffusion
Paper: "TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times"
"""
import math
from typing import Any, cast

import torch
import torch.nn.functional as F

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.attention.selector import (
    global_force_attn_backend,
    get_global_forced_attn_backend,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.pipelines import TrainingBatch

logger = init_logger(__name__)


class LogNormalSampler:
    """Log-normal timestep sampler from TurboDiffusion.
    
    Samples timesteps from a log-normal distribution for training.
    This provides better coverage of the timestep range compared to uniform sampling.
    """
    
    def __init__(self, p_mean: float = 0.0, p_std: float = 1.6):
        self.p_mean = p_mean
        self.p_std = p_std
    
    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from log-normal distribution.
        
        Returns:
            sigma values in [0, 1] range (will be used as t in flow matching)
        """
        u = torch.randn(batch_size, device=device)
        sigma = (u * self.p_std + self.p_mean).exp()
        t = sigma / (sigma + 1)
        return t


class TurboDiffusionSLADistillationPipeline(DistillationPipeline):
    """
    SLA white-box distillation pipeline for TurboDiffusion.
    
    Extends DistillationPipeline to use:
    - real_score_transformer: Teacher with FlashAttention (frozen)
    - fake_score_transformer: Student with SLA attention (trainable)
    
    Training loss: MSE(student_output, teacher_output)
    """
    
    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None):
        """Load modules with different attention backends for teacher vs student.
        
        Teacher (real_score_transformer): Uses FlashAttention
        Student (fake_score_transformer): Uses SLA attention (from env var)
        """
        training_args = cast(TrainingArgs, fastvideo_args)
        
        # First, let parent's parent (TrainingPipeline) load base modules
        # We skip DistillationPipeline.load_modules since we handle teacher/student ourselves
        from fastvideo.training.training_pipeline import TrainingPipeline
        modules = TrainingPipeline.load_modules(self, fastvideo_args, loaded_modules)
        
        # Load teacher (real_score) with FlashAttention
        logger.info("Loading teacher model with FlashAttention backend...")
        original_backend = get_global_forced_attn_backend()
        
        try:
            global_force_attn_backend(AttentionBackendEnum.FLASH_ATTN)
            
            model_path = training_args.real_score_model_path or training_args.model_path
            logger.info("Loading real score transformer from: %s", model_path)
            training_args.override_transformer_cls_name = "WanTransformer3DModel"
            
            self.real_score_transformer = self.load_module_from_path(
                model_path, "transformer", training_args)
            modules["real_score_transformer"] = self.real_score_transformer
            self.real_score_transformer.requires_grad_(False)
            self.real_score_transformer.eval()
            logger.info("Teacher loaded with FlashAttention (frozen)")
            
        finally:
            global_force_attn_backend(original_backend)
        
        # Load student (fake_score) with current backend (should be SLA from env var)
        logger.info("Loading student model with current attention backend (SLA)...")
        model_path = training_args.fake_score_model_path or training_args.model_path
        logger.info("Loading fake score transformer from: %s", model_path)
        
        self.fake_score_transformer = self.load_module_from_path(
            model_path, "transformer", training_args)
        modules["fake_score_transformer"] = self.fake_score_transformer
        logger.info("Student loaded with SLA attention (trainable)")
        
        # No transformer_2 support for SLA (single model)
        self.real_score_transformer_2 = None
        self.fake_score_transformer_2 = None
        
        return modules
    
    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize SLA distillation training pipeline.
        
        This skips the DMD-specific parts of DistillationPipeline.initialize_training_pipeline
        and only uses the parts needed for SLA training.
        """
        from fastvideo.training.training_pipeline import TrainingPipeline
        from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler)
        from fastvideo.training.training_utils import get_scheduler
        from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
        
        logger.info("Initializing TurboDiffusion SLA distillation pipeline...")
        
        # Call grandparent (TrainingPipeline) initialization
        TrainingPipeline.initialize_training_pipeline(self, training_args)
        
        # Setup from DistillationPipeline that we need
        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)
        
        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=self.timestep_shift)
        
        # Ensure teacher is frozen
        self.real_score_transformer.requires_grad_(False)
        self.real_score_transformer.eval()
        
        # Apply gradient checkpointing if enabled
        if training_args.enable_gradient_checkpointing_type is not None:
            self.fake_score_transformer = apply_activation_checkpointing(
                self.fake_score_transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type)
            self.real_score_transformer = apply_activation_checkpointing(
                self.real_score_transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type)
        
        # Initialize optimizer for student (fake_score_transformer)
        fake_score_params = list(
            filter(lambda p: p.requires_grad, self.fake_score_transformer.parameters()))
        
        fake_score_lr = training_args.fake_score_learning_rate
        if fake_score_lr == 0.0:
            fake_score_lr = training_args.learning_rate
            
        betas_str = training_args.fake_score_betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))
        
        self.fake_score_optimizer = torch.optim.AdamW(
            fake_score_params,
            lr=fake_score_lr,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        
        self.fake_score_lr_scheduler = get_scheduler(
            training_args.fake_score_lr_scheduler,
            optimizer=self.fake_score_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )
        
        # SLA-specific setup
        self.generator_update_interval = training_args.generator_update_interval
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps
        self.min_timestep = int(training_args.min_timestep_ratio * self.num_train_timestep)
        self.max_timestep = int(training_args.max_timestep_ratio * self.num_train_timestep)
        
        # No denoising_step_list needed for SLA (we use log-normal sampling)
        self.denoising_step_list = None
        
        # Add log-normal timestep sampler for TurboDiffusion-style training
        self.timestep_sampler = LogNormalSampler(p_mean=0.0, p_std=1.6)
        
        # No EMA for SLA training by default 
        self.generator_ema = None
        self.generator_ema_2 = None
        
        logger.info("SLA distillation pipeline initialized")
        logger.info("  Teacher: real_score_transformer (FlashAttention, frozen)")
        logger.info("  Student: fake_score_transformer (SLA, trainable)")
        logger.info("  Learning rate: %s", fake_score_lr)
    
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline for SLA training.
        
        Uses WanPipeline (not WanDMDPipeline) with the student model since
        SLA training uses standard flow matching, not DMD distillation.
        """
        from copy import deepcopy
        from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
        
        logger.info("Initializing validation pipeline for SLA training...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        
        # Use the student model for validation with standard WanPipeline
        validation_pipeline = WanPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,
            inference_mode=True,
            loaded_modules={"transformer": self.fake_score_transformer},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)
        
        self.validation_pipeline = validation_pipeline
        logger.info("Validation pipeline initialized with student model")
    
    def _generator_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student model with log-normal timestep sampling.
        
        SLA training uses log-normal timestep distribution instead of discrete denoising steps.
        """
        from fastvideo.forward_context import set_forward_context
        
        latents = training_batch.latents  # [B, C, T, H, W]
        batch_size = latents.shape[0]
        dtype = latents.dtype
        device = latents.device
        
        # Sample timesteps from log-normal distribution (TurboDiffusion style)
        t = self.timestep_sampler(batch_size, device)
        t = t.clamp(0.0, 1.0)
        training_batch.dmd_latent_vis_dict["generator_timestep"] = t
        
        # Scale to scheduler timestep range
        timestep = (t * self.num_train_timestep).long()
        training_batch.dmd_latent_vis_dict["dmd_timestep"] = timestep
        
        # Expand for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t_expanded = t.view(batch_size, 1, 1, 1, 1)
        
        # Create noisy input: x_t = (1-t)*x0 + t*noise (flow matching)
        noise = torch.randn_like(latents)
        noisy_latent = (1 - t_expanded) * latents + t_expanded * noise
        
        # Ensure bfloat16 dtype for model compatibility
        noisy_latent = noisy_latent.to(torch.bfloat16)
        
        # Build input kwargs using parent's utility method
        training_batch = self._build_distill_input_kwargs(
            noisy_latent, timestep, training_batch.conditional_dict, training_batch)
        
        # Forward pass through student (fake_score_transformer)
        with set_forward_context(current_timestep=t, attn_metadata=None):
            pred_output = self.fake_score_transformer(**training_batch.input_kwargs)
            if isinstance(pred_output, tuple):
                pred_output = pred_output[0]
            pred_output = pred_output.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        
        return pred_output
    
    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch: TrainingBatch) -> torch.Tensor:
        """Compute SLA alignment loss (overrides DMD loss).
        
        SLA training uses simple MSE alignment between student and teacher predictions.
        
        Args:
            generator_pred_video: Student model prediction
            training_batch: Training batch with input kwargs
            
        Returns:
            MSE loss between student and teacher predictions
        """
        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            teacher_output = self.real_score_transformer(
                **training_batch.input_kwargs)
            
            # Handle tuple output
            if isinstance(teacher_output, tuple):
                teacher_output = teacher_output[0]
            
            # Permute to match generator_pred_video format
            teacher_output = teacher_output.permute(0, 2, 1, 3, 4)
        
        # Compute MSE alignment loss
        loss = F.mse_loss(generator_pred_video.float(), teacher_output.float())
        
        # Store for visualization if needed
        training_batch.dmd_latent_vis_dict.update({
            "generator_pred_video": generator_pred_video.detach(),
            "teacher_pred_video": teacher_output.detach(),
        })
        
        return loss


def main(args) -> None:
    """Main entry point for SLA distillation training."""
    logger.info("Starting TurboDiffusion SLA distillation pipeline...")
    
    pipeline = TurboDiffusionSLADistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    
    pipeline.train()
    
    logger.info("SLA distillation pipeline completed")


if __name__ == "__main__":
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    main(args)
