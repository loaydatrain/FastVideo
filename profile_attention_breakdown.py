
import torch
import math
import argparse
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionBackend, VideoSparseAttentionMetadataBuilder, VSA_TILE_SIZE
from fastvideo.attention.backends.sla import SLAAttentionBackend, SLAAttentionMetadataBuilder, get_block_map
from fastvideo_kernel import video_sparse_attn
# Import SLA kernel directly from triton_kernels
from fastvideo_kernel.triton_kernels.sla_triton import _attention as sla_kernel

# Patch get_sp_group for VSA
import fastvideo.distributed
import fastvideo.attention.backends.video_sparse_attn
class DummyGroup:
    world_size = 1
def dummy_get_sp_group():
    return DummyGroup()
fastvideo.distributed.get_sp_group = dummy_get_sp_group
fastvideo.attention.backends.video_sparse_attn.get_sp_group = dummy_get_sp_group

def measure_time(func, *args, **kwargs):
    # Warmup
    for _ in range(5):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(50):
        func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / 50.0

def profile_vsa(batch_size, T, H, W, device, dtype, sparsity=0.95):
    seq_len = T * H * W
    print(f"\nProfiling VSA (Sparsity {sparsity*100}%) | T={T}, H={H}, W={W} (L={seq_len})...")
    
    # Setup VSA impl
    impl = VideoSparseAttentionBackend.get_impl_cls()(
        num_heads=16, head_size=128, causal=False, softmax_scale=128**-0.5
    )
    if isinstance(impl, torch.nn.Module):
        impl = impl.to(device).to(dtype)
    
    # Metadata construction for video geometry
    # raw_latent_shape needs to match T, H, W after patching?
    # standard patch size is (1, 2, 2)
    patch_size = (1, 2, 2)
    # The builder takes 'raw_latent_shape'. 
    # If we want the *sequence* dimensions to be T, H, W, we must construct raw_latent_shape such that:
    # raw_T // patch_t = T
    # raw_H // patch_h = H
    # raw_W // patch_w = W
    
    raw_T = T * patch_size[0]
    raw_H = H * patch_size[1]
    raw_W = W * patch_size[2]
    
    builder = VideoSparseAttentionMetadataBuilder()
    metadata = builder.build(
        current_timestep=0,
        raw_latent_shape=(raw_T, raw_H, raw_W),
        patch_size=patch_size,
        VSA_sparsity=sparsity,
        device=device
    )
    
    num_heads = 16
    head_dim = 128
    
    # Inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    gate = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # 1. Preprocess (Tiling)
    # VSA requires preprocessing if we want to measure kernel speed accurately or end-to-end properly
    t_preprocess = measure_time(impl.preprocess_qkv, q, metadata)
    print(f"Preprocess (Tile): {t_preprocess:.3f} ms")
    
    # Tiled inputs
    q_tiled = impl.preprocess_qkv(q, metadata)
    k_tiled = impl.preprocess_qkv(k, metadata)
    v_tiled = impl.preprocess_qkv(v, metadata)
    gate_tiled = impl.preprocess_qkv(gate, metadata)
    
    # 2. Transpose overhead inside forward
    def transpose_op(x):
        return x.transpose(1, 2).contiguous()
    
    t_transpose = measure_time(transpose_op, q_tiled)
    print(f"Transpose Overhead (per tensor): {t_transpose:.3f} ms")
    
    # 3. Kernel Execution
    q_tr = q_tiled.transpose(1, 2).contiguous()
    k_tr = k_tiled.transpose(1, 2).contiguous()
    v_tr = v_tiled.transpose(1, 2).contiguous()
    gate_tr = gate_tiled.transpose(1, 2).contiguous()
    
    cur_topk = math.ceil(
            (1 - sparsity) *
            (metadata.total_seq_length / math.prod(VSA_TILE_SIZE)))
    
    def run_kernel():
        return video_sparse_attn(
            q_tr, k_tr, v_tr,
            metadata.variable_block_sizes,
            metadata.variable_block_sizes,
            cur_topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=gate_tr).transpose(1, 2)

    t_kernel = measure_time(run_kernel)
    print(f"Kernel Only: {t_kernel:.3f} ms")
    
    # Total
    t_total = measure_time(impl.forward, q_tiled, k_tiled, v_tiled, gate_tiled, metadata)
    print(f"Full Forward (impl.forward): {t_total:.3f} ms")
    
    print(f"-> Kernel is {t_kernel/t_total:.1%} of total time")


def profile_sla(batch_size, T, H, W, device, dtype, sparsity=0.95):
    seq_len = T * H * W
    print(f"\nProfiling SLA (Sparsity {sparsity*100}%) | T={T}, H={H}, W={W} (L={seq_len})...")
    
    num_heads = 16
    head_dim = 128
    
    impl = SLAAttentionBackend.get_impl_cls()(
        num_heads=num_heads, head_size=head_dim, causal=False, use_bf16=(dtype==torch.bfloat16),
        topk_ratio=(1-sparsity), BLKQ=128, BLKK=64
    ).to(device).to(dtype)
    
    # Inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    metadata = SLAAttentionMetadataBuilder().build(current_timestep=0, topk_ratio=(1-sparsity))
    
    # 1. Transpose overhead
    def transpose_op(x):
        return x.transpose(1, 2).contiguous()
    t_transpose = measure_time(transpose_op, q)
    print(f"Transpose Overhead (per tensor): {t_transpose:.3f} ms")
    
    q_tr = transpose_op(q)
    k_tr = transpose_op(k)
    v_tr = transpose_op(v)
    
    # 2. Block Map Calculation
    def run_block_map():
        return get_block_map(q_tr, k_tr, topk_ratio=(1-sparsity), BLKQ=128, BLKK=64)
    
    t_blockmap = measure_time(run_block_map)
    print(f"Block Map Calc: {t_blockmap:.3f} ms")
    
    sparse_map, lut, real_topk = get_block_map(q_tr, k_tr, topk_ratio=(1-sparsity), BLKQ=128, BLKK=64)
    
    # 3. Sparse Attention Kernel (Tritech)
    def run_sparse_kernel():
        # _attention.apply is triton kernel wrapper
        return sla_kernel.apply(
            q_tr.to(dtype), k_tr.to(dtype), v_tr.to(dtype), 
            sparse_map, lut, real_topk, 128, 64
        )
        
    t_kernel = measure_time(run_sparse_kernel)
    print(f"Sparse Kernel Only: {t_kernel:.3f} ms")
    
    # 4. Linear Attention
    def run_linear():
        q_l = impl.feature_map_q(q_tr)
        k_l = impl.feature_map_k(k_tr)
        return impl._calc_linear_attention(q_l, k_l, v_tr)
        
    t_linear = measure_time(run_linear)
    print(f"Linear Attn: {t_linear:.3f} ms")
    
    # Total
    t_total = measure_time(impl.forward, q, k, v, metadata)
    print(f"Full Forward (impl.forward): {t_total:.3f} ms")


def profile_flash(batch_size, T, H, W, device, dtype):
    seq_len = T * H * W
    print(f"\nProfiling FlashAttention | T={T}, H={H}, W={W} (L={seq_len})...")
    
    num_heads = 16
    head_dim = 128

    from fastvideo.attention.backends.flash_attn import FlashAttentionBackend, FlashAttnMetadata
    impl = FlashAttentionBackend.get_impl_cls()(
        num_heads=num_heads, head_size=head_dim, causal=False, softmax_scale=head_dim**-0.5
    )
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    metadata = FlashAttnMetadata(current_timestep=0)
    
    t_kernel = measure_time(impl.forward, q, k, v, metadata)
    print(f"Flash Kernel (Total): {t_kernel:.3f} ms")
    return t_kernel


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Video Workloads (Latent Dimensions)
    # Assuming patch size overlap is handled by user intent (these are sequence dims)
    video_workloads = [
        # (T, H, W)
        (16, 32, 32),    # 16k tokens (Small Video)
        (32, 64, 64),    # 131k tokens (Medium Video)
        (64, 64, 64),    # 262k tokens (Large Video T)
    ]
    
    for T, H, W in video_workloads:
        seq_len = T * H * W
        print(f"\n{'='*60}")
        print(f" WORKLOAD: T={T}, H={H}, W={W}  (Tokens={seq_len})")
        print(f"{'='*60}")
        
        try:
           profile_flash(1, T, H, W, device, dtype)
        except Exception as e:
           print(f"Flash failed: {e}")

        try:
           profile_sla(1, T, H, W, device, dtype, sparsity=0.95)
        except Exception as e:
           print(f"SLA failed: {e}")

        # Benchmark VSA for all video sizes
        try:
            profile_vsa(1, T, H, W, device, dtype, sparsity=0.95)
        except Exception as e:
            print(f"VSA failed: {e}")

