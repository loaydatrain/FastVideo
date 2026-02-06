
import torch
import math
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionBackend, VideoSparseAttentionMetadataBuilder, VSA_TILE_SIZE
from fastvideo.attention.backends.sla import SLAAttentionBackend, SLAAttentionMetadataBuilder, get_block_map
from fastvideo_kernel import video_sparse_attn
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
    """Profile VSA kernel only."""
    seq_len = T * H * W
    print(f"\nProfiling VSA (Sparsity {sparsity*100}%) | T={T}, H={H}, W={W} (L={seq_len})...")
    
    num_heads = 12
    head_dim = 128
    
    impl = VideoSparseAttentionBackend.get_impl_cls()(
        num_heads=num_heads, head_size=head_dim, causal=False, softmax_scale=head_dim**-0.5
    )
    
    # Build metadata
    patch_size = (1, 2, 2)
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
    
    # Inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    gate = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Preprocess (tile) the inputs
    q_tiled = impl.preprocess_qkv(q, metadata)
    k_tiled = impl.preprocess_qkv(k, metadata)
    v_tiled = impl.preprocess_qkv(v, metadata)
    gate_tiled = impl.preprocess_qkv(gate, metadata)
    
    # Prepare for kernel call (transpose like impl.forward does)
    q_tr = q_tiled.transpose(1, 2).contiguous()
    k_tr = k_tiled.transpose(1, 2).contiguous()
    v_tr = v_tiled.transpose(1, 2).contiguous()
    gate_tr = gate_tiled.transpose(1, 2).contiguous()
    
    cur_topk = math.ceil(
        (1 - sparsity) *
        (metadata.total_seq_length / math.prod(VSA_TILE_SIZE)))
    
    # Measure kernel only
    def run_vsa_kernel():
        return video_sparse_attn(
            q_tr, k_tr, v_tr,
            metadata.variable_block_sizes,
            metadata.variable_block_sizes,
            cur_topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=gate_tr
        )
    
    t_kernel = measure_time(run_vsa_kernel)
    print(f"  VSA Kernel Only: {t_kernel:.3f} ms")
    
    # Measure full forward
    t_forward = measure_time(impl.forward, q_tiled, k_tiled, v_tiled, gate_tiled, metadata)
    print(f"  VSA Forward: {t_forward:.3f} ms")
    return t_kernel, t_forward


def profile_sla(batch_size, T, H, W, device, dtype, sparsity=0.95):
    """Profile SLA kernel only."""
    seq_len = T * H * W
    print(f"\nProfiling SLA (Sparsity {sparsity*100}%) | T={T}, H={H}, W={W} (L={seq_len})...")
    
    num_heads = 12
    head_dim = 128
    BLKQ, BLKK = 128, 64
    
    impl = SLAAttentionBackend.get_impl_cls()(
        num_heads=num_heads, head_size=head_dim, causal=False, use_bf16=(dtype==torch.bfloat16),
        topk_ratio=(1-sparsity), BLKQ=BLKQ, BLKK=BLKK
    ).to(device).to(dtype)
    
    # Inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    metadata = SLAAttentionMetadataBuilder().build(current_timestep=0, topk_ratio=(1-sparsity))
    
    # Prepare for kernel call (transpose like impl.forward does)
    q_tr = q.transpose(1, 2).contiguous()
    k_tr = k.transpose(1, 2).contiguous()
    v_tr = v.transpose(1, 2).contiguous()
    
    # Get block map
    sparse_map, lut, real_topk = get_block_map(q_tr, k_tr, topk_ratio=(1-sparsity), BLKQ=BLKQ, BLKK=BLKK)
    
    # Measure sparse kernel only
    def run_sla_kernel():
        return sla_kernel.apply(
            q_tr.to(dtype), k_tr.to(dtype), v_tr.to(dtype),
            sparse_map, lut, real_topk, BLKQ, BLKK
        )
    
    t_kernel = measure_time(run_sla_kernel)
    print(f"  SLA Kernel Only: {t_kernel:.3f} ms")
    
    # Measure block map calculation
    def run_block_map():
        return get_block_map(q_tr, k_tr, topk_ratio=(1-sparsity), BLKQ=BLKQ, BLKK=BLKK)
    t_blockmap = measure_time(run_block_map)
    print(f"  SLA Block Map: {t_blockmap:.3f} ms")
    
    # Measure linear attention branch (feature maps + linear attn + projection)
    q_linear = impl.feature_map_q(q_tr).contiguous().to(dtype)
    k_linear = impl.feature_map_k(k_tr).contiguous().to(dtype)
    def run_linear_attn():
        
        o_l = impl._calc_linear_attention(q_linear, k_linear, v_tr)
        with torch.amp.autocast('cuda', dtype=dtype):
            o_l = impl.proj_l(o_l)
        # return o_l
    t_linear = measure_time(run_linear_attn)
    print(f"  SLA Linear Attn: {t_linear:.3f} ms")
    
    # Measure full forward
    t_forward = measure_time(impl.forward, q, k, v, metadata)
    print(f"  SLA Forward: {t_forward:.3f} ms")
    return t_kernel, t_forward


def profile_flash(batch_size, T, H, W, device, dtype):
    """Profile FlashAttention kernel only."""
    seq_len = T * H * W
    print(f"\nProfiling FlashAttention | T={T}, H={H}, W={W} (L={seq_len})...")
    
    num_heads = 12
    head_dim = 128

    from fastvideo.attention.backends.flash_attn import FlashAttentionBackend, FlashAttnMetadata
    impl = FlashAttentionBackend.get_impl_cls()(
        num_heads=num_heads, head_size=head_dim, causal=False, softmax_scale=head_dim**-0.5
    )
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    metadata = FlashAttnMetadata(current_timestep=0)
    
    t_forward = measure_time(impl.forward, q, k, v, metadata)
    print(f"  Flash Forward: {t_forward:.3f} ms")
    return t_forward


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Video Workloads (Latent Dimensions)
    video_workloads = [
        # (T, H, W)
        (16, 32, 32),    # 16k tokens
        (21, 30, 52),    # 32,760 tokens (exact inference geometry)
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

        try:
            profile_vsa(1, T, H, W, device, dtype, sparsity=0.95)
        except Exception as e:
            print(f"VSA failed: {e}")
