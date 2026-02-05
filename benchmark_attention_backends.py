
import os
import time
import torch
import math
import argparse

BACKENDS = {}

try:
    from fastvideo.attention.backends.flash_attn import FlashAttentionBackend
    BACKENDS["flash"] = FlashAttentionBackend
except ImportError:
    print("FlashAttention backend not available")

try:
    from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionBackend, VideoSparseAttentionMetadataBuilder, VSA_TILE_SIZE
    BACKENDS["vsa"] = VideoSparseAttentionBackend
except ImportError:
    print("VSA backend not available")

try:
    from fastvideo.attention.backends.sla import SLAAttentionBackend, SLAAttentionMetadataBuilder
    BACKENDS["sla"] = SLAAttentionBackend
except ImportError:
    print("SLA backend not available")

try:
    from fastvideo.attention.backends.sla import SageSLAAttentionBackend
    BACKENDS["sage_sla"] = SageSLAAttentionBackend
except ImportError:
    print("SageSLA backend not available")

try:
    from fastvideo.attention.backends.sage_attn import SageAttentionBackend
    BACKENDS["sage"] = SageAttentionBackend
except ImportError:
    print("SageAttention backend not available")

try:
    from fastvideo.attention.backends.sage_attn3 import SageAttention3Backend
    BACKENDS["sage3"] = SageAttention3Backend
except ImportError:
    print("SageAttention3 backend not available")

# Patch get_sp_group for VSA
import fastvideo.distributed
import fastvideo.attention.backends.video_sparse_attn

class DummyGroup:
    world_size = 1
def dummy_get_sp_group():
    return DummyGroup()

fastvideo.distributed.get_sp_group = dummy_get_sp_group
fastvideo.attention.backends.video_sparse_attn.get_sp_group = dummy_get_sp_group

def measure_cuda_time(func, *args, **kwargs):
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

def benchmark_backend(name, batch_size, seq_len, num_heads, head_dim, device, dtype):
    print(f"Benchmarking {name}...")
    backend_cls = BACKENDS.get(name)
    if not backend_cls:
        print(f"Skipping {name} (not found/configured)")
        return
    
    try:
        impl_cls = backend_cls.get_impl_cls()
    except Exception as e:
        print(f"Skipping {name}: {e}")
        return

    # Instantiate impl
    try:
        if name in ["sla", "sage_sla"]:
             impl = impl_cls(num_heads=num_heads, head_size=head_dim, causal=False, use_bf16=(dtype==torch.bfloat16))
        elif name == "vsa":
             impl = impl_cls(num_heads=num_heads, head_size=head_dim, causal=False, softmax_scale=head_dim**-0.5)
        else:
             impl = impl_cls(num_heads=num_heads, head_size=head_dim, causal=False, softmax_scale=head_dim**-0.5)
        
        if isinstance(impl, torch.nn.Module):
            impl = impl.to(device).to(dtype)
            
    except Exception as e:
        print(f"Failed to init {name}: {e}")
        return

    # Prepare inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Metadata and special args prep
    kwargs = {}
    
    if name == "vsa":
        # Prepare VSA inputs
        gate_compress = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        
        # Build VSA metadata
        # Assume patch size (1, 2, 2)
        # raw latent shape must match seq_len
        # seq_len = T * H * W
        # Let's assume T=1, and H=W=sqrt(seq_len) * 2 (since patch=2)
        # 1024 -> 32*32. H_lat = 64, W_lat = 64.
        side = int(math.sqrt(seq_len))
        raw_T, raw_H, raw_W = 1, side*2, side*2
        
        builder = VideoSparseAttentionMetadataBuilder()
        metadata = builder.build(
            current_timestep=0,
            raw_latent_shape=(raw_T, raw_H, raw_W),
            patch_size=(1, 2, 2),
            VSA_sparsity=0.7, # Example sparsity
            device=device
        )
        kwargs["attn_metadata"] = metadata
        kwargs["gate_compress"] = gate_compress
        
        # VSA requires TILING input for forward()!
        # preprocess_qkv tiles it.
        # Impl does have preprocess_qkv
        q = impl.preprocess_qkv(q, metadata)
        k = impl.preprocess_qkv(k, metadata)
        v = impl.preprocess_qkv(v, metadata)
        gate_compress = impl.preprocess_qkv(gate_compress, metadata)
        kwargs["gate_compress"] = gate_compress

    elif name in ["sla", "sage_sla"]:
        builder = SLAAttentionMetadataBuilder()
        metadata = builder.build(current_timestep=0, topk_ratio=0.1)
        kwargs["attn_metadata"] = metadata
        # SLA input format: (B, L, H, D) - already correct
        
    elif name == "flash":
        from fastvideo.attention.backends.abstract import AttentionMetadata
        kwargs["attn_metadata"] = AttentionMetadata(current_timestep=0)

    elif name in ["sage", "sage3"]:
        from fastvideo.attention.backends.abstract import AttentionMetadata
        kwargs["attn_metadata"] = AttentionMetadata(current_timestep=0)

    # Benchmark
    try:
        t = measure_cuda_time(impl.forward, q, k, v, **kwargs)
        print(f" -> {name}: {t:.3f} ms")
        return t
    except Exception as e:
        print(f"Failed to run {name}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    print(f"Benchmarking with B={args.batch_size}, L={args.seq_len}, H={args.num_heads}, D={args.head_dim}, dtype={dtype}")
    
    results = {}
    for name in ["flash", "vsa", "sla", "sage_sla", "sage", "sage3"]:
        t = benchmark_backend(name, args.batch_size, args.seq_len, args.num_heads, args.head_dim, device, dtype)
        if t is not None:
            results[name] = t
            
    print("\nSummary:")
    for k, v in results.items():
        print(f"{k}: {v:.3f} ms")
