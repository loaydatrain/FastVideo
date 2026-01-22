
import torch
import torch.nn as nn
from fastvideo.attention.backends.sla import SLAAttentionImpl, SLAAttentionBackend

def test_sla_gradients():
    print("Initializing SLA Attention...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create SLA implementation with learnable projection
    sla = SLAAttentionImpl(dim=128, num_heads=4, head_size=32, topk_ratio=0.1)
    sla.to(device)
    
    # Enable gradients for proj_l
    sla.proj_l.weight.requires_grad = True
    
    # Create dummy inputs
    B, L, H, D = 1, 64, 4, 32
    q = torch.randn(B, L, H, D, device=device, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, requires_grad=True)
    v = torch.randn(B, L, H, D, device=device, requires_grad=True)
    
    # Metadata (optional for this simple test)
    attn_metadata = None
    
    print("Running Forward Pass...")
    output = sla(q, k, v, attn_metadata=attn_metadata)
    
    print(f"Output mean: {output.mean().item()}")
    
    # Compute loss
    loss = output.mean()
    
    print("Running Backward Pass...")
    loss.backward()
    
    # Check gradients
    proj_l_grad = sla.proj_l.weight.grad
    q_grad = q.grad
    
    if proj_l_grad is not None:
        grad_norm = proj_l_grad.norm().item()
        print(f"SLA proj_l gradient norm: {grad_norm}")
        if grad_norm > 0:
            print("SUCCESS: Gradients are flowing to proj_l")
        else:
            print("FAILURE: Gradients are zero for proj_l")
    else:
        print("FAILURE: No gradient for proj_l")
        
    if q_grad is not None:
         print(f"Query gradient norm: {q_grad.norm().item()}")
    else:
         print("FAILURE: No gradient for query")

if __name__ == "__main__":
    test_sla_gradients()
