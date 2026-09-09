import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    mask_t = mask.bool
    score = (Q @ K.transpose(-2, -1)) / (Q.shape[-1] ** .5)

    if mask is not None:
        mask_t = torch.tensor(mask, dtype=torch.bool)
        score = score.masked_fill(mask_t, -1e9)
    
    smx = torch.softmax(score, dim=-1)

    output = smx @ V

    return output
