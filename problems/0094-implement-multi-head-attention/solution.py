import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q

    K = X @ W_k

    V = X @ W_v

    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]
    
    scaled_dp = (Q @ (K.transpose(-2, -1)))

    scaled_dp = scaled_dp / (d_k ** .5)

    attention_weights = torch.softmax(scaled_dp, dim=-1)

    attention_output = attention_weights @ V

    return attention_output


def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape

    assert d_model % n_heads == 0

    d_head = d_model // n_heads

    Q = Q.reshape(seq_len, n_heads, d_head).transpose(0, 1)
    K = K.reshape(seq_len, n_heads, d_head).transpose(0, 1)
    V = V.reshape(seq_len, n_heads, d_head).transpose(0, 1)

    head_outputs = self_attention(Q, K, V)

    head_outputs = head_outputs.transpose(0, 1)

    output = head_outputs.reshape(seq_len, d_model)

    return output

