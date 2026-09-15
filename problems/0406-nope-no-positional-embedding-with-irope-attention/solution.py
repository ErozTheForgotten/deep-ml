import torch
import torch.nn.functional as F

def irope_attention(Q: list, K: list, V: list, positions: list, layer_index: int, rope_layers: list, base: float = 10000.0) -> dict:
    """
    Compute attention for a single layer in an iRoPE transformer.

    Args:
        Q: Query matrix, shape (seq_len, d_head)
        K: Key matrix, shape (seq_len, d_head)
        V: Value matrix, shape (seq_len, d_head)
        positions: Position indices, shape (seq_len,)
        layer_index: Current layer index (0-indexed)
        rope_layers: List of layer indices that use RoPE
        base: Base frequency for RoPE

    Returns:
        Dictionary with 'output', 'attention_weights', and 'uses_rope' """


    result = {}
    ## Step 1 instantiate params
   
    Q = torch.tensor(Q, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    V = torch.tensor(V, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)
    
    ## Step 2 Define Vars related to Tensors
    d_head = Q.shape[-1]
    seq_len = Q.shape[-2]

    
    ## Step 3 Positional Encoding
    if rope_layers is not None and layer_index in rope_layers:
        freq_idx = torch.arange(0, d_head, 2)

        inverse_freq = 1 / (base ** (freq_idx / d_head))

    ## Step 4 Calculate Angles
        pos_idx = positions.unsqueeze(1)

        inverse_freq = inverse_freq.unsqueeze(0)

        rotation_angles = pos_idx * inverse_freq

    ## Step 5 Pull Cos, Sin from Angles 

        cos_angle = torch.cos(rotation_angles)
        sin_angle = torch.sin(rotation_angles)

        q_even = Q[..., 0::2]
        q_odd = Q[..., 1::2]

        k_even = K[..., 0::2]
        k_odd = K[..., 1::2]

        assert q_even.shape == q_odd.shape
        assert k_even.shape == k_odd.shape

        rotated_q_even = q_even * cos_angle - q_odd * sin_angle
        rotated_q_odd = q_even * sin_angle + q_odd * cos_angle

        rotated_k_even = k_even * cos_angle - k_odd * sin_angle
        rotated_k_odd = k_even * sin_angle + k_odd * cos_angle

        q_pairs = torch.stack([rotated_q_even, rotated_q_odd],dim=-1)
        k_pairs = torch.stack([rotated_k_even, rotated_k_odd],dim=-1)

        rotated_q = q_pairs.reshape(seq_len, d_head)
        rotated_k = k_pairs.reshape(seq_len, d_head)

        score = rotated_q @ rotated_k.transpose(-2, -1)

        attention_weights = torch.softmax((score / (d_head ** 0.5)), dim=-1)

        attention = attention_weights @ V
        
        rounded_attention = [[round(x, 4) for x in row] for row in attention.tolist()]
        
        rounded_attention_weights = [[round(x, 4) for x in row] for row in attention_weights.tolist()]



        result.update({
            "output": rounded_attention,
        "attention_weights": rounded_attention_weights, 
        "uses_rope": True
        })
    
    


    else:
        score = Q @ K.transpose(-2, -1)

        attention_weights = torch.softmax((score / (d_head ** 0.5)), dim=-1)


        attention = attention_weights @ V
        
        rounded_attention = [[round(x, 4) for x in row] for row in attention.tolist()]
        
        rounded_attention_weights = [[round(x, 4) for x in row] for row in attention_weights.tolist()]
        
       

        result.update({"output": rounded_attention, 
        "attention_weights": rounded_attention_weights,
        "uses_rope": False})
    
    return result
    

