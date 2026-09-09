import torch
import torch.nn.functional as F

def pattern_weaver(n: int, crystal_values: list, dimension: int) -> torch.Tensor:
    """
    Implements a simplified self-attention mechanism for crystal values.
    
    Args:
        n: Number of crystals
        crystal_values: List of crystal values
        dimension: Scaling dimension for attention scores
    
    Returns:
        torch.Tensor of final weighted patterns for each crystal
    """
    # Convert inputs to tensor
    values = torch.as_tensor(crystal_values, dtype=torch.float64)
    # Compute pairwise attention scores and apply softmax
    # Hint: Use torch.outer() for pairwise scores, F.softmax() for normalization,
    #       and torch.matmul() for weighted sum

    score = torch.outer(values, values)

    attn = score / (dimension ** .5)

    softmx = F.softmax(attn, dim=1)

    output = torch.matmul(softmx, values)

    return output



        
        
        

