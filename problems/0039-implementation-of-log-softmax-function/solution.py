import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    A_t = torch.tensor(scores, dtype=torch.float32)

    l_sftmx_A = torch.nn.functional.log_softmax(A_t, dim=0)

    res = l_sftmx_A.numpy()

    return res




    
    
