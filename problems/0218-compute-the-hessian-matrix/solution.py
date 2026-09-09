import torch
from typing import Callable

def compute_hessian(f: Callable[[torch.Tensor], torch.Tensor], point: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hessian matrix using PyTorch autograd.
    
    Args:
        f: A scalar function that takes a tensor and returns a scalar tensor
        point: The point at which to compute the Hessian
        
    Returns:
        The Hessian matrix as a tensor
    """
    # Your code here - use torch.autograd.functional.hessian or manual double backward

    hessian_matrix = torch.autograd.functional.hessian(f, point)
    
    return hessian_matrix



    