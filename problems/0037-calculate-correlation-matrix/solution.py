import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """

    import torch

def calculate_correlation_matrix(X, Y=None):
    X = torch.as_tensor(X, dtype=torch.float32)

    if Y is None:
        return torch.corrcoef(X.T)

    Y = torch.as_tensor(Y, dtype=torch.float32)

    combined = torch.cat((X, Y), dim=1)

    return torch.corrcoef(combined.T)
        




    
