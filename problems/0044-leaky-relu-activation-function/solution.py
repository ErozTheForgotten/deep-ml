import torch
import torch.nn.functional as F

def leaky_relu(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Implements the Leaky ReLU activation function using PyTorch.
    
    Args:
        z: Input tensor (scalar or any shape)
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Output tensor after applying Leaky ReLU
    """
    # Your implementation here
    
    z_t = torch.tensor(z, dtype=torch.float32)
    alpha_t = torch.tensor(alpha, dtype=torch.float32)

    act_func = torch.nn.functional.leaky_relu(z_t, alpha_t)

    return act_func
