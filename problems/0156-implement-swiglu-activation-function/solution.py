import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    x1, x2 = x.chunk(2, dim=-1)

    output = x1 * torch.nn.functional.silu(x2)

    return output
    