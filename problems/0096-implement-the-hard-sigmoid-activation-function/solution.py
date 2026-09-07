import torch

def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function using PyTorch.
    Uses the Keras convention: 0.2*x + 0.5, clamped to [0, 1].

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    # Your code here
    x_t = torch.tensor(x, dtype=torch.float32)

    act_func = (.2 * x_t) + .5

    act_func = torch.clamp(act_func, min=0.0, max=1.0)

    res = act_func.item()

    return res