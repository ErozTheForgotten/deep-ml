import torch

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    # Your code here
    z_t = torch.tensor(z, dtype=torch.float32)

    act_func = torch.relu(z_t)

    return act_func



