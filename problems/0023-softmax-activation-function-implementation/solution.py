import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here

    scores_t = torch.tensor(scores, dtype=torch.float32)
    
    res_t = torch.softmax(scores_t, dim=0)

    res_l = res_t.tolist()


    return res_l
