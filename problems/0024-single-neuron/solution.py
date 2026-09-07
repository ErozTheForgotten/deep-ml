import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE

    features_t = torch.tensor(features, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    binary_labels_t = torch.tensor(labels, dtype=torch.float32)
    bias_t = torch.tensor(bias, dtype=torch.float32)

    loss_fn = torch.nn.MSELoss()

    weighted_sum = features_t @ weights_t + bias_t

    s = torch.sigmoid(weighted_sum)
    
    mse = loss_fn(s, binary_labels_t)

    res_s = torch.round(s, decimals=4).tolist()
    
    res_MSE = round(mse.item(), 4)

    
    return res_s, res_MSE
    




