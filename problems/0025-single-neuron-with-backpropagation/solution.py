import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here

    weights = initial_weights.clone().detach().requires_grad_(True)
    bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)

    mse_values = []

    
    for epoch in range(epochs):
        
        weighted_sum = features @ weights + bias
        
        y_pred = torch.sigmoid(weighted_sum)

        loss_fn = torch.nn.MSELoss()

        loss = loss_fn(y_pred, labels)

        mse_values.append(round(loss.item(), 4))

        loss.backward()

        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad

            weights.grad.zero_()
            bias.grad.zero_()

    updated_w = [round(w, 4) for w in weights.detach().tolist()]

    updated_b = round(bias.detach().item(), 4)

    

    return updated_w, updated_b, mse_values






        



    