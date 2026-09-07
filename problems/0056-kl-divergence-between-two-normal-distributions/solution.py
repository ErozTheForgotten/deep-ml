import torch

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) -> torch.Tensor:
    """
    Compute the KL divergence between two normal distributions P and Q.
    
    Args:
        mu_p: Mean of distribution P
        sigma_p: Standard deviation of distribution P
        mu_q: Mean of distribution Q
        sigma_q: Standard deviation of distribution Q
    
    Returns:
        torch.Tensor: KL divergence KL(P || Q)
    """
    mu_p_t = torch.tensor(mu_p, dtype=torch.float32)
    sigma_p_t = torch.tensor(sigma_p, dtype=torch.float32)
    mu_q_t = torch.tensor(mu_q, dtype=torch.float32)
    sigma_q_t = torch.tensor(sigma_q, dtype=torch.float32)

    kl = (
        torch.log(sigma_q_t / sigma_p_t)
        + (
            sigma_p_t**2
            + (mu_p_t - mu_q_t)**2
        ) / (2 * sigma_q_t**2)
        - 0.5
    )

    
    return kl


