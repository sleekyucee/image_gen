import torch
import torch.nn.functional as F
import yaml

def vae_loss(recon_x, x, mu, log_var, beta=1):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kld_loss

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)