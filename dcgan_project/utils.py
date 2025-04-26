import yaml
import torch
import torch.nn.functional as F

def load_config(config_path):
    """
    Loads a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def dcgan_discriminator_loss(real_output, fake_output):
    """
    Computes the discriminator loss for DCGAN using Binary Cross-Entropy loss.
    
    For real images, the target is 1; for fake images, the target is 0.
    
    Args:
        real_output (Tensor): Discriminator output for real images.
        fake_output (Tensor): Discriminator output for generated images.
    
    Returns:
        Tensor: Combined discriminator loss.
    """
    loss_real = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    loss_fake = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    return loss_real + loss_fake

def dcgan_generator_loss(fake_output):
    """
    Computes the generator loss for DCGAN using Binary Cross-Entropy loss.
    
    The generator aims to fool the discriminator so that its output is 1 for generated images.
    
    Args:
        fake_output (Tensor): Discriminator output for generated images.
    
    Returns:
        Tensor: Generator loss.
    """
    return F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))

