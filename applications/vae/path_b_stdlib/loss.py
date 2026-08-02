import torch
import torch.nn.functional as F


def elbo_loss(x, x_recon, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    """
    Compute ELBO (Evidence Lower Bound) loss for VAE.

    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
         = Reconstruction Loss - KL Loss

    Args:
        x: Original input (batch_size, input_dim)
        x_recon: Reconstructed output (batch_size, input_dim)
        mu: Mean from encoder (batch_size, latent_dim)
        logvar: Log variance from encoder (batch_size, latent_dim)
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence

    Returns:
        total_loss: ELBO loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """

    # Reconstruction loss: Binary Cross Entropy
    # (assumes values are in [0, 1])
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')

    # KL divergence: KL(N(mu, sigma^2*I) || N(0, I))
    # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # ELBO with weights
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def vae_loss_fn(recon_weight=1.0, kl_weight=1.0):
    """Return a loss function with fixed weights."""
    def loss(x, x_recon, mu, logvar):
        return elbo_loss(x, x_recon, mu, logvar, recon_weight, kl_weight)
    return loss


if __name__ == "__main__":
    # Test loss computation
    batch_size = 4
    input_dim = 784
    latent_dim = 20

    x = torch.randn(batch_size, input_dim).clamp(0, 1)
    x_recon = torch.randn(batch_size, input_dim).clamp(0, 1)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    total_loss, recon_loss, kl_loss = elbo_loss(x, x_recon, mu, logvar)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    print("✓ Loss computation works!")
