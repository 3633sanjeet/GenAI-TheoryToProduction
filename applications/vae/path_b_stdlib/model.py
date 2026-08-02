import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output: mean and log_variance
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        in_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())  # For images in [0, 1]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, latent_dim, decoder_hidden_dims, output_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim, encoder_hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, output_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        """Encode image to latent distribution parameters"""
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        """Decode latent vector to image"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: encode, reparameterize, decode"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        """Sample from prior and generate images"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples


if __name__ == "__main__":
    # Test VAE
    batch_size = 4
    input_dim = 784  # 28x28
    latent_dim = 20

    vae = VAE(
        input_dim=input_dim,
        encoder_hidden_dims=[512, 256],
        latent_dim=latent_dim,
        decoder_hidden_dims=[256, 512],
        output_dim=input_dim
    )

    # Random input
    x = torch.randn(batch_size, input_dim)
    x_recon, mu, logvar = vae(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Mean shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print("✓ Model forward pass works!")
