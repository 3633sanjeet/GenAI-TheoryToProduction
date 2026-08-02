"""
Evaluate trained VAE model: visualize reconstructions, generated samples, and metrics.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

from model import VAE
from data import get_fashion_mnist_loaders


def load_model(checkpoint_path, device):
    """Load trained VAE model."""
    model = VAE(
        input_dim=784,
        encoder_hidden_dims=[512, 256],
        latent_dim=20,
        decoder_hidden_dims=[256, 512],
        output_dim=784,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def visualize_reconstructions(model, test_loader, device, output_dir, num_images=16):
    """Visualize original vs reconstructed images."""
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx > 0:
                break
            x = x.view(x.size(0), -1).to(device)
            x_recon, _, _ = model(x)

            # Reshape for visualization
            x_orig = x[:num_images].view(-1, 1, 28, 28).cpu()
            x_recon = x_recon[:num_images].view(-1, 1, 28, 28).cpu()

            # Plot
            fig, axes = plt.subplots(2, num_images, figsize=(16, 4))
            for i in range(num_images):
                # Original
                axes[0, i].imshow(x_orig[i].squeeze(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')

                # Reconstructed
                axes[1, i].imshow(x_recon[i].squeeze(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')

            plt.suptitle('VAE Reconstructions: Top Row = Original, Bottom Row = Reconstructed')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/reconstructions.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_dir}/reconstructions.png")


def generate_samples(model, device, output_dir, num_samples=64):
    """Generate new samples from random latent vectors."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        samples = model.decoder(z)
        samples = samples.view(-1, 1, 28, 28).cpu()

    # Plot grid of samples
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.suptitle('Generated Samples from VAE')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/generated_samples.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/generated_samples.png")


def compute_metrics(model, test_loader, device):
    """Compute reconstruction error and other metrics on test set."""
    model.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(x.size(0), -1).to(device)
            x_recon, mu, logvar = model(x)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')

            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    avg_recon = total_recon_loss / num_batches
    avg_kl = total_kl_loss / num_batches
    avg_total = avg_recon + avg_kl

    return {
        'reconstruction_loss': avg_recon,
        'kl_loss': avg_kl,
        'total_loss': avg_total,
    }


def plot_training_curves(metrics_file, output_dir):
    """Plot training and validation loss curves."""
    epochs = []
    train_loss = []
    val_loss = []

    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (ELBO)', fontsize=12)
    ax.set_title('VAE Training Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_curves.png")

    # Find best validation epoch
    best_idx = val_loss.index(min(val_loss))
    print(f"\n📊 Training Summary:")
    print(f"  Best validation loss: {val_loss[best_idx]:.4f} (Epoch {epochs[best_idx]})")
    print(f"  Final train loss: {train_loss[-1]:.4f}")
    print(f"  Final val loss: {val_loss[-1]:.4f}")


def main():
    device = torch.device('cpu')
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("VAE EVALUATION: Visualizations & Metrics")
    print("=" * 60)

    # Load model
    checkpoint_path = './results/checkpoints/best_model.pt'
    print(f"\n1. Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("   ✓ Model loaded successfully")

    # Load test data
    print("\n2. Loading test data...")
    _, _, test_loader = get_fashion_mnist_loaders(
        data_dir='../../data/',
        batch_size=128,
        num_workers=0,
    )
    print(f"   ✓ Test set loaded ({len(test_loader)} batches)")

    # Generate visualizations
    print("\n3. Generating visualizations...")
    visualize_reconstructions(model, test_loader, device, str(output_dir), num_images=16)
    generate_samples(model, device, str(output_dir), num_samples=64)

    # Compute metrics
    print("\n4. Computing test set metrics...")
    metrics = compute_metrics(model, test_loader, device)
    print(f"   Reconstruction Loss: {metrics['reconstruction_loss']:.4f}")
    print(f"   KL Divergence Loss: {metrics['kl_loss']:.4f}")
    print(f"   Total ELBO Loss: {metrics['total_loss']:.4f}")

    # Plot training curves
    print("\n5. Plotting training curves...")
    metrics_file = './results/metrics.csv'
    plot_training_curves(metrics_file, str(output_dir))

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {output_dir}/reconstructions.png")
    print(f"  - {output_dir}/generated_samples.png")
    print(f"  - {output_dir}/training_curves.png")


if __name__ == "__main__":
    main()
