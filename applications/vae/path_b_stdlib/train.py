import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import csv
from pathlib import Path

from model import VAE
from loss import elbo_loss
from data import get_fashion_mnist_loaders


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure numeric config values are correct types
    config['model']['input_dim'] = int(config['model']['input_dim'])
    config['model']['latent_dim'] = int(config['model']['latent_dim'])
    config['model']['output_dim'] = int(config['model']['output_dim'])
    config['model']['encoder_hidden_dims'] = [int(x) for x in config['model']['encoder_hidden_dims']]
    config['model']['decoder_hidden_dims'] = [int(x) for x in config['model']['decoder_hidden_dims']]

    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['num_epochs'] = int(config['training']['num_epochs'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['grad_clip'] = float(config['training']['grad_clip'])
    config['training']['early_stopping_patience'] = int(config['training']['early_stopping_patience'])

    config['loss']['reconstruction_weight'] = float(config['loss']['reconstruction_weight'])
    config['loss']['kl_weight'] = float(config['loss']['kl_weight'])

    config['checkpoint']['save_interval'] = int(config['checkpoint']['save_interval'])
    config['validation']['val_interval'] = int(config['validation']['val_interval'])

    return config


def train_epoch(model, train_loader, optimizer, device, recon_weight, kl_weight):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc="Training")):
        x = x.view(x.size(0), -1).to(device)  # Flatten images

        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, logvar = model(x)

        # Compute loss
        loss, recon_loss, kl_loss = elbo_loss(x, x_recon, mu, logvar, recon_weight, kl_weight)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)

    return avg_loss, avg_recon, avg_kl


def validate(model, val_loader, device, recon_weight, kl_weight):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for x, _ in tqdm(val_loader, desc="Validating"):
            x = x.view(x.size(0), -1).to(device)

            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = elbo_loss(x, x_recon, mu, logvar, recon_weight, kl_weight)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    avg_loss = total_loss / len(val_loader)
    avg_recon = total_recon / len(val_loader)
    avg_kl = total_kl / len(val_loader)

    return avg_loss, avg_recon, avg_kl


def main():
    # Load configuration
    config = load_config("config.yaml")

    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        num_workers=config['logging'].get('num_workers', 0),
    )

    # Create model
    print("Creating VAE model...")
    model = VAE(
        input_dim=config['model']['input_dim'],
        encoder_hidden_dims=config['model']['encoder_hidden_dims'],
        latent_dim=config['model']['latent_dim'],
        decoder_hidden_dims=config['model']['decoder_hidden_dims'],
        output_dim=config['model']['output_dim'],
    ).to(device)

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Loss weights
    recon_weight = config['loss']['reconstruction_weight']
    kl_weight = config['loss']['kl_weight']

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    metrics_file = os.path.join(config['logging']['log_dir'], 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_recon', 'train_kl', 'val_loss', 'val_recon', 'val_kl'])

    print("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, recon_weight, kl_weight
        )

        # Validate
        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, recon_weight, kl_weight
        )

        # Log metrics
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_recon:.6f}",
                f"{train_kl:.6f}",
                f"{val_loss:.6f}",
                f"{val_recon:.6f}",
                f"{val_kl:.6f}",
            ])

        if config['logging']['verbose']:
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (recon: {val_recon:.4f}, kl: {val_kl:.4f})")

        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint']['save_dir'],
                f"model_epoch_{epoch+1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            if config['logging']['verbose']:
                print(f"  Saved checkpoint to {checkpoint_path}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            best_model_path = os.path.join(config['checkpoint']['save_dir'], 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            if config['logging']['verbose']:
                print(f"  ✓ New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
