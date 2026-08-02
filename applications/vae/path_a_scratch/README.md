# VAE Path A: NumPy From-Scratch Implementation

**Goal:** Implement VAE using only NumPy to understand the mechanics at a deep level.

## Implementation Strategy

This implementation builds VAE components from scratch:

1. **encoder.py** — Forward pass from image to latent distribution parameters
2. **decoder.py** — Forward pass from latent to reconstructed image
3. **vae.py** — Orchestrate encoder, reparameterization, and decoder
4. **loss.py** — Compute ELBO: reconstruction loss + KL divergence
5. **optimizer.py** — SGD with momentum from scratch
6. **train.py** — Training loop with metrics tracking
7. **data_loader.py** — Load and preprocess Fashion-MNIST

## Key Learning Points

### Forward Pass (NumPy)
- Matrix multiplication: `X @ W + b`
- Activation functions: ReLU, Sigmoid
- Gaussian sampling: `μ + σ * ε` (reparameterization)
- No automatic differentiation

### Backward Pass (NumPy)
- Compute gradients manually for each layer
- Chain rule through encoder, reparameterization, decoder
- Update parameters using optimizer

### Numerical Stability
- Log-sum-exp trick for numerical stability
- Proper weight initialization (Xavier)
- Clip gradients to prevent NaN

## Expected Results

On Fashion-MNIST:
- Final reconstruction loss: ~80-120 (Binary Cross Entropy)
- KL loss: ~10-20 (should decrease as training progresses)
- Convergence in ~50-100 epochs on CPU

## Files

```
path_a_scratch/
├── encoder.py           # Encoder class (NumPy)
├── decoder.py           # Decoder class (NumPy)
├── vae.py              # VAE orchestrator
├── loss.py             # ELBO loss computation
├── optimizer.py        # SGD with momentum
├── train.py            # Training loop
├── data_loader.py      # Data loading and preprocessing
├── config.yaml         # Hyperparameters
├── results/            # Generated images, metrics, model
└── README.md           # This file
```

## Running

```bash
# Set config in config.yaml, then:
python train.py --config config.yaml

# Outputs:
# - results/metrics.csv (loss per epoch)
# - results/samples_epoch_*.png (generated samples)
# - results/best_model.pkl (trained encoder/decoder)
```

## Debugging Tips

1. **Gradients exploding?** Check weight initialization and learning rate
2. **Loss not decreasing?** Try smaller learning rate or check loss computation
3. **NaN values?** Check numerical stability (log of 0, division by 0)
4. **Slow training?** NumPy is slower than PyTorch; this is normal

## Comparison with Theory

Each line of code maps to the theory:
- Encoder forward → q(z|x) = N(μ(x), σ²(x)I)
- Reparameterization → z = μ + σ ⊙ ε
- Decoder forward → p(x|z)
- Loss → ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

---

**Next:** After this works, compare with Path B (PyTorch) to validate correctness.
