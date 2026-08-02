# VAE (Variational Autoencoder) Theory

**Course Reference:** GenAI-IITM.pdf, https://prathosh.in/cce-genai.html  
**Key Papers:** Kingma & Welling (2014), "Auto-Encoding Variational Bayes"

## Problem Statement

We want to learn a generative model p(x) that can:
1. Capture the distribution of observed data x
2. Sample new data points from this distribution
3. Learn a meaningful latent representation z

Traditional autoencoders learn deterministic mappings. VAE extends this with a **probabilistic framework** where the latent space has a known prior distribution.

## Core Concepts

### 1. Latent Variable Model

We assume data x is generated from a latent variable z:

```
p(x) = ∫ p(x|z) p(z) dz
```

Where:
- **p(z)** = prior distribution over latent variables (typically N(0, I))
- **p(x|z)** = likelihood/decoder network (what we learn)
- **p(x)** = marginal likelihood (intractable - we can't compute this directly!)

**Problem:** The integral is intractable for complex models. We need an alternative approach.

### 2. Variational Inference & ELBO

Instead of computing p(x) directly, we introduce an inference/encoder network q(z|x) to approximate the true posterior p(z|x).

**Kullback-Leibler (KL) Divergence** measures distance between distributions:

```
KL(q(z|x) || p(z|x)) = E_q[ log q(z|x) - log p(z|x) ]
```

This is always ≥ 0, and equals 0 only when q(z|x) = p(z|x).

### 3. Evidence Lower Bound (ELBO) Derivation

Starting from the definition of KL divergence and using Bayes' rule:

```
KL(q(z|x) || p(z|x)) = E_q[ log q(z|x) ] - E_q[ log p(z|x) ]
                      = E_q[ log q(z|x) ] - E_q[ log p(x,z) ] + log p(x)
                      = E_q[ log q(z|x) ] - E_q[ log p(x|z) ] - E_q[ log p(z) ] + log p(x)
```

Rearranging:

```
log p(x) = KL(q(z|x) || p(z|x)) + E_q[ log p(x|z) ] - E_q[ log q(z|x) ] + E_q[ log p(z) ]
log p(x) = KL(q(z|x) || p(z|x)) + E_q[ log p(x|z) - log q(z|x) + log p(z) ]
```

Since KL ≥ 0:

```
log p(x) ≥ E_q[ log p(x|z) - log q(z|x) + log p(z) ]
log p(x) ≥ E_q[ log p(x|z) ] - KL(q(z|x) || p(z))

This is the ELBO (Evidence Lower Bound):
ELBO(x) = E_q[ log p(x|z) ] - KL(q(z|x) || p(z))
```

**Interpretation:**
- By maximizing ELBO, we maximize a lower bound on log p(x)
- First term: **Reconstruction Loss** — decoder should reconstruct x from z
- Second term: **KL regularization** — encoder should not diverge too far from prior

### 4. Reparameterization Trick

To optimize the encoder, we need to take gradients through sampling z ~ q(z|x). But sampling is not differentiable!

**Solution:** Instead of sampling z directly, parameterize it as:

```
z = μ(x) + σ(x) ⊙ ε,    where ε ~ N(0, I)
```

Where:
- **μ(x)** = mean vector from encoder
- **σ(x)** = standard deviation from encoder
- **⊙** = element-wise multiplication
- **ε** = standard normal noise (differentiable to resample, not to sample from)

This makes z differentiable w.r.t. the encoder parameters, even though it involves randomness.

### 5. KL Divergence (Gaussian Case)

For our choice of q(z|x) = N(μ, σ²I) and p(z) = N(0, I):

```
KL(N(μ, σ²I) || N(0, I)) = -1/2 * Σ_j [ 1 + log(σ_j²) - μ_j² - σ_j² ]
                           = -1/2 * Σ_j [ 1 + log(σ_j²) - (μ_j² + σ_j²) ]
```

This has a **closed-form solution** — no need for sampling!

### 6. Reconstruction Loss

The decoder p(x|z) is typically:
- **Gaussian:** log p(x|z) = -||x - decoder(z)||² / (2σ²) → use MSE loss
- **Bernoulli:** log p(x|z) = x·log(p) + (1-x)·log(1-p) → use Binary Cross Entropy

For images with values in [0,1], we use **Binary Cross Entropy** (BCE) loss.

## Complete VAE Loss Function

```
L(x, θ, φ) = E_ε[ -log p(x|z=g(μ_φ(x), σ_φ(x), ε)) ] + KL(q_φ(z|x) || p(z))
            = Reconstruction Loss + KL Loss
```

Where:
- θ = decoder parameters
- φ = encoder parameters
- g() = reparameterization function

## Training Procedure

1. **Encode:** Given x, compute μ and σ from encoder
2. **Sample:** z = μ + σ ⊙ ε (with ε ~ N(0,I))
3. **Decode:** Reconstruct x̂ = decoder(z)
4. **Compute Loss:** L = BCE(x, x̂) + KL(q(z|x) || p(z))
5. **Backprop:** Update encoder and decoder parameters
6. **Repeat:** Next batch

## Architecture (for Fashion-MNIST)

```
Encoder:
  Input: 28×28 image (784 dims)
  FC: 784 → 512 (ReLU)
  FC: 512 → 256 (ReLU)
  Output: 2×latent_dim (μ and log σ²)

Decoder:
  Input: latent_dim
  FC: latent_dim → 256 (ReLU)
  FC: 256 → 512 (ReLU)
  Output: 784 (sigmoid activation)
```

## Key Hyperparameters

- **Latent dimension:** 20 (default, can vary)
- **Batch size:** 128
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Loss weights:** Reconstruction loss and KL loss are equally weighted (can use β-VAE for different weights)

## References

1. **Kingma & Welling (2014):** Auto-Encoding Variational Bayes
2. **Rezende et al. (2014):** Stochastic Backpropagation and Variational Inference in Deep Networks
3. **Doersch (2016):** Tutorial on Variational Autoencoders

---

**Next Steps:**
- Path A: Implement from scratch using NumPy
- Path B: Implement using PyTorch
- Comparison: Validate both on Fashion-MNIST
