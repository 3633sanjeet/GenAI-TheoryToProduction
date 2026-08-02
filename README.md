# Generative AI: From Theory to Implementation

**Course:** Generative AI (IITM)  
**Author:** Sanjeet Phoughat  

A comprehensive learning project implementing three foundational generative AI architectures (**VAE**, **Diffusion Models**, **GAN**) from first principles, with both scratch and production implementations.

## Project Overview

This project follows a **dual-path learning strategy**:

- **Path A (Scratch):** Implement from scratch using NumPy to understand the mechanics
- **Path B (Production):** Implement using PyTorch following industry best practices
- **Validation:** Compare both approaches on real-world datasets

### Why Two Implementations?

Understanding generative AI requires learning at multiple levels:

1. **Mathematical Theory** — hand-derived equations and concepts
2. **Mechanical Implementation** — how the math translates to code (NumPy path)
3. **Production Patterns** — how to build scalable systems (PyTorch path)
4. **Practical Validation** — proving both approaches work on real data

## Project Structure

```
applications/
├── vae/                    # Variational Autoencoders
├── diffusion/              # Diffusion Models (DDPM/DDIM)
└── gan/                    # Generative Adversarial Networks

Each application contains:
├── theory/                 # Hand-derived mathematical derivations
├── path_a_scratch/         # NumPy-only implementation
├── path_b_stdlib/          # PyTorch production implementation
└── comparison/             # Metrics and analysis comparing both paths
```

## Applications

| Model | Dataset | Key Topics |
|-------|---------|-----------|
| **VAE** | Fashion-MNIST | ELBO, reparameterization trick, latent representations |
| **Diffusion** | CIFAR-10 | Noise schedules, score matching, iterative refinement |
| **GAN** | MNIST | Adversarial training, game theory, mode collapse |

## Getting Started

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed instructions.

### 2. Explore an Application

Each application is self-contained. Start with VAE:

```bash
cd applications/vae

# Read the theory
cat theory/README.md

# Run Path A (NumPy scratch implementation)
python path_a_scratch/train.py --config path_a_scratch/config.yaml

# Run Path B (PyTorch production implementation)
python path_b_stdlib/train.py --config path_b_stdlib/config.yaml

# Compare results
python comparison/compare_paths.py
```

## Key Features

### Mathematical Rigor
- Hand-derived equations from first principles
- Reference to course materials and literature
- Clear derivation of loss functions and sampling procedures

### Educational Implementation (Path A)
- NumPy-only code, no deep learning frameworks
- Detailed comments explaining each mathematical operation
- Forward and backward passes built from scratch
- Custom SGD optimizer implementation

### Production Patterns (Path B)
- PyTorch best practices (nn.Module, DataLoader, checkpointing)
- Modular, scalable code architecture
- Proper training loops with metrics tracking
- Deployment-ready implementations

### Rigorous Validation
- Both implementations trained on identical datasets
- Side-by-side metrics comparison
- Visual quality assessment of generated samples
- Analysis of convergence, speed, and scalability

## Deliverables

For each of the 3 applications (VAE, Diffusion, GAN):

1. **Theory Document** — hand-derived mathematics with intuitive explanations
2. **Scratch Implementation** — NumPy code demonstrating core concepts
3. **Production Implementation** — PyTorch code showing best practices
4. **Comparison Report** — metrics, visualizations, and insights

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy, scikit-image, matplotlib
- See [requirements.txt](requirements.txt) for complete list

## Learning Outcomes

After completing this project, you will understand:

- ✓ The mathematical foundations of VAEs, Diffusion Models, and GANs
- ✓ How to implement these from scratch using basic libraries
- ✓ Modern PyTorch patterns for generative modeling
- ✓ How to evaluate and compare different implementations
- ✓ Real-world considerations: scalability, reproducibility, deployment

## References

**Courses:**
- [IITM - Generative AI Course](local_reference/GenAI-IITM.pdf) (enrolled course materials)
- [Open-Source Generative AI Course](https://prathosh.in/cce-genai.html) (public reference)

**Papers:**
- Kingma & Welling (2014): [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Ho et al. (2020): [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Goodfellow et al. (2014): [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

## Progress

- [x] Project initialization and structure
- [ ] VAE: Theory → Path A → Path B → Comparison
- [ ] Diffusion: Theory → Path A → Path B → Comparison
- [ ] GAN: Theory → Path A → Path B → Comparison
- [ ] Responsible AI deployment guide

---

**Last Updated:** August 2026
