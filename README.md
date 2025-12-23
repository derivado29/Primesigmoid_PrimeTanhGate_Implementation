# Mitigating the Saturation Gap: Inverse Prime-Density Scaling for High-Stakes Probabilistic Modeling

**Authors:** Jaime Aguilar Ortiz & Manuel Francisco Gutierrez Salinas  
**Affiliation:** Universidad Politécnica Metropolitana de Hidalgo (UPMH)  
**Journal:** International Journal of Combinatorial Optimization Problems and Informatics (IJCOPI), 2026.

---

##  Abstract
Learning under severe class imbalance exposes a fundamental limitation in standard probabilistic output gates (like Sigmoid or Probit): exponential saturation leads to vanishing gradients and systematically low recall for minority classes. 

This repository contains the official implementation of **PrimeSigmoid**, a logarithmically moderated output nonlinearity inspired by the **Prime Number Theorem (PNT)**. By embedding inverse prime-density scaling into a sigmoidal core, PrimeSigmoid preserves gradient sensitivity in high-activation regimes.

We also introduce **PrimeTanhGate**, a self-regularized hidden activation function that complements the output gate in deep architectures.

##  Mathematical Formulation

### 1. PrimeSigmoid (Output Gate)
Instead of the linear growth $x$ in the standard sigmoid exponent, we utilize a sublinear scaling derived from prime number density estimation:

$$g(x) = \frac{\pi x}{\ln(2 + |x| + \epsilon)}$$

$$\text{PrimeSigmoid}(x) = \frac{1}{1 + e^{-g(x)}}$$

### 2. PrimeTanhGate (Hidden Activation)
A gated activation mechanism for internal representations:

$$\text{PrimeTanhGate}(z) = z \cdot \left( \frac{1 + \tanh(g(z))}{2} \right)$$

##  Repository Structure

```text
.
├── data/
│   └── (Data is downloaded automatically from the script)
├── experiments/
│   ├── 01_Linear_Output_Gate_Benchmark.py  # Exp I: Logistic Regression Baseline
│   └── 02_MLP_Structural_Robustness.py     # Exp II: MLP Grid Search (PyTorch)
├── README.md
└── requirements.txt
