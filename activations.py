import math
import numpy as np
import torch
import torch.nn as nn

# =========================
# NumPy output gates
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def primesigmoid(z, eps=1e-8):
    return 1 / (1 + np.exp(-(np.pi * z) / np.log(2 + np.abs(z) + eps)))

def tanh_rescaled(z):
    return (1 + np.tanh(z)) / 2

def hard_sigmoid(z):
    return np.clip(0.2 * z + 0.5, 0, 1)

def probit(z):
    return 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))

# =========================
# Torch output gates
# =========================
class SigmoidGate(nn.Module):
    def forward(self, z):
        return torch.sigmoid(z)

class PrimeSigmoidGate(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        denom = torch.log(2 + torch.abs(z) + self.eps)
        return 1 / (1 + torch.exp(-(torch.pi * z) / denom))

class TanhRescaledGate(nn.Module):
    def forward(self, z):
        return (1 + torch.tanh(z)) / 2

class HardSigmoidGate(nn.Module):
    def forward(self, z):
        return torch.clamp(0.2 * z + 0.5, 0, 1)

class ProbitGate(nn.Module):
    def forward(self, z):
        return 0.5 * (1 + torch.erf(z / np.sqrt(2)))

# =========================
# Prime-based hidden activations
# =========================
def prime_core(z, eps=1e-8):
    return (torch.pi * z) / torch.log(2 + torch.abs(z) + eps)

class PrimeGateSoftplus(nn.Module):
    def forward(self, z):
        gate = torch.tanh(torch.nn.functional.softplus(prime_core(z)))
        return z * gate

class PrimeTanhGate(nn.Module):
    def forward(self, z):
        gate = (1 + torch.tanh(prime_core(z))) / 2
        return z * gate
