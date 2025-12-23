import numpy as np
import torch
import torch.nn as nn

# =========================
# Logistic regression
# =========================
class LogisticCustom:
    def __init__(self, activation, lr=0.05, epochs=5000,
                 patience=20, window=10, rel_tol=1e-3):
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.window = window
        self.rel_tol = rel_tol

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        losses, wait = [], 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = np.clip(self.activation(z), 1e-8, 1-1e-8)

            loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
            losses.append(loss)

            dw = np.mean((p-y)[:,None]*X, axis=0)
            db = np.mean(p-y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if len(losses) > self.window:
                prev = losses[-self.window]
                if (prev - loss) / prev < self.rel_tol:
                    wait += 1
                else:
                    wait = 0
                if wait >= self.patience:
                    break

    def predict_proba(self, X):
        return np.clip(self.activation(X @ self.w + self.b), 1e-8, 1-1e-8)

# =========================
# MLP
# =========================
class MLP(nn.Module):
    def __init__(self, hidden_act, output_gate, in_dim=3, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.h = hidden_act
        self.o = output_gate

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.h(z1)
        z2 = self.fc2(a1)
        return self.o(z2)
