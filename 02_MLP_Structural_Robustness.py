# ============================================================
# MATRIZ COMPLETA: Hidden activation × Output gate  -COLAB READY-
# Objetivo: comparar pares 
# Dataset: Default_Fin.csv (Employed, Bank_Balance, Annual_Salary)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, log_loss,
    cohen_kappa_score, matthews_corrcoef
)

# ---------------- CONFIG ----------------
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATA ----------------
url = "https://raw.githubusercontent.com/derivado29/Loan-Default-Prediction/main/Default_Fin.csv"
df = pd.read_csv(url, encoding="latin1", engine="python")
df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("?", "", regex=False)
)

X = df[["Employed", "Bank_Balance", "Annual_Salary"]].values
y = df["Defaulted"].values.astype(int)

print("Dataset:", X.shape, "  Clases:", np.bincount(y))

# ---------------- UTILS ----------------
def clip01(p):
    return np.clip(p, 1e-8, 1 - 1e-8)

# ---------------- OUTPUT GATES ----------------
class SigmoidGate(nn.Module):
    def forward(self, z):
        return torch.sigmoid(z)

class SigmoidTempGate(nn.Module):
    def __init__(self, T=2.0):
        super().__init__()
        self.T = float(T)
    def forward(self, z):
        return torch.sigmoid(z / self.T)

class PrimeSigmoidGate(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        denom = torch.log(2 + torch.abs(z) + self.eps)
        return 1 / (1 + torch.exp(-(torch.pi * z) / denom))

class TanhRescaledGate(nn.Module):
    # (1 + tanh(z))/2  -> (0,1)
    def forward(self, z):
        return (1 + torch.tanh(z)) / 2

class HardSigmoidGate(nn.Module):
    # clip(0.2z + 0.5, 0, 1)
    def forward(self, z):
        return torch.clamp(0.2 * z + 0.5, 0.0, 1.0)

class ProbitGate(nn.Module):
    # Normal CDF: 0.5*(1 + erf(z/sqrt(2)))
    def forward(self, z):
        return 0.5 * (1 + torch.erf(z / np.sqrt(2)))

# ---------------- PRIME CORE (for hidden family) ----------------
def prime_core(z, eps=1e-8):
    return (torch.pi * z) / torch.log(2 + torch.abs(z) + eps)

# ---------------- HIDDEN ACTIVATIONS ----------------
class PrimeGateSoftplus(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        c = prime_core(z, self.eps)
        gate = torch.tanh(torch.nn.functional.softplus(c))
        return z * gate

class PrimeTanhGate(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        c = prime_core(z, self.eps)
        gate = (1 + torch.tanh(c)) / 2
        return z * gate

# ---------------- MLP ----------------
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
        p  = self.o(z2)
        return p

# ---------------- K-FOLD EVAL ----------------
def evaluate_combo_kfold(X, y, hidden_act, output_gate, name,
                         epochs=300, lr=1e-2, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)

    metrics = {k: [] for k in [
        "Accuracy","Balanced_Accuracy","Precision","Recall",
        "F1","Kappa","MCC","AUC","LogLoss"
    ]}

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(y[tr], dtype=torch.float32, device=device).view(-1,1)

        model = MLP(hidden_act, output_gate).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            p = torch.clamp(model(Xtr_t), 1e-8, 1-1e-8)
            loss = loss_fn(p, ytr_t)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p = model(Xte_t).detach().cpu().numpy().ravel()

        p = clip01(p)
        y_pred = (p >= 0.5).astype(int)
        yte = y[te]

        metrics["Accuracy"].append(accuracy_score(yte, y_pred))
        metrics["Balanced_Accuracy"].append(balanced_accuracy_score(yte, y_pred))
        metrics["Precision"].append(precision_score(yte, y_pred, zero_division=0))
        metrics["Recall"].append(recall_score(yte, y_pred, zero_division=0))
        metrics["F1"].append(f1_score(yte, y_pred, zero_division=0))
        metrics["Kappa"].append(cohen_kappa_score(yte, y_pred))
        metrics["MCC"].append(matthews_corrcoef(yte, y_pred))
        metrics["AUC"].append(roc_auc_score(yte, p))
        metrics["LogLoss"].append(log_loss(yte, p))

    row = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}
    row["Model"] = name
    return row

# ---------------- GRID ----------------
hidden_acts = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "ELU": nn.ELU(),
    "Tanh": nn.Tanh(),
    "GELU": nn.GELU(),
    "Swish(SiLU)": nn.SiLU(),
    "Mish": nn.Mish(),
    "PrimeGateSoftplus": PrimeGateSoftplus(),
    "PrimeTanhGate": PrimeTanhGate(),
}

output_gates = {
    "Sigmoid": SigmoidGate(),
    "PrimeSigmoid": PrimeSigmoidGate(),
    "TanhRescaled": TanhRescaledGate(),
    "HardSigmoid": HardSigmoidGate(),
    "Probit(NormalCDF)": ProbitGate(),
    "Sigmoid_T=2.0": SigmoidTempGate(T=2.0),
    "Sigmoid_T=0.5": SigmoidTempGate(T=0.5),
}

# ---------------- RUN ALL ----------------
rows = []
for h_name, h_act in hidden_acts.items():
    for o_name, o_gate in output_gates.items():
        name = f"MLP {h_name} + {o_name}"
        rows.append(evaluate_combo_kfold(X, y, h_act, o_gate, name))

df_res = pd.DataFrame(rows)

# extra: rankings (convert mean string -> float mean)
def mean_from_pm(s):
    # "0.1234 ± 0.0567" -> 0.1234
    return float(str(s).split("±")[0].strip())

for col in ["Accuracy","Balanced_Accuracy","Precision","Recall","F1","Kappa","MCC","AUC","LogLoss"]:
    df_res[col+"_mean"] = df_res[col].apply(mean_from_pm)

# Mostrar tabla completa (y rankings)
display(df_res[["Model","Accuracy","Balanced_Accuracy","Precision","Recall","F1","Kappa","MCC","AUC","LogLoss"]])

print("\nTOP 10 por F1:")
display(df_res.sort_values("F1_mean", ascending=False)[["Model","F1","MCC","Balanced_Accuracy","Recall","Precision","AUC","LogLoss"]].head(10))

print("\nTOP 10 por MCC:")
display(df_res.sort_values("MCC_mean", ascending=False)[["Model","MCC","F1","Balanced_Accuracy","Recall","Precision","AUC","LogLoss"]].head(10))

print("\nResumen por gate de salida (promedio sobre activaciones internas):")
summary_out = df_res.groupby(df_res["Model"].str.split(" + ").str[-1])[
    ["F1_mean","MCC_mean","Balanced_Accuracy_mean","Recall_mean","Precision_mean","AUC_mean","LogLoss_mean"]
].mean().sort_values("F1_mean", ascending=False)
display(summary_out)
