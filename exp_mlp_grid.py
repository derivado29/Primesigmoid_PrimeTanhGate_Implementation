import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score
)

from activations import (
    SigmoidGate, PrimeSigmoidGate,
    TanhRescaledGate, HardSigmoidGate, ProbitGate,
    PrimeGateSoftplus, PrimeTanhGate
)
from models import MLP

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data
# =========================
url = "https://raw.githubusercontent.com/derivado29/Loan-Default-Prediction/main/Default_Fin.csv"
df = pd.read_csv(url, encoding="latin1", engine="python")
df.columns = df.columns.str.strip().str.replace(" ", "_")

X = df[["Employed", "Bank_Balance", "Annual_Salary"]].values
y = df["Defaulted"].values.astype(int)

# =========================
# Config
# =========================
hidden_acts = {
    "ReLU": nn.ReLU(),
    "GELU": nn.GELU(),
    "PrimeGateSoftplus": PrimeGateSoftplus(),
    "PrimeTanhGate": PrimeTanhGate(),
}

output_gates = {
    "Sigmoid": SigmoidGate(),
    "PrimeSigmoid": PrimeSigmoidGate(),
    "TanhRescaled": TanhRescaledGate(),
    "HardSigmoid": HardSigmoidGate(),
    "Probit": ProbitGate(),
}

# =========================
# Evaluation
# =========================
def eval_combo(hidden_act, output_gate):
    skf = StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED)
    f1s, mccs = [], []

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = torch.tensor(scaler.fit_transform(X[tr]), dtype=torch.float32, device=device)
        Xte = torch.tensor(scaler.transform(X[te]), dtype=torch.float32, device=device)
        ytr = torch.tensor(y[tr], dtype=torch.float32, device=device).view(-1,1)

        model = MLP(hidden_act, output_gate).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.BCELoss()

        for _ in range(300):
            opt.zero_grad()
            p = torch.clamp(model(Xtr), 1e-8, 1-1e-8)
            loss = loss_fn(p, ytr)
            loss.backward()
            opt.step()

        with torch.no_grad():
            p = torch.clamp(model(Xte), 1e-8, 1-1e-8).cpu().numpy().ravel()

        y_pred = (p >= 0.5).astype(int)
        f1s.append(f1_score(y[te], y_pred))
        mccs.append(matthews_corrcoef(y[te], y_pred))

    return np.mean(f1s), np.mean(mccs)

rows = []
for h_name, h in hidden_acts.items():
    for o_name, o in output_gates.items():
        f1, mcc = eval_combo(h, o)
        rows.append({
            "Model": f"{h_name} + {o_name}",
            "F1": f1,
            "MCC": mcc
        })

df = pd.DataFrame(rows).sort_values("F1", ascending=False)
print(df)
