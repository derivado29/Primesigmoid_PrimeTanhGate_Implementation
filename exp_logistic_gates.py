import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score
)

from activations import (
    sigmoid, primesigmoid, tanh_rescaled,
    hard_sigmoid, probit
)
from models import LogisticCustom

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# =========================
# Data
# =========================
url = "https://raw.githubusercontent.com/derivado29/Loan-Default-Prediction/main/Default_Fin.csv"
df = pd.read_csv(url, encoding="latin1", engine="python")

df.columns = df.columns.str.strip().str.replace(" ", "_")
X = df[["Employed", "Bank_Balance", "Annual_Salary"]].values
y = df["Defaulted"].values.astype(int)

# =========================
# Evaluation
# =========================
def evaluate_gate(name, activation):
    skf = StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED)
    metrics = {k: [] for k in [
        "Accuracy","Balanced_Accuracy","Precision","Recall",
        "F1","Kappa","MCC","AUC","LogLoss"
    ]}

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        model = LogisticCustom(activation)
        model.fit(Xtr, y[tr])

        p = model.predict_proba(Xte)
        y_pred = (p >= 0.5).astype(int)

        metrics["Accuracy"].append(accuracy_score(y[te], y_pred))
        metrics["Balanced_Accuracy"].append(balanced_accuracy_score(y[te], y_pred))
        metrics["Precision"].append(precision_score(y[te], y_pred, zero_division=0))
        metrics["Recall"].append(recall_score(y[te], y_pred, zero_division=0))
        metrics["F1"].append(f1_score(y[te], y_pred, zero_division=0))
        metrics["Kappa"].append(cohen_kappa_score(y[te], y_pred))
        metrics["MCC"].append(matthews_corrcoef(y[te], y_pred))
        metrics["AUC"].append(roc_auc_score(y[te], p))
        metrics["LogLoss"].append(log_loss(y[te], p))

    return {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k,v in metrics.items()}

gates = {
    "Sigmoid": sigmoid,
    "PrimeSigmoid": primesigmoid,
    "TanhRescaled": tanh_rescaled,
    "HardSigmoid": hard_sigmoid,
    "Probit": probit
}

rows = []
for name, fn in gates.items():
    row = evaluate_gate(name, fn)
    row["Model"] = name
    rows.append(row)

df_results = pd.DataFrame(rows)
print(df_results)
