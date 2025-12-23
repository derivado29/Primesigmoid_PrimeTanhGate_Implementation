# ============================================================
# REGRESIÓN LOGÍSTICA -COLAB READY-
# Comparación completa de GATES DE SALIDA
# ============================================================
import math
import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score,
    confusion_matrix
)

# ---------------- CONFIG ----------------
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

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

print("Dataset:", X.shape, " Clases:", np.bincount(y))

# ---------------- OUTPUT GATES ----------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def primesigmoid(z):
    return 1 / (1 + np.exp(-(np.pi * z) / np.log(2 + np.abs(z) + 1e-8)))

def sigmoid_T2(z):
    return 1 / (1 + np.exp(-(z / 2.0)))

def sigmoid_T05(z):
    return 1 / (1 + np.exp(-(z / 0.5)))

def tanh_rescaled(z):
    return (1 + np.tanh(z)) / 2

def hard_sigmoid(z):
    return np.clip(0.2 * z + 0.5, 0, 1)

def probit(z):
    return 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))


# ---------------- LOGISTIC MODEL ----------------
class LogisticCustom:
    def __init__(
        self,
        activation,
        lr=0.05,
        epochs=5000,
        patience=20,
        window=10,
        rel_tol=1e-3
    ):
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.window = window
        self.rel_tol = rel_tol

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        losses = []
        wait = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_hat = self.activation(z)
            y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)

            loss = -np.mean(
                y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
            )
            losses.append(loss)

            dw = np.mean((y_hat - y)[:, None] * X, axis=0)
            db = np.mean(y_hat - y)

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
        return np.clip(self.activation(X @ self.w + self.b), 1e-8, 1 - 1e-8)

# ---------------- K-FOLD EVAL ----------------
def evaluate_kfold(X, y, activation, name):
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

    return {
        "Model": name,
        **{k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}
    }

# ---------------- CONFUSION MATRIX ----------------
def confusion_matrix_kfold(X, y, activation):
    skf = StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED)
    cm_total = np.zeros((2,2), dtype=int)

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        model = LogisticCustom(activation)
        model.fit(Xtr, y[tr])

        y_pred = (model.predict_proba(Xte) >= 0.5).astype(int)
        cm_total += confusion_matrix(y[te], y_pred)

    return cm_total

# ---------------- RUN ALL ----------------
gates = {
    "Sigmoid": sigmoid,
    "PrimeSigmoid": primesigmoid,
    "Sigmoid_T=2.0": sigmoid_T2,
    "Sigmoid_T=0.5": sigmoid_T05,
    "TanhRescaled": tanh_rescaled,
    "Probit": probit,
    "HardSigmoid": hard_sigmoid
}

results = []
cms = {}

for name, fn in gates.items():
    results.append(evaluate_kfold(X, y, fn, name))
    cms[name] = confusion_matrix_kfold(X, y, fn)

df_results = pd.DataFrame(results)
display(df_results)

print("\n=== MATRICES DE CONFUSIÓN (AGREGADAS) ===")
for name, cm in cms.items():
    print(f"\n{name}")
    print(cm)
