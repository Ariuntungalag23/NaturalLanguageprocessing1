# CBOW_run.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from torch.utils.data import DataLoader, TensorDataset

# ===============================
# PATHS (Windows-д тааруулсан)
# ===============================
ART_DIR = r"C:/Users/User/OneDrive - Mongolian University of Science and Technology/Documents/vscode/NLP/artifacts"

X_PATH = os.path.join(ART_DIR, "w2v_cbow_embeddings.npy")
Y_PATH = os.path.join(ART_DIR, "w2v_cbow_labels.npy")

# ===============================
# LOAD DATA
# ===============================
X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# CROSS VALIDATION (LogReg)
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=300, solver="liblinear"),
    param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=2
)

print("[INFO] Running Cross-Validation for Logistic Regression...")
grid.fit(X_train, y_train)

cv_df = pd.DataFrame(grid.cv_results_)
top10_C = (
    cv_df.sort_values("mean_test_score", ascending=False)
         .head(10)["param_C"]
         .astype(float)
         .tolist()
)

# ===============================
# TINY LSTM (LIGHT)
# ===============================
class TinyLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


def train_eval_lstm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyLSTM(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.from_numpy(X_train).unsqueeze(1)
    ytr = torch.from_numpy(y_train.astype(np.float32))

    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=128,
        shuffle=True
    )

    model.train()
    for _ in range(2):  # хөнгөн
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).unsqueeze(1).to(device)
        preds = (torch.sigmoid(model(Xt)) > 0.5).cpu().numpy()

    return preds


# ===============================
# 10 PARAM × 4 MODEL = 40 RUNS
# ===============================
rows = []

for run_id, C in enumerate(top10_C, 1):
    print(f"[RUN {run_id}/10] C={C}")

    # Logistic Regression
    lr = LogisticRegression(C=C, max_iter=300, solver="liblinear")
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    rows.append([
        run_id, C, "LogisticRegression",
        accuracy_score(y_test, pred),
        f1_score(y_test, pred)
    ])

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=80,
        random_state=run_id,
        n_jobs=2
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rows.append([
        run_id, C, "RandomForest",
        accuracy_score(y_test, pred),
        f1_score(y_test, pred)
    ])

    # AdaBoost
    ada = AdaBoostClassifier(
        n_estimators=80,
        random_state=run_id
    )
    ada.fit(X_train, y_train)
    pred = ada.predict(X_test)
    rows.append([
        run_id, C, "AdaBoost",
        accuracy_score(y_test, pred),
        f1_score(y_test, pred)
    ])

    # LSTM
    pred = train_eval_lstm()
    rows.append([
        run_id, C, "LSTM",
        accuracy_score(y_test, pred),
        f1_score(y_test, pred)
    ])

# ===============================
# SAVE RESULTS
# ===============================
df = pd.DataFrame(
    rows,
    columns=["run", "param_C", "model", "accuracy", "f1"]
)

OUT_PATH = os.path.join(ART_DIR, "w2v_cbow_40_results.csv")
df.to_csv(OUT_PATH, index=False)

print("[DONE] Results saved to:", OUT_PATH)
print("Total rows:", len(df))
