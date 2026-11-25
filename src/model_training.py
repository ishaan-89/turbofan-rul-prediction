"""
model_training.py â€” Training, tuning, timing and saving for the 4 core RUL models.

Why this file exists: FD001-FD004 all use the same 4 algorithms (Linear Regression,
Random Forest, XGBoost, LSTM). Keeping the training logic here means each dataset
notebook only calls functions, and the FastAPI backend can re-use the LSTM class and
the loaders to rebuild exactly the same models from disk.

Validation protocol (identical for all 4 models):
  * 80 training engines  -> inner split: 64 "fit" engines + 16 "tune" engines
  * hyperparameters are chosen on the 16 tune engines (test-like cut points)
  * the chosen configuration is re-trained on all 80 training engines
  * the 20 validation engines (200 cut points) are used ONLY for the final comparison
  * the official NASA test set is NOT touched in this phase
"""
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import evaluation as ev


# ----------------------------------------------------------------------------- reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------- metrics
def point_metrics(y_true, y_pred, cap: float) -> dict:
    """Metrics on test-like cut points, against the TRUE (uncapped) RUL.
    Primary selection metric: RMSE on points with true RUL <= cap (maintenance-relevant zone)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true <= cap
    return {
        "RMSE_le_cap": ev.rmse(y_true[m], y_pred[m]),
        "MAE_le_cap": ev.mae(y_true[m], y_pred[m]),
        "NASA_le_cap": ev.nasa_score(y_true[m], y_pred[m]),
        "bias_le_cap": float(np.mean(y_pred[m] - y_true[m])),
        "RMSE_all": ev.rmse(y_true, y_pred),
        "MAE_all": ev.mae(y_true, y_pred),
        "n_points": int(len(y_true)),
        "n_points_le_cap": int(m.sum()),
    }


def time_inference(predict_fn, n_repeats: int = 5):
    """Run predict_fn several times; return (predictions, median seconds)."""
    times, pred = [], None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        pred = predict_fn()
        times.append(time.perf_counter() - t0)
    return pred, float(np.median(times))


# ----------------------------------------------------------------------------- tabular models
def tune_tabular(make_model, param_grid, X_fit, y_fit, X_tune, y_tune_true, cap, fit_kwargs_fn=None,
                 verbose=True):
    """Try every config in `param_grid` (list of dicts): fit on fit-engines, score on tune cut points.
    Returns (results DataFrame sorted best-first, best params dict, best fitted model)."""
    rows, best = [], (np.inf, None, None)
    for i, params in enumerate(param_grid, 1):
        model = make_model(**params)
        t0 = time.perf_counter()
        kwargs = fit_kwargs_fn(model) if fit_kwargs_fn else {}
        model.fit(X_fit, y_fit, **kwargs)
        fit_s = time.perf_counter() - t0
        pred = np.clip(model.predict(X_tune), 0, None)
        m = point_metrics(y_tune_true, pred, cap)
        extra = {}
        if hasattr(model, "best_iteration") and getattr(model, "best_iteration", None) is not None:
            extra["best_iteration"] = int(model.best_iteration)
        rows.append({**params, **extra, "tune_RMSE_le_cap": m["RMSE_le_cap"],
                     "tune_NASA_le_cap": m["NASA_le_cap"], "fit_s": fit_s})
        if verbose:
            print(f"  [{i}/{len(param_grid)}] {params} {extra} -> tune RMSE<=cap {m['RMSE_le_cap']:.2f} "
                  f"({fit_s:.1f}s)")
        if m["RMSE_le_cap"] < best[0]:
            best = (m["RMSE_le_cap"], {**params, **extra}, model)
    res = pd.DataFrame(rows).sort_values("tune_RMSE_le_cap").reset_index(drop=True)
    return res, best[1], best[2]


# ----------------------------------------------------------------------------- LSTM
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_lstm(n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
    """LSTM regressor: sequence [batch, seq_len, n_features] -> RUL / cap (scalar)."""
    torch, nn = _torch()

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.lstm(x)              # out: [batch, seq_len, hidden]
            return self.head(out[:, -1]).squeeze(-1)   # use the LAST time step's summary

    return LSTMRegressor()


def lstm_predict(model, X, cap: float, device: str = "cpu", batch_size: int = 2048):
    torch, _ = _torch()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).float().cpu().numpy())
    return np.clip(np.concatenate(preds) * cap, 0, None)


def train_lstm(config: dict, X_train, y_train, cap: float, X_val=None, y_val=None, epochs: int = 50,
               patience: int = 8, lr: float = 1e-3, batch_size: int = 256, device: str = "cpu",
               seed: int = 42, verbose=True):
    """Train with MSE loss on RUL/cap (targets scaled to ~[0,1] for stable training).
    If validation windows are given: early stopping on validation loss, best weights restored.
    Returns (model, history DataFrame, best_epoch)."""
    torch, nn = _torch()
    set_seed(seed)
    model = build_lstm(X_train.shape[2], **config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train / cap, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    history, best_loss, best_state, best_epoch, bad = [], np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train_loss": total / len(Xt)}
        if X_val is not None:
            pv = lstm_predict(model, X_val, cap, device) / cap
            row["val_loss"] = float(np.mean((pv - y_val / cap) ** 2))
            if row["val_loss"] < best_loss - 1e-5:
                best_loss, best_epoch, bad = row["val_loss"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(row)
        if verbose and (epoch == 1 or epoch % 5 == 0):
            msg = f"    epoch {epoch:3d}  train {row['train_loss']:.4f}"
            if "val_loss" in row:
                msg += f"  val {row['val_loss']:.4f}  (best epoch {best_epoch})"
            print(msg)
        if X_val is not None and bad >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch}; best epoch {best_epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    return model, pd.DataFrame(history), best_epoch


# ----------------------------------------------------------------------------- saving
def save_model_bundle(directory, model, kind: str, metadata: dict):
    """Save a trained model + metadata.json. kind: 'sklearn' | 'xgboost' | 'lstm'."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "sklearn":
        joblib.dump(model, directory / "model.joblib", compress=3)
        metadata["model_file"] = "model.joblib"
    elif kind == "xgboost":
        model.save_model(directory / "model.json")
        metadata["model_file"] = "model.json"
    elif kind == "lstm":
        torch, _ = _torch()
        torch.save(model.state_dict(), directory / "model.pt")
        metadata["model_file"] = "model.pt"
    else:
        raise ValueError(kind)
    metadata["preprocessing_dir"] = "../preprocessing"
    metadata["feature_config"] = "../feature_config.json"
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    size_mb = sum(p.stat().st_size for p in directory.iterdir()) / 1e6
    return size_mb


def load_lstm(directory, device: str = "cpu"):
    torch, _ = _torch()
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    model = build_lstm(meta["n_features"], **meta["architecture"])
    model.load_state_dict(torch.load(directory / "model.pt", map_location=device))
    return model.to(device).eval(), meta
