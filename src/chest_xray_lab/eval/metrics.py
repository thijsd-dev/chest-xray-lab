import numpy as np, torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60); return 1.0/(1.0+np.exp(-x))

def compute_epoch_metrics(y_true, y_logits, thresh=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _sigmoid_np(np.asarray(y_logits, dtype=float))
    y_pred = (y_prob >= thresh).astype(int)
    out = {}
    try: out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception: out["auroc"] = float("nan")
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1"]  = float(f1_score(y_true, y_pred, zero_division=0))
    return out

@torch.no_grad()
def collect_logits(model, loader, device):
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(1).detach().cpu().numpy()
        y = yb.squeeze(1).detach().cpu().numpy()
        logits_all.append(logits); y_all.append(y)
    return (np.concatenate(y_all).astype(np.float32),
            np.concatenate(logits_all).astype(np.float32))

def compute_shuf_auc(y_true, y_logits, seed=42):
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(y_true)
    return float(roc_auc_score(y_perm, _sigmoid_np(y_logits)))
