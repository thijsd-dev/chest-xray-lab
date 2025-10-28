# split.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import re
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Robust patient-id extractor covering common CXR filenames
_PATIENT_PATTERNS = [
    re.compile(r"(person\d+)", re.I),        # person1234_*
    re.compile(r"^(IM-\d+-\d+)", re.I),      # IM-1234-5678
    re.compile(r"^(uid_\w+)", re.I),         # uid_* (fallback if present)
]

def patient_id_from_path(p: str) -> str:
    name = Path(p).stem
    for rx in _PATIENT_PATTERNS:
        m = rx.search(name)
        if m:
            return m.group(1).lower()
    return name.lower()  # final fallback: whole stem

def infer_label_from_path(p: Path) -> int:
    parts = [q.lower() for q in p.parts]
    if "pneumonia" in parts or "bacteria" in p.name.lower() or "virus" in p.name.lower():
        return 1
    if "normal" in parts or "normal" in p.name.lower():
        return 0
    raise ValueError(f"Cannot infer label for: {p}")

def collect_all_images(raw_root: Path) -> List[Tuple[str, int, str]]:
    """
    Walk raw_root/{train,val,test}/{NORMAL,PNEUMONIA}/**/* and
    return list of (abs_path, label, patient_id).
    The original Kaggle split is *ignored* after collection.
    """
    items: List[Tuple[str, int, str]] = []
    for split in ("train", "val", "test"):
        base = raw_root / split
        if not base.exists(): 
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                y = infer_label_from_path(p)
                pid = patient_id_from_path(str(p))
                items.append((str(p.resolve()), y, pid))
    items.sort()
    return items

def summarize_split(tag: str, idxs: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    ys = y[idxs]
    n_neg = int((ys == 0).sum())
    n_pos = int((ys == 1).sum())
    tot   = int(len(idxs))
    pos_pct = 100.0 * n_pos / max(1, tot)
    n_pids  = len(set(groups[idxs]))
    return dict(tag=tag, total=tot, neg=n_neg, pos=n_pos, pos_pct=pos_pct, patients=n_pids)

def split_by_patient(
    raw_root: Path,
    seed: int = 1337,
    test_frac: float = 0.11,
    val_frac: float  = 0.09,
) -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Returns dict with keys 'train','val','test'.
    Each value is a list of (abs_path, label, patient_id).
    Does not touch the filesystem beyond reading.
    """
    items = collect_all_images(raw_root)
    if not items:
        raise RuntimeError(f"No images found under {raw_root}")

    X = np.array([p for p, _, _ in items], dtype=object)
    y = np.array([int(lbl) for _, lbl, _ in items], dtype=np.int32)
    groups = np.array([pid for _, _, pid in items], dtype=object)

    # 1) hold-out test by patient
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(X, y, groups))

    # 2) split train/val by patient
    val_size_rel = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss2.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    train_idx = trainval_idx[train_idx_rel]
    val_idx   = trainval_idx[val_idx_rel]

    # summaries + leakage checks
    S_tr = summarize_split("train", train_idx, y, groups)
    S_va = summarize_split("val",   val_idx,   y, groups)
    S_te = summarize_split("test",  test_idx,  y, groups)

    pid_train = set(groups[train_idx]); pid_val = set(groups[val_idx]); pid_test = set(groups[test_idx])
    leakages = (len(pid_train & pid_val), len(pid_train & pid_test), len(pid_val & pid_test))
    print("== Patient-level split summary ==")
    print(f"{'split':<6} {'total':>6} {'NORMAL':>8} {'PNEUM.':>8} {'Pos%':>7} {'patients':>10}")
    for S in (S_tr, S_va, S_te):
        print(f"{S['tag']:<6} {S['total']:>6} {S['neg']:>8} {S['pos']:>8} {S['pos_pct']:>6.1f}% {S['patients']:>10}")
    print("\nPatient overlap (should be 0): train∩val=%d, train∩test=%d, val∩test=%d" % leakages)

    def pack(idxs):
        return [(X[i], int(y[i]), str(groups[i])) for i in idxs]

    return {
        "train": pack(train_idx),
        "val":   pack(val_idx),
        "test":  pack(test_idx),
        "pos_weight": float((y[train_idx] == 0).sum() / max(1, (y[train_idx] == 1).sum())),
    }
