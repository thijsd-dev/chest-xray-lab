# chest_xray_lab/explain/sofi_core.py
from __future__ import annotations
from pathlib import Path
import json, random
from typing import Literal, Optional, Dict, Any

import numpy as np
import torch

from chest_xray_lab.data import load_gray01
from chest_xray_lab.explain.segmentation import compute_superpixels_from_gray
from chest_xray_lab.model import prob_pos_batch  # <- adjust to your actual location

# ---------------------------------------------------------------------
# Baseline configs
# ---------------------------------------------------------------------
class SOFIBaseline:
    """
    Tiny baseline provider so we can say:
        baseline = SOFIBaseline.kind("dataset_mean", value=0.4321)
    """
    def __init__(self, kind: str, value: float):
        self.kind = kind
        self.value = float(value)

    @classmethod
    def dataset_mean(cls, value: float):
        return cls("dataset_mean", value)

    @classmethod
    def per_image(cls):
        return cls("per_image_mean", -1.0)

    @classmethod
    def constant(cls, value: float):
        return cls("constant", value)

    def resolve(self, g0: np.ndarray) -> float:
        if self.kind == "per_image_mean":
            return float(g0.mean())
        return self.value


# ---------------------------------------------------------------------
# low-level helpers (same logic you had)
# ---------------------------------------------------------------------
def auc_uniform_01(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size < 2:
        return float(y.mean() if y.size else 0.0)
    dx = 1.0 / (len(y) - 1)
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y, dx=dx))

def _segment_indices_torch(segmap: np.ndarray, device: torch.device):
    H, W = segmap.shape
    K = int(segmap.max()) + 1
    flat = torch.from_numpy(segmap.reshape(-1).astype(np.int64)).to(device)
    idxs = [(flat == k).nonzero(as_tuple=True)[0] for k in range(K)]
    return idxs, H, W, K

@torch.no_grad()
def _img_to_tensor(gray01: np.ndarray, device: torch.device):
    t = torch.from_numpy(gray01.astype(np.float32, copy=False)).to(device)
    t = t.unsqueeze(0).unsqueeze(0).contiguous(memory_format=torch.channels_last)
    return t

@torch.no_grad()
def morf_curve(model, gray01, segmap, order, baseline_value, device,
               early_stop_p=None, chunk=128,
               enforce_monotone=False, monotone_tol=1e-3):
    cur = _img_to_tensor(gray01, device)
    idxs, H, W, K = _segment_indices_torch(segmap, device)
    p0 = prob_pos_batch(model, cur)[0]
    eps = 1e-8

    vals = [1.0]
    total = len(order)
    s = 0
    while s < total:
        upto = min(total, s + chunk)
        B = upto - s
        batch = cur.repeat(B, 1, 1, 1)
        flat = batch.view(B, -1)
        flat[0, idxs[order[s]]] = float(baseline_value)
        for bi in range(1, B):
            flat[bi] = flat[bi-1]
            flat[bi, idxs[order[s+bi]]] = float(baseline_value)
        probs = prob_pos_batch(model, batch)
        vals.extend((probs / max(p0, eps)).tolist())
        cur = batch[-1:].contiguous()
        s = upto
        if early_stop_p is not None and probs[-1] <= early_stop_p:
            vals.extend([vals[-1]] * (total - len(vals) + 1))
            break

    y = np.asarray(vals, dtype=np.float32)
    if enforce_monotone:
        y = np.minimum.accumulate(y + monotone_tol)
    return y, auc_uniform_01(y), float(p0)

@torch.no_grad()
def seed_order_greedy_cumulative(model, gray01, segmap, baseline_value, device, batch_sz=128):
    cur = _img_to_tensor(gray01, device)
    idxs, H, W, K = _segment_indices_torch(segmap, device)
    remaining = list(range(K))
    order = []
    p_curr = prob_pos_batch(model, cur)[0]
    flat_cur_1 = cur.view(1, -1)
    for _ in range(K):
        best_sid, best_drop = None, -1.0
        for b0 in range(0, len(remaining), batch_sz):
            cand = remaining[b0:b0+batch_sz]
            B = len(cand)
            batch = cur.repeat(B, 1, 1, 1)
            flat = batch.view(B, -1)
            for bi, sid in enumerate(cand):
                flat[bi, idxs[sid]] = float(baseline_value)
            probs = prob_pos_batch(model, batch)
            drops = np.maximum(0.0, p_curr - probs)
            j = int(np.argmax(drops))
            if drops[j] > best_drop:
                best_drop = float(drops[j]); best_sid = int(cand[j])
        flat_cur_1[0, idxs[best_sid]] = float(baseline_value)
        p_curr = prob_pos_batch(model, cur)[0]
        order.append(best_sid)
        remaining.remove(best_sid)
    return order

def bounded_monotone_curve(y: np.ndarray, monotone_tol: float = 1e-3) -> np.ndarray:
    yb = np.asarray(y, dtype=np.float32)
    yb = np.clip(yb, 0.0, 1.0)
    yb = np.minimum.accumulate(yb + monotone_tol)
    return np.clip(yb, 0.0, 1.0)

def k_at_tau(y_bounded: np.ndarray, tau: float = 0.1) -> int:
    hit = np.flatnonzero(y_bounded <= float(tau))
    return int(hit[0]) if hit.size else (len(y_bounded) - 1)

# ---------------------------------------------------------------------
# high-level runner
# ---------------------------------------------------------------------
def run_sofi_for_image(
    model,
    img_path: str,
    device: torch.device,
    *,
    n_segments: int = 100,
    compactness: float = 10.0,
    sigma: float = 1.0,
    use_lab: bool = False,
    baseline: SOFIBaseline = SOFIBaseline.per_image(),
) -> Dict[str, Any]:
    # 1) load & segment
    g0 = load_gray01(img_path).astype(np.float32)
    seg0, _ = compute_superpixels_from_gray(
        g0, n_segments=n_segments, compactness=compactness, sigma=sigma, use_lab=use_lab
    )
    seg0 = seg0.astype(np.int32)
    K = int(seg0.max()) + 1

    base_val = baseline.resolve(g0)

    # 2) seed
    order_seed = seed_order_greedy_cumulative(model, g0, seg0, base_val, device)
    y_seed, auc_seed, p0 = morf_curve(model, g0, seg0, order_seed, base_val, device, enforce_monotone=False)
    yb_seed = bounded_monotone_curve(y_seed)

    return {
        "image": img_path,
        "g0": g0,
        "segmap": seg0,
        "baseline_value": float(base_val),
        "order_seed": order_seed,
        "y_seed": y_seed,
        "yb_seed": yb_seed,
        "auc_seed_raw": float(auc_seed),
        "p0": float(p0),
        "params": {
            "n_segments": n_segments,
            "compactness": compactness,
            "sigma": sigma,
            "use_lab": use_lab,
            "baseline_kind": baseline.kind,
        },
    }
