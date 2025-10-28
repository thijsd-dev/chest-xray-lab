# preproc.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

# -------------------------------
# Config & hashing
# -------------------------------
@dataclass(frozen=True)
class PreprocConfig:
    # crop
    border_ratio: float = 0.10
    gauss_ks: int = 3
    window: int = 7
    dark_frac: float = 0.8
    min_keep_run: int = 8
    max_crop_frac: float = 0.12
    pad_after_crop: int = 4

    # resize
    target_hw: tuple[int, int] = (224, 224)  # (H,W)
    resize_mode: str = "fit_pad"             # {"fit_pad","cover_crop"}
    pad_mode: str = "reflect"                # {"reflect","constant"}
    pad_value: int | None = None             # used if pad_mode=="constant"

    # augment (train-only)
    aug_hflip_p: float = 0.5
    aug_rotate_deg: int = 5
    aug_rotate_p: float = 0.5

def preproc_hash(cfg: PreprocConfig) -> str:
    d = asdict(cfg)
    # normalize tuple to list for stable JSON
    d["target_hw"] = list(d["target_hw"])
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:10]

# -------------------------------
# Core ops (your algorithms)
# -------------------------------
def crop_dark_frame(
    img_bgr_or_gray: np.ndarray,
    *,
    border_ratio: float = 0.10,
    gauss_ks: int = 3,
    window: int = 7,
    dark_frac: float = 0.8,
    min_keep_run: int = 8,
    max_crop_frac: float = 0.12,
    pad: int = 4,
) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    """Remove dark canvas/outlines from all four sides. Returns (cropped_img, (y0,y1,x0,x1))."""
    img = img_bgr_or_gray
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = g.shape
    if h < 16 or w < 16:
        return g, (0, h, 0, w)

    bw, bh = max(2, int(border_ratio * w)), max(2, int(border_ratio * h))
    band = np.zeros_like(g, np.uint8)
    band[:, :bw] = 1; band[:, w - bw:] = 1; band[:bh, :] = 1; band[h - bh:, :] = 1

    ks = max(1, gauss_ks | 1)
    sm = cv2.GaussianBlur(g, (ks, ks), 0)
    inv = 255 - sm
    inv_band = inv[band > 0]
    if inv_band.size == 0:
        return g, (0, h, 0, w)

    thr_val, _ = cv2.threshold(inv_band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (inv >= thr_val).astype(np.uint8)

    def _stop_index_along_rows(mask, from_top=True):
        row_frac = mask.mean(axis=1)
        k = max(1, window)
        kernel = np.ones(k) / k
        smoothed = np.convolve(row_frac, kernel, mode="same")
        idxs = range(0, h) if from_top else range(h - 1, -1, -1)
        keep_run = 0; cut = 0
        for i in idxs:
            is_border = smoothed[i] >= dark_frac
            keep_run = 0 if is_border else (keep_run + 1)
            cut += 1
            if keep_run >= min_keep_run:
                cut -= keep_run
                break
        cut = min(cut + pad, int(max_crop_frac * h))
        return (h - cut) if not from_top else cut

    def _stop_index_along_cols(mask, from_left=True):
        col_frac = mask.mean(axis=0)
        k = max(1, window)
        kernel = np.ones(k) / k
        smoothed = np.convolve(col_frac, kernel, mode="same")
        idxs = range(0, w) if from_left else range(w - 1, -1, -1)
        keep_run = 0; cut = 0
        for j in idxs:
            is_border = smoothed[j] >= dark_frac
            keep_run = 0 if is_border else (keep_run + 1)
            cut += 1
            if keep_run >= min_keep_run:
                cut -= keep_run
                break
        cut = min(cut + pad, int(max_crop_frac * w))
        return (w - cut) if not from_left else cut

    y0 = _stop_index_along_rows(dark, from_top=True)
    y1 = _stop_index_along_rows(dark, from_top=False)
    x0 = _stop_index_along_cols(dark, from_left=True)
    x1 = _stop_index_along_cols(dark, from_left=False)

    y0 = int(np.clip(y0, 0, h - 2))
    y1 = int(np.clip(y1, y0 + 1, h))
    x0 = int(np.clip(x0, 0, w - 2))
    x1 = int(np.clip(x1, x0 + 1, w))
    return g[y0:y1, x0:x1], (y0, y1, x0, x1)

def resize_cover_then_center_crop(img: np.ndarray, out=(224, 224)) -> np.ndarray:
    th, tw = out
    H, W = img.shape[:2]
    scale = max(th / H, tw / W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    inter = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    r = cv2.resize(img, (newW, newH), interpolation=inter)
    y0 = max(0, (newH - th) // 2); x0 = max(0, (newW - tw) // 2)
    return r[y0:y0 + th, x0:x0 + tw]

def resize_fit_then_pad(img: np.ndarray, out=(224, 224), pad_mode="reflect", pad_value=None) -> np.ndarray:
    th, tw = out
    H, W = img.shape[:2]
    scale = min(th / H, tw / W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    inter = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    r = cv2.resize(img, (newW, newH), interpolation=inter)
    top = (th - newH) // 2; bottom = th - newH - top
    left = (tw - newW) // 2; right = tw - newW - left
    if pad_mode == "constant":
        if pad_value is None:
            pad_value = int(np.median(img))
        value = pad_value if r.ndim == 2 else [pad_value] * 3
        return cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_REFLECT_101)

def normalize01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    out = img.astype(np.float32)
    if out.size == 0:
        return out
    if out.max() <= 1.0 and out.min() >= 0.0:
        return out
    return np.clip(out, 0.0, 255.0) / 255.0

# -------------------------------
# Augmentations (train-only)
# -------------------------------
def maybe_hflip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    if p <= 0: return img
    if np.random.rand() < p: return cv2.flip(img, 1)
    return img

def rotate_small(img: np.ndarray, max_deg: int = 5) -> np.ndarray:
    if max_deg <= 0: return img
    h, w = img.shape[:2]
    angle = float(np.random.uniform(-max_deg, max_deg))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def maybe_rotate(img: np.ndarray, max_deg: int = 5, p: float = 0.5) -> np.ndarray:
    if p <= 0 or max_deg <= 0: return img
    if np.random.rand() > p: return img
    return rotate_small(img, max_deg=max_deg)

def maybe_flip_and_rotate(img: np.ndarray, p_hflip=0.5, rotate_max_deg=5, p_rotate=0.5) -> np.ndarray:
    img = maybe_hflip(img, p=p_hflip)
    img = maybe_rotate(img, max_deg=rotate_max_deg, p=p_rotate)
    return img

# -------------------------------
# Pipeline glue
# -------------------------------
def apply_resize(img: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    if cfg.resize_mode == "cover_crop":
        return resize_cover_then_center_crop(img, out=cfg.target_hw)
    if cfg.resize_mode == "fit_pad":
        return resize_fit_then_pad(img, out=cfg.target_hw, pad_mode=cfg.pad_mode, pad_value=cfg.pad_value)
    raise ValueError(f"Unknown resize_mode: {cfg.resize_mode}")

def preprocess_image_array(img_bgr_or_gray: np.ndarray, cfg: PreprocConfig, *, train: bool) -> np.ndarray:
    """Full pipeline: crop → (train aug) → resize → normalize01. Returns float32 [0..1] HxW."""
    g, _ = crop_dark_frame(
        img_bgr_or_gray,
        border_ratio=cfg.border_ratio,
        gauss_ks=cfg.gauss_ks,
        window=cfg.window,
        dark_frac=cfg.dark_frac,
        min_keep_run=cfg.min_keep_run,
        max_crop_frac=cfg.max_crop_frac,
        pad=cfg.pad_after_crop,
    )
    if train:
        g = maybe_flip_and_rotate(g, p_hflip=cfg.aug_hflip_p, rotate_max_deg=cfg.aug_rotate_deg, p_rotate=cfg.aug_rotate_p)
    g = apply_resize(g, cfg)
    g = normalize01(g)
    return g

# -------------------------------
# Cache helpers (split-agnostic)
# -------------------------------
def cache_path(proc_dir: Path, cfg: PreprocConfig, rel_path: str) -> Path:
    """Deterministic cached path for a RAW rel_path under a given config."""
    h = preproc_hash(cfg)
    return Path(proc_dir) / f"cache_{h}" / "images" / rel_path

def ensure_cached_png(src_abs: Path, dst_abs: Path, cfg: PreprocConfig, *, train_aug: bool = False) -> Path:
    """
    Read RAW image, run preprocessing (if not cached), and write grayscale PNG (uint8 after inverse of normalize01).
    For caching: write *post-crop+resize but pre-normalize* PNG to keep mean/std calculation flexible.
    """
    if dst_abs.exists():
        return dst_abs
    dst_abs.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_abs), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(src_abs)
    # crop + optional aug + resize
    g, _ = crop_dark_frame(
        img,
        border_ratio=cfg.border_ratio,
        gauss_ks=cfg.gauss_ks,
        window=cfg.window,
        dark_frac=cfg.dark_frac,
        min_keep_run=cfg.min_keep_run,
        max_crop_frac=cfg.max_crop_frac,
        pad=cfg.pad_after_crop,
    )
    if train_aug:
        # DO NOT augment in cache; cache should be deterministic. Leave augmentations to runtime if needed.
        pass
    g = apply_resize(g, cfg)
    # write as uint8 PNG; normalization happens downstream when loading tensors
    out = g if g.dtype == np.uint8 else np.clip(np.round(g).astype(np.uint8), 0, 255)
    ok = cv2.imwrite(str(dst_abs), out)
    if not ok:
        raise IOError(f"Failed to write cache file: {dst_abs}")
    return dst_abs

# -------------------------------
# Tiny self-sanity (optional)
# -------------------------------
def _quick_sanity(img_path: Path):
    cfg = PreprocConfig()
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"read fail: {img_path}"
    out = preprocess_image_array(img, cfg, train=False)
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape == cfg.target_hw, (out.shape, cfg.target_hw)
    return True
