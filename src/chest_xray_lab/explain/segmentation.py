# src/chest_xray_lab/explain/segmentation.py
from __future__ import annotations
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab

__all__ = [
    "compute_superpixels_from_gray",
    "masks_from_segments",
    "overlay_boundaries",
]

def compute_superpixels_from_gray(
    gray01_hw: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0,
    sigma: float = 1.0,
    use_lab: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args
    ----
    gray01_hw : float32 [H,W] in [0,1]
    n_segments : desired superpixels
    compactness : SLIC compactness
    sigma : Gaussian smoothing (passed to SLIC)
    use_lab : if True, convert grayscale-3ch to LAB before SLIC

    Returns
    -------
    seg : [H,W] int32 labels 0..K-1
    rgb : [H,W,3] float32 image used for SLIC (for visualization)
    """
    g = gray01_hw.astype(np.float32)
    rgb = np.dstack([g, g, g])  # SLIC expects multi-channel by default
    img_for_slic = rgb2lab(rgb) if use_lab else rgb
    seg = slic(
        img_for_slic,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,  # last axis has channels
    )
    return seg.astype(np.int32), rgb

def masks_from_segments(seg: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    seg : [H,W] labels 0..K-1
    Returns (labels, masks) where each mask is uint8 {0,1}.
    """
    labels = np.unique(seg)
    masks = [(seg == k).astype(np.uint8) for k in labels]
    return labels, masks

def overlay_boundaries(gray01_hw: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Convenience: returns an RGB overlay with superpixel boundaries.
    """
    g = gray01_hw.astype(np.float32)
    rgb = np.dstack([g, g, g])
    over = mark_boundaries(rgb, seg, color=(1, 0, 0))  # red borders
    return over
