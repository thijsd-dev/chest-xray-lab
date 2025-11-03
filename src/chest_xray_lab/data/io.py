# src/chest_xray_lab/data/io.py
from pathlib import Path
import cv2
import numpy as np


def load_gray01(path: str | Path) -> np.ndarray:
    """
    Load an X-ray image as single-channel float32 in [0, 1].

    - accepts absolute or relative path
    - if image is RGB, convert to gray
    - if image is already gray, keep it
    - guarantees: np.float32, 2D, 0..1
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # handle 3-channel → gray
    if img.ndim == 3:
        # OpenCV loads as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img is now HxW, uint8 or uint16
    img = img.astype(np.float32)

    # normalize to 0..1
    mx = img.max()
    if mx > 0:
        img /= mx
    else:
        # very defensive: avoid div0
        img[:] = 0.0

    return img
