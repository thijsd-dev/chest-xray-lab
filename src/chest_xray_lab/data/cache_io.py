from pathlib import Path
import cv2, numpy as np

def to_cached_pairs(items_abs_y_pid, RAW_DIR: Path, PROC_DIR: Path, CFG, cache_path_fn):
    out = []
    for abs_path, y, _pid in items_abs_y_pid:
        rel = str(Path(abs_path).resolve().relative_to(RAW_DIR))
        out.append((str(cache_path_fn(PROC_DIR, CFG, rel)), int(y)))
    return out

def mean_std_from_cached_pairs(pairs):
    s = s2 = 0.0; n = 0
    for p, _ in pairs:
        a = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if a is None: raise FileNotFoundError(p)
        x = a.astype(np.float32)/255.0
        s  += float(x.sum()); s2 += float((x*x).sum()); n += x.size
    mean = s/n; var = max(s2/n - mean*mean, 1e-12)
    return float(mean), float(var**0.5)
