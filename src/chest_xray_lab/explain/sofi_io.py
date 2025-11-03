# chest_xray_lab/explain/sofi_io.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List

def sofi_npz_path(cache_dir: Path, fname: str) -> Path:
    return cache_dir / fname

def load_sofi_cache(cache_dir: Path, fname: str):
    npz_p = cache_dir / fname
    z = np.load(npz_p, allow_pickle=False)
    meta_p = npz_p.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return z, meta

def list_sofi_cache(cache_dir: Path) -> List[str]:
    return sorted([p.name for p in cache_dir.glob("*.npz")])

def common_sofi_files(*dirs: Path) -> List[str]:
    sets = []
    for d in dirs:
        sets.append({p.name for p in d.glob("*.npz")})
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(list(common))
