# src/chest_xray_lab/config.py
from pathlib import Path
import os, torch

PROJ_ROOT  = Path(os.environ.get("CXR_PROJ_ROOT", Path.cwd())).resolve()
RAW_DIR    = Path(os.environ.get("CXR_RAW_DIR",  PROJ_ROOT / "data" / "raw" / "chest_xray")).resolve()
PROC_DIR   = Path(os.environ.get("CXR_PROC_DIR", PROJ_ROOT / "data" / "processed" / "chest_xray_split")).resolve()
MANIFESTS  = Path(os.environ.get("CXR_MANIFESTS", PROJ_ROOT / "data" / "processed" / "manifests")).resolve()
CKPT_PATH  = Path(os.environ.get("CXR_CKPT_PATH", PROJ_ROOT / "checkpoints" / "best_model.pt")).resolve()

SEED   = int(os.environ.get("CXR_SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
