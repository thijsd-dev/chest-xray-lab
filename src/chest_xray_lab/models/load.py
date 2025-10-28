from pathlib import Path
import torch
from torch import nn

def load_checkpoint(model: nn.Module, ckpt_path: Path, map_location=None) -> nn.Module:
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=map_location)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
