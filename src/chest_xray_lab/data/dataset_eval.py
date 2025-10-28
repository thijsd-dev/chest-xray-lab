import cv2, torch, numpy as np
from torch.utils.data import Dataset, DataLoader

class CachedEvalDataset(Dataset):
    def __init__(self, pairs): self.pairs = list(pairs)   # [(png_path, label)]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p, y = self.pairs[i]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None: raise FileNotFoundError(p)
        x = torch.from_numpy(g.astype(np.float32)/255.0).unsqueeze(0)   # [1,H,W]
        y = torch.tensor([int(y)], dtype=torch.float32)                 # [1]
        return x, y

def make_eval_loader(pairs, batch_size=32, pin_memory=None):
    if pin_memory is None: pin_memory = torch.cuda.is_available()
    return DataLoader(CachedEvalDataset(pairs), batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=pin_memory)
