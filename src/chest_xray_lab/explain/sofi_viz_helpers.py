# chest_xray_lab/explain/sofi_viz_helpers.py
import numpy as np
import matplotlib.pyplot as plt

def progressive_mask(gray01, segmap, order, baseline_value, steps):
    imgs = []
    for t in steps:
        out = gray01.copy()
        for sid in order[:t]:
            out[segmap == sid] = baseline_value
        imgs.append(out)
    return imgs

def plot_progressive(masked_imgs, titles=None, row_title=None):
    n = len(masked_imgs)
    plt.figure(figsize=(2.5*n, 2.5))
    for i, im in enumerate(masked_imgs, 1):
        plt.subplot(1, n, i)
        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
        if titles:
            plt.title(titles[i-1])
        plt.axis("off")
    if row_title:
        plt.suptitle(row_title)
    plt.tight_layout()
    plt.show()
