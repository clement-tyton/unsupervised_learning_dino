"""
Shared visualization primitives used by both the pre-training (method-explaining)
and post-training (pretrained-vs-finetuned) suites.

Keep this dependency-light: tile IO, ImageNet normalization constants, the single
forward-ready tensor builder, the PCA->RGB patch colouring, and tile selection.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def tile_to_pil(path):
    """Read a GeoTIFF tile's RGB, per-band min-max stretch to uint8, return PIL."""
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype("float32")
    for i in range(3):
        b = arr[i]
        lo, hi = b.min(), b.max()
        if hi > lo:
            arr[i] = (b - lo) / (hi - lo) * 255.0
    return Image.fromarray(arr.astype("uint8").transpose(1, 2, 0))


def _global_tensor(path, size=224):
    """A single normalized global crop tensor (1,3,size,size) for a model forward."""
    img = tile_to_pil(path).resize((size, size))
    t = TF.to_tensor(img)
    t = TF.normalize(t, MEAN.tolist(), STD.tolist())
    return t.unsqueeze(0)


def _embed_tensor(path, img_size=224):
    """Deterministic input (3,H,W): stretch -> resize -> ImageNet-normalize, NO augmentation."""
    img = tile_to_pil(path).resize((img_size, img_size))
    return TF.normalize(TF.to_tensor(img), MEAN.tolist(), STD.tolist())


def _feature_pca_rgb(patches, g):
    """patches (N,D) -> (g,g,3) RGB image via PCA to 3 comps, per-channel min-max."""
    f = patches - patches.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(f.float(), q=3)        # top-3 principal directions
    proj = f.float() @ V[:, :3]                         # (N, 3)
    lo, hi = proj.min(0).values, proj.max(0).values
    proj = (proj - lo) / (hi - lo + 1e-8)              # -> [0,1]
    return proj.reshape(g, g, 3).cpu().numpy()


def _black_fraction(path):
    """Fraction of fully no-data (RGB all 0) pixels — high for AOI-edge tiles."""
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3])
    return float((arr.sum(0) == 0).mean())


def pick_full_tiles(tiles_dir, n=4, max_black=0.005, seed=0):
    """Pick n (nearly) fully-covered tiles, skipping AOI-edge tiles with no-data corners."""
    files = sorted(Path(tiles_dir).glob("*.tif"))
    order = np.random.default_rng(seed).permutation(len(files))
    out = []
    bar = tqdm(order, desc="picking full tiles")
    for i in bar:
        if _black_fraction(files[i]) <= max_black:
            out.append(str(files[i]))
            if len(out) >= n:
                break
    bar.close()
    return out
