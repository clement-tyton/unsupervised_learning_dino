"""
Per-tile DINO feature cache — compute patch + CLS features ONCE per (model variant,
tile) and store them on disk, so the post-training viz functions never re-run the
backbone twice for the same tile.

A "variant" is either the PRETRAINED backbone (LoRA=0) or a finetuned CHECKPOINT. The
on-disk key per tile = variant tag (weights / ckpt identity, incl. ckpt mtime) + a hash
of the tile's absolute path, so different sites / checkpoints never collide and a changed
checkpoint invalidates its own cache automatically.

Public entry point:
    pre_ft_features(weights, ckpt, tiles) -> (pre, ft, g)
        pre, ft = (patches (T,N,D) float32 tensor, cls (T,D) float32 tensor)
        g       = patches per side (N = g*g)
    Only cache MISSES trigger a forward pass, and the heavy model is built only if at
    least one tile is missing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.visualisation.common import _embed_tensor

CACHE_ROOT = Path("outputs/_feature_cache")


# ── model loading ────────────────────────────────────────────────────────────────
def _load_pre_ft(weights: str, ckpt: str, device: str) -> tuple:
    """Build the student at the checkpoint's out_dim and return (student, teacher).

    The student starts PRETRAINED (LoRA=0); call load_checkpoint(s, t, ckpt) to switch
    it to the finetuned state. One code path shared by every comparison.
    """
    from src.train.dino import build_student_teacher  # lazy: heavy import

    ck = torch.load(ckpt, map_location="cpu")
    out_dim = ck["student"]["head.prototypes.weight"].shape[0]
    has_ibot = any("ibot_head" in k for k in ck["student"])
    s, t = build_student_teacher(weights=weights, pretrained=True, out_dim=out_dim, use_ibot=has_ibot)
    return s.to(device), t


# ── cache keys ────────────────────────────────────────────────────────────────────
def _variant_tag(weights: str, ckpt: str | None) -> str:
    """Stable short id for a model variant (pretrained vs a specific, mtimed checkpoint)."""
    if ckpt is None:
        key = f"pretrained::{Path(weights).name}"
    else:
        key = f"finetuned::{Path(ckpt).parent.name}::{int(os.path.getmtime(ckpt))}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _tile_key(tile: str) -> str:
    """Collision-proof per-tile filename stem: tile name + hash of its absolute path."""
    h = hashlib.md5(str(Path(tile).resolve()).encode()).hexdigest()[:8]
    return f"{Path(tile).stem}_{h}"


def _cache_file(cache_root: Path, variant_tag: str, tile: str) -> Path:
    """Where one tile's features live for a given variant."""
    return cache_root / variant_tag / f"{_tile_key(tile)}.npz"


# ── disk I/O (isolated at the edges) ───────────────────────────────────────────────
def load_tile_features(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read cached (patches, cls) for one tile, or None if it isn't cached yet."""
    if not path.exists():
        return None
    z = np.load(path)
    return z["patches"], z["cls"]


def save_tile_features(path: Path, patches: np.ndarray, cls: np.ndarray) -> None:
    """Write one tile's features to disk (patches as float16 to halve size, cls float32)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, patches=patches.astype("float16"), cls=cls.astype("float32"))


# ── compute (the only place that runs the backbone) ────────────────────────────────
def compute_tile_features(model, tiles: list[str], device: str = "cpu", img_size: int = 224,
                          batch_size: int = 16, to_tensor=_embed_tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward `tiles` through `model` once -> [(patches (N,D), cls (D,)) per tile].

    to_tensor(path, img_size) -> (3,H,W) builds the model input; swap it to read a
    different source (e.g. .npz imagery instead of .tif)."""
    model.eval()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in tqdm(range(0, len(tiles), batch_size), desc="forward (cache miss)"):
        xb = torch.stack([to_tensor(p, img_size) for p in tiles[i:i + batch_size]]).to(device)
        with torch.no_grad():
            res = model(xb)
        cls = res[1].float().cpu().numpy()                                   # [1] = CLS
        pat = res[2].float().cpu().numpy()                                   # [2] = patches
        out.extend((pat[j], cls[j]) for j in range(len(cls)))
    return out


# ── cache orchestration ────────────────────────────────────────────────────────────
def _missing(cache_root: Path, variant_tag: str, tiles: list[str]) -> list[tuple[int, str]]:
    """(index, tile) pairs whose features aren't on disk yet for this variant."""
    return [(i, t) for i, t in enumerate(tiles)
            if not _cache_file(cache_root, variant_tag, t).exists()]


def _fill_cache(model, items: list[tuple[int, str]], cache_root: Path, variant_tag: str,
                device: str, img_size: int, to_tensor=_embed_tensor) -> None:
    """Compute + save features for the given (index, tile) misses."""
    feats = compute_tile_features(model, [t for _, t in items], device, img_size, to_tensor=to_tensor)
    for (_, tile), (patches, cls) in zip(items, feats):
        save_tile_features(_cache_file(cache_root, variant_tag, tile), patches, cls)


def _stack_cached(cache_root: Path, variant_tag: str, tiles: list[str]) -> tuple:
    """Load every tile's cached features and stack -> (patches (T,N,D), cls (T,D)) tensors."""
    pats, clss = [], []
    for tile in tiles:
        patches, cls = load_tile_features(_cache_file(cache_root, variant_tag, tile))
        pats.append(torch.from_numpy(patches.astype("float32")))
        clss.append(torch.from_numpy(cls))
    return torch.stack(pats), torch.stack(clss)


def pre_ft_features(weights: str, ckpt: str, tiles: list[str], device: str = "cpu",
                    img_size: int = 224, cache_root: Path = CACHE_ROOT,
                    to_tensor=_embed_tensor) -> tuple:
    """(pre, ft, g): cached patch+CLS features for the pretrained and finetuned variants.

    pre, ft = (patches (T,N,D), cls (T,D)) float32 tensors; g = patches per side.
    Only cache misses run the backbone, and the model is built only when something is
    missing -> a fully cached call does no model work at all. to_tensor lets a caller
    read a different source (.npz vs .tif) — give it a DEDICATED cache_root so the
    features don't mix with the default .tif cache.
    """
    cache_root = Path(cache_root)
    pre_tag = _variant_tag(weights, None)
    ft_tag = _variant_tag(weights, ckpt)
    pre_miss = _missing(cache_root, pre_tag, tiles)
    ft_miss = _missing(cache_root, ft_tag, tiles)

    if pre_miss or ft_miss:                                  # build heavy model only on a miss
        from src.train.dino import load_checkpoint
        s, t = _load_pre_ft(weights, ckpt, device)
        if pre_miss:
            _fill_cache(s, pre_miss, cache_root, pre_tag, device, img_size, to_tensor)
        if ft_miss:
            load_checkpoint(s, t, ckpt, map_location=device)
            _fill_cache(s, ft_miss, cache_root, ft_tag, device, img_size, to_tensor)

    g = img_size // 16
    return (_stack_cached(cache_root, pre_tag, tiles),
            _stack_cached(cache_root, ft_tag, tiles), g)


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
#  Line 2 (cold) builds the model + caches; line 3 (warm) should be instant, no model.
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.visualisation.common import pick_full_tiles

    CONFIG = {
        "weights": "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "ckpt": "checkpoints/monrovia__r16-K65k-dino+ibot+gram+koleo-sat/final.pt",
        "tiles_dir": "input_site_data/monrovia/RGB/study_area",
        "device": "cpu",
    }

    tiles = pick_full_tiles(CONFIG["tiles_dir"], n=4, seed=1)        # 1) a few real tiles

    (fpre, cpre), (fft, cft), g = pre_ft_features(                   # 2) cold: caches misses
        CONFIG["weights"], CONFIG["ckpt"], tiles, CONFIG["device"])
    print("patches:", tuple(fpre.shape), "cls:", tuple(cpre.shape), "g:", g)

    (fpre2, _), (fft2, _), _ = pre_ft_features(                      # 3) warm: should be instant
        CONFIG["weights"], CONFIG["ckpt"], tiles, CONFIG["device"])
    print("warm read matches cold:", torch.allclose(fpre, fpre2), torch.allclose(fft, fft2))
