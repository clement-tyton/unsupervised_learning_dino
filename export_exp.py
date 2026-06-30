"""
Export a <site>_exp/ bundle: RGB tiles + 2 DINO embedding sets (pretrained & finetuned),
matched 1:1 by filename stem — the same layout as monrovia_exp / manned_bens_oasis_exp.

  <out>/tif_tiles/<tile>.tif              RGB imagery tile (input)
  <out>/embeddings_pretrained/<tile>.npz  DINOv3-SAT pretrained features (LoRA=0)
  <out>/embeddings_finetuned/<tile>.npz   finetuned checkpoint features
Each .npz: 'patches' (N,1024) float16, 'cls' (1024,) float32 — N = (img_size/16)^2.

Reuses the per-tile feature cache (src.visualisation.features.pre_ft_features), so the heavy
backbone runs at most once per (variant, tile) and a re-run is instant. Pure functions + a flat
RUN block at the bottom — step through one line at a time (Shift+Enter).
"""

import shutil
from pathlib import Path

import numpy as np


def collect_tiles(tiles_dir: str) -> list[str]:
    """Sorted .tif tile paths under tiles_dir (recursive)."""
    return [str(p) for p in sorted(Path(tiles_dir).rglob("*.tif"))]


def _save_emb(path: Path, patches: np.ndarray, cls: np.ndarray) -> None:
    """Write one tile's features in the _exp npz format (patches float16, cls float32)."""
    np.savez(path, patches=patches.astype("float16"), cls=cls.astype("float32"))


def _write_readme(out_dir: Path, weights: str, ckpt: str, img_size: int, g: int, n: int) -> None:
    """Drop a README.txt describing the bundle layout + provenance."""
    (out_dir / "README.txt").write_text(
        f"{out_dir.name} — matched per-tile data (align the 3 dirs by filename stem)\n\n"
        f"  tif_tiles/<tile>.tif                RGB imagery tile (input)\n"
        f"  embeddings_pretrained/<tile>.npz    DINOv3-SAT pretrained features (LoRA=0)\n"
        f"  embeddings_finetuned/<tile>.npz     finetuned checkpoint features\n\n"
        f"Each .npz has keys: 'patches' (N,1024) float16, 'cls' (1024,) float32.\n"
        f"N = (img_size/16)^2 = {g * g} patches at img_size={img_size} ({g}x{g} grid).\n"
        f"tiles   : {n}\n"
        f"weights : {Path(weights).name}\n"
        f"ckpt    : {ckpt}\n")


def _embed_and_save(model, tiles: list[str], out_subdir: Path, device: str,
                    img_size: int, batch_size: int = 8) -> None:
    """Forward `tiles` through `model` in batches (on `device`) and save each tile's npz as we go.
    STREAMS: never holds more than `batch_size` tiles' features in RAM — safe at full resolution."""
    from src.visualisation.features import compute_tile_features
    out_subdir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(tiles), batch_size):
        chunk = tiles[i:i + batch_size]
        for tile, (patches, cls) in zip(chunk, compute_tile_features(model, chunk, device, img_size,
                                                                      batch_size=batch_size)):
            _save_emb(out_subdir / f"{Path(tile).stem}.npz", patches, cls)


def export_exp(tiles: list[str], weights: str, ckpt: str, out_dir: str,
               device: str = "cuda:1", img_size: int = 512, batch_size: int = 8) -> Path:
    """Build out_dir/{tif_tiles,embeddings_pretrained,embeddings_finetuned} + README from `tiles`.
    Builds the backbone ONCE on `device` (GPU), embeds all tiles pretrained, then switches the same
    student to the finetuned checkpoint and embeds again — streaming each tile to disk so RAM stays
    flat even at full 512 resolution (1024 patches/tile)."""
    from src.train.dino import load_checkpoint
    from src.visualisation.features import _load_pre_ft
    out = Path(out_dir)
    tif_dir = out / "tif_tiles"
    tif_dir.mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        shutil.copy(tile, tif_dir / f"{Path(tile).stem}.tif")
    print(f"loading backbone on {device} ...")
    student, teacher = _load_pre_ft(weights, ckpt, device)              # pretrained student, on GPU
    _embed_and_save(student, tiles, out / "embeddings_pretrained", device, img_size, batch_size)
    load_checkpoint(student, teacher, ckpt, map_location=device)        # switch student -> finetuned
    _embed_and_save(student, tiles, out / "embeddings_finetuned", device, img_size, batch_size)
    g = img_size // 16
    _write_readme(out, weights, ckpt, img_size, g, len(tiles))
    print(f"-> {out}  ({len(tiles)} tiles x 2 sets @ {img_size}px = {g * g} patches/tile)")
    return out


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
#  Line 2 (cold) builds the model + caches; a re-run is instant (cache hit).
# ════════════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "tiles_dir": "input_site_data/curepto_chile/RGB",       # the site's .tif tiles (must be local)
    "weights": "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "ckpt": "checkpoints/curepto_chile/final.pt",
    "out_dir": "curepto_chile_exp",
    "device": "cuda:1",                                     # GPU forward (set "cpu" only as fallback)
    "img_size": 512,                                        # FULL res -> 1024 patches/tile (32x32 grid)
    "batch_size": 8,                                        # tiles/forward; lower if GPU OOM at 512
}

tiles = collect_tiles(CONFIG["tiles_dir"])
print(f"{len(tiles)} tiles under {CONFIG['tiles_dir']}")

out = export_exp(tiles, CONFIG["weights"], CONFIG["ckpt"], CONFIG["out_dir"],
                 device=CONFIG["device"], img_size=CONFIG["img_size"], batch_size=CONFIG["batch_size"])
