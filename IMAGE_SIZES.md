# Image sizes across the pipeline

Every stage feeds DINOv3 ViT-L/**16** (patch size 16). The patch grid is always
`g = img_size / 16`, so the number of patch tokens per tile is `g²`. The **source `.tif`
tiles are 512×512 on disk** — each stage then resizes them to whatever `img_size` it needs
before the backbone runs. Nothing is cropped to reach a target size (except DINO training's
multi-crop); it's a plain **resize (downscale/upscale)**.

| img_size | grid `g` | patches `g²` | px/patch (of a 512 source) |
|---|---|---|---|
| 96  | 6×6   | 36   | ~85 px |
| 224 | 14×14 | 196  | ~37 px |
| 512 | 32×32 | 1024 | 16 px |
| 1024| 64×64 | 4096 | 8 px (upscaled) |
| 2048| 128×128 | 16384 | 4 px (upscaled) |

## Master table — what each stage uses

| Stage | img_size | Patches/tile | Level | Set in |
|---|---|---|---|---|
| **Source tiles on disk** | 512 | — | — | `download_imagery_tiles(patch=512)` |
| **DINO finetuning (training)** | **224 global + 96 local** | 196 / 36 | crops | `MultiCropTransform` (data.py) |
| **Linear probe** | **512** | 1024 | patch | `linear_probe_confusion(img_size=512)` |
| **Clustering (NMI/ARI/purity)** | **512** | 1024 | patch | `cluster_vs_annotation(img_size=512)` |
| **Seg comparison (DINO heads + UNet)** | **512** | 1024 → upsampled 512² | pixel | `compare_*(img_size=512)` |
| **Visual analysis (PCA, mosaics, cosine)** | **224** | 196 | patch | `_embed_tensor` default 224 |
| **Export `_exp` bundles** | 224 (old) / **512** (curepto) | 196 / 1024 | patch | `scripts/export_exp.py` CONFIG |

## Stage detail

### DINO finetuning — 224 + 96 (NOT 512)
The 512 tile is only a pixel reservoir. `MultiCropTransform` takes `RandomResizedCrop`s:
- **2 global crops** → resized to **224** (cover 40–100 % of the tile),
- **4 local crops** → resized to **96** (cover 5–40 %).

The teacher sees only the **224** globals; the student sees **all** crops (224 + 96) and must
match the teacher. So the backbone is optimized at **224/96** — it never sees 512 in training.

### Linear probe & clustering — 512, patch-level
DINO runs the tile at **512** → 32×32 = 1024 patch features (1024-d each). The mask is pooled
to the 32×32 grid (`pool_mask`, majority class per 16-px block). Cache: `EVAL_CACHE`
(`outputs/_feature_cache_eval`), separate from the viz cache.
- **Probe**: StandardScaler → LogisticRegression on the 1024-d patch features.
- **Clustering**: global k-means over all patches → NMI / ARI / purity vs annotation
  (assignment-free, unsupervised).

### Segmentation comparison — 512, pixel-level
Same 512 features, but `ConvSegHead` learns to **upsample** 32×32×1024 → 512×512×K1, so the
DINO heads and the UNet are scored at full pixel resolution. **Training**: full 512 tiles
(`crop=None`), no data_balance by default. **Eval**: full 512 tiles, no augmentation.

### Visual analysis — 224, rescaled
All qualitative figures default to **img_size=224**, i.e. the 512 tile is **resized to 224**
(`tile_to_pil(path).resize((224,224))`) for both the features AND the background image under the
heatmap, so the 14×14 map overlays the displayed tile pixel-for-pixel. Two PCA flavours:
- **Local PCA** (per-tile basis, `_feature_pca_rgb`) → `pca_grid.png`.
- **Pooled/global PCA** (one basis fit on ~50k patches pooled across all tiles, jointly on
  pre+ft so colours are comparable, `_aligned_pca_maps`) → `feature_mosaic.png`, `pca_explosion.png`.

Figures affected by 224: `pca_grid`, `feature_mosaic`, `cosine_change`, `changed`,
`expr_changed`, `pca_explosion` (sections 1–3 of the report).

### Export `_exp` bundles
`scripts/export_exp.py` runs the tile at its CONFIG `img_size` and saves per-tile
`patches (g²,1024)` + `cls (1024,)`. Current state on disk:
- `monrovia_exp`, `manned_bens_oasis_exp` → **224** (196 patches),
- `curepto_chile_exp` → **512** (1024 patches).
→ the three bundles are **not** on the same grid; re-export to harmonise if needed.

## The rule of thumb
- **Anything scored against annotations** (probe, clustering, seg) → **512** (finest spatial grid).
- **DINO self-supervised training** → **224/96** (fixed by multi-crop design).
- **Qualitative figures** → **224** (cheap, enough to see structure).
- Resizing from 512 to the target is a plain PIL resize; the ViT (patch 16) accepts any
  multiple of 16 because DINOv3 interpolates its position embeddings.
