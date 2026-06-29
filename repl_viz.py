"""
REPL: produce the explanatory plot suite for a site, plus pretrained-vs-finetuned
comparisons — PCA feature maps AND patch k-means segmentation.

Re-run after editing src/* : the reload block below picks up your edits WITHOUT
restarting the REPL (reload order = dependencies first).

INPUT  tiles come from input_site_data/<site>/RGB/<area>/   (the .tif tiles)
OUTPUT pictures go to outputs/<site>/   (kept separate from the input data)

Outputs (per area, into outputs/<site>/<area>/):
  overview.png, multicrop_<tile>.png, distill_<tile>.png, ibot_<tile>.png
Plus, at outputs/<site>/:
  _pca_grid.png            input | pretrained | finetuned  (PCA of patch features)
  _patch_kmeans_grid.png   input | pretrained | finetuned  (patch k-means segmentation)
  _feature_mosaic_*.png    whole-site PCA-RGB feature mosaic (pretrained / finetuned)
"""

import importlib
import json
from pathlib import Path

# --- reload src modules so the REPL sees your latest edits (deps first) -------
import src.visualisation.common
import src.visualisation.features
import src.visualisation.pretraining
import src.visualisation.posttraining

importlib.reload(src.visualisation.common)          # before features/pretraining/posttraining
importlib.reload(src.visualisation.features)        # before posttraining (it imports from it)
importlib.reload(src.visualisation.pretraining)
importlib.reload(src.visualisation.posttraining)
# importlib.reload(src.train.dino)                  # uncomment if you edited src/train/dino.py

from src.visualisation.common import pick_full_tiles
from src.visualisation.pretraining import plot_suite
from src.visualisation.posttraining import (compare_features_grid, compare_feature_mosaic,
                                            compare_patch_cosine, patch_cluster_grid)

# --- params -------------------------------------------------------------------
SITE = Path("input_site_data/monrovia")             # INPUT: tiles / config
OUT = Path("outputs") / SITE.name                   # OUTPUT: pictures
OUT.mkdir(parents=True, exist_ok=True)
N_EXAMPLES = 2
config = json.loads((SITE / "site_config.json").read_text())
print("fgbs:", config["fgbs"])

# --- 1) model-free suite per .fgb area ----------------------------------------
for fgb in config["fgbs"]:
    area = Path(fgb).stem
    tiles = SITE / "RGB" / area
    print(f"\n=== {area} ===")
    plot_suite(tiles, out_dir=OUT / area, n_examples=N_EXAMPLES)

# --- 2) pretrained vs finetuned (needs a checkpoint for THIS site) ------------
# CKPT must be trained on the site whose tiles you compare. The Monrovia one below
# is the model we have; change SITE + CKPT together for another site.
WEIGHTS = "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
CKPT = "checkpoints/monrovia__r16-K65k-dino+ibot+gram+koleo-sat/final.pt"
DEVICE = "cuda"                              # set "cuda" if the GPU is free (faster)
cmp_tiles = pick_full_tiles(SITE / "RGB" / Path(config["fgbs"][0]).stem, n=30, seed=1)

# 2a) PCA feaure maps
compare_features_grid(WEIGHTS, CKPT, cmp_tiles[25:], device=DEVICE,
                      out_png=OUT / "_pca_grid.png")

# 2b) PATCH k-means segmentation (global k-means -> same color = same cluster)
patch_cluster_grid(WEIGHTS, CKPT, cmp_tiles[25:], n_clusters=12, device=DEVICE,
                   out_png=OUT / "_patch_kmeans_grid.png")

# 2c) whole-site FEATURE MOSAIC: every tile as its PCA-RGB patch-feature map at its
# geo position (pretrained vs finetuned). Pass ALL tiles for a dense map -> run on GPU.
all_tiles = pick_full_tiles(SITE / "RGB" / Path(config["fgbs"][0]).stem, n=10**9)
compare_feature_mosaic(WEIGHTS, CKPT, all_tiles, device=DEVICE,
                       out_png=OUT / "_feature_mosaic.png")

# 2d) WHERE did finetuning change features? per-patch cosine(pretrained, finetuned),
# drawn as a whole-site heatmap (dark = changed most) + a histogram of the shifts.
compare_patch_cosine(WEIGHTS, CKPT, all_tiles, device=DEVICE,
                     out_png=OUT / "_cosine_change.png")

# --- optional: tile-level (CLS) embedding map, pretrained vs finetuned --------
# from src.visualisation.posttraining import compare_embeddings
# compare_embeddings(WEIGHTS, CKPT, all_tiles, device=DEVICE, method="tsne",
#                    n_clusters=8, out_png=OUT / "_embeddings_compare.png")
