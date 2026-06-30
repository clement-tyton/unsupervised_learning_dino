"""
REPL: download a site's tiles from its webmap (S3) to local GeoTIFFs.

One webmap per site, ONE OR MORE .fgb areas (config["fgbs"]). Each .fgb -> its own
bbox -> grid -> tiles under RGB/<area>/. Logic lives in tytonai_utils.webmap.
Needs valid AWS_* creds in .env (temporary S3 creds expire -> refresh on 401/403).
Step through the blocks in a REPL.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from tytonai_utils.webmap import build_grid, download_grid, plot_grid, preview_tiles

# --- params -------------------------------------------------------------------
SITE = Path("input_site_data/manned_bens_oasis_wet")         # INPUT: .fgb / config / RGB tiles
OUT = Path("outputs") / SITE.name                        # OUTPUT: previews / pictures
OUT.mkdir(parents=True, exist_ok=True)
PATCH = 512              # tile size in pixels
WORKERS = 8             # parallel ranged reads
SKIP_EMPTY = True       # don't save tiles with no data

# --- setup: env + config ------------------------------------------------------
load_dotenv(".env", override=True)                       # AWS_* creds for /vsis3
config = json.loads((SITE / "site_config.json").read_text())
RES = config["resolution_m"]                             # metres per pixel
webmap = config["webmap_s3"]                             # /vsis3/<bucket>/<key>.tif
print("webmap :", webmap)
print("fgbs   :", config["fgbs"])

# --- download every .fgb area (same webmap, one grid per .fgb) -----------------
for fgb in config["fgbs"]:
    # fgb = config["fgbs"][0]   # <- uncomment this and skip the loop to step ONE area
    name = Path(fgb).stem
    print(f"\n=== {fgb} ===")
    grid, study_area = build_grid(SITE / fgb, RES, PATCH)
    print(f"grid : {len(grid)} tiles of {PATCH}px ({PATCH * RES:.1f} m) | crs {study_area.crs}")
    plot_grid(grid, study_area, f"{SITE.name}/{name}", OUT / f"grid_preview_{name}.png")
    written = download_grid(grid, webmap, SITE / "RGB" / name, bands=[1, 2, 3], workers=WORKERS, skip_empty=SKIP_EMPTY)
    print(f"wrote {len(written)} / {len(grid)} tiles -> {SITE / 'RGB' / name}")
    preview_tiles(SITE / "RGB" / name, downscale=16, out_png=OUT / f"overview_{name}.png")


# fgb = config["fgbs"][1]   # <- uncomment this and skip the loop to step ONE area
# name = Path(fgb).stem
# print(f"\n=== {fgb} ===")
# grid, study_area = build_grid(SITE / fgb, RES, PATCH)
# print(f"grid : {len(grid)} tiles of {PATCH}px ({PATCH * RES:.1f} m) | crs {study_area.crs}")
# plot_grid(grid, study_area, f"{SITE.name}/{name}", SITE / f"grid_preview_{name}.png")

# written = download_grid(grid, webmap, SITE / "RGB" / name, bands=[1, 2, 3], workers=WORKERS, skip_empty=SKIP_EMPTY)
# print(f"wrote {len(written)} / {len(grid)} tiles -> {SITE / 'RGB' / name}")

# preview_tiles(SITE / "RGB" / name, downscale=16, out_png=SITE / f"overview_{name}.png")


from tytonai_utils.webmap import build_grid, download_grid, plot_grid, preview_tiles
from tytonai_utils.viz import plot_image_mask_pairs, plot_image_mask_tiles
from tytonai_utils.rollup import (
    CLASS_NAMES, RND_NAMES_6CLASS, RND_NAMES_7CLASS,
    RND_REMAP_6CLASS, RND_REMAP_7CLASS, rollup_annotations, rollup_mask,
)

preview_tiles(
    "input_site_data/EM2020_Jimblebar_Rail/RGB/study_area", downscale=16, out_png="preview.png"
)


plot_image_mask_pairs(
    "input_site_data/monrovia/annotations",
    "input_site_data/monrovia/dataset.json",
    [0,1,2],
    CLASS_NAMES,
    out_png= "lol.png"
)


plot_image_mask_pairs(
    "input_site_data/monrovia/annotations",
    "input_site_data/monrovia/dataset.json",
    [0,1,2],
    ,
    out_png= "lol.png"
)

site = "manned_bens_oasis_wet"
plot_image_mask_tiles(f'input_site_data/{site}/RGB/study_area_1', 
                      f'input_site_data/{site}/masks/study_area_1', n=6,
                      out_png="pairs_aligned.png")