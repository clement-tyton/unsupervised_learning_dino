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
