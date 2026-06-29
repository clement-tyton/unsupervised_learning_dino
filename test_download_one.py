"""
Smallest end-to-end test: download ONE tile from a site's webmap (S3) and inspect it.

Mirrors the real pipeline (build_grid -> windowed /vsis3 read -> GeoTIFF) but for a
single tile, so you can check the creds + endpoint work before running the full
download. Step through it in a REPL, or run it: .venv/bin/python test_download_one.py
"""

import json
from pathlib import Path

import rasterio
from dotenv import load_dotenv

from tytonai_utils.webmap import build_grid, download_grid

# --- pick a site -------------------------------------------------------------
SITE = Path("input_site_data/monrovia")
PATCH = 512                                          # tile size in pixels

load_dotenv(".env", override=True)                   # AWS_* creds + AWS_S3_ENDPOINT for /vsis3
cfg = json.loads((SITE / "site_config.json").read_text())
RES = cfg["resolution_m"]
webmap = cfg["webmap_s3"]
fgb = cfg["fgbs"][0]
print(f"site   : {SITE.name}")
print(f"webmap : {webmap}")
print(f"res    : {RES} m/px | patch {PATCH}px ({PATCH * RES:.2f} m)")

# --- 1) build the grid (no download yet) -------------------------------------
grid, study_area = build_grid(SITE / fgb, RES, PATCH)
print(f"grid   : {len(grid)} candidate tiles | crs {study_area.crs}")

# --- 2) download ONE tile -----------------------------------------------------
# Start at the grid centre (most likely to have coverage) and walk outward until a
# non-empty tile is written. download_grid does the windowed /vsis3 read for us.
out_dir = Path("outputs") / SITE.name / "_one_tile_test"
order = sorted(range(len(grid)), key=lambda i: abs(i - len(grid) // 2))
written = []
for i in order[:25]:                                 # try up to 25 central tiles
    written = download_grid(grid.iloc[[i]], webmap, out_dir, bands=[1, 2, 3], workers=1)
    if written:
        print(f"got coverage at grid tile #{i} -> {out_dir / written[0]}")
        break
assert written, "no covered tile found among the 25 central candidates"

# --- 3) inspect the downloaded tile ------------------------------------------
with rasterio.open(out_dir / written[0]) as src:
    print("count x H x W :", (src.count, src.height, src.width))
    print("dtype         :", src.dtypes[0])
    print("crs           :", src.crs)
    print("bounds        :", src.bounds)
    print("pixel res     :", src.res)
    print("min/max band1 :", src.read(1).min(), src.read(1).max())
