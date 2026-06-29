"""
Annotation alignment — project glue over `tytonai_utils.align` (the core package).

Cut the georeferenced npz masks onto the SAME webmap grid as the imagery tiles (site_config
`resolution_m` + patch), so each imagery RGB/<area>/tile_NNNNN.tif pairs 1:1 by index with a
mask masks/<area>/tile_NNNNN.tif. After re-tiling, a grid cell no longer carries the
manifest's `training_area_id` (it may merge several annotation tiles), so we also write a
sidecar {tile_stem: training_area_id} by footprint containment — this keeps the honest
area-level eval split (see [[eval-split-by-training-area]]) working on the aligned tiles.

`patch` MUST match the patch the imagery was downloaded with, or indices won't pair.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from tytonai_utils.align import realign_annotations_to_grid
from tytonai_utils.webmap import build_grid

AREA_SIDECAR = "_training_areas.json"


def _manifest_footprints(dataset, annotations_dir, srid):
    """GeoDataFrame of each manifest tile's geographic footprint + its training_area_id.

    Footprint = the mask's geotransform origin + its (H,W) pixel size; CRS from the tile srid.
    """
    tiles = json.loads(Path(dataset).read_text())
    geoms, areas = [], []
    for t in tiles:
        gt = t["geotransform"]                          # GDAL: [ox, rx, 0, oy, 0, ry]
        h, w = np.load(Path(annotations_dir) / t["mask_file"])["CLASSIFY"].shape
        ox, rx, oy, ry = gt[0], gt[1], gt[3], gt[5]
        geoms.append(box(ox, oy + h * ry, ox + w * rx, oy))   # ry<0 -> oy+h*ry is the bottom
        areas.append(t["training_area_id"])
    return gpd.GeoDataFrame({"training_area_id": areas}, geometry=geoms, crs=f"EPSG:{srid}")


def _training_area_map(grid, written, dataset, annotations_dir, srid):
    """{tile_stem: training_area_id} for the written grid cells, by MAX-overlap footprint.

    512px cells straddle the 384px annotation footprints, so a cell takes the area whose
    footprint it overlaps most (centroid-in-footprint would miss the many straddlers).
    """
    idx = [int(name.split("_")[1].split(".")[0]) for name in written]
    cells = grid.loc[idx, ["geometry"]].copy()
    cells["stem"] = [Path(n).stem for n in written]
    fp = _manifest_footprints(dataset, annotations_dir, srid).to_crs(cells.crs)  # RangeIndex
    j = gpd.sjoin(cells, fp.reset_index(drop=True), how="inner", predicate="intersects")
    j["overlap"] = [cell.intersection(fp.geometry.iloc[ir]).area
                    for cell, ir in zip(j.geometry, j["index_right"])]
    j = j.sort_values("overlap").drop_duplicates("stem", keep="last")            # max overlap wins
    return dict(zip(j["stem"], j["training_area_id"]))


def align_site_annotations(site_config, dataset, annotations_dir, masks_root, patch=512,
                           mask_key="CLASSIFY", overlapping="first"):
    """Realign the manifest masks onto each .fgb area's imagery grid.

    Writes mask tiles to masks_root/<area>/tile_NNNNN.tif (pairs by index with the imagery
    RGB/<area>/tile_NNNNN.tif) plus a _training_areas.json sidecar. Returns {area: masks_dir}.
    `patch` MUST match the imagery download patch (default 512).
    """
    site_config = Path(site_config)
    cfg = json.loads(site_config.read_text())
    res = cfg["resolution_m"]
    out = {}
    for fgb in cfg["fgbs"]:
        area = Path(fgb).stem
        grid, _ = build_grid(site_config.parent / fgb, res, patch)
        dst = Path(masks_root) / area
        written = realign_annotations_to_grid(grid, annotations_dir, dataset, dst,
                                              mask_key=mask_key, overlapping=overlapping)
        srid = json.loads(Path(dataset).read_text())[0]["srid"]
        amap = _training_area_map(grid, written, dataset, annotations_dir, srid)
        (dst / AREA_SIDECAR).write_text(json.dumps(amap))
        print(f"[align] {area}: {len(written)} mask tiles, {len(set(amap.values()))} training areas")
        out[area] = dst
    return out


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    CONFIG = {
        "site_config": Path("input_site_data/monrovia/site_config.json"),
        "dataset": Path("input_site_data/monrovia/dataset.json"),
        "annotations_dir": Path("input_site_data/monrovia/annotations"),
        "masks_root": Path("input_site_data/monrovia/masks"),
    }
    dirs = align_site_annotations(CONFIG["site_config"], CONFIG["dataset"],
                                  CONFIG["annotations_dir"], CONFIG["masks_root"])
    print(dirs)
