"""Whole-site classification maps: color every tile by its predicted class index and draw it at
its geographic position (the same geo-mosaic trick as compare_feature_mosaic), then write PNG
rasters. Used by the pipeline report to compare the FINETUNED-DINO seg head vs the UNet baseline
wall-to-wall across the site.

Predictions are in the shared CONTIGUOUS class space (index 0 = ignore, 1..K = the data-driven
classes from class_mapping), so `names[i]` labels index i and the palette is fixed BY INDEX — the
same class is the same color in every map and in the legend.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch


def _class_palette(n_classes):
    """List of RGBA colors over the K1 contiguous indices. Index 0 (ignore/no-data) -> light grey;
    1..K -> tab20 (the repo's qualitative convention). Fixed by index so colors are stable across
    maps and match the legend. tab20 saturates at 20 hues — a site with >20 real classes will reuse
    colors (rare here)."""
    base = plt.get_cmap("tab20", max(n_classes - 1, 1))
    return [(0.85, 0.85, 0.85, 1.0)] + [base(i) for i in range(n_classes - 1)]


def _class_rgba(pred, colors):
    """(H,W) class-index map -> (H,W,4) RGBA via the fixed palette."""
    arr = np.asarray(colors)                                     # (K1, 4)
    return arr[np.clip(pred, 0, len(colors) - 1)]


def _tile_extents(tiles):
    """(left, right, bottom, top) geo-extent per tile, read ONCE (imshow wants this order)."""
    ext = []
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
        ext.append((b.left, b.right, b.bottom, b.top))
    return ext


def _draw_map(ax, preds, extents, colors, title):
    """Draw one classification mosaic onto `ax`: each tile's colored prediction at its geo extent."""
    for pred, ext in zip(preds, extents):
        ax.imshow(_class_rgba(pred, colors), extent=ext, origin="upper")
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])


def _map_png(preds, extents, colors, title, out_png, dpi=200, figsize=(12, 12)):
    """One whole-site classification raster to its own PNG (full resolution)."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_map(ax, preds, extents, colors, f"{title}  ({len(preds)} tiles)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _legend_handles(names, colors):
    """Patch handles for the real classes (index 1..K; ignore/index-0 dropped)."""
    return [Patch(facecolor=colors[i], edgecolor="none", label=names[i]) for i in range(1, len(names))]


def _legend_png(names, colors, out_png, dpi=200):
    """Shared class legend as its own PNG."""
    handles = _legend_handles(names, colors)
    fig = plt.figure(figsize=(3, 0.32 * len(handles) + 0.6))
    fig.legend(handles=handles, loc="center", frameon=False, fontsize=10, title="classes")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _combined_png(panels, extents, colors, names, out_png, dpi=200):
    """DINO-ft vs UNet side by side (same extents, same palette) + a shared class legend."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 6.5), squeeze=False)
    for ax, (title, preds) in zip(axes[0], panels):
        _draw_map(ax, preds, extents, colors, title)
    fig.legend(handles=_legend_handles(names, colors), loc="center right",
               frameon=False, fontsize=8, title="classes")
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def classification_maps(dino_preds, unet_preds, tiles, names, out_dir, dpi=200):
    """Write the whole-site classification rasters: the finetuned-DINO map, the UNet map, a combined
    side-by-side, and a shared legend. `*_preds` are per-tile (H,W) class-index maps aligned 1:1 with
    `tiles` (pass None to skip a model); `names` is the class_mapping names (index 0 = ignore).
    Returns {label: png_path} of everything written."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    colors = _class_palette(len(names))
    extents = _tile_extents(tiles)                               # read geo bounds once, reuse everywhere
    out = {"legend": _legend_png(names, colors, out_dir / "legend.png", dpi)}
    panels = []
    if dino_preds is not None:
        out["dino_finetuned"] = _map_png(dino_preds, extents, colors, "DINO finetuned — ConvSegHead",
                                         out_dir / "map_dino_finetuned.png", dpi)
        panels.append(("DINO finetuned", dino_preds))
    if unet_preds is not None:
        out["unet"] = _map_png(unet_preds, extents, colors, "UNet baseline",
                               out_dir / "map_unet.png", dpi)
        panels.append(("UNet baseline", unet_preds))
    if len(panels) == 2:                                         # the comparison the report is about
        out["combined"] = _combined_png(panels, extents, colors, names, out_dir / "map_compare.png", dpi)
    return out
