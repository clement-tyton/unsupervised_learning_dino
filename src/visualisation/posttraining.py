"""
Post-training visualizations: compare the PRETRAINED backbone (SAT, LoRA=0) against
the SAME backbone with a finetuned checkpoint loaded, to see what training changed.

Every function builds the model at the checkpoint's own out_dim, reads the patch/CLS
features BEFORE loading the adapters (pretrained) and AFTER (finetuned).

  compare_features / compare_features_grid -> PCA patch-feature maps (per tile)
  patch_cluster_grid / compare_patch_exemplars -> global patch k-means segmentation
  compare_feature_mosaic -> whole-site PCA-RGB mosaic at geo positions
  embedding_png / compare_embeddings -> CLS embedding 2D maps (k-means colored)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.visualisation.common import _embed_tensor, _feature_pca_rgb, tile_to_pil
from src.visualisation.features import CACHE_ROOT, pre_ft_features


# ── PCA patch-feature maps (per tile) ─────────────────────────────────────────────
def compare_features(weights, ckpt, tile, device="cpu", img_size=224, out_png=None,
                     cache_root=CACHE_ROOT):
    """PCA feature map: the pretrained backbone (LoRA=0) vs the SAME backbone with a
    finetuned checkpoint loaded -> see what the finetuning changed on one tile.

    Features come from the per-tile cache (computed once). NOTE: PCA bases are independent
    per panel, so judge the STRUCTURE (coherence of regions), not the exact colors.
    """
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, [tile], device, img_size, cache_root)
    img = tile_to_pil(tile).resize((img_size, img_size))
    pre = _feature_pca_rgb(fpre[0], g)               # SAT, adapters = 0
    ft = _feature_pca_rgb(fft[0], g)                 # SAT + finetuned adapters

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, im, ti in zip(ax, [img, pre, ft],
                         ["input", "pretrained (SAT)", f"finetuned ({Path(ckpt).parent.name})"]):
        a.imshow(im); a.set_title(ti, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("pretrained vs finetuned patch features (independent PCA bases; compare STRUCTURE)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def compare_features_grid(weights, ckpt, tiles, device="cpu", img_size=224, out_png=None,
                          cache_root=CACHE_ROOT):
    """Grid of PCA feature maps: rows = tiles, cols = [input | pretrained | finetuned].

    Features come from the per-tile cache. PCA basis is per-panel -> compare STRUCTURE.
    """
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    imgs = [tile_to_pil(p).resize((img_size, img_size)) for p in tiles]
    pre = [_feature_pca_rgb(fpre[r], g) for r in range(len(tiles))]      # SAT, adapters = 0
    ft = [_feature_pca_rgb(fft[r], g) for r in range(len(tiles))]        # SAT + finetuned

    nrow = len(tiles)
    fig, axes = plt.subplots(nrow, 3, figsize=(8.5, 2.9 * nrow), squeeze=False)
    cols = ["input", "pretrained (SAT)", "finetuned"]
    for r in range(nrow):
        for c, im in enumerate([imgs[r], pre[r], ft[r]]):
            ax = axes[r][c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
    fig.suptitle("pretrained vs finetuned patch features  (per-panel PCA basis -> compare STRUCTURE)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


# ── CLS embeddings (tile-level) ───────────────────────────────────────────────────
def _embed_tensor(path, img_size=224):
    """Deterministic input: stretch->resize->ImageNet-normalize (no augmentation)."""
    img = tile_to_pil(path).resize((img_size, img_size))
    return TF.normalize(TF.to_tensor(img), MEAN.tolist(), STD.tolist())   # (3,H,W)


def extract_embeddings(model, tiles, device="cpu", batch_size=16, img_size=224):
    """(N, 1024) CLS embeddings for the given tile paths."""
    model.eval()
    out = []
    for i in tqdm(range(0, len(tiles), batch_size), desc="cls embeddings"):
        xb = torch.stack([_embed_tensor(p, img_size) for p in tiles[i:i + batch_size]]).to(device)
        with torch.no_grad():
            out.append(model(xb)[1].float().cpu())                        # [1] = cls
    return torch.cat(out).numpy()


def _project(emb, method="pca", seed=0):
    if method == "tsne":
        perp = min(30, max(5, len(emb) // 4))
        return TSNE(n_components=2, init="pca", perplexity=perp, random_state=seed).fit_transform(emb)
    return PCA(n_components=2, random_state=seed).fit_transform(emb)


def embedding_map(emb, tiles, ax, method="pca", n_clusters=8, n_thumb=0, seed=0, title=""):
    """Project emb to 2D, k-means cluster, scatter (optionally overlay n_thumb tiles)."""
    coords = _project(emb, method, seed)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(emb)
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    if n_thumb:
        for i in np.linspace(0, len(tiles) - 1, min(n_thumb, len(tiles))).astype(int):
            im = np.asarray(tile_to_pil(tiles[i]).resize((26, 26)))
            ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=1), coords[i], frameon=False))
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    return coords, labels


def embedding_png(model, tiles, out_png, device="cpu", method="pca", n_clusters=8,
                  n_thumb=0, seed=0):
    """Compute embeddings for ALL tiles and draw one 2D map -> PNG. Returns (fig, emb)."""
    emb = extract_embeddings(model, tiles, device)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    embedding_map(emb, tiles, ax, method, n_clusters, n_thumb, seed,
                  f"DINO CLS embeddings — {len(tiles)} tiles ({method.upper()}, k={n_clusters})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig, emb


def compare_embeddings(weights, ckpt, tiles, device="cpu", method="pca", n_clusters=8,
                       n_thumb=0, out_png=None, seed=0, cache_root=CACHE_ROOT):
    """Two embedding maps side by side: pretrained vs finetuned. Returns (fig, emb_pre, emb_ft)."""
    (_, cpre), (_, cft), _ = pre_ft_features(weights, ckpt, tiles, device, cache_root=cache_root)
    emb_pre = cpre.numpy()                                        # SAT, adapters = 0
    emb_ft = cft.numpy()                                          # SAT + finetuned adapters

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    embedding_map(emb_pre, tiles, axes[0], method, n_clusters, n_thumb, seed, "pretrained (SAT)")
    embedding_map(emb_ft, tiles, axes[1], method, n_clusters, n_thumb, seed,
                  f"finetuned ({Path(ckpt).parent.name})")
    fig.suptitle(f"DINO CLS embeddings ({len(tiles)} tiles) — {method.upper()} 2D, k-means k={n_clusters}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig, emb_pre, emb_ft


# ── patch k-means segmentation ────────────────────────────────────────────────────
def _patch_cluster_maps(feats, n_clusters, seed=0):
    """GLOBAL k-means over all patches of all tiles -> per-tile (g,g) label maps.

    One global fit means a cluster id == the same semantic group across every tile,
    so the colors are comparable tile-to-tile (segmentation-like).
    """
    T, N, D = feats.shape
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        feats.reshape(T * N, D).numpy())
    g = int(round(N ** 0.5))
    return labels.reshape(T, g, g)


def patch_cluster_grid(weights, ckpt, tiles, n_clusters=8, device="cpu", img_size=224,
                       out_png=None, seed=0, cache_root=CACHE_ROOT):
    """Grid like _pca_grid but with PATCH k-means: rows = tiles, cols = input |
    pretrained patch-clusters | finetuned patch-clusters. Global k-means -> colors are
    consistent across tiles (same color = same cluster)."""
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    maps_pre = _patch_cluster_maps(fpre, n_clusters, seed)
    maps_ft = _patch_cluster_maps(fft, n_clusters, seed)

    imgs = [tile_to_pil(p).resize((img_size, img_size)) for p in tiles]
    cmap = plt.get_cmap("tab20", n_clusters)
    cols = ["input", "pretrained patch k-means", "finetuned patch k-means"]
    nrow = len(tiles)
    fig, axes = plt.subplots(nrow, 3, figsize=(8.5, 2.9 * nrow), squeeze=False)
    for r in range(nrow):
        axes[r][0].imshow(imgs[r])
        axes[r][1].imshow(maps_pre[r], cmap=cmap, vmin=0, vmax=n_clusters - 1, interpolation="nearest")
        axes[r][2].imshow(maps_ft[r], cmap=cmap, vmin=0, vmax=n_clusters - 1, interpolation="nearest")
        for c in range(3):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(cols[c], fontsize=11)
    fig.suptitle(f"patch k-means (k={n_clusters}, global over {len(tiles)} tiles) — "
                 "same color = same cluster", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


# ── whole-site PCA-RGB mosaic ─────────────────────────────────────────────────────
def _global_pca_proj(feats, seed=0, sample=50000):
    """(T,N,D) -> (T*N, 3) PCA projection (raw, unnormalized). One global PCA."""
    T, N, D = feats.shape
    flat = feats.reshape(T * N, D).numpy()
    idx = np.random.default_rng(seed).choice(T * N, min(sample, T * N), replace=False)
    pca = PCA(n_components=3, random_state=seed).fit(flat[idx])           # fit on a subsample
    return pca.transform(flat)


def _global_pca_maps(feats, g, seed=0, sample=50000):
    """(T,N,D) patch feats -> (T,g,g,3) RGB maps via ONE global PCA (consistent colors)."""
    proj = _global_pca_proj(feats, seed, sample)
    lo, hi = proj.min(0), proj.max(0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return proj.reshape(feats.shape[0], g, g, 3)


def _aligned_pca_maps(fpre, fft, g, seed=0, sample=50000):
    """PCA both, then ALIGN the finetuned color axes to the pretrained ones via
    orthogonal Procrustes on the corresponding patches (same tile positions), and
    normalize jointly -> R/G/B mean the same thing in both maps. Returns (pre, ft)."""
    from scipy.linalg import orthogonal_procrustes

    pre = _global_pca_proj(fpre, seed, sample)                           # (M, 3)
    ft = _global_pca_proj(fft, seed, sample)                            # (M, 3), same rows
    R, _ = orthogonal_procrustes(ft, pre)                              # best rotation/flip ft->pre
    ft = ft @ R
    both = np.concatenate([pre, ft], axis=0)                            # shared color range
    lo, hi = both.min(0), both.max(0)
    def norm(x):
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    T = fpre.shape[0]
    return norm(pre).reshape(T, g, g, 3), norm(ft).reshape(T, g, g, 3)


def _mosaic_png(maps, tiles, title, out_png, figsize=(12, 12), dpi=200):
    """Draw one whole-site feature mosaic to its own PNG (full resolution)."""
    fig, ax = plt.subplots(figsize=figsize)
    for fmap, tile in zip(maps, tiles):
        with rasterio.open(tile) as src:
            b = src.bounds
        ax.imshow(fmap, extent=(b.left, b.right, b.bottom, b.top), origin="upper")
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_title(f"{title}  ({len(tiles)} tiles, PCA-RGB patch features)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def compare_feature_mosaic(weights, ckpt, tiles, device="cpu", img_size=224,
                           out_png=None, seed=0, dpi=200, align=True, cache_root=CACHE_ROOT):
    """Whole-site feature mosaic: every tile drawn as its PCA-RGB patch-feature map at
    its geo position (like overview.png, but features instead of imagery).

    Saves TWO separate PNGs (full resolution each): pretrained and finetuned. From
    out_png='..._mosaic.png' you get '..._mosaic_pretrained.png' and '..._finetuned.png'.
    align=True (default): the finetuned PCA color axes are Procrustes-aligned to the
    pretrained ones (same patches), so R/G/B mean the same thing in both maps -> you
    can compare them directly. Returns (pretrained_path, finetuned_path).
    """
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    if align:                                                            # shared color axes
        maps_pre, maps_ft = _aligned_pca_maps(fpre, fft, g, seed)
    else:                                                                # independent per map
        maps_pre, maps_ft = _global_pca_maps(fpre, g, seed), _global_pca_maps(fft, g, seed)
    del fpre, fft

    out_pre = out_ft = None
    if out_png:
        p = Path(out_png)
        out_pre = p.with_name(f"{p.stem}_pretrained{p.suffix}")
        out_ft = p.with_name(f"{p.stem}_finetuned{p.suffix}")
    _mosaic_png(maps_pre, tiles, "pretrained (SAT)", out_pre, dpi=dpi)
    _mosaic_png(maps_ft, tiles, "finetuned", out_ft, dpi=dpi)
    return out_pre, out_ft


# ── patch-cluster exemplars ───────────────────────────────────────────────────────
def _cluster_composition(feats, n_clusters, seed=0):
    """Global patch k-means -> per-tile (T, K) fraction of patches in each cluster."""
    T, N, D = feats.shape
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        feats.reshape(T * N, D).numpy()).reshape(T, N)
    comp = np.stack([(labels == c).mean(1) for c in range(n_clusters)], axis=1)   # (T, K)
    return comp


def _exemplar_montage(comp, tiles, n_clusters, per_cluster, title, out_png, thumb=64):
    """Rows = clusters; for each, the tiles with the highest fraction of that cluster."""
    cmap = plt.get_cmap("tab20", n_clusters)
    fig, axes = plt.subplots(n_clusters, per_cluster + 1,
                             figsize=(1.6 * (per_cluster + 1), 1.6 * n_clusters), squeeze=False)
    for c in range(n_clusters):
        sw = axes[c][0]
        sw.imshow(np.full((8, 8, 3), cmap(c)[:3]))
        sw.set_ylabel(f"cluster {c}", fontsize=9); sw.set_xticks([]); sw.set_yticks([])
        top = np.argsort(comp[:, c])[::-1][:per_cluster]                 # tiles most in cluster c
        for j, ti in enumerate(top):
            ax = axes[c][j + 1]
            ax.imshow(tile_to_pil(tiles[ti]).resize((thumb, thumb)))
            ax.set_title(f"{comp[ti, c] * 100:.0f}%", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title} — patch clusters: tiles most dominated by each (% = patch share)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_png:
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def compare_patch_exemplars(weights, ckpt, tiles, n_clusters=8, per_cluster=6,
                            device="cpu", img_size=224, out_png=None, seed=0,
                            cache_root=CACHE_ROOT):
    """For each PATCH cluster, show the tiles most dominated by it -> interpret the
    clusters. Saves TWO PNGs (pretrained, finetuned). Returns (pre_path, ft_path)."""
    (fpre, _), (fft, _), _ = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    comp_pre = _cluster_composition(fpre, n_clusters, seed)
    comp_ft = _cluster_composition(fft, n_clusters, seed)

    out_pre = out_ft = None
    if out_png:
        p = Path(out_png)
        out_pre = p.with_name(f"{p.stem}_pretrained{p.suffix}")
        out_ft = p.with_name(f"{p.stem}_finetuned{p.suffix}")
    _exemplar_montage(comp_pre, tiles, n_clusters, per_cluster, "pretrained (SAT)", out_pre)
    _exemplar_montage(comp_ft, tiles, n_clusters, per_cluster, "finetuned", out_ft)
    return out_pre, out_ft


# ── cross-model per-patch cosine change ───────────────────────────────────────────
def _patch_cosine(fpre, fft):
    """(T,N,D) pre & ft -> (T, g, g) cosine similarity between the SAME patch position
    in both models. The two come from the SAME input, so a low value means finetuning
    rotated that patch's representation a lot (= changed it)."""
    sim = (F.normalize(fpre, dim=-1) * F.normalize(fft, dim=-1)).sum(-1)   # (T, N)
    g = int(round(fpre.shape[1] ** 0.5))
    return sim.reshape(fpre.shape[0], g, g).numpy()


def _cosine_mosaic_png(sim_maps, tiles, vmin, vmax, out_png,
                       figsize=(12, 12), dpi=200, cmap="magma"):
    """Whole-site heatmap of the per-patch cosine change at each tile's geo position."""
    fig, ax = plt.subplots(figsize=figsize)
    im = None
    for smap, tile in zip(sim_maps, tiles):
        with rasterio.open(tile) as src:
            b = src.bounds
        im = ax.imshow(smap, extent=(b.left, b.right, b.bottom, b.top), origin="upper",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_title(f"per-patch cosine(pretrained, finetuned) — {len(tiles)} tiles "
                 "(dark = changed most)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="cosine similarity")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def compare_patch_cosine(weights, ckpt, tiles, device="cpu", img_size=224,
                         out_png=None, seed=0, dpi=200, cache_root=CACHE_ROOT):
    """WHERE did finetuning change the features? For every patch, the cosine similarity
    between its pretrained and finetuned embedding (same position, same input image).

    Saves a whole-site geo HEATMAP (dark = big change) and a histogram of all per-patch
    similarities, and prints the mean. Returns (mosaic_path, hist_path, mean_sim).
    """
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    sim = _patch_cosine(fpre, fft)                       # (T, g, g)

    flat = sim.reshape(-1)
    mean_sim = float(flat.mean())
    vmin = float(np.percentile(flat, 1))                 # robust low end -> contrast
    print(f"mean per-patch cosine(pretrained, finetuned) = {mean_sim:.4f}  "
          f"(1st pct {vmin:.3f}); lower = more changed by finetuning")

    out_mosaic = out_hist = None
    if out_png:
        p = Path(out_png)
        out_mosaic = p.with_name(f"{p.stem}_mosaic{p.suffix}")
        out_hist = p.with_name(f"{p.stem}_hist{p.suffix}")
    _cosine_mosaic_png(sim, tiles, vmin, 1.0, out_mosaic, dpi=dpi)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flat, bins=80, color="#444")
    ax.axvline(mean_sim, color="#e41a1c", lw=2, label=f"mean = {mean_sim:.3f}")
    ax.set_xlabel("cosine(pretrained patch, finetuned patch)")
    ax.set_ylabel("patch count")
    ax.set_title(f"how much finetuning moved each patch ({len(tiles)} tiles)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    if out_hist:
        fig.savefig(out_hist, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_mosaic, out_hist, mean_sim


# ── the most-changed patches: where they are + how they moved ──────────────────────
def _changed_montage_png(tiles, order, maps_pre, maps_ft, mask, g, img_size, out_png, noun="changed"):
    """Rows = the selected tiles; cols = RGB | pretrained PCA-RGB | finetuned PCA-RGB
    (the feature-colour shift between the last two is the per-patch change)."""
    cols = ["input", "pretrained PCA-RGB", "finetuned PCA-RGB"]
    fig, axes = plt.subplots(len(order), 3, figsize=(9, 3 * len(order)), squeeze=False)
    for row, ti in enumerate(order):
        img = tile_to_pil(tiles[ti]).resize((img_size, img_size))
        axes[row][0].imshow(img)
        axes[row][1].imshow(maps_pre[ti])
        axes[row][2].imshow(maps_ft[ti])
        for c in range(3):
            axes[row][c].set_xticks([]); axes[row][c].set_yticks([])
            if row == 0:
                axes[row][c].set_title(cols[c], fontsize=11)
        axes[row][0].set_ylabel(f"{int(mask[ti].sum())} {noun}", fontsize=9)
    verb = "changed most" if noun == "changed" else "stayed most stable"
    fig.suptitle(f"tiles that {verb}: pretrained vs finetuned feature colour", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if out_png:
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _migration_scatter_png(fpre, fft, low, out_png, seed=0, sample=8000, n_arrows=300):
    """Shared-PCA scatter of where the most-changed patches sat (pretrained) and moved to
    (finetuned), with arrows. TWO planes side by side — PC1-PC2 and PC1-PC3 — so extra
    differentiation that shows up along PC3 (not only PC2) is visible. Background grey = all
    patches (pretrained). One PCA + one Procrustes alignment shared by both planes."""
    from scipy.linalg import orthogonal_procrustes

    pre = _global_pca_proj(fpre, seed)                       # (M, 3)
    ft = _global_pca_proj(fft, seed)
    R, _ = orthogonal_procrustes(ft, pre)                   # align ft axes onto pre
    ft = ft @ R
    M = pre.shape[0]
    rng = np.random.default_rng(seed)
    bg = rng.choice(M, min(sample, M), replace=False)
    lows = np.where(low.reshape(-1))[0]
    sel = rng.choice(lows, min(n_arrows, len(lows)), replace=False)

    def draw(ax, yc):                                        # yc = y-axis component (1=PC2, 2=PC3)
        # both FULL clouds, subtly distinct: grey = all patches before, pink = all patches after
        ax.scatter(pre[bg, 0], pre[bg, yc], s=3, c="0.78", label="all patches · pretrained")
        ax.scatter(ft[bg, 0], ft[bg, yc], s=3, c="#f0b9b0", alpha=0.45, label="all patches · finetuned")
        for i in sel:
            ax.annotate("", xy=(ft[i, 0], ft[i, yc]), xytext=(pre[i, 0], pre[i, yc]),
                        arrowprops=dict(arrowstyle="->", color="#777", lw=0.4, alpha=0.5))
        ax.scatter(pre[sel, 0], pre[sel, yc], s=10, c="#1f77b4", label="changed: pretrained")
        ax.scatter(ft[sel, 0], ft[sel, yc], s=10, c="#e41a1c", label="changed: finetuned")
        ax.set_xlabel("PCA-1"); ax.set_ylabel(f"PCA-{yc + 1}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw(axes[0], 1); draw(axes[1], 2)
    axes[0].set_title("PC1–PC2 plane"); axes[1].set_title("PC1–PC3 plane (extra differentiation?)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("how the most-changed patches migrate in PCA feature space", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _pick_tiles(counts, k_tiles, randomize, pool_frac, seed):
    """Which tiles to show: the strict top-k by changed-patch count, or — when randomize —
    a random sample drawn from the top `pool_frac` of changed tiles (variety across the
    big changes, not always the same extremes)."""
    ranked = np.argsort(counts)[::-1]
    if not randomize:
        return ranked[:k_tiles]
    pool = ranked[:max(k_tiles, int(len(counts) * pool_frac))]
    pool = pool[counts[pool] > 0]                          # must actually have changed patches
    return np.random.default_rng(seed).choice(pool, min(k_tiles, len(pool)), replace=False)


def changed_patch_story(weights, ckpt, tiles, q=0.05, which="changed", k_tiles=6, randomize=False,
                        pool_frac=0.4, device="cpu", img_size=224, out_png=None, seed=0,
                        cache_root=CACHE_ROOT):
    """Pick the patches that moved most (which='changed', lowest pre-vs-ft cosine) or least
    (which='stable', highest cosine), and show (1) the tiles richest in them, with PCA-RGB
    feature colour before/after, and (2) how they migrate in the shared PCA feature space.

    q = the quantile (0.05 -> the 5% most/least changed patches). randomize=True samples
    the shown tiles from the top `pool_frac` of qualifying tiles (vary seed for new draws).
    Saves two PNGs (_montage, _migration). Returns (montage_path, migration_path, threshold).
    """
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    sim = _patch_cosine(fpre, fft)                          # (T, g, g)
    if which == "changed":
        thr = float(np.quantile(sim, q)); mask = sim <= thr; noun = "changed"
    else:
        thr = float(np.quantile(sim, 1 - q)); mask = sim >= thr; noun = "stable"
    maps_pre, maps_ft = _aligned_pca_maps(fpre, fft, g, seed)
    counts = mask.reshape(len(tiles), -1).sum(1)
    order = _pick_tiles(counts, k_tiles, randomize, pool_frac, seed)
    pick = "random sample" if randomize else "top"
    print(f"{noun} threshold (q={q}) = cosine {thr:.3f}; {int(mask.sum())} / {mask.size} patches "
          f"{noun}; showing {pick} {len(order)} tiles")

    out_mont = out_mig = None
    if out_png:
        p = Path(out_png)
        out_mont = p.with_name(f"{p.stem}_montage{p.suffix}")
        out_mig = p.with_name(f"{p.stem}_migration{p.suffix}")
    _changed_montage_png(tiles, order, maps_pre, maps_ft, mask, g, img_size, out_mont, noun)
    _migration_scatter_png(fpre, fft, mask, out_mig, seed)
    return out_mont, out_mig, thr


# ── basis-free expressiveness of a patch set (is the nuance real or PCA-1/2 only?) ──
def _participation_ratio(feats):
    """Effective dimensionality (sum λ)^2 / sum λ^2 of the covariance — basis-free.
    High = variance spread over many axes; low = concentrated in a few."""
    f = feats.float() - feats.float().mean(0, keepdim=True)
    lam = torch.linalg.svdvals(f) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def _mean_pairwise_cos(feats, sample=2000, seed=0):
    """Mean off-diagonal cosine among a sample of patches — basis-free spread.
    Lower = the patches are more differentiated from each other."""
    f = F.normalize(feats.float(), dim=-1)
    idx = np.random.default_rng(seed).choice(len(f), min(sample, len(f)), replace=False)
    sim = f[idx] @ f[idx].t()
    n = sim.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))          # exclude the diagonal (=1)


def _ev_curve(feats, k=40):
    """Cumulative explained-variance ratio of the first k PCA components (own space)."""
    f = feats.numpy()
    p = PCA(n_components=min(k, f.shape[1], f.shape[0])).fit(f)
    return np.cumsum(p.explained_variance_ratio_)


def changed_patch_expressiveness(weights, ckpt, tiles, q=0.05, which="changed", device="cpu",
                                 img_size=224, out_png=None, seed=0, cache_root=CACHE_ROOT):
    """Is the finetuned 'explosion' real expressiveness, or just PCA-1/2 reallocation?
    For a patch set, measure — in EACH model's OWN full space — the effective
    dimensionality (participation ratio), the basis-free spread (mean pairwise cosine),
    and the explained-variance spectrum. Saves one PNG, returns the metrics.

    which='changed' = the q lowest-cosine (most-moved) patches; 'stable' = the q
    highest-cosine (barely-moved) patches, as a control.
    """
    (fpre, _), (fft, _), _ = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    sim = _patch_cosine(fpre, fft).reshape(-1)
    if which == "changed":
        sel = sim <= np.quantile(sim, q)                   # most-moved
        label = "most-changed"
    else:
        sel = sim >= np.quantile(sim, 1 - q)               # barely-moved (control)
        label = "least-changed (stable)"
    Fpre = fpre.reshape(-1, fpre.shape[-1])[sel]           # (m, D) in pretrained space
    Fft = fft.reshape(-1, fft.shape[-1])[sel]              # (m, D) in finetuned space

    m = dict(n=int(sel.sum()), which=which,
             pr_pre=_participation_ratio(Fpre), pr_ft=_participation_ratio(Fft),
             cos_pre=_mean_pairwise_cos(Fpre, seed=seed), cos_ft=_mean_pairwise_cos(Fft, seed=seed))
    ev_pre, ev_ft = _ev_curve(Fpre), _ev_curve(Fft)
    print(f"{label} patches: {m['n']}")
    print(f"  effective dim (PR):     pretrained {m['pr_pre']:.1f}  ->  finetuned {m['pr_ft']:.1f}")
    print(f"  mean pairwise cosine:   pretrained {m['cos_pre']:.3f}  ->  finetuned {m['cos_ft']:.3f}  "
          f"(lower = more differentiated)")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(range(1, len(ev_pre) + 1), ev_pre, "-o", ms=3,
            label=f"pretrained (PR={m['pr_pre']:.0f}, mean-cos={m['cos_pre']:.2f})")
    ax.plot(range(1, len(ev_ft) + 1), ev_ft, "-o", ms=3,
            label=f"finetuned (PR={m['pr_ft']:.0f}, mean-cos={m['cos_ft']:.2f})")
    ax.set_xlabel("PCA components (own space)"); ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"expressiveness of the {m['n']} {label} patches — each in its OWN space",
                 fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return m


# ── the 'explosion' seen directly: patches scattered in the PC2-PC3 plane ───────────
def _norm_pca3(feats, seed=0, sample=40000):
    """Patches -> PCA(3) AFTER scaling to unit total variance, so the cloud SIZE is
    comparable across models (PC1 is usually brightness; PC2/PC3 carry the structure).
    Returns (proj (M,3), explained_variance_ratio (3,))."""
    f = feats.reshape(-1, feats.shape[-1]).float()
    f = f - f.mean(0, keepdim=True)
    f = (f / (f.pow(2).sum() / f.shape[0]).sqrt()).numpy()       # total variance -> 1
    idx = np.random.default_rng(seed).choice(len(f), min(sample, len(f)), replace=False)
    p = PCA(n_components=3, random_state=seed).fit(f[idx])
    return p.transform(f), p.explained_variance_ratio_


def _explosion_png(proj_pre, evr_pre, pr_pre, proj_ft, evr_ft, pr_ft, out_png, sample=8000, seed=0):
    """Two PC2-vs-PC3 panels (pretrained | finetuned), shared axes, colored by PC1.
    Finetuned filling MORE area = variance pushed off PC1 into more directions = explosion."""
    rng = np.random.default_rng(seed)
    def sub(P):
        return P[rng.choice(len(P), min(sample, len(P)), replace=False)]
    Ppre, Pft = sub(proj_pre), sub(proj_ft)
    lim = np.percentile(np.abs(np.vstack([Ppre[:, 1:3], Pft[:, 1:3]])), 99)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True)
    sc = None
    for ax, P, evr, pr, name in [(axes[0], Ppre, evr_pre, pr_pre, "pretrained"),
                                 (axes[1], Pft, evr_ft, pr_ft, "finetuned")]:
        sc = ax.scatter(P[:, 1], P[:, 2], c=P[:, 0], cmap="Spectral", s=4, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("PC2"); ax.set_ylabel("PC3"); ax.grid(alpha=0.25)
        ax.set_title(f"{name}\nPC2+PC3 = {(evr[1] + evr[2]) * 100:.0f}% of variance | PR = {pr:.0f}",
                     fontsize=11)
    fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label="PC1 (color)")
    fig.suptitle("PCA 'explosion' in the PC2–PC3 plane "
                 "(unit total variance -> comparable; wider cloud = more spread)", fontsize=12)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def pca_explosion(weights, ckpt, tiles, device="cpu", img_size=224, out_png=None, seed=0,
                  cache_root=CACHE_ROOT):
    """SEE the 'explosion': scatter every patch in the PC2-PC3 plane, pretrained vs finetuned.
    PC1 typically encodes brightness/background; finetuning pushes variance off PC1 into
    PC2/PC3+ (more directions), so the cloud spreads. Clouds are normalized to unit total
    variance to be size-comparable, and the participation ratio (PR) quantifies the spread.
    Returns the metrics dict."""
    (fpre, _), (fft, _), _ = pre_ft_features(weights, ckpt, tiles, device, img_size, cache_root)
    proj_pre, evr_pre = _norm_pca3(fpre, seed)
    proj_ft, evr_ft = _norm_pca3(fft, seed)
    pr_pre = _participation_ratio(fpre.reshape(-1, fpre.shape[-1]))
    pr_ft = _participation_ratio(fft.reshape(-1, fft.shape[-1]))
    print(f"PCA explosion: PC2+PC3 variance  pretrained {(evr_pre[1] + evr_pre[2]) * 100:.1f}%  ->  "
          f"finetuned {(evr_ft[1] + evr_ft[2]) * 100:.1f}%   |   PR {pr_pre:.0f} -> {pr_ft:.0f}")
    if out_png:
        _explosion_png(proj_pre, evr_pre, pr_pre, proj_ft, evr_ft, pr_ft, Path(out_png), seed=seed)
    return dict(evr_pre=evr_pre.tolist(), evr_ft=evr_ft.tolist(), pr_pre=pr_pre, pr_ft=pr_ft)
