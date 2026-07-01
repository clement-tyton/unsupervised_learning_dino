"""
Cross the unsupervised patch k-means with the human CLASSIFY annotations.

Runs on the SAME .tif tiles as training/visualisation: the npz masks are realigned onto the
webmap grid (src/train/align.py), so each imagery RGB/<area>/tile_NNNNN.tif pairs 1:1 with a
mask masks/<area>/tile_NNNNN.tif. The pipeline:
  1. run DINO on each imagery .tif at img_size=512 -> 32x32 (=1024) patch features per tile
  2. global k-means over all patches -> one cluster id per patch
  3. majority-pool the aligned mask onto the SAME 32x32 patch grid
  4. score cluster-vs-annotation agreement: NMI / ARI / purity, pretrained vs finetuned

Features are cached (own root, EVAL_CACHE) so reruns are instant. Realignment note: at
img_size=512 DINO makes 32x32 patches over the 512px tile, so patch (r,c) is the mask block
[r*16:(r+1)*16, c*16:(c+1)*16]; majority vote in each block gives the patch's class.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.train.align import AREA_SIDECAR
from src.visualisation.common import _embed_tensor
from src.visualisation.features import pre_ft_features

EVAL_CACHE = Path("outputs/_feature_cache_eval")        # .tif/512 (32x32) features, apart from the 224 viz cache

CLASS_NAMES = {                                          # source class id -> human name (0 = ignore)
    0: "(ignore)",
    1: "Not Erosion", 2: "Ground", 4: "Shrub", 5: "Tree", 6: "Herb", 7: "Grass", 9: "Generic debris",
    14: "Erosion", 40: "Sedge", 100: "Biotic", 200: "Abiotic", 201: "Water", 301: "Tussock",
    10108: "Chile_Tree1", 10109: "Chile_Tree2", 10110: "Chile_Tree3", 10111: "Chile_Shrub1",
    10115: "Eucalyptus spp", 10117: "Melaleuca Argentea", 10118: "Cenchrus spp",
    10119: "Hummock Grass", 10120: "Aerva javanica", 10121: "Annual herbs and grasses",
    10122: "Phoenix dactylifera", 10123: "Eucalyptus camaldulensis", 10124: "Eucalyptus victrix",
    10125: "Mulga", 10126: "Calotropis procera",
}

# Source-id remap applied at mask read (raw id -> canonical id), so a site's odd codes fold into
# the right taxonomy class everywhere (present_classes / targets / F1 / MLflow). E.g. manned's
# 10116 is Tussock grass -> 301. Add entries here as new sites surface non-standard codes.
CLASS_REMAP = {10116: 301}


# ── aligned .tif IO (imagery tile <-> grid-aligned mask tile, paired by stem) ─────────
def aligned_pairs(rgb_root, masks_root):
    """(imagery_paths, mask_paths) for every grid cell that has BOTH an imagery .tif and an
    aligned mask .tif, across all area subdirs of masks_root. Paired by tile stem so imagery
    tile_00571.tif lines up with mask tile_00571.tif (same geographic cell)."""
    rgb_root, masks_root = Path(rgb_root), Path(masks_root)
    imgs, masks = [], []
    for area_dir in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        rgb_dir = rgb_root / area_dir.name
        for mp in sorted(area_dir.glob("*.tif")):
            ip = rgb_dir / mp.name
            if ip.exists():
                imgs.append(str(ip)); masks.append(mp)
    return imgs, masks


def _tif_rgb(tif_path):
    """(H,W,3) uint8 from a 3-band RGB GeoTIFF, per-band min-max stretched (for display)."""
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3]).astype("float32")
    for i in range(3):
        b = arr[i]; lo, hi = b.min(), b.max()
        if hi > lo:
            arr[i] = (b - lo) / (hi - lo) * 255.0
    return arr.astype("uint8").transpose(1, 2, 0)


def read_mask(mask_path):
    """The (H,W) class annotation: band 1 of a grid-aligned mask .tif, with CLASS_REMAP applied
    so non-standard source codes fold into their canonical class (e.g. manned's leftover 10116
    artefacts -> 301 Tussock)."""
    with rasterio.open(mask_path) as src:
        m = src.read(1)
    for src_id, dst_id in CLASS_REMAP.items():
        m[m == src_id] = dst_id
    return m


def _drop_small(imgs, masks, min_side):
    """Drop degenerate edge tiles whose mask is smaller than the gxg patch grid on a side —
    their imagery would be stretched to nonsense and they can't be pooled. Returns the kept
    (imgs, masks)."""
    keep = [i for i, m in enumerate(masks) if min(read_mask(m).shape) >= min_side]
    if len(keep) < len(masks):
        print(f"[eval] dropped {len(masks) - len(keep)} tiles smaller than {min_side}px grid")
    return [imgs[i] for i in keep], [masks[i] for i in keep]


# ── realignment: 384 mask -> g x g patch grid ──────────────────────────────────────
def pool_mask(mask, g):
    """Majority class of each patch block -> (g, g) annotation at the patch resolution.
    An empty block (a tiny edge tile with fewer than g pixels on a side) -> class 0, which
    the eval already ignores, so degenerate slivers never crash and never score."""
    H, W = mask.shape
    ys = np.linspace(0, H, g + 1).astype(int)
    xs = np.linspace(0, W, g + 1).astype(int)
    out = np.zeros((g, g), dtype=mask.dtype)
    for r in range(g):
        for c in range(g):
            block = mask[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].ravel()
            if block.size:
                vals, cnts = np.unique(block, return_counts=True)
                out[r, c] = vals[cnts.argmax()]
    return out


# ── clustering + scoring ────────────────────────────────────────────────────────────
def _kmeans_maps(feats, n_clusters, seed):
    """Global k-means over all patches -> per-tile (T, g, g) cluster-id maps."""
    T, N, D = feats.shape
    labels = KMeans(n_clusters, n_init=10, random_state=seed).fit_predict(feats.reshape(T * N, D).numpy())
    g = int(round(N ** 0.5))
    return labels.reshape(T, g, g)


def _purity(true, pred):
    """Fraction of patches in the majority class of their assigned cluster."""
    cm = contingency_matrix(true, pred)
    return float(cm.max(axis=0).sum() / cm.sum())


def _scores(annot, clusters, ignore):
    """NMI / ARI / purity of clusters vs annotation, dropping the `ignore` classes."""
    keep = ~np.isin(annot, list(ignore))
    a, c = annot[keep], clusters[keep]
    return dict(nmi=float(normalized_mutual_info_score(a, c)),
                ari=float(adjusted_rand_score(a, c)),
                purity=_purity(a, c), n=int(keep.sum()))


# ── figures ──────────────────────────────────────────────────────────────────────
def _scores_bar_png(res, n_clusters, out_png):
    metrics = ["nmi", "ari", "purity"]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for off, name in [(-w / 2, "pretrained"), (w / 2, "finetuned")]:
        vals = [res[name][m] for m in metrics]
        ax.bar(x + off, vals, w, label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["NMI", "ARI", "purity"])
    ax.set_ylabel("agreement with CLASSIFY (higher = better)")
    ax.set_title(f"patch k-means (k={n_clusters}) vs annotation — pretrained vs finetuned", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


def _examples_png(imgs, annot_maps, cluster_maps, n_clusters, k, seed, out_png):
    """k tiles: imagery | CLASSIFY annotation | finetuned k-means (both at patch grid)."""
    idx = np.random.default_rng(seed).choice(len(imgs), min(k, len(imgs)), replace=False)
    classes = np.unique(annot_maps)
    cidx = {c: i for i, c in enumerate(classes)}
    acmap = plt.get_cmap("tab20", len(classes))
    kcmap = plt.get_cmap("tab20", n_clusters)
    titles = ["imagery", "CLASSIFY annotation", "finetuned k-means"]
    fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)), squeeze=False)
    for row, ti in enumerate(idx):
        axes[row][0].imshow(_tif_rgb(imgs[ti]))
        amap = np.vectorize(cidx.get)(annot_maps[ti])
        axes[row][1].imshow(amap, cmap=acmap, vmin=0, vmax=len(classes) - 1, interpolation="nearest")
        axes[row][2].imshow(cluster_maps[ti], cmap=kcmap, vmin=0, vmax=n_clusters - 1, interpolation="nearest")
        for c in range(3):
            axes[row][c].set_xticks([]); axes[row][c].set_yticks([])
            if row == 0:
                axes[row][c].set_title(titles[c], fontsize=11)
    fig.suptitle("imagery vs annotation vs unsupervised clusters (patch grid)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


# ── contingency: where each annotation class lands among the clusters ──────────────
def _contingency_png(M, classes, n_clusters, model, out_png):
    """Heatmap: annotation class (rows) x cluster id (cols), each ROW normalized so it
    reads 'what % of this class's patches fall in each cluster'. No cluster->class
    assignment — just the raw distribution."""
    counts = M.sum(1)
    Mn = M / M.sum(1, keepdims=True).clip(min=1)
    rows = [f"{CLASS_NAMES.get(int(c), c)} ({int(c)})  n={int(n)}" for c, n in zip(classes, counts)]
    fig, ax = plt.subplots(figsize=(1.1 * n_clusters + 3, 0.6 * len(classes) + 2))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_clusters)); ax.set_xticklabels([f"c{j}" for j in range(n_clusters)])
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlabel("k-means cluster (unsupervised, not assigned to any class)")
    ax.set_title(f"where each annotation class lands among the {model} clusters\n"
                 "(row-normalized: % of the class's patches per cluster)", fontsize=10)
    for i in range(len(classes)):
        for j in range(n_clusters):
            v = Mn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="row fraction")
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def confusion_cluster_vs_annotation(weights, ckpt, rgb_root, masks_root, model="finetuned",
                                    n_clusters=8, img_size=512, ignore=(0,), device="cpu",
                                    out_png=None, seed=0, max_tiles=None):
    """Contingency matrix annotation-class x cluster (no assignment). Returns (M, classes)
    with M[i,j] = #patches of class i that k-means put in cluster j, for `model` in
    {'pretrained','finetuned'}. Shows how each labelled class spreads over the clusters."""
    imgs, masks = aligned_pairs(rgb_root, masks_root)
    if max_tiles:
        imgs, masks = imgs[:max_tiles], masks[:max_tiles]
    imgs, masks = _drop_small(imgs, masks, img_size // 16)
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, imgs, device, img_size,
                                             EVAL_CACHE, to_tensor=_embed_tensor)
    feats = {"pretrained": fpre, "finetuned": fft}[model]
    clusters = _kmeans_maps(feats, n_clusters, seed).reshape(-1)
    annot = np.stack([pool_mask(read_mask(m), g) for m in masks]).reshape(-1)
    keep = ~np.isin(annot, list(ignore))
    classes = np.unique(annot[keep])
    M = contingency_matrix(annot[keep], clusters[keep])            # (n_classes, n_clusters)
    print(f"{model}: contingency {M.shape} (classes x clusters), {keep.sum()} patches")
    if out_png:
        _contingency_png(M, classes, n_clusters, model, Path(out_png))
    return M, classes


# ── supervised linear probe: orient the split by the LABELS ────────────────────────
def _area_index(masks):
    """Map each aligned tile to an integer training-AREA index, read from the per-area-dir
    `_training_areas.json` sidecar (written by src/train/align.py via footprint overlap).
    Holding out whole areas — not single tiles — keeps neighbouring tiles from straddling the
    split (see [[eval-split-by-training-area]]). Returns (area_idx (T,), n_areas)."""
    sidecars, uids = {}, []
    for mp in masks:
        d = Path(mp).parent
        if d not in sidecars:
            sidecars[d] = json.loads((d / AREA_SIDECAR).read_text())
        uids.append(sidecars[d].get(Path(mp).stem))
    order = {u: i for i, u in enumerate(dict.fromkeys(uids))}    # stable int per area
    return np.array([order[u] for u in uids]), len(order)


def _patch_table(feats, masks, g, ignore, tile_group=None):
    """Flatten (T,N,D) features + pooled masks -> (X, y, group) over non-ignored patches.
    group is the split key per patch: per-tile by default, or `tile_group` (length T, e.g. a
    training-area index) to hold out whole areas instead of single tiles."""
    T, N = feats.shape[0], feats.shape[1]
    X = feats.reshape(-1, feats.shape[-1]).numpy()
    y = np.stack([pool_mask(read_mask(m), g) for m in masks]).reshape(-1)
    groups = np.arange(T) if tile_group is None else np.asarray(tile_group)
    gid = np.repeat(groups, N)
    keep = ~np.isin(y, list(ignore))
    return X[keep], y[keep], gid[keep]


def _tile_split(group_id, test_frac, seed):
    """Split BY GROUP so every patch of a group (tile or area) lands on ONE side only — no
    group leaks across train/test. Returns (train_mask, test_mask) over patches and the
    (train_groups, test_groups) id arrays. Deterministic in seed, so callers agree on it."""
    groups = np.unique(group_id)
    n_test = max(1, round(len(groups) * test_frac))
    test_groups = np.sort(np.random.default_rng(seed).choice(groups, n_test, replace=False))
    test_mask = np.isin(group_id, test_groups)
    return ~test_mask, test_mask, np.setdiff1d(groups, test_groups), test_groups


def _tile_centers(imgs):
    """(T,2) geographic (x,y) center of each tile from its GeoTIFF bounds."""
    centers = []
    for p in imgs:
        with rasterio.open(p) as src:
            b = src.bounds
        centers.append(((b.left + b.right) / 2, (b.bottom + b.top) / 2))
    return np.array(centers)


def _split_map_png(centers, train_tiles, test_tiles, out_png, title):
    """Map tile centers: train=red, validation=green (unused=grey) — eyeball/control the split."""
    tr, te = set(train_tiles.tolist()), set(test_tiles.tolist())
    is_tr = np.array([i in tr for i in range(len(centers))])
    is_te = np.array([i in te for i in range(len(centers))])
    grey = ~(is_tr | is_te)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    if grey.any():
        ax.scatter(centers[grey, 0], centers[grey, 1], c="lightgrey", s=14, label=f"unused ({grey.sum()})")
    ax.scatter(centers[is_tr, 0], centers[is_tr, 1], c="red", s=22, label=f"train ({is_tr.sum()})")
    ax.scatter(centers[is_te, 0], centers[is_te, 1], c="green", s=22, label=f"validation ({is_te.sum()})")
    ax.set_aspect("equal"); ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    ax.set_title(title, fontsize=11); ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def _probe_confusion_png(cm, classes, model, acc, out_png, subtitle=""):
    """True-class x predicted-class heatmap (row-normalized) of a linear probe."""
    names = [f"{CLASS_NAMES.get(int(c), c)} ({int(c)})" for c in classes]
    fig, ax = plt.subplots(figsize=(0.9 * len(classes) + 3, 0.7 * len(classes) + 2))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title(f"linear probe on {model} patch features — overall acc {acc:.2f}{subtitle}\n"
                 "(row-normalized: where each TRUE class gets predicted)", fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="row fraction")
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def _split_groups(masks, split):
    """(group_per_tile (T,), unit_label) for a split mode: 'area' = contiguous training-area
    blocks (honest, no neighbour leak), 'tile' = one group per tile, 'patch' = per-tile here
    (the leaky per-patch shuffle is applied later)."""
    if split == "area":
        return _area_index(masks)[0], "area"
    return np.arange(len(masks)), "tile"


def _probe_fit_predict(X, y, tr, te, seed, sub=50_000, batch=20_000, passes=5):
    """Memory-bounded linear probe: streaming logistic regression via SGDClassifier(log-loss) +
    partial_fit. Standardization stats come from a `sub`-patch subsample, and both fit and predict
    run one `batch`-sized slice at a time — so the full standardized matrix is NEVER materialized
    (peak RAM = one batch, not the whole train set, regardless of site size). It's a separability
    CHECK, not a tuned model, so the SGD approximation of logistic regression is fine.
    Returns (yp aligned with np.where(te)[0], classes)."""
    from sklearn.linear_model import SGDClassifier
    rng = np.random.default_rng(seed)
    tr_i, te_i = np.where(tr)[0], np.where(te)[0]
    ssub = rng.choice(tr_i, min(sub, len(tr_i)), replace=False)   # scaler stats from a bounded subsample
    scaler = StandardScaler().fit(X[ssub])
    classes = np.unique(y)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed)
    for _ in range(passes):                                       # a few streaming epochs
        rng.shuffle(tr_i)
        for i in range(0, len(tr_i), batch):                     # one standardized batch in RAM at a time
            b = tr_i[i:i + batch]
            clf.partial_fit(scaler.transform(X[b]), y[b], classes=classes)
    yp = (np.concatenate([clf.predict(scaler.transform(X[te_i[i:i + batch]]))
                          for i in range(0, len(te_i), batch)]) if len(te_i) else np.array([], int))
    return yp, classes


def linear_probe_confusion(weights, ckpt, rgb_root, masks_root, model="finetuned",
                           img_size=512, ignore=(0,), device="cpu", out_png=None, seed=0,
                           max_tiles=None, test_frac=0.3, split="area"):
    """Supervised linear probe (standardize -> logistic regression) on patch features ->
    annotation class. Oriented BY THE LABELS, so its class x class confusion shows whether
    a class (e.g. Grass) is linearly separable or a genuine catch-all.

    split='area' (default, HONEST): hold out whole training-area BLOCKS (manifest
    training_area_id), so neighbouring tiles never straddle the split. split='tile' holds
    out single tiles (adjacent tiles can still land on opposite sides). split='patch' is the
    old per-patch shuffle (leaky, inflates accuracy). Returns (cm, classes, acc, test_groups)."""
    imgs, masks = aligned_pairs(rgb_root, masks_root)
    if max_tiles:
        imgs, masks = imgs[:max_tiles], masks[:max_tiles]
    imgs, masks = _drop_small(imgs, masks, img_size // 16)
    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, imgs, device, img_size,
                                             EVAL_CACHE, to_tensor=_embed_tensor)
    feats = {"pretrained": fpre, "finetuned": fft}[model]
    groups, unit = _split_groups(masks, split)
    X, y, gid = _patch_table(feats, masks, g, ignore, tile_group=groups)
    if split == "patch":                                          # leaky per-patch shuffle
        idx = np.arange(len(y))
        itr, ite = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
        tr, te = np.isin(idx, itr), np.isin(idx, ite)
        gtr, gte = np.unique(gid[tr]), np.unique(gid[te])
    else:                                                         # hold out whole groups (area/tile)
        tr, te, gtr, gte = _tile_split(gid, test_frac, seed)
    yp, classes = _probe_fit_predict(X, y, tr, te, seed)          # streaming SGD log-loss -> flat RAM
    y_te = y[te]
    acc = float((yp == y_te).mean()) if len(yp) else 0.0
    cm = confusion_matrix(y_te, yp, labels=classes, normalize="true")
    print(f"{model} linear probe ({split} split): acc {acc:.3f} | "
          f"{len(gtr)} train / {len(gte)} val {unit}s, {te.sum()} val patches")
    for c, row in zip(classes, cm):
        print(f"  {CLASS_NAMES.get(int(c), str(c)):16s} recall {row[list(classes).index(c)]:.2f}")
    if out_png:
        leak = {"area": "no area leak", "tile": "no tile leak", "patch": "leaky per-patch"}[split]
        _probe_confusion_png(cm, classes, model, acc, Path(out_png), subtitle=f"  ({split} split, {leak})")
    return cm, classes, acc, gte


def probe_split_map(rgb_root, masks_root, img_size=512, ignore=(0,), test_frac=0.3,
                    seed=0, max_tiles=None, out_png=None, split="area"):
    """Map the probe split on the tiles' geographic positions: train=red, validation=green.
    split='area' colours whole contiguous training-area blocks together (so blocks are
    uniformly red or green, never interleaved). Model-free and uses the SAME deterministic
    split as the probe. Returns (train_groups, test_groups)."""
    imgs, masks = aligned_pairs(rgb_root, masks_root)
    if max_tiles:
        imgs, masks = imgs[:max_tiles], masks[:max_tiles]
    imgs, masks = _drop_small(imgs, masks, img_size // 16)
    g = img_size // 16
    groups, unit = _split_groups(masks, split)
    y = np.stack([pool_mask(read_mask(m), g) for m in masks]).reshape(-1)
    gid = np.repeat(groups, g * g)
    keep = ~np.isin(y, list(ignore))
    _, _, gtr, gte = _tile_split(gid[keep], test_frac, seed)
    tr_tiles = np.where(np.isin(groups, gtr))[0]                  # colour each TILE by its group's side
    te_tiles = np.where(np.isin(groups, gte))[0]
    centers = _tile_centers(imgs)
    title = (f"probe split by {unit} — {len(gtr)} train / {len(gte)} val {unit}s  "
             f"(train={len(tr_tiles)} red tiles / val={len(te_tiles)} green tiles)")
    if out_png:
        _split_map_png(centers, tr_tiles, te_tiles, Path(out_png), title)
    return gtr, gte


# ── main ─────────────────────────────────────────────────────────────────────────
def cluster_vs_annotation(weights, ckpt, rgb_root, masks_root, n_clusters=12,
                          img_size=512, ignore=(0,), device="cpu", out_png=None, seed=0,
                          k_examples=4, max_tiles=None):
    """Score patch k-means against the CLASSIFY annotations, pretrained vs finetuned.

    img_size=512 -> 32x32 patches, exact 16-px blocks of the 512px tile. ignore drops
    classes (default {0} = background/no-data) from the scoring. max_tiles truncates the
    set (handy for a quick CPU check). Saves a metrics bar PNG + an examples PNG, and
    returns {model: {nmi, ari, purity, n}}.
    """
    imgs, masks = aligned_pairs(rgb_root, masks_root)
    if max_tiles:
        imgs, masks = imgs[:max_tiles], masks[:max_tiles]
    imgs, masks = _drop_small(imgs, masks, img_size // 16)
    print(f"{len(imgs)} annotated tiles | img_size {img_size} | k={n_clusters} | ignore={ignore}")

    (fpre, _), (fft, _), g = pre_ft_features(weights, ckpt, imgs, device, img_size,
                                             EVAL_CACHE, to_tensor=_embed_tensor)
    annot_maps = np.stack([pool_mask(read_mask(m), g) for m in masks])       # (T, g, g)
    annot = annot_maps.reshape(-1)

    res, cluster_maps = {}, {}
    for name, feats in [("pretrained", fpre), ("finetuned", fft)]:
        maps = _kmeans_maps(feats, n_clusters, seed)
        cluster_maps[name] = maps
        res[name] = _scores(annot, maps.reshape(-1), ignore)
        r = res[name]
        print(f"  {name:10s}  NMI {r['nmi']:.3f}   ARI {r['ari']:.3f}   purity {r['purity']:.3f}   (n={r['n']})")

    if out_png:
        p = Path(out_png)
        _scores_bar_png(res, n_clusters, p)
        _examples_png(imgs, annot_maps, cluster_maps["finetuned"], n_clusters, k_examples,
                      seed, p.with_name(f"{p.stem}_examples{p.suffix}"))
    return res


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    CONFIG = {
        "weights": "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "ckpt": "checkpoints/monrovia__r16-K65k-dino+ibot+gram+koleo-sat/final.pt",
        "rgb_root": "input_site_data/monrovia/RGB",       # imagery .tif tiles (same as training)
        "masks_root": "input_site_data/monrovia/masks",   # grid-aligned mask .tif (src/train/align.py)
        "device": "cpu",
    }

    EVAL = Path("outputs/monrovia/report/04_evaluation"); EVAL.mkdir(parents=True, exist_ok=True)
    cfg = (CONFIG["rgb_root"], CONFIG["masks_root"])

    imgs, masks = aligned_pairs(*cfg)                                        # 1) paired aligned tiles
    print(f"{len(imgs)} aligned imagery/mask tiles")
    print("mask classes (tile 0):", np.unique(read_mask(masks[0])))

    # 2) the area-level split, drawn (model-free) — train=red, val=green, control before probing
    train_groups, test_groups = probe_split_map(*cfg, out_png=EVAL / "probe_split_map.png")

    # 3) honest linear probe (no area leaks across the split) vs the leaky per-patch split
    cm_a, classes, acc_area, _ = linear_probe_confusion(                     # area split (honest)
        CONFIG["weights"], CONFIG["ckpt"], *cfg, model="finetuned", split="area",
        device="cuda", out_png=EVAL / "probe_finetuned.png")
    cm_p, _, acc_patch, _ = linear_probe_confusion(                         # patch split (leaky, for contrast)
        CONFIG["weights"], CONFIG["ckpt"], *cfg, model="finetuned", split="patch", device="cuda")
    print(f"finetuned probe accuracy: area-split {acc_area:.3f}  vs  patch-split {acc_patch:.3f} (leaky)")

    res_full = cluster_vs_annotation(                                       # 4) clustering vs annotation (full)
        CONFIG["weights"], CONFIG["ckpt"], *cfg,
        n_clusters=12, device="cuda", out_png=EVAL / "eval_separability.png")
