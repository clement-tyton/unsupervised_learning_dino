"""
Pre-training visualizations: explain HOW the DINOv3 self-distillation method works,
on our GeoTIFF tiles. These are mostly model-free (or only need a forward pass) and
are meant to teach the mechanism before/around training.

The story (per image):
  - RandomResizedCrop makes 2 GLOBAL crops (224px, big regions) + N LOCAL crops
    (96px, small zoomed regions), each with flip / color-jitter / grayscale.
  - TEACHER sees only the 2 global crops -> they become the soft targets.
  - STUDENT sees ALL crops (globals + locals).
  - Loss: every student view's output is pulled toward the teacher's output on the
    global crops (cross-view self-distillation). A tiny local patch must "reproduce"
    what the teacher gets from the whole-ish global view.

Helpers:
  - show_crops(paths)        -> where crops come from + what the network sees
  - show_distillation(path)  -> teacher targets vs student views (who matches whom)
  - show_ibot / show_gram    -> the iBOT and Gram loss intuitions
  - plot_suite(tiles_dir)    -> render the whole explanatory suite for a site
"""

from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.visualisation.common import (MEAN, STD, _feature_pca_rgb, _global_tensor,
                                       tile_to_pil)

GREEN, BLUE = "#2ca02c", "#1f77b4"  # teacher/global, student/local
GLOBAL_COLORS = ["#e41a1c", "#ffcc00"]  # global 1 = red, global 2 = yellow


def _aug(size):
    """Shared DINO augmentations after the crop (matches data.py)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD.tolist()),
    ])


def make_views(img, global_size=224, local_size=96,
               global_scale=(0.4, 1.0), local_scale=(0.05, 0.4), n_local=6):
    """Return list of (augmented_tensor, source_box, kind). box = (x, y, w, h) on img."""
    views = []
    g_aug, l_aug = _aug(global_size), _aug(local_size)
    for size, scale, aug, kind, n in [
        (global_size, global_scale, g_aug, "global", 2),
        (local_size, local_scale, l_aug, "local", n_local),
    ]:
        for _ in range(n):
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=(3 / 4, 4 / 3))
            crop = TF.resized_crop(img, i, j, h, w, [size, size])
            views.append((aug(crop), (j, i, w, h), kind))
    return views


def _denorm(t):
    """Normalized CHW tensor -> HWC float [0,1] for display (what the net sees)."""
    return np.clip(t.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


def show_one_example(path, n_local=6, out_png=None):
    """One example, 3 columns:
      row 0: original (2 global boxes in red & yellow) | global crop 1 | global crop 2
      rows 1..: the local crops, 3 per row
    Globals bordered red/yellow, locals blue.
    """
    img = tile_to_pil(path)
    views = make_views(img, n_local=n_local)
    globals_ = [v for v in views if v[2] == "global"][:2]
    locals_ = [v for v in views if v[2] == "local"]

    ncol = 3
    nrow = 1 + (len(locals_) + ncol - 1) // ncol  # 1 (orig+globals) + ceil(n_local/3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    # row 0, col 0: original with the 2 global boxes (red, yellow) + faint local boxes
    ax = axes[0][0]
    ax.imshow(img)
    for li, (_, (x, y, w, h), _) in enumerate(locals_):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=BLUE, linewidth=1.6))
        ax.text(x + 2, y + 12, str(li + 1), color=BLUE, fontsize=8, fontweight="bold")
    for k, (_, (x, y, w, h), _) in enumerate(globals_):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=GLOBAL_COLORS[k], linewidth=3))
    ax.set_title("original + crop regions", fontsize=10)

    # row 0, cols 1-2: the 2 global crops, bordered red / yellow
    for k, (tensor, _, _) in enumerate(globals_):
        ax = axes[0][k + 1]
        ax.axis("on")
        ax.imshow(_denorm(tensor))
        ax.set_title(f"global {k + 1} (224)", fontsize=10, color=GLOBAL_COLORS[k])
        for s in ax.spines.values():
            s.set_edgecolor(GLOBAL_COLORS[k]); s.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    # rows 1..: local crops, 3 per row, blue border
    for idx, (tensor, _, _) in enumerate(locals_):
        r, c = 1 + idx // ncol, idx % ncol
        ax = axes[r][c]
        ax.axis("on")
        ax.imshow(_denorm(tensor))
        ax.set_title(f"local {idx + 1} (96)", fontsize=9, color=BLUE)
        for s in ax.spines.values():
            s.set_edgecolor(BLUE); s.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{_Path(path).stem}  —  red/yellow = GLOBAL (teacher+student), blue = LOCAL (student)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def show_ibot(path, img_size=224, patch=16, mask_ratio=0.3, seed=0, out_png=None):
    """Illustrate iBOT: the SAME crop seen by teacher (full) vs student (masked), on the
    shared (img_size/patch)^2 grid. Grey cells = masked in the student -> the positions
    where the student must reproduce the teacher's per-patch distribution."""
    img = tile_to_pil(path).resize((img_size, img_size))
    g = img_size // patch                      # patches per side (14 @ 224/16)
    N = g * g
    rng = np.random.default_rng(seed)
    masked = np.zeros(N, dtype=bool)
    masked[rng.choice(N, int(N * mask_ratio), replace=False)] = True

    fig, (axt, axs) = plt.subplots(1, 2, figsize=(9, 5))
    for ax, title in [(axt, f"TEACHER: full crop ({N} patches)"),
                      (axs, f"STUDENT: {int(masked.sum())} patches masked")]:
        ax.imshow(img)
        for k in range(1, g):                  # the shared patch grid
            ax.axhline(k * patch, color="w", lw=0.5, alpha=0.5)
            ax.axvline(k * patch, color="w", lw=0.5, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # grey out masked cells on the student panel
    for idx in np.where(masked)[0]:
        r, c = idx // g, idx % g
        axs.add_patch(Rectangle((c * patch, r * patch), patch, patch, color="0.5", alpha=0.95))
    fig.suptitle("iBOT — same crop, same grid: patch i <-> patch i. "
                 "Match teacher's K-dim distribution at the masked (grey) positions.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def show_crops(paths, n_local=6, out_dir=None):
    """One PNG per example. Saves multicrop_<tilename>.png in out_dir. Returns the figs."""
    figs = []
    for path in paths:
        out = (_Path(out_dir) / f"multicrop_{_Path(path).stem}.png") if out_dir else None
        figs.append(show_one_example(path, n_local=n_local, out_png=out))
    return figs


def show_batch(crops, n_global=2, out_png=None):
    """Visualize the ACTUAL collated batch you have in the REPL.

    crops: list of (B,C,H,W) tensors (the loader output, already augmented+normalized).
    Renders one row per IMAGE (the B batch elements) and one column per CROP, so you
    see exactly the '2 images x 6 crops' that go into the model. Globals get a green
    border, locals blue.
    """
    B = crops[0].shape[0]
    ncol = len(crops)
    fig, axes = plt.subplots(B, ncol, figsize=(1.5 * ncol, 1.8 * B), squeeze=False)
    for row in range(B):
        for col, crop in enumerate(crops):
            ax = axes[row][col]
            ax.imshow(_denorm(crop[row].detach().cpu().float()))
            color = GREEN if col < n_global else BLUE
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(2)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title("global" if col < n_global else "local", fontsize=8, color=color)
        axes[row][0].set_ylabel(f"image {row}", fontsize=8)
    fig.suptitle(f"batch: {B} images x {ncol} crops — green=global (teacher+student), blue=local",
                 fontsize=10)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
    return fig


def show_distillation(path, n_local=6, out_png=None):
    """One image: teacher targets (2 globals) on top, all student views below,
    with arrows showing that every student view must match the teacher targets."""
    img = tile_to_pil(path)
    views = make_views(img, n_local=n_local)
    globals_ = [v for v in views if v[2] == "global"]
    students = views  # student sees everything

    fig = plt.figure(figsize=(2 * max(len(students), 3), 5.5))
    gs = fig.add_gridspec(2, len(students), height_ratios=[1, 1], hspace=0.5)

    # top: teacher targets (centered)
    off = (len(students) - len(globals_)) // 2
    for k, (t, _, _) in enumerate(globals_):
        ax = fig.add_subplot(gs[0, off + k])
        ax.imshow(_denorm(t))
        ax.set_title(f"TEACHER target {k + 1}\n(global)", fontsize=8, color=GREEN)
        for s in ax.spines.values():
            s.set_edgecolor(GREEN); s.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])

    # bottom: student views, all pointing up to the targets
    for k, (t, _, kind) in enumerate(students):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(_denorm(t))
        color = GREEN if kind == "global" else BLUE
        ax.set_title("student\n" + ("global" if kind == "global" else "local"),
                     fontsize=8, color=color)
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.annotate("", xy=(0.5, 1.35), xytext=(0.5, 1.02), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    fig.suptitle("What the student must reproduce: every view -> teacher's global targets",
                 fontsize=11)
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def show_gram(student, teacher, path, device="cpu", out_png=None):
    """Visualize the GRAM loss: the (DxD) channel-correlation matrix of the patch
    features, for teacher vs student, and their |difference| (what the MSE penalizes).
    Needs the models (Gram is computed from real patch features, not the raw crop).
    """
    x = _global_tensor(path).to(device)
    student.eval(); teacher.eval()
    with torch.no_grad():
        _, _, ps = student(x)          # (1, N, D)
        _, _, pt = teacher(x)

    def gram(p):
        p = p[0]                        # (N, D)
        return (p.t() @ p / p.shape[0]).cpu().numpy()   # (D, D)

    Gs, Gt = gram(ps), gram(pt)
    diff = np.abs(Gs - Gt)
    mse = float(((Gs - Gt) ** 2).mean())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (G, title) in zip(axes, [(Gt, "teacher Gram (DxD)"),
                                     (Gs, "student Gram (DxD)"),
                                     (diff, f"|teacher - student|   (MSE={mse:.2e})")]):
        im = ax.imshow(G, cmap="viridis")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Gram loss = MSE between teacher & student channel-correlation matrices "
                 "of the patch features (preserves dense structure)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def show_feature_pca(models, path, device="cpu", labels=None, img_size=224, out_png=None):
    """Compare what models 'see': PCA(patch features)->RGB map, one panel per model.

    The iconic DINO dense-feature view. Similar colors = similar features -> coherent
    regions mean the backbone groups the scene semantically. Great for pretrained vs
    random vs finetuned. (PCA sign is arbitrary, so absolute colors differ between
    panels; look at the STRUCTURE/coherence, not the exact hue.)
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    labels = labels or [f"model {i}" for i in range(len(models))]
    img = tile_to_pil(path).resize((img_size, img_size))
    x = _global_tensor(path, img_size).to(device)
    g = img_size // 16

    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.2 * (len(models) + 1), 3.4))
    axes[0].imshow(img); axes[0].set_title("input", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, m, lab in zip(axes[1:], models, labels):
        m.eval()
        with torch.no_grad():
            patches = m(x)[2][0]                        # (N, D)
        ax.imshow(_feature_pca_rgb(patches, g))
        ax.set_title(lab, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("PCA of patch features (top-3 -> RGB): coherent regions = semantic grouping",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def plot_suite(tiles_dir, out_dir=None, n_examples=2, n_local=6, models=None,
               device="cpu", seed=0):
    """Generate the whole explanatory plot suite for a site's tiles, reusable anywhere.

    Saves into out_dir (default: <tiles_dir>/../viz):
      - overview.png                     coarse mosaic of the site
      - multicrop_<tile>.png             CLS/DINO loss: crops + provenance (per example)
      - distill_<tile>.png               who matches whom (per example)
      - ibot_<tile>.png                  masked-patch story (per example)
      - gram_<tile>.png                  Gram channel-correlation (only if models given)
    `models` = (student, teacher) to also render the Gram panels.
    """
    from tytonai_utils.webmap import preview_tiles  # lazy: avoids import coupling at module load
    tiles_dir = _Path(tiles_dir)
    out_dir = _Path(out_dir) if out_dir else tiles_dir.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(tiles_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no .tif in {tiles_dir}")
    rng = np.random.default_rng(seed)
    picks = [files[i] for i in rng.choice(len(files), min(n_examples, len(files)), replace=False)]

    plt.close(preview_tiles(tiles_dir, out_png=out_dir / "overview.png").figure)
    for p in picks:
        plt.close(show_one_example(p, n_local=n_local, out_png=out_dir / f"multicrop_{p.stem}.png"))
        plt.close(show_distillation(str(p), n_local=n_local, out_png=out_dir / f"distill_{p.stem}.png"))
        plt.close(show_ibot(str(p), out_png=out_dir / f"ibot_{p.stem}.png"))
        if models is not None:
            plt.close(show_gram(models[0], models[1], str(p), device=device,
                                out_png=out_dir / f"gram_{p.stem}.png"))
    print(f"saved plot suite for {len(picks)} examples -> {out_dir}")
    return out_dir
