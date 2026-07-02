"""
Segmentation comparison — UNet vs a seg head on frozen DINO (pretrained & finetuned).

Goal (built incrementally): on the grid-aligned annotation tiles, compare which segments best
across THREE pixel-level models, and — if accuracy is close — make the point on training time
(the DINO heads train in seconds on CACHED frozen features; the UNet trains end-to-end):
  1. UNet                       (smp; reuses the TRAINED Mega Model encoder+decoder from the
                                 config's checkpoint + a fresh head, fine-tuned end-to-end)
  2. ConvSegHead on PRETRAINED  DINO features (frozen) + learned upsampler
  3. ConvSegHead on FINETUNED   DINO features (frozen) + learned upsampler

This file is STEP 1 — instantiation only: build every model and shape-check one forward pass.
Nothing trains here; the dataset, training loops, mIoU/IoU, figures and the pipeline step come
later. The fairness lever is that ConvSegHead decodes the 32x32x1024 frozen-feature map up to
512x512xK1 — the SAME output shape as the UNet — so all three are scored at full pixel res.

Shared label space: the K real classes map to contiguous indices 1..K, with index 0 = ignore,
so every model outputs K1 = K+1 channels and later trains with CrossEntropyLoss(ignore_index=0).
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.separability import CLASS_NAMES


# ── shared class space (derived from the DATA we train on, not the config) ──────────
def present_classes(masks_root, ignore=(0,)):
    """The real class ids actually present in the aligned masks under masks_root — i.e. the
    class list we will train on. The model config's `class_list` is NOT used (it can describe a
    different model). Scans every mask .tif (band 1) and returns the sorted non-ignore ids."""
    import numpy as np
    from src.evaluation.separability import read_mask
    seen = set()
    for mp in sorted(Path(masks_root).rglob("*.tif")):
        seen |= set(np.unique(read_mask(mp)).tolist())
    return sorted(seen - set(ignore))


def class_mapping(masks_root=None, ignore=(0,)):
    """(ids, id2idx, names) — the class space shared by all three models, derived from DATA.
    `ids` = the real classes present in the aligned masks under masks_root (the classes we train
    on), or sorted from CLASS_NAMES if no masks_root. `id2idx` maps each sparse id -> a contiguous
    index with index 0 reserved for ignore (labels 1..K, K1 = K+1). `names[i]` is index i's label.
    The same K1 sizes both ConvSegHead and the UNet head so their outputs are comparable.
    """
    if masks_root is not None and Path(masks_root).exists():
        ids = present_classes(masks_root, ignore)
    else:
        ids = sorted(set(CLASS_NAMES) - set(ignore))
    id2idx = {0: 0, **{sid: i + 1 for i, sid in enumerate(ids)}}
    names = ["(ignore)"] + [CLASS_NAMES.get(s, str(s)) for s in ids]
    return ids, id2idx, names


# ── DINO segmentation head: learned upsampler (full-res, fair vs UNet) ───────────────
class ConvSegHead(nn.Module):
    """Frozen-DINO seg head: a small learned decoder mapping a patch-feature map to full res.
    Input  (B, N, D) patch tokens (N = g*g) -> reshape to a (B, D, g, g) feature map ->
    1x1 channel-reduce -> log2(out_size/g) upsampling blocks (Upsample 2x + Conv3x3 + BN + ReLU)
    -> 1x1 classifier -> (B, n_classes, out_size, out_size). Same output shape as the UNet, so
    DINO competes at the pixel level. Trains on CACHED features (the ViT backbone never runs).
    """
    def __init__(self, in_dim=1024, n_classes=9, g=32, out_size=512, widths=(256, 128, 64, 32)):
        super().__init__()
        self.g = g
        n_up = int(round(math.log2(out_size / g)))                  # 32 -> 512 = 4 doublings
        chans = list(widths) + [widths[-1]] * (n_up - (len(widths) - 1))  # pad to n_up+1 entries
        self.proj = nn.Conv2d(in_dim, chans[0], kernel_size=1)
        blocks = []
        for i in range(n_up):
            blocks += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, padding=1),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(chans[n_up], n_classes, kernel_size=1)
    def forward(self, patches):
        """(B, N, D) -> (B, n_classes, out_size, out_size)."""
        B, N, D = patches.shape
        x = patches.transpose(1, 2).reshape(B, D, self.g, self.g)   # tokens -> (B, D, g, g) map
        x = self.decoder(self.proj(x))
        return self.classifier(x)


# The UNet is built straight from the package — no wrapper. Download the trained Mega Model
# weights, then load_model_with_fresh_head_from_config(...num_classes=K1, freeze_encoder=False):
# it reuses the checkpoint's encoder + decoder and gives a fresh head for OUR K1 classes, all
# trainable. Set the config's "activation" to null so the head emits logits for CrossEntropyLoss.


# ── targets: remap masks to the shared 0..K label space, at the model's output resolution ──
def remap_mask(mask, id2idx):
    """(H,W) sparse class ids -> contiguous indices (0 = ignore); ids not in id2idx -> 0."""
    import numpy as np
    out = np.zeros_like(mask, dtype=np.int64)
    for sid, idx in id2idx.items():
        if idx:                                                  # 0 stays 0 (ignore)
            out[mask == sid] = idx
    return out


def build_targets(masks, id2idx, out_size=512):
    """(T, out_size, out_size) long targets: each aligned mask remapped to 0..K, then
    nearest-resized to the model's output grid (matches ConvSegHead / UNet output H,W)."""
    import torch.nn.functional as F
    from src.evaluation.separability import read_mask
    ts = []
    for m in masks:
        r = torch.from_numpy(remap_mask(read_mask(m), id2idx)).float()[None, None]   # (1,1,H,W)
        ts.append(F.interpolate(r, size=(out_size, out_size), mode="nearest")[0, 0].long())
    return torch.stack(ts)                                       # (T, out, out)


# ── class balancing: shared inverse-frequency CE weights (same recipe for every model) ──
def class_weights(targets, tr_idx, K1):
    """Inverse-frequency CrossEntropy class weights from the TRAIN tiles only (no val leakage).
    weight_c ∝ 1 / pixel_count_c over the real classes (mean-normalised ~1); index 0 = ignore -> 0.
    Computed ONCE and passed to every model so all three train with the identical balanced loss."""
    import numpy as np
    cnt = np.bincount(targets[tr_idx].numpy().ravel(), minlength=K1).astype(float)
    w = np.zeros(K1, dtype=np.float32)
    real = cnt[1:]
    nz = real > 0
    w[1:][nz] = real[nz].sum() / (nz.sum() * real[nz])           # inverse freq, averages to ~1
    return torch.from_numpy(w)


def class_pixel_counts(targets, tr_idx, K1):
    """Per-real-class pixel count over the annotated TRAIN tiles (index 0 = ignore, dropped).
    This is the absolute training signal per class behind the macro-F1: a class with very few
    pixels is the one that drags the macro mean down (or comes out NaN in a fold with no val
    pixels). Train only (no val) — it describes what the models actually learned from."""
    import numpy as np
    cnt = np.bincount(targets[tr_idx].numpy().ravel(), minlength=K1).astype(float)
    return cnt[1:]                                                # real classes only (drop ignore)


# ── DINO head training (on FROZEN cached features — fast) ───────────────────────────
def _aug_feats(xb, yb, g):
    """One random dihedral transform (rot90^k + optional H-flip) applied IDENTICALLY to the patch
    tokens (reshaped to their g×g grid, same layout ConvSegHead uses) and to the (B,H,W) label.
    Recovers the UNet's flip augmentation for the frozen head at ~zero cost (pure tensor ops on
    cached features — the backbone never runs). ViT features aren't perfectly flip-equivariant, so
    it acts as regularization rather than an exact symmetry — which is what a 1.5M head on ~90
    tiles needs. Returns (tokens (B,N,D), label (B,H,W))."""
    B, N, D = xb.shape
    x = xb.transpose(1, 2).reshape(B, D, g, g)                  # tokens -> grid (matches ConvSegHead)
    k = int(torch.randint(0, 4, (1,)))
    x, yb = torch.rot90(x, k, (2, 3)), torch.rot90(yb, k, (1, 2))
    if torch.rand(1) < 0.5:
        x, yb = x.flip(3), yb.flip(2)                           # grid width <-> label width
    return x.reshape(B, D, N).transpose(1, 2).contiguous(), yb.contiguous()


def train_dino_head(feats, targets, tr_idx, K1, device="cpu", epochs=30, lr=1e-3,
                    batch_size=4, class_weight=None, seed=0, aug=True):
    """Train a fresh ConvSegHead on FROZEN features `feats` (T,N,D) -> full-res logits, against
    the remapped `targets` (T,out,out), with class-weighted CrossEntropyLoss(ignore_index=0). The
    ViT backbone never runs (features are cached), so this is cheap. `aug` adds feature-space
    dihedral augmentation (the frozen-head analog of the UNet's flips — see _aug_feats), still
    seconds/epoch. Times ONLY the optimization loop. Returns (head, {train_time_s, loss_curve,
    final_loss, n_params, n_train})."""
    torch.manual_seed(seed)
    head = ConvSegHead(in_dim=feats.shape[-1], n_classes=K1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    cw = class_weight.to(device) if class_weight is not None else None
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=0)
    Xtr, Ytr = feats[tr_idx], targets[tr_idx]                    # kept on CPU; batches move to device
    n = len(tr_idx)
    head.train()
    curve = []
    t0 = time.perf_counter()
    bar = tqdm(range(epochs), desc="dino head", leave=False)
    for _ in bar:
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            xb, yb = Xtr[b].to(device), Ytr[b].to(device)
            if aug:
                xb, yb = _aug_feats(xb, yb, head.g)             # feature-space flip/rotate (cheap)
            loss = ce(head(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        curve.append(total / n)
        bar.set_postfix(loss=f"{curve[-1]:.3f}")
    return head, {"train_time_s": time.perf_counter() - t0, "loss_curve": curve,
                  "final_loss": curve[-1], "n_params": sum(p.numel() for p in head.parameters()),
                  "n_train": n}


# Segmentation runs log to their OWN MLflow experiment (the comparison) on the SAME server as
# the LoRA training — one run per model variant, so the MLflow UI lines them up.
SEG_EXPERIMENT = "dino_segmentation_comparison"


def _log_seg_run(site, scenario, metrics, config, experiment=SEG_EXPERIMENT, tracking_uri=None):
    """Log one segmentation model's training to MLflow: per-epoch loss + final/val metrics, into
    `experiment` on the configured server (default the same server as training). Best-effort —
    a tracking-server hiccup is reported but never sinks the training."""
    import mlflow as mlf
    from dotenv import load_dotenv
    from src.train.tracking import TRACKING_URI, DinoRun
    try:
        load_dotenv(".env", override=True)                      # MLFLOW_TRACKING_* creds
        with DinoRun(site, scenario, config=config, experiment=experiment,
                     tracking_uri=tracking_uri or TRACKING_URI) as run:
            mlf.set_tags({"task": "segmentation", "model": config.get("model", scenario)})
            for ep, lo in enumerate(metrics["loss_curve"]):
                run.log({"train_loss": lo}, step=ep)
            scalars = {"train_time_s": metrics["train_time_s"], "final_loss": metrics["final_loss"],
                       "params_M": metrics["n_params"] / 1e6}
            for k in ("val_f1", "val_pixel_acc"):               # validation metrics if present
                if k in metrics:
                    scalars[k] = metrics[k]
            for nm, v in metrics.get("per_f1", {}).items():     # per-class F1 (skip NaN classes)
                if not (isinstance(v, float) and math.isnan(v)):
                    scalars[f"f1/{nm.replace(' ', '_')}"] = v    # MLflow-safe metric key
            for nm, v in metrics.get("train_px_M", {}).items():  # per-class train pixels (millions)
                scalars[f"train_px_M/{nm.replace(' ', '_')}"] = v
            run.log(scalars)
        print(f"    logged -> mlflow[{experiment}] {site}__{scenario}")
    except Exception as e:
        print(f"    mlflow logging skipped ({type(e).__name__}: {e})")


# ── validation: per-class F1 / macro-F1 / pixel-accuracy on the held-out tiles ──────
def seg_scores(cm, K1):
    """From a (K1,K1) confusion (rows=true, cols=pred, index 0=ignore): (per_class_F1[1..K],
    macro_F1, pixel_acc). F1_c = 2*TP / (2*TP + FP + FN) = 2*inter / (gt + pr). Ignore (0) is
    excluded; classes absent from val (gt+pr==0) get F1=NaN, dropped from the macro mean."""
    import numpy as np
    inter = np.diag(cm)[1:].astype(float)                       # TP per real class
    gt = cm[1:, :].sum(1).astype(float)                         # TP+FN (all GT pixels per class)
    pr = cm[:, 1:].sum(0).astype(float)                         # TP+FP (all pixels predicted as class)
    denom = gt + pr
    f1 = np.full(inter.shape, np.nan)                            # classes absent from val stay NaN
    np.divide(2 * inter, denom, out=f1, where=denom > 0)         # divide only where denom>0 (no 0/0 warning)
    return f1, float(np.nanmean(f1)), (float(inter.sum() / gt.sum()) if gt.sum() else 0.0)


def _fmt_per_class(per_f1, names):
    """'Ground 0.81 · Shrub 0.42 · Water —' over the K real classes (names[1:]); NaN -> —."""
    import numpy as np
    return " · ".join(f"{n} {'—' if np.isnan(f) else f'{f:.2f}'}"
                      for n, f in zip(names[1:], per_f1))


def _fmt_train_balance(counts, names):
    """'Ground 12.34M (62.4%) · Sedge 0.03M (0.3%)' — per-class train pixels in millions (2 dp)
    with the share in %, over the K real classes (names[1:])."""
    tot = sum(counts) or 1.0
    return " · ".join(f"{n} {c / 1e6:.2f}M ({100 * c / tot:.1f}%)"
                      for n, c in zip(names[1:], counts))


@torch.no_grad()
def _eval_seg(predict, te_idx, targets, K1):
    """Accumulate the (K1,K1) confusion over the val tiles and score it. predict(idx) -> (H,W)
    predicted class ids; scored against targets[idx] where GT > 0 (ignore dropped)."""
    import numpy as np
    cm = np.zeros((K1, K1), dtype=np.int64)
    for idx in te_idx:
        pred = predict(int(idx))
        true = targets[idx].numpy()
        keep = true > 0
        cm += np.bincount(true[keep] * K1 + pred[keep], minlength=K1 * K1).reshape(K1, K1)
    return seg_scores(cm, K1)


@torch.no_grad()
def _tta_logits(head, x):
    """Test-time augmentation: mean logits over the 4 flips (id, H, V, HV). Flip the feature grid,
    run the head, flip the logits back to the original frame, average. Head-only (no backbone), so
    it's nearly free and gives a small, consistent macro-F1 bump. x is (1,N,D)."""
    B, N, D = x.shape
    g = head.g
    xm = x.transpose(1, 2).reshape(B, D, g, g)                  # tokens -> grid
    acc = None
    for dims in ((), (3,), (2,), (2, 3)):                       # grid dims 2/3 == output H/W
        xf = torch.flip(xm, dims) if dims else xm
        lf = head(xf.reshape(B, D, N).transpose(1, 2).contiguous())      # (1,K1,512,512), flipped frame
        lf = torch.flip(lf, dims) if dims else lf               # un-flip back to the original frame
        acc = lf if acc is None else acc + lf
    return acc


def eval_head(head, feats, targets, te_idx, K1, device="cpu", tta=True):
    """Validate a trained ConvSegHead on the val tiles -> (per_f1, macro_f1, pix_acc). `tta`
    flip-averages the logits (see _tta_logits) — cheap, no backbone, small consistent gain."""
    head.eval()

    def predict(idx):
        x = feats[idx:idx + 1].to(device)
        logits = _tta_logits(head, x) if tta else head(x)
        return logits.argmax(1)[0].cpu().numpy()
    return _eval_seg(predict, te_idx, targets, K1)


def eval_unet(model, imgs, targets, te_idx, K1, model_config, device="cpu", img_size=512):
    """Validate a trained UNet on the val tiles -> (per_f1, macro_f1, pix_acc). Uses the SAME
    preprocessing as train_unet (_rgb_raw + site-stat Normalize, no augmentation) so the train and
    eval input distributions can't diverge — a mismatch there silently tanks val F1."""
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2
    from tytonai_utils.model import read_model_config
    cfg = read_model_config(model_config)["config"]
    norm = albu.Compose([albu.Normalize(mean=cfg["train_mean"], std=cfg["train_std"],
                                        max_pixel_value=1.0), ToTensorV2()])   # identical to training
    model.eval()

    def predict(idx):
        x = norm(image=_rgb_raw(imgs[idx], img_size))["image"][None].to(device)
        return model(x).argmax(1)[0].cpu().numpy()
    return _eval_seg(predict, te_idx, targets, K1)


# ── whole-site inference: run a TRAINED model over EVERY tile (not just val) ─────────
@torch.no_grad()
def predict_dino_site(head, weights, ckpt, tiles, device="cpu", img_size=512, tta=True):
    """Per-tile (out,out) predicted class-index maps from the trained FINETUNED ConvSegHead over an
    arbitrary tile list (the whole site, not just the val split). Same prediction path as eval_head
    (optional flip-TTA), on the finetuned cached features. Returns a list of (out,out) int arrays
    aligned 1:1 with `tiles`, in the contiguous class space (0 = ignore, 1..K = class_mapping)."""
    _, ft = _dino_features(weights, ckpt, tiles, device, img_size)   # finetuned feats (pre dropped)
    head.eval()
    preds = []
    for i in range(len(tiles)):
        x = ft[i:i + 1].to(device)
        logits = _tta_logits(head, x) if tta else head(x)
        preds.append(logits.argmax(1)[0].cpu().numpy())
    return preds


@torch.no_grad()
def predict_unet_site(model, model_config, tiles, device="cpu", img_size=512):
    """Per-tile (img_size,img_size) predicted class-index maps from the trained UNet over an arbitrary
    tile list. SAME preprocessing as eval_unet (_rgb_raw + site-stat Normalize) so inference matches
    training. Returns a list aligned 1:1 with `tiles`, in the contiguous class space."""
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2
    from tytonai_utils.model import read_model_config
    cfg = read_model_config(model_config)["config"]
    norm = albu.Compose([albu.Normalize(mean=cfg["train_mean"], std=cfg["train_std"],
                                        max_pixel_value=1.0), ToTensorV2()])   # identical to eval_unet
    model.eval()
    preds = []
    for t in tiles:
        x = norm(image=_rgb_raw(t, img_size))["image"][None].to(device)
        preds.append(model(x).argmax(1)[0].cpu().numpy())
    return preds


# ── single-model trainers (the maps step's retrain-if-absent path; no ablation ladder) ──
def finetuned_head(weights, ckpt, rgb_root, masks_root, device="cpu", img_size=512,
                   epochs=50, batch_size=4, test_frac=0.3, seed=0):
    """Train JUST the finetuned-DINO ConvSegHead on the aligned split (no ablation ladder, no UNet).
    For run_classification_maps when a trained head wasn't handed in. Returns (head, names)."""
    d = _prepare_data(rgb_root, masks_root, img_size, test_frac, seed)
    _, ft = _dino_features(weights, ckpt, d["imgs"], device, img_size)
    cw = class_weights(d["targets"], d["tr_idx"], d["K1"])
    head, _ = train_dino_head(ft, d["targets"], d["tr_idx"], d["K1"], device, epochs,
                              batch_size=batch_size, class_weight=cw, seed=seed, aug=True)
    return head, d["names"]


def baseline_unet(rgb_root, masks_root, model_config, device="cpu", img_size=512,
                  unet_epochs=5, unet_batch=16, test_frac=0.3, seed=0):
    """Train JUST the UNet baseline on the aligned split (for run_classification_maps when a trained
    UNet wasn't handed in). Raises if the S3 weights / 'epoch_file_key' are unavailable — the caller
    decides whether to skip. Returns the trained model."""
    d = _prepare_data(rgb_root, masks_root, img_size, test_frac, seed)
    cw = class_weights(d["targets"], d["tr_idx"], d["K1"])
    model, _ = train_unet(d["imgs"], d["targets"], d["tr_idx"], model_config, d["K1"], device,
                          unet_epochs, batch_size=unet_batch, class_weight=cw, seed=seed)
    return model


# ── shared setup (same tiles, features, targets and area split for EVERY model) ─────
def _prepare_data(rgb_root, masks_root, img_size, test_frac, seed):
    """Model-AGNOSTIC base shared by every segmentation run: aligned image/mask pairs (small ones
    dropped), the data-driven class space, the remapped targets, and the honest area-level split.
    No model, no DINO, no UNet — both the DINO heads and the UNet baseline build on this, so they
    score the SAME tiles on the SAME split. Safe to reuse on its own for any new seg model."""
    import numpy as np
    from src.evaluation.separability import (_area_index, _drop_small, _tile_split, aligned_pairs)
    imgs, masks = aligned_pairs(rgb_root, masks_root)
    imgs, masks = _drop_small(imgs, masks, img_size // 16)
    ids, id2idx, names = class_mapping(masks_root)
    K1 = len(ids) + 1
    targets = build_targets(masks, id2idx, out_size=img_size)
    area_idx, n_areas = _area_index(masks)
    tr_mask, te_mask, tr_groups, te_groups = _tile_split(area_idx, test_frac, seed)
    tr_idx, te_idx = np.where(tr_mask)[0], np.where(te_mask)[0]
    print(f"{len(imgs)} tiles | train {len(tr_idx)} / val {len(te_idx)} | areas {n_areas} "
          f"({len(tr_groups)} train / {len(te_groups)} val) | K1={K1}")
    return dict(imgs=imgs, masks=masks, targets=targets, tr_idx=tr_idx, te_idx=te_idx,
                ids=ids, id2idx=id2idx, names=names, K1=K1)


def _dino_features(weights, ckpt, imgs, device, img_size):
    """DINO-ONLY: the frozen PRETRAINED and FINETUNED patch features for the ConvSegHeads (cached
    in EVAL_CACHE). Kept apart from _prepare_data so the UNet baseline never triggers a backbone
    pass. Returns (pre, ft)."""
    from src.evaluation.separability import EVAL_CACHE
    from src.visualisation.common import _embed_tensor
    from src.visualisation.features import pre_ft_features
    (pre, _), (ft, _), _ = pre_ft_features(weights, ckpt, imgs, device, img_size,
                                           EVAL_CACHE, _embed_tensor)
    return pre, ft


# ── UNet end-to-end training (whole net runs; reuses the trained Mega Model weights) ──
def _balance_pool(targets, tr_idx, K1, seed=112, upsample_power=0.2, data_control=0.2):
    """data_balance: oversample the TRAIN tiles by class pixel-counts -> a multiset of TILE indices
    (rare-class tiles repeated). Mirrors DataBalance.start()'s target computation. Returns the
    balanced index array (values are indices into imgs/targets, with repetition)."""
    import math
    import numpy as np
    from data_balance.sampling import calculate_sample, select_tiles
    np.random.seed(seed)
    Y = targets[tr_idx].numpy()                                  # (n_tr, H, W) remapped labels
    tile_arr = np.stack([np.bincount(y.ravel(), minlength=K1)[1:] for y in Y]).astype(np.float64)  # drop ignore
    lifeform, total = tile_arr.sum(0), tile_arr.sum()
    prob = np.where(lifeform > 0, (lifeform / total) ** upsample_power, 0.0)
    target = np.array([math.ceil(data_control * total * 3 * p) for p in prob], dtype=np.int32)
    samp = calculate_sample(len(tile_arr), tile_arr, target, lifeform)
    bal = select_tiles(tile_arr, samp, target, lifeform, max(2 * len(tile_arr), 500_000))
    return np.asarray(tr_idx)[np.asarray(bal, dtype=np.int64)]


def _rgb512(path, size=512):
    """(size,size,3) uint8 RGB (per-band stretched, resized) — input to albumentations."""
    import numpy as np
    from PIL import Image
    from src.evaluation.separability import _tif_rgb
    return np.array(Image.fromarray(_tif_rgb(path)).resize((size, size)))


def _rgb_raw(path, size=512):
    """(size,size,3) uint8 RGB read straight from bands [1,2,3] with NO per-band stretch, then
    resized. Pixels stay on the native 0–255 scale the prod model's train_mean/train_std were
    computed on — so (px - train_mean)/train_std reproduces the production normalization."""
    import numpy as np
    import rasterio
    from PIL import Image
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).transpose(1, 2, 0).astype("uint8")   # (H,W,3), native 0–255
    return np.array(Image.fromarray(arr).resize((size, size)))


def train_unet(imgs, targets, tr_idx, model_config, K1, device="cpu", epochs=30, lr=1e-3,
               batch_size=16, grad_accum=16, samples_per_epoch=6000, crop=None, balance=False,
               class_weight=None, weights_dir="model_weight/unet", seed=0):
    """UNet fine-tune: trained Mega Model encoder+decoder + fresh head (NOT frozen). By DEFAULT it
    now trains on FULL 512 tiles (crop=None) with NO data_balance (balance=False) — same data the
    DINO heads see, for a clean comparison. Augmentation = H/V flips (+ optional RandomCrop if
    `crop` is set); micro-batch `batch_size` with grad accumulation `grad_accum` (effective batch =
    batch*accum = 16*16 = 256); `samples_per_epoch` random draws/epoch; bf16 autocast; class-weighted
    CE(ignore_index=0) — SAME loss/eval as the DINO heads. NOTE: full 512 is memory-heavy (~10 GB at
    batch 16); drop to batch 8 / grad_accum 32 if you OOM, or set crop=256/balance=True for the
    prod-like recipe. Returns (model, metrics)."""
    import albumentations as albu
    import numpy as np
    from albumentations.pytorch import ToTensorV2
    from dotenv import load_dotenv
    from tytonai_utils.model import (download_model_weights_from_config,
                                     load_model_with_fresh_head_from_config, read_model_config)
    load_dotenv(".env", override=True)                          # AWS creds for the weights download
    cfg = read_model_config(model_config)
    if "epoch_file_key" not in cfg:
        raise KeyError("model config needs 'epoch_file_key' (s3://...pth) — rename your "
                       "'epoch_v2_file_key' to 'epoch_file_key'.")
    mean, std = cfg["config"]["train_mean"], cfg["config"]["train_std"]   # prod normalization stats
    torch.manual_seed(seed)
    wpath = download_model_weights_from_config(model_config, weights_dir)
    model = load_model_with_fresh_head_from_config(model_config, wpath, num_classes=K1,
                                                   freeze_encoder=False).to(device)
    if hasattr(model, "segmentation_head"):                     # CE needs logits: drop config activation
        model.segmentation_head[-1] = nn.Identity()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    cw = class_weight.to(device) if class_weight is not None else None
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=0)

    pool = _balance_pool(targets, tr_idx, K1, seed) if balance else np.asarray(tr_idx)
    rgb = {int(j): _rgb_raw(imgs[j]) for j in np.unique(pool)}   # cache raw (512,512,3) RGB per used tile
    tf = [albu.RandomCrop(crop, crop)] if crop else []          # crop=None -> full 512 tile (parity w/ DINO heads)
    tf += [albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5),
           albu.Normalize(mean=mean, std=std, max_pixel_value=1.0),   # prod site stats — MUST match eval_unet
           ToTensorV2()]
    aug = albu.Compose(tf)
    amp = "cuda" in str(device)
    steps = samples_per_epoch // batch_size
    model.train(); curve = []; t0 = time.perf_counter()
    bar = tqdm(total=epochs * steps, desc="unet train")
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        draw = rng.choice(pool, size=samples_per_epoch, replace=True)   # 6000 balanced draws / epoch
        total = 0.0; opt.zero_grad()
        for s in range(steps):
            sel = draw[s * batch_size:(s + 1) * batch_size]
            au = [aug(image=rgb[int(j)], mask=targets[j].numpy()) for j in sel]
            xb = torch.stack([a["image"] for a in au]).to(device)
            yb = torch.stack([a["mask"] for a in au]).long().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                loss = ce(model(xb), yb) / grad_accum            # scale for accumulation
            loss.backward()
            total += loss.item() * grad_accum
            if (s + 1) % grad_accum == 0:                        # optimizer step every `grad_accum`
                opt.step(); opt.zero_grad()
            bar.update(1); bar.set_postfix(ep=f"{ep + 1}/{epochs}", loss=f"{total / (s + 1):.3f}")
        curve.append(total / steps)
    bar.close()
    return model, {"train_time_s": time.perf_counter() - t0, "loss_curve": curve,
                   "final_loss": curve[-1], "n_params": sum(p.numel() for p in model.parameters()),
                   "n_train": len(tr_idx), "n_pool": len(pool)}


def train_dino_heads(weights, ckpt, rgb_root, masks_root, device="cpu", img_size=512,
                     epochs=30, batch_size=4, test_frac=0.3, seed=0, mlflow=False, site=None,
                     experiment=SEG_EXPERIMENT):
    """Train + VALIDATE a ConvSegHead on the PRETRAINED and FINETUNED frozen features, on the
    honest area split. Reports macro-F1 / per-class F1 / pixel-acc per variant; with mlflow=True
    logs each to `experiment` (one run per variant, same server). Returns ({variant: (head, metrics)}, split)."""
    site = site or Path(rgb_root).parent.name                    # e.g. .../monrovia/RGB -> "monrovia"
    d = _prepare_data(rgb_root, masks_root, img_size, test_frac, seed)
    pre, ft = _dino_features(weights, ckpt, d["imgs"], device, img_size)
    cw = class_weights(d["targets"], d["tr_idx"], d["K1"])       # one balanced loss for both heads
    res = {}
    for name, feats in [("pretrained", pre), ("finetuned", ft)]:
        head, m = train_dino_head(feats, d["targets"], d["tr_idx"], d["K1"], device, epochs,
                                  batch_size=batch_size, class_weight=cw, seed=seed)
        per_f1, m["val_f1"], m["val_pixel_acc"] = eval_head(head, feats, d["targets"], d["te_idx"],
                                                            d["K1"], device)
        m["per_f1"] = dict(zip(d["names"][1:], per_f1.tolist()))
        print(f"  {name:10s} {epochs} ep in {m['train_time_s']:5.1f}s | loss {m['loss_curve'][0]:.3f}"
              f" -> {m['final_loss']:.3f} | val F1 {m['val_f1']:.3f} pixAcc {m['val_pixel_acc']:.3f}"
              f" | head {m['n_params']/1e6:.2f}M")
        print(f"             per-class F1: {_fmt_per_class(per_f1, d['names'])}")
        if mlflow:
            _log_seg_run(site, f"seg_dino_{name}", m, experiment=experiment,
                         config=dict(model=f"dino_{name}_convseghead", epochs=epochs, lr=1e-3,
                                     batch_size=batch_size, test_frac=test_frac, seed=seed, K1=d["K1"]))
        res[name] = (head, m)
    return res, dict(tr_idx=d["tr_idx"], te_idx=d["te_idx"], ids=d["ids"],
                     id2idx=d["id2idx"], names=d["names"])


def _split_info(d):
    """The shared class-space + area-split descriptor every comparison run returns."""
    return dict(tr_idx=d["tr_idx"], te_idx=d["te_idx"], ids=d["ids"],
                id2idx=d["id2idx"], names=d["names"])


def _finish_seg(res, names, site, mlflow, experiment, label, net, m, val, cfg, train_px_M=None):
    """Attach validation scores to `m`, print the per-class summary, optionally log to MLflow,
    and stash (net, m) under `label` in `res`. Shared by both comparison halves. `train_px_M` (the
    per-class train pixels in millions, same for every model) is stashed on `m` so MLflow logs it."""
    per_f1, m["val_f1"], m["val_pixel_acc"] = val
    m["per_f1"] = dict(zip(names[1:], per_f1.tolist()))
    if train_px_M is not None:
        m["train_px_M"] = train_px_M
    print(f"  {label:16s} {m['train_time_s']:6.1f}s | val F1 {m['val_f1']:.3f} "
          f"pixAcc {m['val_pixel_acc']:.3f} | {m['n_params']/1e6:.2f}M params")
    print(f"                   per-class F1: {_fmt_per_class(per_f1, names)}")
    if mlflow:
        _log_seg_run(site, f"seg_{label}", m, experiment=experiment, config=cfg)
    res[label] = (net, m)


def compare_dino_heads(weights, ckpt, rgb_root, masks_root, device="cpu", img_size=512,
                       epochs=50, dino_batch=4, test_frac=0.3, seed=0, mlflow=False,
                       site=None, experiment=SEG_EXPERIMENT):
    """Train + VALIDATE an ABLATION LADDER of frozen-DINO ConvSegHeads on the aligned tiles + area
    split: the 2 existing baselines (pretrained, finetuned) + 4 cheap upgrades — finetuned+aug,
    finetuned+aug+TTA, the SAT493M⊕finetuned bundle, and bundle+aug+TTA. All run on cached features
    (the backbone never runs), so every head still trains in seconds. Reports macro-F1 / per-class
    F1 / pixel-acc; with mlflow=True logs one run per head so the incremental gains line up."""
    site = site or Path(rgb_root).parent.name
    d = _prepare_data(rgb_root, masks_root, img_size, test_frac, seed)
    pre, ft = _dino_features(weights, ckpt, d["imgs"], device, img_size)
    cw = class_weights(d["targets"], d["tr_idx"], d["K1"])       # balanced loss shared by all heads
    counts = class_pixel_counts(d["targets"], d["tr_idx"], d["K1"])
    px_M = dict(zip(d["names"][1:], (counts / 1e6).tolist()))    # {class: train pixels in millions}
    print(f"  train pixels: {_fmt_train_balance(counts, d['names'])}")   # context for the macro-F1
    # SAT493M-pretrained ⊕ finetuned features per patch (2048-d) — both already cached, so the
    # bundle is nearly free (no backbone pass); the head just picks the best of each.
    combo = torch.cat([pre, ft], dim=-1)
    # Ablation ladder: the 2 existing baselines, then each cheap improvement isolated so its F1
    # gain is readable in MLflow — (label, features, aug, tta). All train on cached features.
    variants = [
        ("dino_pretrained",        pre,   False, False),   # exists: SAT493M features, no tricks
        ("dino_finetuned",         ft,    False, False),   # exists: finetuned features, no tricks
        ("dino_finetuned_aug",     ft,    True,  False),   # + feature-space augmentation
        ("dino_finetuned_aug_tta", ft,    True,  True),    # + augmentation + flip-TTA
        ("dino_pre+ft",            combo, False, False),   # SAT493M ⊕ finetuned bundle, no tricks
        ("dino_pre+ft_aug_tta",    combo, True,  True),    # bundle + augmentation + flip-TTA
    ]
    res = {}
    for label, feats, aug, tta in variants:
        head, m = train_dino_head(feats, d["targets"], d["tr_idx"], d["K1"], device, epochs,
                                  batch_size=dino_batch, class_weight=cw, seed=seed, aug=aug)
        val = eval_head(head, feats, d["targets"], d["te_idx"], d["K1"], device, tta=tta)
        _finish_seg(res, d["names"], site, mlflow, experiment, label, head, m, val,
                    dict(model=label, epochs=epochs, lr=1e-3, batch_size=dino_batch,
                         test_frac=test_frac, seed=seed, K1=d["K1"], aug=aug, tta=tta),
                    train_px_M=px_M)
    return res, _split_info(d)


def compare_unet(rgb_root, masks_root, model_config, device="cpu", img_size=512,
                 unet_epochs=5, unet_batch=16, test_frac=0.3, seed=0, mlflow=False,
                 site=None, experiment=SEG_EXPERIMENT):
    """Train + VALIDATE the prod-like UNet BASELINE (trained Mega Model body + fresh head) on the
    aligned tiles + area split. Fully self-contained — NO DINO weights/features, so this chain
    stands alone and can be reused for other UNet work. Needs the S3 weights + 'epoch_file_key' in
    `model_config`. Same data/split/loss as compare_dino_heads (deterministic), so it stays
    apples-to-apples. Reports macro-F1 / per-class F1 / pixel-acc; logs one run if mlflow=True."""
    site = site or Path(rgb_root).parent.name
    
    # rgb_root, masks_root, model_config = paths["rgb"], paths["masks"], paths["model_config"]
    # site  = paths["site"]
    
    d = _prepare_data(rgb_root, masks_root, img_size, test_frac, seed)
    cw = class_weights(d["targets"], d["tr_idx"], d["K1"])
    counts = class_pixel_counts(d["targets"], d["tr_idx"], d["K1"])
    px_M = dict(zip(d["names"][1:], (counts / 1e6).tolist()))    # {class: train pixels in millions}
    print(f"  train pixels: {_fmt_train_balance(counts, d['names'])}")   # context for the macro-F1
    res = {}
    model, m = train_unet(d["imgs"], d["targets"], d["tr_idx"], model_config, d["K1"],
                          device, unet_epochs, batch_size=unet_batch, class_weight=cw, seed=seed)
    val = eval_unet(model, d["imgs"], d["targets"], d["te_idx"], d["K1"], model_config, device, img_size)
    _finish_seg(res, d["names"], site, mlflow, experiment, "unet", model, m, val,
                dict(model="unet_megamodel_freshhead", epochs=unet_epochs, lr=1e-3,
                     batch_size=unet_batch, test_frac=test_frac, seed=seed, K1=d["K1"]), train_px_M=px_M)
    return res, _split_info(d)


def compare_segmentation(weights, ckpt, rgb_root, masks_root, model_config, device="cpu",
                         img_size=512, epochs=50, unet_epochs=5, dino_batch=4, unet_batch=16,
                         test_frac=0.3, seed=0, mlflow=False, site=None, experiment=SEG_EXPERIMENT):
    """Convenience wrapper: run BOTH halves (the two DINO heads, then the UNet) on the same split.
    The UNet is best-effort (skipped if its S3 weights / 'epoch_file_key' are unavailable). Prefer
    calling compare_dino_heads / compare_unet directly when you want them independently."""
    res, split = compare_dino_heads(weights, ckpt, rgb_root, masks_root, device, img_size,
                                    epochs, dino_batch, test_frac, seed, mlflow, site, experiment)
    try:                                                         # UNet — heavier, best-effort
        ures, _ = compare_unet(rgb_root, masks_root, model_config, device, img_size,
                               unet_epochs, unet_batch, test_frac, seed, mlflow, site, experiment)
        res.update(ures)
    except Exception as e:
        print(f"  unet skipped ({type(e).__name__}: {e})")
    return res, split


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — verify every model instantiates and forward-passes to the right shape.
#  Step through one line at a time (Shift+Enter). Nothing trains.
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    CONFIG = {
        "model_config": "input_site_data/monrovia/unet_config.json",  # smp arch (UNet skipped if None/missing)
        "masks_root": "input_site_data/monrovia/masks/study_area",               # classes derived from THESE masks
        "weights": "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "ckpt": "checkpoints/monrovia__r16-K65k-dino+ibot+gram+koleo-sat/final.pt",
        "tiles_dir": "input_site_data/monrovia/RGB/study_area",
        "device": "cuda:1",
    }
    
    
    # 1) shared class space — derived from the aligned masks (NOT the config) ----------
    ids, id2idx, names = class_mapping(CONFIG["masks_root"])
    K1 = len(ids) + 1
    print(f"classes (from data): {ids}  | K={len(ids)}  K1={K1}\nnames: {names}")

    # 2) DINO head (ConvSegHead) — no config needed ------------------------------------
    conv = ConvSegHead(in_dim=1024, n_classes=K1)
    out = conv(torch.randn(2, 1024, 1024))                          # (B=2, N=32*32, D=1024)
    assert out.shape == (2, K1, 512, 512), out.shape
    print(f"ConvSegHead: randn(2,1024,1024) -> {tuple(out.shape)}  "
          f"| params {sum(p.numel() for p in conv.parameters())/1e6:.2f}M")

    # 3) UNet — trained Mega Model encoder+decoder + fresh head (no wrapper, NO freeze) -
    if CONFIG["model_config"] and Path(CONFIG["model_config"]).exists():
        from dotenv import load_dotenv
        from tytonai_utils.model import (download_model_weights_from_config,
                                         load_model_with_fresh_head_from_config, read_model_config)

        from src.train.dino import count_params
        load_dotenv(".env", override=True)                          # AWS creds for the weights download
        if "epoch_file_key" not in read_model_config(CONFIG["model_config"]):
            print("UNet: config needs 'epoch_file_key' (s3://...pth) to fetch the trained weights "
                  "— yours has 'epoch_v2_file_key'. Rename/add it. Skipping UNet.")
        else:
            wpath = download_model_weights_from_config(CONFIG["model_config"], "model_weight/unet")
            unet = load_model_with_fresh_head_from_config(CONFIG["model_config"], wpath,
                                                          num_classes=K1, freeze_encoder=False)
            uout = unet(torch.randn(1, 3, 512, 512))
            assert uout.shape == (1, K1, 512, 512), uout.shape
            tr, tot = count_params(unet)
            print(f"UNet (trained enc+dec, fresh head, NOT frozen): randn(1,3,512,512) -> "
                  f"{tuple(uout.shape)} (matches ConvSegHead) | {tr:.1f}M trainable / {tot:.1f}M")
    else:
        print("UNet: skipped (no model_config — provide the smp config JSON to test it)")

    # 4) optional real-feature sanity (cached pretrained & finetuned DINO features) -----
    try:
        from src.evaluation.separability import EVAL_CACHE
        from src.visualisation.common import _embed_tensor, pick_full_tiles
        from src.visualisation.features import pre_ft_features
        tiles = pick_full_tiles(CONFIG["tiles_dir"], n=1, seed=0)
        (pre, _), (ft, _), g = pre_ft_features(CONFIG["weights"], CONFIG["ckpt"], tiles,
                                               CONFIG["device"], 512, EVAL_CACHE, _embed_tensor)
        for name, feats in [("pretrained", pre), ("finetuned", ft)]:
            o = conv(feats)                                         # (1, N, D) -> (1, K1, 512, 512)
            assert o.shape == (1, K1, 512, 512), o.shape
            print(f"ConvSegHead on real {name} features {tuple(feats.shape)} -> {tuple(o.shape)}")
    except Exception as e:
        print(f"real-feature sanity skipped ({type(e).__name__}: {e})")

    print("all models instantiate + forward OK")

    # 5) train the DINO heads on the cached frozen features (fast — backbone never runs) -
    #    pretrained vs finetuned, on the honest area-split train tiles. Eval/mIoU comes next.
    res, split = train_dino_heads(CONFIG["weights"], CONFIG["ckpt"],
                                  "input_site_data/monrovia/RGB", "input_site_data/monrovia/masks",
                                  device=CONFIG["device"], epochs=30, mlflow = True)
