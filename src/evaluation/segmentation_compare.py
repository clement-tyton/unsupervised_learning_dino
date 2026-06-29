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


# ── DINO head training (on FROZEN cached features — fast) ───────────────────────────
def train_dino_head(feats, targets, tr_idx, K1, device="cpu", epochs=30, lr=1e-3,
                    batch_size=4, seed=0):
    """Train a fresh ConvSegHead on FROZEN features `feats` (T,N,D) -> full-res logits, against
    the remapped `targets` (T,out,out), with CrossEntropyLoss(ignore_index=0). The ViT backbone
    never runs (features are cached), so this is cheap. Times ONLY the optimization loop.
    Returns (head, {train_time_s, loss_curve, final_loss, n_params, n_train})."""
    torch.manual_seed(seed)
    head = ConvSegHead(in_dim=feats.shape[-1], n_classes=K1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=0)
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
                    scalars[f"f1/{nm}"] = v
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
    f1 = np.where(denom > 0, 2 * inter / denom, np.nan)
    return f1, float(np.nanmean(f1)), (float(inter.sum() / gt.sum()) if gt.sum() else 0.0)


def _fmt_per_class(per_f1, names):
    """'Ground 0.81 · Shrub 0.42 · Water —' over the K real classes (names[1:]); NaN -> —."""
    import numpy as np
    return " · ".join(f"{n} {'—' if np.isnan(f) else f'{f:.2f}'}"
                      for n, f in zip(names[1:], per_f1))


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


def eval_head(head, feats, targets, te_idx, K1, device="cpu"):
    """Validate a trained ConvSegHead on the val tiles -> (per_f1, macro_f1, pix_acc)."""
    head.eval()

    def predict(idx):
        return head(feats[idx:idx + 1].to(device)).argmax(1)[0].cpu().numpy()
    return _eval_seg(predict, te_idx, targets, K1)


def eval_unet(model, imgs, targets, te_idx, K1, device="cpu", img_size=512):
    """Validate a trained UNet on the val tiles -> (per_f1, macro_f1, pix_acc)."""
    from src.visualisation.common import _embed_tensor
    model.eval()

    def predict(idx):
        x = _embed_tensor(imgs[idx], img_size)[None].to(device)
        return model(x).argmax(1)[0].cpu().numpy()
    return _eval_seg(predict, te_idx, targets, K1)


# ── shared setup (same tiles, features, targets and area split for EVERY model) ─────
def _prepare(weights, ckpt, rgb_root, masks_root, device, img_size, test_frac, seed):
    """Aligned pairs (+drop small), frozen DINO features (pre/ft), remapped targets, and the
    honest area split — identical for every model so the comparison is apples-to-apples."""
    import numpy as np
    from src.evaluation.separability import (EVAL_CACHE, _area_index, _drop_small, _tile_split,
                                             aligned_pairs)
    from src.visualisation.common import _embed_tensor
    from src.visualisation.features import pre_ft_features
    imgs, masks = aligned_pairs(rgb_root, masks_root)
    imgs, masks = _drop_small(imgs, masks, img_size // 16)
    ids, id2idx, names = class_mapping(masks_root)
    K1 = len(ids) + 1
    (pre, _), (ft, _), g = pre_ft_features(weights, ckpt, imgs, device, img_size,
                                           EVAL_CACHE, _embed_tensor)
    targets = build_targets(masks, id2idx, out_size=img_size)
    area_idx, n_areas = _area_index(masks)
    tr_mask, te_mask, tr_groups, te_groups = _tile_split(area_idx, test_frac, seed)
    tr_idx, te_idx = np.where(tr_mask)[0], np.where(te_mask)[0]
    print(f"{len(imgs)} tiles | train {len(tr_idx)} / val {len(te_idx)} | areas {n_areas} "
          f"({len(tr_groups)} train / {len(te_groups)} val) | K1={K1}")
    return dict(imgs=imgs, masks=masks, targets=targets, tr_idx=tr_idx, te_idx=te_idx,
                ids=ids, id2idx=id2idx, names=names, K1=K1, pre=pre, ft=ft)


# ── UNet end-to-end training (whole net runs; reuses the trained Mega Model weights) ──
def train_unet(imgs, targets, tr_idx, model_config, K1, device="cpu", epochs=30, lr=1e-4,
               batch_size=2, weights_dir="model_weight/unet", seed=0):
    """Fine-tune the UNet end-to-end (RGB 512 -> mask). Reuses the trained Mega Model
    encoder+decoder + a fresh head (NOT frozen) from the config; the WHOLE net runs (no feature
    cache), bf16 autocast on CUDA, CrossEntropyLoss(ignore_index=0). Times the loop. Returns
    (model, metrics). batch_size is small (2) — res2net101 UNet @512 is memory-heavy."""
    from dotenv import load_dotenv
    from tytonai_utils.model import (download_model_weights_from_config,
                                     load_model_with_fresh_head_from_config, read_model_config)
    from src.visualisation.common import _embed_tensor
    load_dotenv(".env", override=True)                          # AWS creds for the weights download
    if "epoch_file_key" not in read_model_config(model_config):
        raise KeyError("model config needs 'epoch_file_key' (s3://...pth) — yours has "
                       "'epoch_v2_file_key'; rename/add it.")
    torch.manual_seed(seed)
    wpath = download_model_weights_from_config(model_config, weights_dir)
    model = load_model_with_fresh_head_from_config(model_config, wpath, num_classes=K1,
                                                   freeze_encoder=False).to(device)
    if hasattr(model, "segmentation_head"):                     # CE needs logits: drop config activation
        model.segmentation_head[-1] = nn.Identity()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    tr_imgs, Ytr = [imgs[i] for i in tr_idx], targets[tr_idx]
    n, amp = len(tr_idx), "cuda" in str(device)
    model.train()
    curve = []
    t0 = time.perf_counter()
    n_batches = (n + batch_size - 1) // batch_size
    bar = tqdm(total=epochs * n_batches, desc="unet train")           # per-batch (UNet is slow)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size].tolist()
            xb = torch.stack([_embed_tensor(tr_imgs[j], 512) for j in b]).to(device)
            yb = Ytr[b].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                loss = ce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
            bar.update(1)
            bar.set_postfix(ep=f"{ep + 1}/{epochs}", loss=f"{total / min(i + batch_size, n):.3f}")
        curve.append(total / n)
    bar.close()
    return model, {"train_time_s": time.perf_counter() - t0, "loss_curve": curve,
                   "final_loss": curve[-1], "n_params": sum(p.numel() for p in model.parameters()),
                   "n_train": n}


def train_dino_heads(weights, ckpt, rgb_root, masks_root, device="cpu", img_size=512,
                     epochs=30, batch_size=4, test_frac=0.3, seed=0, mlflow=False, site=None,
                     experiment=SEG_EXPERIMENT):
    """Train + VALIDATE a ConvSegHead on the PRETRAINED and FINETUNED frozen features, on the
    honest area split. Reports macro-F1 / per-class F1 / pixel-acc per variant; with mlflow=True
    logs each to `experiment` (one run per variant, same server). Returns ({variant: (head, metrics)}, split)."""
    site = site or Path(rgb_root).parent.name                    # e.g. .../monrovia/RGB -> "monrovia"
    d = _prepare(weights, ckpt, rgb_root, masks_root, device, img_size, test_frac, seed)
    res = {}
    for name, feats in [("pretrained", d["pre"]), ("finetuned", d["ft"])]:
        head, m = train_dino_head(feats, d["targets"], d["tr_idx"], d["K1"], device, epochs,
                                  batch_size=batch_size, seed=seed)
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


def compare_segmentation(weights, ckpt, rgb_root, masks_root, model_config, device="cpu",
                         img_size=512, epochs=30, dino_batch=4, unet_batch=2, test_frac=0.3,
                         seed=0, mlflow=False, site=None, experiment=SEG_EXPERIMENT):
    """Train + VALIDATE all THREE models (DINO pretrained head, DINO finetuned head, UNet) on the
    SAME aligned tiles and SAME area split. Reports macro-F1 / per-class F1 / pixel-acc; with
    mlflow=True logs each as a run in the comparison experiment. The UNet is best-effort (needs S3
    weights + 'epoch_file_key'); if it can't build, the DINO results still come through. Returns
    ({model: (net, metrics)}, split)."""
    site = site or Path(rgb_root).parent.name
    d = _prepare(weights, ckpt, rgb_root, masks_root, device, img_size, test_frac, seed)
    res = {}

    def _finish(label, net, m, val, cfg):
        per_f1, m["val_f1"], m["val_pixel_acc"] = val
        m["per_f1"] = dict(zip(d["names"][1:], per_f1.tolist()))
        print(f"  {label:16s} {m['train_time_s']:6.1f}s | val F1 {m['val_f1']:.3f} "
              f"pixAcc {m['val_pixel_acc']:.3f} | {m['n_params']/1e6:.2f}M params")
        print(f"                   per-class F1: {_fmt_per_class(per_f1, d['names'])}")
        if mlflow:
            _log_seg_run(site, f"seg_{label}", m, experiment=experiment, config=cfg)
        res[label] = (net, m)

    for label, feats in [("dino_pretrained", d["pre"]), ("dino_finetuned", d["ft"])]:
        head, m = train_dino_head(feats, d["targets"], d["tr_idx"], d["K1"], device, epochs,
                                  batch_size=dino_batch, seed=seed)
        val = eval_head(head, feats, d["targets"], d["te_idx"], d["K1"], device)
        _finish(label, head, m, val, dict(model=label, epochs=epochs, lr=1e-3,
                                          batch_size=dino_batch, test_frac=test_frac, seed=seed, K1=d["K1"]))

    try:                                                         # UNet — heavier, best-effort
        model, m = train_unet(d["imgs"], d["targets"], d["tr_idx"], model_config, d["K1"],
                              device, epochs, batch_size=unet_batch, seed=seed)
        val = eval_unet(model, d["imgs"], d["targets"], d["te_idx"], d["K1"], device, img_size)
        _finish("unet", model, m, val, dict(model="unet_megamodel_freshhead", epochs=epochs,
                                            lr=1e-4, batch_size=unet_batch, test_frac=test_frac,
                                            seed=seed, K1=d["K1"]))
    except Exception as e:
        print(f"  unet skipped ({type(e).__name__}: {e})")

    return res, dict(tr_idx=d["tr_idx"], te_idx=d["te_idx"], ids=d["ids"],
                     id2idx=d["id2idx"], names=d["names"])


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
