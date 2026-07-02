"""
End-to-end per-site pipeline: download -> train -> analyse -> evaluate.

Everything is parameterised by the SITE directory, so a new site is just a new CONFIG
block (+ its site_config.json and dataset.json). Each step is a small function that
takes explicit paths and returns its output; the RUN block at the bottom wires them so
you can step through one line at a time (Shift+Enter), cheap -> expensive.

A site directory holds:
  input_site_data/<site>/site_config.json   webmap + .fgb areas + resolution (for .tif tiles)
  input_site_data/<site>/<area>.fgb         study-area polygons
  input_site_data/<site>/dataset.json       npz imagery+mask manifest (for the annotated eval)
Outputs (pictures, checkpoints) go to outputs/<site>/ and checkpoints/<site>/.
"""

import json
from pathlib import Path

WEIGHTS = "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


def site_paths(site_dir) -> dict:
    """All the paths the pipeline needs, derived from one site directory."""
    site_dir = Path(site_dir)
    return {
        "site": site_dir,
        "name": site_dir.name,
        "site_config": site_dir / "site_config.json",
        "dataset": site_dir / "dataset.json",
        "rgb": site_dir / "RGB",
        "annotations": site_dir / "annotations",
        "masks": site_dir / "masks",                       # npz masks realigned onto the imagery grid
        "model_config": site_dir / "unet_config.json",     # smp UNet config (for the seg comparison)
        "out": Path("outputs") / site_dir.name,
        "ckpt": Path("checkpoints") / site_dir.name / "final.pt",
    }


# ── 1) data preparation ─────────────────────────────────────────────────────────
def download_imagery_tiles(paths, patch=512, workers=8):
    """Download the .tif tiles for every .fgb area from the site webmap (S3 /vsis3)."""
    from dotenv import load_dotenv
    from tytonai_utils.webmap import build_grid, download_grid, preview_tiles
    load_dotenv(".env", override=True)
    cfg = json.loads(paths["site_config"].read_text())
    res, webmap = cfg["resolution_m"], cfg["webmap_s3"]
    paths["out"].mkdir(parents=True, exist_ok=True)
    for fgb in cfg["fgbs"]:
        area = Path(fgb).stem
        grid, _ = build_grid(paths["site"] / fgb, res, patch)
        download_grid(grid, webmap, paths["rgb"] / area, bands=[1, 2, 3], workers=workers)  # RGB only
        preview_tiles(paths["rgb"] / area, out_png=paths["out"] / f"overview_{area}.png")
    return paths["rgb"]


def download_annotations(paths, workers=8):
    """Download the npz imagery+mask annotations referenced by dataset.json."""
    from dotenv import load_dotenv
    from tytonai_utils.manifest import download_annotations_from_dataset_manifest
    if not paths["dataset"].exists():
        print(f"[pipeline] no dataset.json at {paths['dataset']} — skipping annotations")
        return None
    load_dotenv(".env", override=True)
    return download_annotations_from_dataset_manifest(paths["dataset"], paths["annotations"], workers=workers)


def align_annotations(paths, patch=512, mask_key="CLASSIFY", overlapping="first"):
    """Realign the npz masks onto the imagery webmap grid -> mask tiles paired 1:1 (by index)
    with the .tif imagery, so the eval scores the SAME tiles used for training. `patch` MUST
    match download_imagery_tiles' patch. Writes masks/<area>/tile_NNNNN.tif (+ area sidecar)."""
    from src.train.align import align_site_annotations
    if not paths["dataset"].exists():
        print("[pipeline] no dataset.json — skipping annotation alignment")
        return None
    return align_site_annotations(paths["site_config"], paths["dataset"], paths["annotations"],
                                  paths["masks"], patch=patch, mask_key=mask_key, overlapping=overlapping)


# ── 2) training ────────────────────────────────────────────────────────────────
def train_site(paths, weights=WEIGHTS, area=None, epochs=50, n_local=4, batch_size=2,
               grad_accum=4, max_steps=None, save_every_epochs=10, mlflow=True):
    """LoRA-finetune DINOv3 on a site's .tif tiles. Logs to the MLflow experiment
    'dino_lora_finetune' (run name = <site>__<scenario>) and drops a DISTINCT checkpoint
    every `save_every_epochs` (epoch010, epoch020, ...) plus final.pt. Returns final.pt."""
    from dotenv import load_dotenv
    from src.train.trainer import train
    load_dotenv(".env", override=True)                     # MLflow creds (+ AWS) from .env
    cfg = json.loads(paths["site_config"].read_text())
    area = area or Path(cfg["fgbs"][0]).stem
    save_dir = f"checkpoints/{paths['name']}"
    train(paths["name"], paths["rgb"] / area, weights, epochs=epochs, n_local=n_local,
          batch_size=batch_size, grad_accum=grad_accum, max_steps=max_steps,
          save_every_epochs=save_every_epochs, mlflow_enabled=mlflow, save_dir=save_dir)
    return Path(save_dir) / "final.pt"


# ── 3) visual analysis (pretrained vs finetuned, on the .tif tiles) ──────────────
#  The report is organised one SECTION per folder under outputs/<site>/report/.
#  Each report_* function owns exactly one section and returns the folder it filled;
#  run_visual_analysis just wires them, isolating failures so one can't sink the rest.

def report_method(tiles_dir, dst, n_examples=2):
    """Section 0 — model-free method figures (multicrop / iBOT / distillation)."""
    from src.visualisation.pretraining import plot_suite
    plot_suite(tiles_dir, out_dir=dst, n_examples=n_examples)
    return dst


def report_features(weights, ckpt, tiles, all_tiles, dst, device):
    """Section 1 — PCA-RGB feature maps: a 6-tile grid + the whole-site mosaic."""
    from src.visualisation.posttraining import compare_feature_mosaic, compare_features_grid
    compare_features_grid(weights, ckpt, tiles[:6], device=device, out_png=dst / "pca_grid.png")
    compare_feature_mosaic(weights, ckpt, all_tiles, device=device, out_png=dst / "feature_mosaic.png")
    return dst


def report_change(weights, ckpt, tiles, all_tiles, dst, device):
    """Section 2 — what finetuning changed: cosine heatmap + most-changed patch story."""
    from src.visualisation.posttraining import changed_patch_story, compare_patch_cosine
    compare_patch_cosine(weights, ckpt, all_tiles, device=device, out_png=dst / "cosine_change.png")
    changed_patch_story(weights, ckpt, tiles, device=device, out_png=dst / "changed.png")
    return dst


def report_expressiveness(weights, ckpt, tiles, dst, device):
    """Section 3 — expressiveness of the most-changed patches (PR + pairwise cosine) and the
    PC2-PC3 'explosion' scatter (where finetuning spread the patch cloud)."""
    from src.visualisation.posttraining import changed_patch_expressiveness, pca_explosion
    changed_patch_expressiveness(weights, ckpt, tiles, device=device, out_png=dst / "expr_changed.png")
    pca_explosion(weights, ckpt, tiles, device=device, out_png=dst / "pca_explosion.png")
    return dst


def _make_sections(report_dir, names):
    """Create report_dir/<name> for each section name; return {name: path}."""
    dirs = {name: report_dir / name for name in names}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _run_isolated(sections):
    """Run each (label, thunk); announce its folder, report+skip a failure. Returns the ones that ran."""
    import traceback
    done = []
    for label, thunk in sections:
        print(f"[viz] {label} ...", flush=True)
        try:
            dst = thunk()
            done.append(dst)
            print(f"[viz]   -> {dst}", flush=True)
        except Exception:                                   # isolate: report and keep going
            print(f"[viz]   FAILED ({label}):", flush=True)
            traceback.print_exc()
    return done


def run_visual_analysis(paths, ckpt, weights=WEIGHTS, device="cpu", n_tiles=30):
    """Build the pretrained-vs-finetuned report under outputs/<site>/report/, one folder
    per section. Each section is isolated, so a single failure never leaves you nothing."""
    from src.visualisation.common import pick_full_tiles
    cfg = json.loads(paths["site_config"].read_text())
    area = Path(cfg["fgbs"][0]).stem
    tiles_dir = paths["rgb"] / area
    tiles = pick_full_tiles(tiles_dir, n=n_tiles, seed=1)
    all_tiles = pick_full_tiles(tiles_dir, n=10**9)
    report = paths["out"] / "report"
    d = _make_sections(report, ["00_method", "01_features", "02_change", "03_expressiveness"])
    print(f"[viz] {paths['name']}/{area}: {len(all_tiles)} tiles ({len(tiles)} sampled) on {device} -> {report}")
    sections = [
        ("method",         lambda: report_method(tiles_dir, d["00_method"])),
        ("features",       lambda: report_features(weights, ckpt, tiles, all_tiles, d["01_features"], device)),
        ("change",         lambda: report_change(weights, ckpt, tiles, all_tiles, d["02_change"], device)),
        ("expressiveness", lambda: report_expressiveness(weights, ckpt, tiles, d["03_expressiveness"], device)),
    ]
    done = _run_isolated(sections)
    print(f"[viz] {paths['name']}: {len(done)}/{len(sections)} sections built -> {report}")
    return report


# ── 4) annotated evaluation (cluster / confusion / probe, on the aligned .tif tiles) ──
def _has_aligned_tiles(paths) -> bool:
    """True only if at least one imagery tile has a matching aligned mask. The masks/ dir can
    exist yet be empty (e.g. a site with no CLASSIFY annotations), so checking the directory
    isn't enough — an empty pair list crashes the eval downstream (empty torch.stack)."""
    if not paths["masks"].exists():
        return False
    from src.evaluation.separability import aligned_pairs
    imgs, _ = aligned_pairs(paths["rgb"], paths["masks"])
    return len(imgs) > 0



def run_evaluation(paths, ckpt, weights=WEIGHTS, device="cpu", n_clusters=8):
    """Cluster-vs-annotation + contingency + linear probe, on the .tif imagery tiles against
    the grid-aligned masks (run align_annotations first). Needs dataset.json + masks."""
    from src.evaluation.separability import (cluster_vs_annotation, confusion_cluster_vs_annotation,
                                             linear_probe_confusion, probe_split_map)
    if not _has_aligned_tiles(paths):
        print("[pipeline] no aligned tiles — run align_annotations first; skipping evaluation")
        return None
    rgb, masks = paths["rgb"], paths["masks"]
    dst = _make_sections(paths["out"] / "report", ["04_evaluation"])["04_evaluation"]
    res = cluster_vs_annotation(weights, ckpt, rgb, masks, n_clusters=n_clusters, device=device,
                                out_png=dst / "eval_separability.png")
    probe_split_map(rgb, masks, out_png=dst / "probe_split_map.png")   # control: train(red)/val(green) areas
    for model in ("pretrained", "finetuned"):
        confusion_cluster_vs_annotation(weights, ckpt, rgb, masks, model=model, n_clusters=n_clusters,
                                        device=device, out_png=dst / f"confusion_{model}.png")
        linear_probe_confusion(weights, ckpt, rgb, masks, model=model, device=device,   # area-level split (honest)
                               out_png=dst / f"probe_{model}.png")
    print(f"[eval] {paths['name']}: evaluation -> {dst}")
    return res


# ── 5) segmentation comparison (UNet vs DINO seg heads, on the aligned tiles) ─────────
def run_dino_seg_heads(paths, ckpt, weights=WEIGHTS, device="cpu", epochs=50,
                       dino_batch=4, mlflow=True):
    """Train + validate the two frozen-DINO ConvSegHeads (pretrained & finetuned) on the aligned
    tiles + honest area split. Cheap (cached features). Logs each head to the MLflow comparison
    experiment ('dino_segmentation_comparison'). Needs aligned masks (run align_annotations)."""
    from src.evaluation.segmentation_compare import compare_dino_heads
    if not _has_aligned_tiles(paths):
        print("[pipeline] no aligned tiles — run align_annotations first; skipping DINO seg heads")
        return None
    res, split = compare_dino_heads(weights, ckpt, paths["rgb"], paths["masks"],
                                    device=device, epochs=epochs, dino_batch=dino_batch,
                                    mlflow=mlflow, site=paths["name"])
    return res


def run_unet_seg(paths, device="cpu", unet_epochs=5, unet_batch=16, mlflow=True):
    """Train + validate the prod-like UNet BASELINE (trained Mega Model body + fresh head) on the
    aligned tiles + area split. Self-contained — NO DINO checkpoint needed (this chain stands on
    its own). `unet_epochs` is small (6000 samples/epoch). Logs to the MLflow comparison experiment.
    Needs aligned masks AND unet_config.json (epoch_file_key + activation:null)."""
    from src.evaluation.segmentation_compare import compare_unet
    if not _has_aligned_tiles(paths):
        print("[pipeline] no aligned tiles — run align_annotations first; skipping UNet seg")
        return None
    if not paths["model_config"].exists():
        print("[pipeline] no unet_config.json — skipping UNet seg")
        return None
    res, split = compare_unet(paths["rgb"], paths["masks"], paths["model_config"],
                              device=device, unet_epochs=unet_epochs, unet_batch=unet_batch,
                              mlflow=mlflow, site=paths["name"])
    return res


# ── 5c) whole-site classification maps (finetuned-DINO head vs UNet, colored geo-rasters) ──
def _site_tiles(paths, n=10**9):
    """Every (nearly) full imagery tile across ALL area subdirs of the site — the whole-site
    inference set for the classification maps (edge/black tiles skipped, like the feature mosaic)."""
    from src.visualisation.common import pick_full_tiles
    tiles = []
    for area_dir in sorted(p for p in paths["rgb"].iterdir() if p.is_dir()):
        tiles += pick_full_tiles(area_dir, n=n)
    return tiles


def run_classification_maps(paths, ckpt, weights=WEIGHTS, device="cpu", dino_res=None,
                            unet_res=None, epochs=50, unet_epochs=10, img_size=512):
    """Whole-site classification maps: run the FINETUNED-DINO seg head and the UNet baseline over
    EVERY site tile and write colored geo-rasters (report/06_classification_maps/: map_dino_finetuned,
    map_unet, map_compare, legend). REUSES the models already trained by run_dino_seg_heads /
    run_unet_seg when their result dicts are passed as `dino_res`/`unet_res`; trains only what's
    missing (the DINO head is cheap; the UNet is best-effort and skipped if it can't be sourced).
    Needs aligned masks (for the class space) + the .tif tiles."""
    from src.evaluation.segmentation_compare import (baseline_unet, class_mapping, finetuned_head,
                                                     predict_dino_site, predict_unet_site)
    from src.visualisation.classification import classification_maps
    if not _has_aligned_tiles(paths):
        print("[pipeline] no aligned tiles — run align_annotations first; skipping classification maps")
        return None
    _, _, names = class_mapping(paths["masks"])                # shared contiguous class space (0=ignore)

    # DINO finetuned head — reuse 5a's if handed in, else train just this one (cheap, cached feats)
    head = dino_res.get("dino_finetuned", (None,))[0] if dino_res else None
    if head is None:
        print("[maps] no DINO head passed — training the finetuned head")
        head, names = finetuned_head(weights, ckpt, paths["rgb"], paths["masks"],
                                     device=device, img_size=img_size, epochs=epochs)

    # UNet — reuse 5b's if handed in, else train it (best-effort; needs unet_config + S3 weights)
    unet = unet_res.get("unet", (None,))[0] if unet_res else None
    if unet is None and paths["model_config"].exists():
        try:
            print("[maps] no UNet passed — training the baseline UNet")
            unet = baseline_unet(paths["rgb"], paths["masks"], paths["model_config"],
                                 device=device, img_size=img_size, unet_epochs=unet_epochs)
        except Exception as e:                                 # unavailable weights/config -> skip UNet panel
            print(f"[maps] UNet unavailable ({type(e).__name__}: {e}) — DINO map only")

    tiles = _site_tiles(paths)
    if not tiles:
        print("[maps] no full tiles found — skipping classification maps")
        return None
    print(f"[maps] {paths['name']}: classifying {len(tiles)} site tiles on {device}")
    dino_preds = predict_dino_site(head, weights, ckpt, tiles, device, img_size) if head else None
    unet_preds = predict_unet_site(unet, paths["model_config"], tiles, device, img_size) if unet else None

    dst = _make_sections(paths["out"] / "report", ["06_classification_maps"])["06_classification_maps"]
    out = classification_maps(dino_preds, unet_preds, tiles, names, dst)
    print(f"[maps] {paths['name']}: {len([k for k in out if k != 'legend'])} maps -> {dst}")
    return out


# ── all post-training analysis for one site / many sites (NO download, NO train) ──────
def run_site_analysis(paths, ckpt, device="cuda:1", epochs=50, unet_epochs=10, mlflow=True):
    """Full per-site flow EXCEPT the unsupervised finetuning: data prep (download imagery +
    annotations, align masks) then the post-training steps (visual analysis, annotated evaluation,
    the 2 DINO seg heads, the UNet baseline). The finetuning is NEVER (re)done here — it reuses the
    existing checkpoint and raises FileNotFoundError if it's missing (we don't silently retrain).
    Every step self-skips when its own inputs are missing (no dataset.json -> no annotations/align;
    no aligned masks -> eval/DINO/UNet skip and only visual analysis runs)."""
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"no finetuned checkpoint at {ckpt} — finetune the site first; "
                                f"this step never retrains")
    # data prep — IDEMPOTENT: only fetch/align what's missing (download_grid re-writes every tile,
    # so we guard on existing outputs to avoid re-pulling thousands of tiles from S3 each re-run).
    if not any(paths["rgb"].rglob("*.tif")):                   # 1a) imagery absent -> download .tif tiles
        download_imagery_tiles(paths)
    if not paths["masks"].exists():                            # 1b+c) masks absent -> annotations + align
        download_annotations(paths)                            #       (both self-skip if no dataset.json)
        align_annotations(paths)
    run_visual_analysis(paths, ckpt, device=device)            # 3) pretrained-vs-finetuned pictures
    run_evaluation(paths, ckpt, device=device)                 # 4) cluster / confusion / probe (needs masks)
    dino_res = run_dino_seg_heads(paths, ckpt, device=device, epochs=epochs, mlflow=mlflow)   # 5a) DINO seg heads
    unet_res = run_unet_seg(paths, device=device, unet_epochs=unet_epochs, mlflow=mlflow)     # 5b) UNet baseline
    run_classification_maps(paths, ckpt, device=device, dino_res=dino_res, unet_res=unet_res,  # 5c) whole-site maps
                            epochs=epochs, unet_epochs=unet_epochs)


def run_all_sites(site_dirs, device="cuda:1", epochs=50, unet_epochs=10, mlflow=True):
    """run_site_analysis over several sites — the unsupervised finetuning is NEVER (re)done here,
    each site reuses its existing checkpoint (checkpoints/<site>/final.pt). FAILS FAST: if ANY site
    lacks a finetuned checkpoint, raises FileNotFoundError listing them (no silent skip, no retrain)
    before doing any work. Once past that gate, per-site analysis failures are isolated so one site
    can't sink the rest. Prints a per-site summary at the end."""
    import traceback
    paths_by_site = {sd: site_paths(sd) for sd in site_dirs}
    missing = [p["name"] for p in paths_by_site.values() if not p["ckpt"].exists()]
    if missing:                                                # finetuned model unavailable -> error
        raise FileNotFoundError(f"no finetuned checkpoint for: {', '.join(missing)} "
                                f"(expected checkpoints/<site>/final.pt). Finetune those first — "
                                f"this loop never retrains.")
    summary = {}
    for site_dir, paths in paths_by_site.items():
        print(f"\n{'=' * 22} {paths['name']} {'=' * 22}", flush=True)
        try:
            run_site_analysis(paths, paths["ckpt"], device=device, epochs=epochs,
                              unet_epochs=unet_epochs, mlflow=mlflow)
            summary[paths["name"]] = "ok"
        except Exception:                                      # isolate: one site's failure ≠ all
            traceback.print_exc()
            summary[paths["name"]] = "FAILED"
    print("\n=== summary ===")
    for name, status in summary.items():
        print(f"  {name:28s} {status}")
    return summary


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — pick a SITE config, then run the lines below one at a time (Shift+Enter).
#  cheap -> expensive: prep -> train (GPU) -> analyse/eval (cached after 1st pass).
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # --- choose ONE site ---
    
    CONFIG = {
        "site_dir": "input_site_data/monrovia",
        #"site_dir": "input_site_data/manned_bens_oasis_wet",
        #"site_dir": "input_site_data/monrovia",
        # 50 epochs, MLflow on (experiment 'dino_lora_finetune'), checkpoint every 10 epochs
        "train": dict(epochs=50, n_local=4, batch_size=2, grad_accum=4,
                      max_steps=None, save_every_epochs=10, mlflow=True),
        "device": "cuda:1",                                   # analysis/eval (cached -> cpu is fine)
    }

    paths = site_paths(CONFIG["site_dir"])
    
    download_imagery_tiles(paths)                           # 1a) .tif tiles for training + viz
    download_annotations(paths)                             # 1b) npz annotations
    align_annotations(paths)                                # 1c) realign npz masks onto the .tif imagery grid

    # ── eyeball the data (optional) — viz helpers straight from tytonai_utils ──────────
    # from tytonai_utils.viz import plot_image_mask_pairs, plot_image_mask_tiles
    # from tytonai_utils.rollup import CLASS_NAMES                 # {raw id -> name} for the legend
    # after 1b) — raw manifest .npz pairs (imagery + CLASSIFY mask, BEFORE re-tiling):
    # plot_image_mask_pairs(paths["annotations"], paths["dataset"], n=6,
    #                       class_names=CLASS_NAMES, out_png=paths["out"] / "annot_pairs.png")
    # after 1c) — grid-aligned .tif tiles (imagery tile_NNNNN.tif <-> mask tile_NNNNN.tif):
    # area = Path(json.loads(paths["site_config"].read_text())["fgbs"][0]).stem
    # plot_image_mask_tiles(paths["rgb"] / area, paths["masks"] / area, n=6,
    #                       class_names=CLASS_NAMES, out_png=paths["out"] / "aligned_tiles.png")

    ckpt = train_site(paths, **CONFIG["train"])            # 2) finetune -> checkpoints/<site>/{epoch010..050,final}.pt
    ckpt = paths["ckpt"]
    # ckpt  =  "checkpoints/monrovia__r16-K65k-dino+ibot+gram+koleo-sat/final.pt"  
    
    run_visual_analysis(paths, ckpt, device=CONFIG["device"])   # 3) pretrained-vs-finetuned pictures
    run_evaluation(paths, ckpt, device=CONFIG["device"])        # 4) cluster / confusion / probe on aligned tiles
    dino_res = run_dino_seg_heads(paths, ckpt, device="cuda:1", epochs=50)   # 5a) 2 DINO seg heads (cheap, cached -> MLflow)
    unet_res = run_unet_seg(paths, device="cuda:1", unet_epochs=10)          # 5b) prod-like UNet baseline (standalone -> MLflow)
    run_classification_maps(paths, ckpt, device="cuda:1",                    # 5c) whole-site maps (DINO-ft vs UNet, colored rasters)
                            dino_res=dino_res, unet_res=unet_res)

    # ── OR: relaunch the analysis for ALL sites at once, NO download / NO training ─────
    #  Uses each site's existing checkpoint. Sites without aligned masks (curepto, EM2020)
    #  run visual analysis only; the eval / DINO / UNet steps self-skip for them.
    SITES = [
        # "input_site_data/curepto_chile",
        # "input_site_data/EM2020_Jimblebar_Rail",
        "input_site_data/manned_bens_oasis_wet",
        "input_site_data/monrovia",
    ]
    run_all_sites(SITES, device="cuda:1", epochs=50, unet_epochs=10)

