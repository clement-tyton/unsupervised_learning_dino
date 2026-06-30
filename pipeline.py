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
def run_evaluation(paths, ckpt, weights=WEIGHTS, device="cpu", n_clusters=8):
    """Cluster-vs-annotation + contingency + linear probe, on the .tif imagery tiles against
    the grid-aligned masks (run align_annotations first). Needs dataset.json + masks."""
    from src.evaluation.separability import (cluster_vs_annotation, confusion_cluster_vs_annotation,
                                             linear_probe_confusion, probe_split_map)
    if not paths["masks"].exists():
        print("[pipeline] no aligned masks — run align_annotations first; skipping evaluation")
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
def run_segmentation_comparison(paths, ckpt, weights=WEIGHTS, device="cpu", epochs=50,
                                unet_epochs=5, dino_batch=4, unet_batch=32, mlflow=True):
    """Train + validate UNet vs ConvSegHead on frozen DINO (pretrained & finetuned), on the
    aligned tiles + honest area split. `epochs` = DINO heads (cheap full passes); `unet_epochs`
    = the prod-like UNet (6000 samples/epoch, so far fewer). Logs each model to the MLflow
    comparison experiment ('dino_segmentation_comparison'). Needs aligned masks (run
    align_annotations); the UNet also needs unet_config.json (epoch_file_key + activation:null)."""
    from src.evaluation.segmentation_compare import compare_segmentation
    if not paths["masks"].exists():
        print("[pipeline] no aligned masks — run align_annotations first; skipping seg comparison")
        return None
    mc = paths["model_config"] if paths["model_config"].exists() else None
    res, split = compare_segmentation(weights, ckpt, paths["rgb"], paths["masks"], mc,
                                      device=device, epochs=epochs, unet_epochs=unet_epochs,
                                      dino_batch=dino_batch, unet_batch=unet_batch,
                                      mlflow=mlflow, site=paths["name"])
    return res


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — pick a SITE config, then run the lines below one at a time (Shift+Enter).
#  cheap -> expensive: prep -> train (GPU) -> analyse/eval (cached after 1st pass).
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # --- choose ONE site ---
    CONFIG = {
        #"site_dir": "input_site_data/EM2020_Jimblebar_Rail",
        "site_dir": "input_site_data/manned_bens_oasis_wet",
        #l"site_dir": "input_site_data/monrovia",
        # 50 epochs, MLflow on (experiment 'dino_lora_finetune'), checkpoint every 10 epochs
       "train": dict(epochs=50, n_local=4, batch_size=2, grad_accum=4,
                      max_steps=None, save_every_epochs=10, mlflow=True),
        "device": "cpu",                                   # analysis/eval (cached -> cpu is fine)
    }

    paths = site_paths(CONFIG["site_dir"])
    
    download_imagery_tiles(paths)                           # 1a) .tif tiles for training + viz
    download_annotations(paths)                             # 1b) npz annotations
    align_annotations(paths)                                # 1c) realign npz masks onto the .tif imagery grid

    ckpt = train_site(paths, **CONFIG["train"])            # 2) finetune -> checkpoints/<site>/{epoch010..050,final}.pt
    ckpt = paths["ckpt"]
    # ckpt  =  "checkpoints/monrovia__r16-K65k-dino+ibot+gram+koleo-sat/final.pt"  
    
    run_visual_analysis(paths, ckpt, device=CONFIG["device"])   # 3) pretrained-vs-finetuned pictures
    run_evaluation(paths, ckpt, device=CONFIG["device"])        # 4) cluster / confusion / probe on aligned tiles
    run_segmentation_comparison(paths, ckpt, device="cuda:1", epochs=50, unet_epochs=5)  # 5) UNet vs DINO seg heads (-> MLflow)

