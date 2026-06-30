"""
Prod-like segmentation comparison: data_balance -> object_train (production UNet) vs DINO heads.

This wires the REAL production activity packages the way the platform chains them, as an example
you can adapt — it is NOT meant to be a one-click run (object_train is heavy + multi-label):

  1. data_balance  : from per-tile class pixel-counts (dataset.json), oversample rare-class tiles
                     -> a balanced multiset of tiles (a `count` per tile).      [RUNNABLE here]
  2. object_train  : train the PRODUCTION UNet (smp, MULTI-LABEL one-hot masks, focal/weighted CE,
                     LR schedule, Lightning Fabric) on the balanced npz tiles.   [wiring shown]
  3. DINO heads    : the frozen-DINO ConvSegHead comparison from src/evaluation/segmentation_compare.

IMPORTANT caveats (so the comparison isn't misread):
  - object_train is MULTI-LABEL (sigmoid per class, MultilabelF1 @0.5); the DINO heads here are
    SINGLE-LABEL (softmax/argmax, my F1). Different label semantics + metric -> the prod UNet number
    is a reference, not a like-for-like cell in the DINO F1 table.
  - object_train consumes the 384 npz tiles (input_site_data/<site>/annotations, native format);
    the DINO heads run on the 512 aligned .tif tiles. Different tilings.
  - object_train normally pulls tiles from S3; here we point TrainingData at the LOCAL npz so it
    runs offline. Construct Config/Metadata to match YOUR installed object_train version.
"""

import json
import math
from pathlib import Path

import numpy as np


# ── 1) data_balance: per-tile class counts -> a balanced multiset of tiles ──────────
def manifest_class_counts(dataset_json, class_list):
    """(tiles, tile_arr): the manifest tile dicts + a (n_tiles, n_classes) float64 count matrix
    (tile_arr[i, j] = pixels of class_list[j] in tile i), exactly what data_balance consumes."""
    tiles = json.loads(Path(dataset_json).read_text())
    tile_arr = np.array(
        [[t["class_counts"].get(str(c), 0) for c in class_list] for t in tiles], dtype=np.float64)
    return tiles, tile_arr


def balance_tiles(tile_arr, upsample_power=0.2, data_control=0.2, seed=112):
    """data_balance core (no ActivityApp): oversample rare-class tiles. Replicates
    DataBalance.start()'s target computation, then calculate_sample + select_tiles. Returns a 1D
    array of tile indices WITH REPETITION (rare-class tiles appear several times)."""
    from data_balance.sampling import calculate_sample, select_tiles
    np.random.seed(seed)
    n_tiles = len(tile_arr)
    lifeform = tile_arr.sum(0)                                   # total pixels per class
    total = tile_arr.sum()
    prob = np.where(lifeform > 0, (lifeform / total) ** upsample_power, 0.0)
    target = np.array([math.ceil(data_control * total * 3 * p) for p in prob], dtype=np.int32)
    samp = calculate_sample(n_tiles, tile_arr, target, lifeform)
    idx = select_tiles(tile_arr, samp, target, lifeform, max(2 * n_tiles, 500_000))
    return np.asarray(idx, dtype=np.int64)


def balance_report(tile_arr, balanced_idx, class_list, names=None):
    """Print the class pixel-distribution BEFORE vs AFTER balancing (as % of total)."""
    before = tile_arr.sum(0)
    after = tile_arr[balanced_idx].sum(0)
    nm = names or [str(c) for c in class_list]
    print(f"  {'class':14s} {'before %':>9s} {'after %':>9s}  (x tiles)")
    for j, c in enumerate(class_list):
        b = 100 * before[j] / before.sum()
        a = 100 * after[j] / after.sum()
        print(f"  {nm[j]:14s} {b:8.1f}% {a:8.1f}%   x{after[j] / before[j]:.1f}" if before[j] else
              f"  {nm[j]:14s} {b:8.1f}% {a:8.1f}%   (absent)")
    print(f"  tiles: {len(tile_arr)} unique -> {len(balanced_idx)} sampled "
          f"({len(balanced_idx) / len(tile_arr):.1f}x)")


# ── 2) object_train: build the production UNet inputs from the balanced sample ───────
def balanced_training_data(tiles, balanced_idx, annotations_dir, bands):
    """list[object_train.TrainingData] with a `count` per tile (= how many times the balancer
    picked it). object_train's Dataset repeats each tile `count` times. Points at the LOCAL npz."""
    from object_train.train_types import TrainingData
    counts = np.bincount(balanced_idx, minlength=len(tiles))
    ann = Path(annotations_dir)
    return [TrainingData(bands=bands, imagery_file=str(ann / tiles[i]["imagery_file"]),
                         mask_file=str(ann / tiles[i]["mask_file"]), count=int(counts[i]))
            for i in range(len(tiles)) if counts[i] > 0]


def train_unet_prod(training_data, class_list, bands, train_mean, train_std,
                    num_epochs=30, batch_size=8, tile_size=384, device=1, seed=112):
    """PROD path: build the smp UNet via object_train and run its Trainer.fit on the balanced
    npz tiles (multi-label, focal/weighted CE, LR schedule, Fabric). This is the real call
    sequence object_train.ObjectTrain.start() uses, minus the S3/activity context. Adjust the
    Config/Metadata fields to your installed object_train version. Returns the trained model.

    NOTE: heavy (res2net101 @384, multi-label) + needs lightning Fabric — run on GPU.
    """
    from lightning.fabric import Fabric, seed_everything
    from torch.utils.data import DataLoader
    from tytonai.ml.models import build_new_model

    from object_train.data_generator import Dataset
    from object_train.io_schema.model import Config, Metadata, ModelType
    from object_train.model import setup_model
    from object_train.train_types import TrainSettings
    from object_train.trainer import Trainer

    seed_everything(seed, workers=True)
    fabric = Fabric(accelerator="auto", devices=device, precision="bf16-mixed")
    fabric.launch()

    config = Config(bands=bands, model_type=ModelType.UNET, encoder_type="timm-res2net101_26w_4s",
                    encoder_weights="imagenet", activation="softmax2d")
    model = build_new_model(len(class_list), config.model_type.value, config, load_encoder_weights=True)

    ds = Dataset(training_data, class_list=class_list, bands=bands, train_mean=train_mean,
                 train_std=train_std, seed=seed, augmented_tile_size=256, tile_size=tile_size, train=True)
    loader = fabric.setup_dataloaders(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4))

    metadata = Metadata(batch_size=batch_size, num_epochs=num_epochs, initial_lr=1e-3,
                        lr_schedular_step_size=2, lr_schedular_dec_rate=0.5, seed=seed,
                        focal_loss=False, weight_decay=0.0)        # adjust to your Metadata schema
    model, optimizer, scheduler, loss = setup_model(fabric, model, metadata, steps_per_epoch=len(loader))
    trainer = Trainer(fabric, model, optimizer, scheduler, loss, class_list, config.model_type)

    settings = TrainSettings(config=config, metadata=metadata, train_mean=train_mean, train_std=train_std,
                             cnn_model_id=None, initial_epoch=1, starting_epoch=1, save_every_n_epochs=1,
                             existing_epoch=None, existing_class_list=None)

    def on_epoch_end(d):
        print(f"  [unet] epoch {d.epoch}: loss={d.train_logs.get('loss'):.3f} "
              f"f1={d.train_logs.get('f1/overall')}")

    trainer.fit(loader, None, ds, settings, num_epochs=num_epochs,
                on_epoch_end=on_epoch_end, on_step_end=lambda *_: None)
    return model


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN — step through (Shift+Enter). Part 1 runs offline; Part 2 (UNet) needs a GPU.
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.evaluation.segmentation_compare import CLASS_NAMES, present_classes

    SITE = Path("input_site_data/monrovia")
    DATASET = SITE / "dataset.json"
    ANNOTATIONS = SITE / "annotations"
    class_list = present_classes(SITE / "masks")                 # [2,4,5,7,9,100,200,201]
    names = [CLASS_NAMES.get(c, str(c)) for c in class_list]

    # 1) data_balance — oversample the rare classes (RUNNABLE, offline) ----------------
    tiles, tile_arr = manifest_class_counts(DATASET, class_list)
    balanced_idx = balance_tiles(tile_arr)
    print("=== data_balance: class distribution before vs after ===")
    balance_report(tile_arr, balanced_idx, class_list, names)

    # 2) object_train — build the prod UNet inputs from the balanced sample ------------
    cfg0 = tiles[0]
    bands = cfg0["imagery_bands"]                                 # e.g. ['DSM','RED','GREEN','BLUE']
    training_data = balanced_training_data(tiles, balanced_idx, ANNOTATIONS, bands)
    print(f"\n=== object_train: {len(training_data)} tiles with balance counts "
          f"(total {sum(t.count for t in training_data)} samples) ===")
    # train_unet_prod(training_data, class_list, bands, cfg0["image_mean"], cfg0["image_std"],
    #                 num_epochs=30, device=1)        # <- uncomment to train the prod UNet on GPU

    # 3) DINO heads — the single-label comparison (different tiling/metric, see caveats) -
    # from src.evaluation.segmentation_compare import compare_segmentation
    # res, split = compare_segmentation(WEIGHTS, CKPT, SITE/"RGB", SITE/"masks",
    #                                   model_config=SITE/"unet_config.json", device="cuda:1", epochs=30)
