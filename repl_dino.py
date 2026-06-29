"""
REPL walkthrough: build the DINOv3 ViT-L/16 + LoRA self-distillation pieces and
inspect EVERY shape, step by step. Run it line-by-line (or block-by-block) — each
block prints the dimensions it produces so you can see the whole story.

Objects live in src/dino.py; data utils in src/data.py; plotting in src/viz_multicrop.py.

Weights: PRETRAINED=True downloads the real DINOv3 ViT-L (gated — needs access; set
WEIGHTS to a local .pth you fetched with your HF token if the default URL is blocked).
PRETRAINED=False uses random weights so you can inspect shapes fully offline.
"""

import torch
from torch.utils.data import DataLoader

from src.train.dino import (DINOLoss, build_student_teacher, count_params, describe,
                            gram_loss, iBOTLoss, koleo_loss, make_ibot_masks)
from src.train.data import GeoTIFFDataset, MultiCropTransform, collate_crops
from src.visualisation.pretraining import show_one_example

# --- knobs --------------------------------------------------------------------
TILES = "input_site_data/monrovia/RGB/study_area"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = True     # False = random weights, offline (just to learn the shapes)
WEIGHTS = "model_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"  # local SAT493M (satellite) ckpt
OUT_DIM = 65536       # K prototypes (try 8192 for fast experiments)
BATCH = 2
N_GLOBAL = 2
USE_IBOT = True       # also build + show the iBOT (masked-patch) loss
print("device:", DEVICE)

# === 1) DATA: one multi-crop batch ============================================
transform = MultiCropTransform(n_local_crops=6)
dataset = GeoTIFFDataset(TILES, transform=transform)
print("dataset:", len(dataset), "tiles")
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_crops)

iterator  = iter(loader)
crops = next(iterator)                 # list of 8 tensors (2 global + 6 local)
crops = [c.to(DEVICE) for c in crops]
describe(crops, "crops")                   # [0,1]=(B,3,224,224) globals, [2..7]=(B,3,96,96) locals

# see the FULL multi-crop representation for the first example (original + crop
# regions in red/yellow for the 2 globals + the 6 local crops). Re-samples the crops
# from the tile path (the collated batch tensors carry no original/boxes).
show_one_example(dataset.paths[0], n_local=6,
                 out_png="input_site_data/monrovia/_batch_views.png")

# === 2) MODEL: student (LoRA) + teacher (EMA, frozen) =========================
student, teacher = build_student_teacher(weights=WEIGHTS, pretrained=PRETRAINED, out_dim=OUT_DIM,
                                         use_ibot=USE_IBOT, ibot_out_dim=OUT_DIM)
student, teacher = student.to(DEVICE), teacher.to(DEVICE)
tr, tot = count_params(student)
print(f"student: trainable {tr:.1f}M / {tot:.0f}M  ({100 * tr / tot:.1f}%)  <- LoRA + head only")

# === 3) TEACHER forward (global crops only) -> targets ========================
with torch.no_grad():
    t_proj = [teacher(g)[0] for g in crops[:N_GLOBAL]]   # distributions, 2 x (B, OUT_DIM)
    t_pat = [teacher(g)[2] for g in crops[:N_GLOBAL]]    # patch tokens, 2 x (B, 196, 1024)
describe(t_proj, "teacher proj")
describe(t_pat[0], "teacher patches")

# === 4) STUDENT forward (ALL crops) ===========================================
s_proj, s_cls, s_pat = [], [], []
for c in crops:
    proj, cls, pat = student(c)
    s_proj.append(proj); s_cls.append(cls); s_pat.append(pat)
describe(s_proj, "student proj")           # 8 x (B, OUT_DIM)
describe(s_cls[0], "student cls")          # (B, 1024)  <- the embedding you keep later

# === 5) LOSSES ================================================================
dino_loss = DINOLoss(out_dim=OUT_DIM).to(DEVICE)
l_dino = dino_loss(s_proj, t_proj)                                   # cross-view CE
l_gram = sum(gram_loss(s_pat[i], t_pat[i]) for i in range(N_GLOBAL)) / N_GLOBAL
l_koleo = sum(koleo_loss(s_cls[i]) for i in range(N_GLOBAL)) / N_GLOBAL
loss = l_dino + 0.1 * l_gram + 0.1 * l_koleo
print(f"DINO {l_dino.item():.3f} | Gram {l_gram.item():.4f} | "
      f"KoLeo {l_koleo.item():.3f} | total {loss.item():.3f}")

# === 5b) iBOT (optional): masked-patch self-distillation ======================
if USE_IBOT:
    N = s_pat[0].shape[1]                                  # patches per global crop (196)
    ibot_loss = iBOTLoss(out_dim=OUT_DIM).to(DEVICE)
    l_ibot = 0.0
    for g in crops[:N_GLOBAL]:
        masks = make_ibot_masks(g.shape[0], N, ratio=0.3, device=DEVICE)
        _, _, s_p = student(g, masks=masks)               # student sees MASKED patches
        with torch.no_grad():
            _, _, t_p = teacher(g)                         # teacher sees the FULL patches
        describe(masks, "ibot masks")
        l_ibot = l_ibot + ibot_loss(student.ibot(s_p), teacher.ibot(t_p), masks)
    l_ibot = l_ibot / N_GLOBAL
    loss = loss + 1.0 * l_ibot
    print(f"iBOT {l_ibot.item():.3f} | total+iBOT {loss.item():.3f}")

# === 6) TRAIN with the clean loop (Accelerate bf16 + MLflow) ==================
# Plots the explanatory suite BEFORE and AFTER training on the SAME models, so the
# Gram |teacher-student| panel goes from ~0 (untrained) to a real difference (adapted).
from src.train.trainer import train
from src.visualisation.pretraining import plot_suite

# free the big inspection tensors before training (optional, saves VRAM)
del t_proj, t_pat, s_proj, s_cls, s_pat
torch.cuda.empty_cache()


student, teacher = build_student_teacher(weights=WEIGHTS, pretrained=PRETRAINED, out_dim=OUT_DIM,
                                         use_ibot=USE_IBOT, ibot_out_dim=OUT_DIM)
student, teacher = student.to(DEVICE), teacher.to(DEVICE)

# --- BEFORE: baseline (LoRA adapters = 0 -> student == teacher -> Gram diff ~0) ---
plot_suite(TILES, out_dir="input_site_data/monrovia/viz_before",
           n_examples=1, models=(student, teacher), device=DEVICE)

# --- TRAIN in place (logs every step to MLflow under monrovia__<scenario>) ---
student, teacher = train(
    "monrovia", TILES, WEIGHTS, student=student, teacher=teacher,
    out_dim=OUT_DIM, batch_size=2, grad_accum=6, epochs=20, max_steps=None,
    n_local=4,                      # 6 -> 4 keeps peak ~10.7 GB on 12 GB
    use_ibot=USE_IBOT, num_workers=8, mlflow_enabled=True,
)

# --- AFTER: Gram |teacher-student| should now show real structure ---
plot_suite(TILES, out_dir="input_site_data/monrovia/viz_after",
           n_examples=1, models=(student, teacher), device=DEVICE)
# compare: input_site_data/monrovia/viz_before/gram_*.png  vs  viz_after/gram_*.png
