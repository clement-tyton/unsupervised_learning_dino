"""
Clean DINOv3 ViT-L/16 + LoRA self-distillation training loop.

bf16 mixed precision via Accelerate, MLflow logging (one run per site+scenario),
EMA teacher, gradient accumulation. Losses: DINO (+ optional Gram / KoLeo / iBOT).

Memory (12 GB): bf16 + batch_size=2 + grad_accum fits ViT-L student+teacher with the
multi-crop. If you OOM: drop iBOT, then n_local, then batch_size=1 (but KoLeo needs
batch>=2 -> disable KoLeo if you go to 1).
"""

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train.dino import (DINOLoss, build_student_teacher, count_params, gram_loss,
                            iBOTLoss, koleo_loss, make_ibot_masks, save_checkpoint,
                            update_teacher)
from src.train.tracking import DinoRun, scenario_name
from src.train.data import GeoTIFFDataset, MultiCropTransform, collate_crops


def build_loader(tiles_dir, batch_size, n_local=6, num_workers=4):
    ds = GeoTIFFDataset(tiles_dir, transform=MultiCropTransform(n_local_crops=n_local))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=collate_crops, num_workers=num_workers, pin_memory=True)

CFG = dict(
    # --- model ---
    pretrained=True,
    out_dim=65536,            # K prototypes (DINO + iBOT heads)

    # --- data / batch (12 GB-safe) ---
    n_global=2,
    n_local=4,                # 4 fits 12 GB; 6 OOMs
    batch_size=2,             # physical images per micro-batch
    grad_accum=4,             # -> effective batch = 2*4 = 8
    num_workers=4,

    # --- optimization ---
    epochs=3,
    lr=1e-4,
    weight_decay=0.04,
    teacher_momentum=0.996,
    max_steps=None,           # None = full epochs; 100 = quick smoke

    # --- losses (toggles + weights) ---
    use_ibot=True,  lambda_ibot=1.0,  ibot_ratio=0.3,
    use_gram=True,  lambda_gram=0.1,
    use_koleo=True, lambda_koleo=0.1,

    # --- logging / checkpoints ---
    mlflow_enabled=True,
    log_every=1,
    save_dir=None,            # default: checkpoints/<site>__<scenario>/
    save_every_epochs=1,
)


# site             = "monrovia"
# tiles_dir        = TILES
# weights          = WEIGHTS
# # student, teacher already in your session
# pretrained       = True
# out_dim          = 65536
# n_global         = 2
# n_local          = 4
# batch_size       = 2
# grad_accum       = 4
# epochs           = 3
# lr               = 1e-4
# weight_decay     = 0.04
# teacher_momentum = 0.996
# lambda_gram      = 0.1
# lambda_koleo     = 0.1
# lambda_ibot      = 1.0
# use_ibot         = True
# use_gram         = True
# use_koleo        = True
# ibot_ratio       = 0.3
# num_workers      = 0          # 0 avoids the multiprocessing forkserver issue in a REPL
# max_steps        = 5          # tiny while stepping
# log_every        = 1
# mlflow_enabled   = False      # off while debugging the body
# save_dir         = None
# save_every_epochs= 1

def train(site, tiles_dir, weights, student=None, teacher=None, *, pretrained=True,
          out_dim=65536, n_global=2, n_local=6, batch_size=2, grad_accum=4,
          epochs=3, lr=1e-4, weight_decay=0.04, teacher_momentum=0.996,
          lambda_gram=0.1, lambda_koleo=0.1, lambda_ibot=1.0,
          use_ibot=True, use_gram=True, use_koleo=True, ibot_ratio=0.3,
          num_workers=4, max_steps=None, log_every=1, mlflow_enabled=True,
          save_dir=None, save_every_epochs=1):
    """Train (or continue) a student/teacher on one site's tiles. Returns (student, teacher).

    Pass an existing (student, teacher) to train them in place (e.g. to plot Gram
    before/after on the SAME models); leave None to build fresh from `weights`.
    """
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=grad_accum)
    device = accelerator.device

    loader = build_loader(tiles_dir, batch_size, n_local, num_workers=num_workers)
    if student is None or teacher is None:
        student, teacher = build_student_teacher(weights=weights, pretrained=pretrained,
                                                 out_dim=out_dim, use_ibot=use_ibot,
                                                 ibot_out_dim=out_dim)
    teacher.to(device)

    dino_loss = DINOLoss(out_dim).to(device)
    ibot_loss = iBOTLoss(out_dim).to(device) if use_ibot else None
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    student, optimizer, loader = accelerator.prepare(student, optimizer, loader)

    cfg = dict(out_dim=out_dim, lora_r=16, batch_size=batch_size, grad_accum=grad_accum,
               epochs=epochs, lr=lr, n_local=n_local, teacher_momentum=teacher_momentum,
               use_ibot=use_ibot, use_gram=use_gram, use_koleo=use_koleo, weights_tag="sat")
    scen = scenario_name(cfg)
    run = DinoRun(site, scen, config=cfg) if mlflow_enabled else None
    tr, tot = count_params(accelerator.unwrap_model(student))
    print(f"[{site}__{scen}] trainable {tr:.1f}M / {tot:.0f}M")

    ckpt_dir = save_dir or f"checkpoints/{site}__{scen}"

    def _save(tag):
        path = save_checkpoint(accelerator.unwrap_model(student), teacher, f"{ckpt_dir}/{tag}.pt")
        if run:
            run.log_artifact(path)
        print(f"  saved checkpoint -> {path}")

    step = 0
    student.train()
    for epoch in range(epochs):
        for crops in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            with accelerator.accumulate(student):
                with accelerator.autocast():
                    with torch.no_grad():                          # teacher: globals only
                        t_proj = [teacher(g)[0] for g in crops[:n_global]]
                        t_pat = [teacher(g)[2] for g in crops[:n_global]]
                    # crops = next(iter(loader))
                    s_out = [student(c) for c in crops]            # student: all crops
                    s_proj = [o[0] for o in s_out] # cls token proj into K=65K prototypes
                    s_cls = [o[1] for o in s_out]
                    s_pat = [o[2] for o in s_out]

                    loss = dino_loss(s_proj, t_proj)
                    logs = {"loss/dino": float(loss.detach())}
                    if use_gram:
                        lg = sum(gram_loss(s_pat[i], t_pat[i]) for i in range(n_global)) / n_global
                        loss = loss + lambda_gram * lg
                        logs["loss/gram"] = float(lg.detach())
                    if use_koleo:
                        lk = sum(koleo_loss(s_cls[i]) for i in range(n_global)) / n_global
                        loss = loss + lambda_koleo * lk
                        logs["loss/koleo"] = float(lk.detach())
                    if use_ibot:
                        um = accelerator.unwrap_model(student)
                        N = s_pat[0].shape[1]
                        li = 0.0
                        for g in crops[:n_global]:
                            B = g.shape[0]
                            masks = make_ibot_masks(B, N, ratio=ibot_ratio, device=device)
                            nm = int(masks[0].sum())
                            s_p = student(g, masks=masks)[2]           # (B,N,D) student masked-input
                            with torch.no_grad():
                                t_p = teacher(g)[2]                    # (B,N,D) teacher full
                            # project ONLY the masked tokens: (B,nm,K) instead of (B,N,K)
                            s_m = um.ibot(s_p[masks].view(B, nm, -1))           # student head, grad
                            with torch.no_grad():
                                t_m = teacher.ibot(t_p[masks].view(B, nm, -1))  # TEACHER head, no grad
                            li = li + ibot_loss(s_m, t_m)              # all tokens masked -> plain CE
                        li = li / n_global
                        loss = loss + lambda_ibot * li
                        logs["loss/ibot"] = float(li.detach())
                    logs["loss/total"] = float(loss.detach())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), 3.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:                          # one real optimizer step done
                update_teacher(accelerator.unwrap_model(student), teacher, teacher_momentum)
                if run and step % log_every == 0:
                    run.log(logs, step=step)
                step += 1
                if max_steps and step >= max_steps:
                    break
        # end of epoch: a DISTINCT checkpoint per milestone (epoch010, epoch020, ...)
        # so you can analyse how the model evolves; LoRA+heads only, small.
        if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
            _save(f"epoch{epoch + 1:03d}")
        if max_steps and step >= max_steps:
            break

    _save("final")
    if run:
        run.end()
    return accelerator.unwrap_model(student), teacher
