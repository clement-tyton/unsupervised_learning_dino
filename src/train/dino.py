"""
DINOv3 ViT-L/16 + LoRA self-distillation building blocks.

Everything is a small, inspectable object so you can step through it in a REPL and
print shapes at each stage. Nothing here runs on import — you call the factories.

Pipeline recap (shapes for ViT-L/16 @ 224, batch B):
    crop (B,3,224,224)
      -> backbone.forward_features -> cls (B,1024)  + patches (B,196,1024)
      -> DINOHead(1024 -> 65536)   -> proj (B,65536)   <- the distribution lives here
    Teacher = EMA copy (no grad), sees only the 2 global crops -> targets.
    Student = LoRA-adapted, sees all crops -> must match teacher targets.

Losses: DINO (CLS-level CE on proj), Gram (patch-feature correlation, DINOv3),
KoLeo (spread embeddings apart, anti-collapse).
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import dinov3.hub.backbones as _bb
from peft import LoraConfig, get_peft_model

EMBED_DIM = 1024  # ViT-L/16 hidden size


# ── backbone ──────────────────────────────────────────────────────────────────
def build_backbone(weights=None, pretrained=True):
    """DINOv3 ViT-L/16. `weights`: None -> default LVD1689M enum, or a local .pth path.

    pretrained=False builds the architecture with random weights (handy to inspect
    shapes offline without the gated download).
    """
    if weights is None:
        weights = _bb.Weights.LVD1689M
    return _bb.dinov3_vitl16(pretrained=pretrained, weights=weights)


def forward_features(backbone, x, masks=None):
    """Run the ViT and return (cls, patches). Works whether or not LoRA-wrapped.

    cls:     (B, 1024)        global CLS token
    patches: (B, N, 1024)     N = (img/16)^2 patch tokens (196 @ 224, 36 @ 96)
    masks:   (B, N) bool      iBOT: True positions are replaced by the mask token.
    """
    net = getattr(backbone, "base_model", backbone)        # unwrap peft if present
    net = getattr(net, "model", net)
    out = net.forward_features(x, masks=masks)
    return out["x_norm_clstoken"], out["x_norm_patchtokens"]


# ── DINO projection head ───────────────────────────────────────────────────────
class DINOHead(nn.Module):
    """MLP -> bottleneck -> L2-norm -> weight-normed Linear to `out_dim` prototypes.

    The softmax over its output is the distribution the student reproduces.
    """

    def __init__(self, in_dim=EMBED_DIM, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        # prototypes: trainable directions, L2-normalized at forward (== weight_norm g=1,
        # but deepcopy-safe so the teacher can be deep-copied).
        self.prototypes = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):                                # x: (B, in_dim)
        x = self.mlp(x)                                  # (B, bottleneck_dim)
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(self.prototypes.weight, dim=-1, p=2)
        return F.linear(x, w)                            # (B, out_dim) cosine logits


# ── backbone + head as one model ───────────────────────────────────────────────
class DINOv3Model(nn.Module):
    """Wraps a (possibly LoRA) backbone + a DINO CLS head (+ optional iBOT patch head).

    forward(x, masks) -> (proj, cls, patches). For iBOT, call .ibot(patches) on the
    returned patch tokens to get per-patch prototype logits (B, N, K_ibot).
    """

    def __init__(self, backbone, head, ibot_head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.ibot_head = ibot_head        # None unless use_ibot

    def forward(self, x, masks=None):
        cls, patches = forward_features(self.backbone, x, masks)
        return self.head(cls), cls, patches

    def ibot(self, patches):
        """Per-patch prototype logits (B, N, K_ibot). Needs use_ibot=True."""
        return self.ibot_head(patches)


def build_student_teacher(weights=None, pretrained=True, out_dim=65536,
                          lora_r=16, lora_alpha=32, lora_targets=("qkv", "proj"),
                          lora_dropout=0.1, use_ibot=False, ibot_out_dim=65536):
    """Returns (student, teacher).

    student: DINOv3 ViT-L with LoRA on attention (qkv, proj) + a fresh DINO head -> trainable.
    teacher: deep-copy EMA of student, gradients off.
    use_ibot: also attach an iBOT patch head (per-patch DINOHead) to both.
    """
    
    # weights=WEIGHTS
    # pretrained=PRETRAINED
    # out_dim=OUT_DIM
    # use_ibot=USE_IBOT
    # ibot_out_dim=OUT_DIM
    # lora_r=16
    # lora_alpha=32
    # lora_targets=("qkv", "proj")
    # lora_dropout=0.1
    
    backbone = build_backbone(weights, pretrained)
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=list(lora_targets),
                     lora_dropout=lora_dropout, bias="none")
    backbone = get_peft_model(backbone, cfg)
    ibot_head = DINOHead(EMBED_DIM, ibot_out_dim) if use_ibot else None
    student = DINOv3Model(backbone, DINOHead(EMBED_DIM, out_dim), ibot_head)

    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


def make_ibot_masks(B, N, ratio=0.3, device="cpu", generator=None):
    """Random boolean mask (B, N): ~ratio of patches True (to be masked in the student)."""
    n_mask = max(1, int(N * ratio))
    masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, device=device, generator=generator)[:n_mask]
        masks[b, idx] = True
    return masks


@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    """EMA: teacher = m*teacher + (1-m)*student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


# ── losses ─────────────────────────────────────────────────────────────────────
class DINOLoss(nn.Module):
    """Cross-view CE between sharpened+centered teacher and student distributions."""

    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.tt, self.ts, self.cm = teacher_temp, student_temp, center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_proj, teacher_proj):
        """student_proj: list of (B, K) for ALL crops. teacher_proj: list of (B, K) for 2 globals."""
        s = [F.log_softmax(p / self.ts, dim=-1) for p in student_proj]
        t = [F.softmax((p - self.center) / self.tt, dim=-1).detach() for p in teacher_proj]
        loss, n = 0.0, 0
        for i, ti in enumerate(t):
            for j, sj in enumerate(s):
                if i == j:                               # skip same-crop self-pair
                    continue
                loss = loss - (ti * sj).sum(-1).mean()
                n += 1
        self._update_center(teacher_proj)
        return loss / n

    @torch.no_grad()
    def _update_center(self, teacher_proj):
        c = torch.cat(teacher_proj).mean(0, keepdim=True)
        self.center.mul_(self.cm).add_(c, alpha=1 - self.cm)


class iBOTLoss(nn.Module):
    """Masked-patch self-distillation: student predicts the teacher's patch distribution
    at the MASKED positions (the student saw a mask token there, the teacher saw the
    real patch). Same softmax/centering trick as DINO, but per-patch.
    """

    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.tt, self.ts, self.cm = teacher_temp, student_temp, center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    def forward(self, student_patch_proj, teacher_patch_proj, masks=None):
        """student/teacher_patch_proj: (B, M, K). masks: (B, M) bool, or None when the
        tensors already contain ONLY masked tokens (memory-efficient path)."""
        s = F.log_softmax(student_patch_proj / self.ts, dim=-1)
        t = F.softmax((teacher_patch_proj - self.center) / self.tt, dim=-1).detach()
        ce = -(t * s).sum(-1)                                   # (B, M)
        if masks is None:
            loss = ce.mean()
        else:
            loss = (ce * masks).sum() / masks.sum().clamp(min=1)
        self._update_center(teacher_patch_proj)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_patch_proj):
        c = teacher_patch_proj.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.cm).add_(c, alpha=1 - self.cm)


def gram_matrix(features):
    """features (B,N,D) -> (B,D,D) channel correlation, normalized by N."""
    B, N, D = features.shape
    return torch.bmm(features.permute(0, 2, 1), features) / N


def gram_loss(student_patches, teacher_patches):
    """MSE between Gram matrices -> preserve dense spatial structure (DINOv3)."""
    return F.mse_loss(gram_matrix(student_patches), gram_matrix(teacher_patches).detach())


def koleo_loss(x, eps=1e-8):
    """Spread embeddings: push each sample away from its nearest neighbor in the batch."""
    x = F.normalize(x, dim=-1)
    sim = x @ x.t()
    sim.fill_diagonal_(-2)
    nn_idx = sim.argmax(1)
    d = (x - x[nn_idx]).norm(dim=1)
    return -(d + eps).log().mean()


# ── inspection helpers ─────────────────────────────────────────────────────────
def describe(obj, name="obj"):
    """Print shapes of a tensor / list / dict of tensors — for REPL stepping."""
    if torch.is_tensor(obj):
        print(f"{name:18} tensor {tuple(obj.shape)} {obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"{name:18} {type(obj).__name__} of {len(obj)}:")
        for i, o in enumerate(obj):
            describe(o, f"  [{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            describe(v, f"  {k}")
    else:
        print(f"{name:18} {type(obj).__name__}")


def count_params(model):
    """(trainable, total) parameter counts in millions."""
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr / 1e6, tot / 1e6


# ── checkpointing ───────────────────────────────────────────────────────────────
def _adapter_state(model):
    """The only params worth saving: LoRA adapters + the heads (the frozen 300M
    backbone is already on disk, so we skip it). Works for student AND teacher."""
    sd = model.state_dict()
    return {k: v for k, v in sd.items()
            if "lora_" in k or k.startswith(("head.", "ibot_head."))}


def save_checkpoint(student, teacher, path):
    """Save LoRA adapters + heads of both models into one small .pt file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"student": _adapter_state(student),
                "teacher": _adapter_state(teacher) if teacher is not None else None}, path)
    return path


def load_checkpoint(student, teacher, path, map_location="cpu"):
    """Load a checkpoint INTO existing models (frozen backbone untouched).

    The models must already exist with the same architecture (build_student_teacher).
    strict=False because we only carry the adapter+head keys, not the backbone.
    """
    ck = torch.load(path, map_location=map_location)
    student.load_state_dict(ck["student"], strict=False)
    if teacher is not None and ck.get("teacher") is not None:
        teacher.load_state_dict(ck["teacher"], strict=False)
    return student, teacher
