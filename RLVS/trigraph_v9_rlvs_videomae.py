"""
trigraph_v9_rlvs_videomae.py
Guardian Eye — V9  |  Phase 2: VideoMAE Fine-Tuning + Embedding Extraction
Dataset : RLVS (Real Life Violence Situations)
Hardware: Local workstation — RTX 3090 (24 GB VRAM)

What this script does
─────────────────────
Stage A — Fine-tune VideoMAE-Base (ViT-Base/16, pretrained on Kinetics-400)
  as a standalone binary violence classifier on RLVS.
  Input:  frames_vit[16, 224, 224, 3] from the V9 NPZ cache.
  Output: best checkpoint saved to OUTPUT_DIR/videomae_best.pt.
  Records standalone accuracy as ablation experiment C.

Stage B — Embedding extraction.
  Load the best checkpoint. Run inference over all 2,000 clips (all splits).
  Mean-pool the final ViT layer hidden states -> 768-d float32 vector.
  Write vit_embedding[768] back into every NPZ (atomic in-place update).
  The VideoMAE model is NEVER loaded again after this stage.

Fine-tuning recipe
──────────────────
  AdamW, lr=1e-4, LLRD layer_decay=0.75, cosine decay to 1e-6
  Linear warmup 5 epochs
  weight_decay=0.05, label_smoothing=0.05
  Mixup alpha=0.8 + CutMix alpha=1.0 (equal probability; RWF recipe)
  Attention dropout=0.1, gradient checkpointing enabled
  Gradient accumulation steps=4 (effective batch=64)
  epochs=50, patience=10, min_ckpt_epoch=3
  batch=16 (RTX 3090 24 GB), grad_clip=1.0, EMA decay=0.999
  WeightedRandomSampler + pos_weight (dynamic from train labels)

RLVS is ~1:1 balanced so pos_weight ~1.0, but computed dynamically
to stay correct if the split changes.

Resumability
────────────
  - Stage A writes videomae_last.pt EVERY epoch (full model+ema+optimizer+
    scheduler+epoch state). Re-running auto-resumes from the last epoch.
  - On clean Stage A finish, videomae_stageA.done is written; a later re-run
    then skips straight to Stage B.
  - Stage B skips NPZs that already have a vit_embedding, so an interrupted
    extraction picks up where it stopped.
  Just re-run the same command after any interruption.

Usage
─────
  python trigraph_v9_rlvs_videomae.py
  python trigraph_v9_rlvs_videomae.py --skip-finetune    # Stage B only
  python trigraph_v9_rlvs_videomae.py --force-finetune   # redo Stage A
  python trigraph_v9_rlvs_videomae.py --force-reembed    # redo all of Stage B
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
PREPROC_DIR = Path(r"C:\Violence detection\ashour\fresh\RLVS\preproc_output")
OUTPUT_DIR  = Path(r"C:\Violence detection\ashour\fresh\RLVS\finetune_output")

CACHE_DIR   = PREPROC_DIR / "cache_v9"
SPLIT_CSV   = PREPROC_DIR / "split_rlvs.csv"
VIT_CKPT    = OUTPUT_DIR  / "videomae_best.pt"    # best-val checkpoint (used by Stage B)
VIT_LAST    = OUTPUT_DIR  / "videomae_last.pt"    # full state every epoch (for resume)
VIT_LOG_CSV = OUTPUT_DIR  / "videomae_train_log.csv"
VIT_RESULTS = OUTPUT_DIR  / "videomae_test_results.json"
STAGEA_DONE = OUTPUT_DIR  / "videomae_stageA.done"  # marker: Stage A finished


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class ViTCFG:
    model_name:      str   = "MCG-NJU/videomae-base"
    embed_dim:       int   = 768
    num_frames:      int   = 16
    img_size:        int   = 224

    batch_size:      int   = 16    # RTX 3090 24 GB + gradient checkpointing
    epochs:          int   = 50
    patience:        int   = 10
    min_ckpt_epoch:  int   = 3

    lr:              float = 1e-4
    lr_min:          float = 1e-6
    layer_decay:     float = 0.75
    weight_decay:    float = 0.05
    warmup_epochs:   int   = 5
    accum_steps:     int   = 4     # effective batch = 16 * 4 = 64
    grad_clip:       float = 1.0
    label_smoothing: float = 0.05
    # RWF recipe (balanced data) — RLVS is also 1:1 so no positive-class scarcity
    mixup_alpha:     float = 0.8    # Beta(alpha, alpha) frame-level mixup
    cutmix_alpha:    float = 1.0    # Beta(alpha, alpha) rectangular CutMix
    dropout:         float = 0.1
    ema_decay:       float = 0.999

    num_workers:     int   = 0   # Windows multiprocessing spawn breaks with workers > 0
    seed:            int   = 42


vcfg = ViTCFG()


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _color_jitter(frames: torch.Tensor,
                  brightness: float = 0.2,
                  contrast:   float = 0.2,
                  saturation: float = 0.2) -> torch.Tensor:
    """Consistent colour jitter across all T frames. frames: [T, 3, H, W] float32."""
    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast,   contrast)
    s = 1.0 + random.uniform(-saturation, saturation)
    frames = torch.clamp(frames * b, 0.0, 1.0)
    mean_lum = frames.mean(dim=(2, 3), keepdim=True)
    frames = torch.clamp((frames - mean_lum) * c + mean_lum, 0.0, 1.0)
    gray = (0.2989 * frames[:, 0:1] + 0.5870 * frames[:, 1:2]
            + 0.1140 * frames[:, 2:3])
    return torch.clamp(gray + s * (frames - gray), 0.0, 1.0)


class RLVSVideoDataset(Dataset):
    """
    Loads frames_vit [T_vit, H, W, 3] uint8 into RAM.
    Augmentation is applied on-the-fly so each epoch sees different crops/flips.
    """
    def __init__(self, npz_paths: List[Path], labels: List[int],
                 augment: bool = False):
        self.augment = augment
        self.samples = []
        print(f"  Loading {len(npz_paths)} clips into memory ...")
        for path, lbl in tqdm(zip(npz_paths, labels),
                               total=len(npz_paths),
                               desc="Loading", unit="file"):
            try:
                d = np.load(str(path), allow_pickle=True)
                self.samples.append({
                    "frames": d["frames_vit"],   # uint8 [T, H, W, 3] — half the RAM of float32
                    "label":  float(lbl),
                    "path":   str(path),
                })
            except Exception as e:
                print(f"  WARN: {path}: {e}")
        print(f"  Loaded {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = torch.from_numpy(
            sample["frames"].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)                          # [T, 3, H, W]
        frames = F.pad(frames, (16, 16, 16, 16), mode="reflect")  # -> [T,3,256,256]

        if self.augment:
            h0 = random.randint(0, 32)
            w0 = random.randint(0, 32)
            frames = frames[:, :, h0:h0 + 224, w0:w0 + 224]
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[3])
            frames = _color_jitter(frames)
        else:
            frames = frames[:, :, 16:240, 16:240]      # centre crop 224x224

        frames = (frames - MEAN) / STD
        return {
            "pixel_values": frames,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "label":        torch.stack([b["label"]        for b in batch]),
    }


# ── LLRD parameter groups ─────────────────────────────────────────────────────
def build_llrd_param_groups(model, base_lr: float,
                            layer_decay: float, weight_decay: float):
    num_layers = len(model.encoder.encoder.layer)  # 12 for ViT-Base

    def get_layer_id(name: str) -> int:
        if name.startswith("head") or name.startswith("norm"):
            return 0
        if "encoder.layer." in name:
            parts = name.split(".")
            for j, p in enumerate(parts):
                if p == "layer" and j + 1 < len(parts):
                    try:
                        return num_layers - int(parts[j + 1])
                    except ValueError:
                        pass
        return num_layers + 1

    def is_no_decay(name: str) -> bool:
        return name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid  = get_layer_id(name)
        nowd = is_no_decay(name)
        lr   = base_lr * (layer_decay ** lid)
        wd   = 0.0 if nowd else weight_decay
        key  = (lid, nowd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
        groups[key]["params"].append(param)
    return list(groups.values())


# ── VideoMAE wrapper ──────────────────────────────────────────────────────────
def build_videomae_model(model_name: str, embed_dim: int, dropout: float):
    from transformers import VideoMAEModel

    print(f"  Loading {model_name} from HuggingFace ...")
    base = VideoMAEModel.from_pretrained(model_name)
    base.gradient_checkpointing_enable()

    for layer in base.encoder.layer:
        if hasattr(layer.attention.attention, "dropout"):
            layer.attention.attention.dropout.p = dropout

    class VideoMAEClassifier(nn.Module):
        def __init__(self, encoder, d_model: int, drop: float):
            super().__init__()
            self.encoder = encoder
            self.norm    = nn.LayerNorm(d_model)
            self.drop    = nn.Dropout(drop)
            self.head    = nn.Linear(d_model, 1)

        def forward(self, pixel_values):
            # pixel_values: [B, T, C, H, W]
            out = self.encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state.mean(dim=1)  # [B, 768]
            return self.head(self.drop(self.norm(cls))).squeeze(-1)  # [B]

        def extract_embedding(self, pixel_values):
            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values)
                return out.last_hidden_state.mean(dim=1)  # [B, 768]

    model = VideoMAEClassifier(base, embed_dim, dropout)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── EMA ───────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ── Mixup / CutMix ────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha: float):
    if alpha <= 0:
        return x, y
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def cutmix_batch(x, y, alpha: float):
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    B, T, C, H, W = x.shape
    perm    = torch.randperm(B, device=x.device)
    r       = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(H * r), int(W * r)
    cx, cy  = random.randint(0, W), random.randint(0, H)
    x1, x2  = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    y1, y2  = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x_m     = x.clone()
    x_m[:, :, :, y1:y2, x1:x2] = x[perm, :, :, y1:y2, x1:x2]
    lam_act = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_m, lam_act * y + (1 - lam_act) * y[perm]


# ── Evaluate ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        pv  = batch["pixel_values"].to(device)
        lbl = batch["label"].cpu().numpy().astype(int)
        prb = torch.sigmoid(model(pv)).cpu().numpy()
        y_true.extend(lbl.tolist())
        y_prob.extend(prb.tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if len(y_true) == 0:
        return y_true, y_prob, 0.5, 0.0, 0.0
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.02, 0.91, 0.02):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0
    return y_true, y_prob, best_thr, best_f1, auc


# ── Main ─────────────────────────────────────────────────────────────────────
def main(args) -> None:
    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(vcfg.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Output : {OUTPUT_DIR}")

    # Load split CSV
    if not SPLIT_CSV.exists():
        raise RuntimeError(
            f"split_rlvs.csv not found at {SPLIT_CSV}. "
            "Run Phase 1 (trigraph_v9_rlvs_preprocess.py) first."
        )
    split_df = pd.read_csv(str(SPLIT_CSV))

    def get_split(name: str):
        sub    = split_df[split_df["split"] == name]
        paths  = [CACHE_DIR / f"{cid}.npz" for cid in sub["clip_id"]]
        labels = sub["label"].tolist()
        missing = sum(1 for p in paths if not p.exists())
        if missing:
            print(f"  WARN: {missing} NPZ files missing for split={name}")
        present_paths  = [p for p in paths if p.exists()]
        present_labels = [l for p, l in zip(paths, labels) if p.exists()]
        return present_paths, present_labels

    train_paths, train_labels = get_split("train")
    val_paths,   val_labels   = get_split("val")
    test_paths,  test_labels  = get_split("test")
    print(f"Train={len(train_paths)}  Val={len(val_paths)}  Test={len(test_paths)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage A — Fine-tuning
    # ═══════════════════════════════════════════════════════════════════════
    # Skip Stage A if: (a) user passed --skip-finetune, OR (b) Stage A already
    # completed on a previous run (STAGEA_DONE marker exists). Case (b) makes a
    # crash during Stage B resumable — re-running the script jumps straight to
    # Stage B instead of re-fine-tuning. Pass --force-finetune to override (b).
    stage_a_done = STAGEA_DONE.exists() and VIT_CKPT.exists()
    skip_stage_a = args.skip_finetune or (stage_a_done and not args.force_finetune)

    if skip_stage_a:
        if not VIT_CKPT.exists():
            raise RuntimeError(
                f"Stage A skip requested but no checkpoint at {VIT_CKPT}."
            )
        reason = "--skip-finetune" if args.skip_finetune else "Stage A already complete"
        print(f"\nSkipping Stage A ({reason}): loading checkpoint {VIT_CKPT}")
        ck    = torch.load(str(VIT_CKPT), map_location=DEVICE)
        model = build_videomae_model(vcfg.model_name, vcfg.embed_dim,
                                     vcfg.dropout).to(DEVICE)
        ema   = ModelEMA(model, decay=vcfg.ema_decay)
        ema.model.load_state_dict(ck["ema_state_dict"])
        print(f"  epoch={ck['epoch']}  val_f1={ck['val_macro_f1']:.4f}  "
              f"val_auc={ck.get('val_roc_auc', float('nan')):.4f}  "
              f"threshold={ck['threshold']:.4f}")
    else:
        # ── Datasets and loaders ──────────────────────────────────────────
        train_ds = RLVSVideoDataset(train_paths, train_labels, augment=True)
        val_ds   = RLVSVideoDataset(val_paths,   val_labels,   augment=False)

        # Dynamic pos_weight — RLVS is ~1:1 so pw ~1.0
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pw_val = n_neg / max(n_pos, 1)
        pw_tensor = torch.tensor([pw_val], device=DEVICE)
        print(f"  pos_weight = {pw_val:.3f}  (n_neg={n_neg}, n_pos={n_pos})")

        # WeightedRandomSampler so batches stay balanced even if split drifts slightly
        sample_weights = torch.tensor(
            [pw_val if l == 1 else 1.0 for l in train_labels], dtype=torch.float64)
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        train_loader = DataLoader(train_ds, batch_size=vcfg.batch_size,
                                  sampler=sampler, num_workers=vcfg.num_workers,
                                  pin_memory=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds, batch_size=vcfg.batch_size,
                                  shuffle=False, num_workers=vcfg.num_workers,
                                  pin_memory=True, collate_fn=collate_fn)

        # ── Model ─────────────────────────────────────────────────────────
        model = build_videomae_model(vcfg.model_name, vcfg.embed_dim,
                                     vcfg.dropout).to(DEVICE)
        ema   = ModelEMA(model, decay=vcfg.ema_decay)

        param_groups = build_llrd_param_groups(model, vcfg.lr,
                                               vcfg.layer_decay, vcfg.weight_decay)
        optimizer = AdamW(param_groups, betas=(0.9, 0.95))

        total_steps  = vcfg.epochs * len(train_loader) // vcfg.accum_steps
        warmup_steps = vcfg.warmup_epochs * len(train_loader) // vcfg.accum_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog   = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return max(vcfg.lr_min / vcfg.lr, cosine)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        start_epoch  = 1
        best_val_f1  = -1.0
        patience_cnt = 0
        global_step  = 0
        log_rows     = []

        # Resume from the full-state "last" checkpoint if present. This restores
        # model + EMA + optimizer + scheduler + epoch counter + patience + log,
        # so an interrupted run continues exactly where it stopped (not just from
        # the best epoch). VIT_LAST is written every epoch; VIT_CKPT is best-only.
        if VIT_LAST.exists():
            print(f"\nResuming Stage A from {VIT_LAST}")
            ck = torch.load(str(VIT_LAST), map_location=DEVICE)
            model.load_state_dict(ck["model_state_dict"])
            ema.model.load_state_dict(ck["ema_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch  = ck["epoch"] + 1
            best_val_f1  = ck["best_val_f1"]
            patience_cnt = ck["patience_cnt"]
            global_step  = ck.get("global_step", 0)
            log_rows     = ck.get("log_rows", [])
            print(f"  Resumed after ep={ck['epoch']}  "
                  f"best_val_f1={best_val_f1:.4f}  patience={patience_cnt}  "
                  f"-> continuing from ep={start_epoch}")
        else:
            print("Starting from pretrained Kinetics-400 weights.")

        def smooth_bce(logits, targets):
            a = vcfg.label_smoothing
            t = targets * (1.0 - a) + (1.0 - targets) * a
            return F.binary_cross_entropy_with_logits(logits, t,
                                                      pos_weight=pw_tensor)

        print(f"\n{'='*60}")
        print(f"Stage A: VideoMAE fine-tuning ({vcfg.epochs} epochs, "
              f"lr={vcfg.lr:.1e}, LLRD={vcfg.layer_decay}, "
              f"warmup={vcfg.warmup_epochs} ep)")
        print(f"  mixup={vcfg.mixup_alpha}  cutmix={vcfg.cutmix_alpha}  "
              f"ls={vcfg.label_smoothing}  ema={vcfg.ema_decay}  "
              f"accum={vcfg.accum_steps}  pw={pw_val:.3f}")
        print(f"{'='*60}")

        for epoch in range(start_epoch, vcfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch:03d}", leave=False, unit="batch")

            for step, batch in pbar:
                pv  = batch["pixel_values"].to(DEVICE)  # [B, T, 3, H, W]
                lbl = batch["label"].to(DEVICE)

                if step % vcfg.accum_steps == 0:
                    optimizer.zero_grad()

                use_mix    = vcfg.mixup_alpha > 0.0
                use_cutmix = vcfg.cutmix_alpha > 0.0
                if use_mix and use_cutmix:
                    if random.random() < 0.5:
                        pv_m, lbl_m = mixup_batch(pv, lbl, vcfg.mixup_alpha)
                    else:
                        pv_m, lbl_m = cutmix_batch(pv, lbl, vcfg.cutmix_alpha)
                elif use_mix:
                    pv_m, lbl_m = mixup_batch(pv, lbl, vcfg.mixup_alpha)
                elif use_cutmix:
                    pv_m, lbl_m = cutmix_batch(pv, lbl, vcfg.cutmix_alpha)
                else:
                    pv_m, lbl_m = pv, lbl

                logits = model(pv_m)
                loss   = smooth_bce(logits, lbl_m) / vcfg.accum_steps
                loss.backward()
                epoch_loss += loss.item() * vcfg.accum_steps

                is_last = (step == len(train_loader) - 1)
                if (step + 1) % vcfg.accum_steps == 0 or is_last:
                    nn.utils.clip_grad_norm_(model.parameters(), vcfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    ema.update(model)
                    global_step += 1

                pbar.set_postfix(loss=f"{loss.item() * vcfg.accum_steps:.4f}")

            avg_loss = epoch_loss / max(len(train_loader), 1)
            _, _, val_thr, val_f1, val_auc = evaluate(ema.model, val_loader, DEVICE)

            log_rows.append({
                "epoch": epoch, "train_loss": avg_loss,
                "val_macro_f1": val_f1, "val_roc_auc": val_auc,
                "val_threshold": val_thr,
            })
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
                  f"Val F1 {val_f1:.4f} | AUC {val_auc:.4f} | Thr {val_thr:.2f}")

            improved = False
            if epoch >= vcfg.min_ckpt_epoch and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "epoch": epoch,
                    "ema_state_dict": ema.model.state_dict(),
                    "val_macro_f1":   val_f1,
                    "val_roc_auc":    val_auc,
                    "threshold":      val_thr,
                }, str(VIT_CKPT))
                patience_cnt = 0
                improved = True
                print(f"  Best checkpoint saved (val_f1={val_f1:.4f})")
            elif epoch >= vcfg.min_ckpt_epoch:
                patience_cnt += 1

            # Full-state "last" checkpoint EVERY epoch — this is the resume point.
            # Written atomically (tmp->rename) so a kill mid-save can't corrupt it.
            tmp_last = VIT_LAST.with_suffix(".tmp.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "ema_state_dict":       ema.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1":          best_val_f1,
                "patience_cnt":         patience_cnt,
                "global_step":          global_step,
                "log_rows":             log_rows,
            }, str(tmp_last))
            tmp_last.replace(VIT_LAST)

            # Persist the training log every epoch too, so it survives a kill.
            pd.DataFrame(log_rows).to_csv(str(VIT_LOG_CSV), index=False)

            if patience_cnt >= vcfg.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # Stage A finished cleanly — drop a marker so a later resume skips
        # straight to Stage B without re-running fine-tuning.
        STAGEA_DONE.write_text(f"completed; best_val_f1={best_val_f1:.4f}\n")
        print(f"Training log -> {VIT_LOG_CSV}")

        # Test evaluation — Ablation C
        print("\nLoading best checkpoint for test evaluation (Ablation C) ...")
        ck = torch.load(str(VIT_CKPT), map_location=DEVICE)
        ema.model.load_state_dict(ck["ema_state_dict"])
        best_thr = float(ck["threshold"])

        test_ds     = RLVSVideoDataset(test_paths, test_labels, augment=False)
        test_loader = DataLoader(test_ds, batch_size=vcfg.batch_size,
                                 shuffle=False, num_workers=vcfg.num_workers,
                                 pin_memory=True, collate_fn=collate_fn)

        y_tt, y_pt, _, _, _ = evaluate(ema.model, test_loader, DEVICE)
        if len(y_tt) == 0:
            test_results = {"accuracy": None, "macro_f1": None,
                            "roc_auc": None, "threshold": best_thr,
                            "note": "test loader empty"}
        else:
            preds = (y_pt >= best_thr).astype(int)
            try:
                auc = float(roc_auc_score(y_tt, y_pt))
            except Exception:
                auc = None
            test_results = {
                "accuracy":  float(accuracy_score(y_tt, preds)),
                "macro_f1":  float(f1_score(y_tt, preds, average="macro",
                                            zero_division=0)),
                "roc_auc":   auc,
                "threshold": best_thr,
            }

        print(f"\n{'='*60}")
        print("Ablation C — VideoMAE standalone (TEST split):")
        for k, v in test_results.items():
            print(f"  {k:<15}: {v}")
        print(f"{'='*60}")

        VIT_RESULTS.write_text(json.dumps(test_results, indent=2))
        print(f"VideoMAE results -> {VIT_RESULTS}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage B — Embedding extraction for ALL splits
    # All splits must be processed so Phase 3 can load vit_embedding.
    # ═══════════════════════════════════════════════════════════════════════
    all_paths  = train_paths + val_paths + test_paths
    all_labels = train_labels + val_labels + test_labels

    # Resume: skip NPZs that already carry a vit_embedding (written by a prior
    # interrupted Stage B run). Pass --force-reembed to re-extract everything.
    if not args.force_reembed:
        todo_paths, todo_labels = [], []
        already = 0
        for p, l in zip(all_paths, all_labels):
            try:
                with np.load(str(p), allow_pickle=True) as d:
                    if "vit_embedding" in d:
                        already += 1
                        continue
            except Exception:
                pass  # unreadable -> re-extract
            todo_paths.append(p)
            todo_labels.append(l)
        print(f"\nStage B resume: {already} clips already have embeddings, "
              f"{len(todo_paths)} remaining.")
        all_paths, all_labels = todo_paths, todo_labels

    print(f"\n{'='*60}")
    print(f"Stage B: Extracting 768-d embeddings for {len(all_paths)} clips")
    print(f"{'='*60}")

    if len(all_paths) == 0:
        print("All embeddings already extracted. Phase 2 complete.")
        return

    ema.model.eval()

    class LazyNPZDataset(Dataset):
        """Reads NPZs from disk on demand — avoids loading all ~2,000 clips at once."""
        def __init__(self, paths, labels):
            self.paths  = [str(p) for p in paths]
            self.labels = labels

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            d      = np.load(self.paths[idx], allow_pickle=True)
            frames = torch.from_numpy(
                d["frames_vit"].astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2)
            frames = F.pad(frames, (16, 16, 16, 16), mode="reflect")
            frames = frames[:, :, 16:240, 16:240]
            frames = (frames - MEAN) / STD
            return {"pixel_values": frames,
                    "label":        torch.tensor(float(self.labels[idx])),
                    "path":         self.paths[idx]}

    def collate_lazy(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "label":        torch.stack([b["label"]        for b in batch]),
            "paths":        [b["path"] for b in batch],
        }

    full_ds     = LazyNPZDataset(all_paths, all_labels)
    full_loader = DataLoader(full_ds, batch_size=vcfg.batch_size,
                             shuffle=False, num_workers=vcfg.num_workers,
                             pin_memory=True, collate_fn=collate_lazy)

    write_errors = 0
    written      = 0
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Extracting+writing embeddings",
                          unit="batch"):
            pv   = batch["pixel_values"].to(DEVICE)
            embs = (ema.model.encoder(pixel_values=pv)
                    .last_hidden_state.mean(dim=1).cpu().numpy())  # [B, 768]
            for path_str, emb in zip(batch["paths"], embs):
                npz_path = Path(path_str)
                try:
                    existing = dict(np.load(str(npz_path), allow_pickle=True))
                    existing["vit_embedding"] = emb.astype(np.float32)
                    tmp = npz_path.with_suffix(".tmp.npz")
                    np.savez_compressed(str(tmp), **existing)
                    tmp.replace(npz_path)  # replace() overwrites atomically on Windows; rename() fails if dst exists
                    written += 1
                except Exception as e:
                    print(f"  ERROR updating {npz_path}: {e}")
                    write_errors += 1

    print(f"\nEmbedding extraction complete.")
    print(f"  Written : {written}")
    print(f"  Errors  : {write_errors}")
    print("Phase 2 complete. VideoMAE model will not be loaded again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLVS V9 VideoMAE")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip Stage A and load existing checkpoint for Stage B")
    parser.add_argument("--force-finetune", action="store_true",
                        help="Re-run Stage A even if it previously completed "
                             "(ignores the videomae_stageA.done marker)")
    parser.add_argument("--force-reembed", action="store_true",
                        help="Re-extract embeddings for all clips in Stage B, "
                             "even those that already have vit_embedding")
    args = parser.parse_args()
    main(args)
