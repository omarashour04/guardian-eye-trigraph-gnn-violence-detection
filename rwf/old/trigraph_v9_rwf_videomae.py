"""
trigraph_v9_rwf_videomae.py
Guardian Eye — V9  |  Phase 2: VideoMAE Fine-Tuning + Embedding Extraction
Dataset : RWF-2000
GPU     : A10G  |  ETA ~2 hours

What this script does
─────────────────────
Stage A — Fine-tune VideoMAE-Base (ViT-Base/16, pretrained on Kinetics-400)
  as a standalone binary violence classifier on RWF-2000
  (2,000 videos: 1,600 train + 400 val).
  Input: frames_vit[16, 224, 224, 3] from the V9 NPZ cache.
  Output: best checkpoint saved to MOUNT_PATH/videomae_best.pt.
  This stage also records the standalone accuracy as ablation experiment C.

Stage B — Embedding extraction.
  Load the best checkpoint.  Run inference over all 2,000 videos.
  Mean-pool the final ViT layer hidden states → 768-d float32 vector.
  Write vit_embedding[768] back into every NPZ (compressed in-place update).
  The VideoMAE model is NEVER loaded again after this stage.

Fine-tuning recipe (VideoMAE official protocol)
────────────────────────────────────────────────
  AdamW, lr=1e-4, layer-wise LR decay=0.75, cosine decay to 1e-6
  Linear warmup for 5 epochs
  weight_decay=0.05
  label_smoothing=0.05
  Mixup α=0.8 + CutMix α=1.0 (applied with equal probability)
  Dropout=0.1 (attention)
  Gradient accumulation steps=4 (effective batch=64)
  epochs=50, patience=10, min_ckpt_epoch=3
  batch=16 (fits A10G 24GB VRAM)
  Gradient clipping=1.0
  EMA decay=0.999

Usage
─────
  modal run trigraph_v9_rwf_videomae.py::run_videomae
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import random
from pathlib import Path
from dataclasses import dataclass
import torch

import modal

# ── Modal primitives ──────────────────────────────────────────────────────────
APP_NAME      = "guardian-eye-v9-rwf-videomae"
VOL_NAME_PROC = "rwf-2000-processed"

app      = modal.App(APP_NAME)
vol_proc = modal.Volume.from_name(VOL_NAME_PROC, create_if_missing=True)

PROC_MOUNT  = Path("/data/proc")
CACHE_DIR   = PROC_MOUNT / "cache_v9"
SPLIT_CSV   = PROC_MOUNT / "split_v9.csv"
VIT_CKPT    = PROC_MOUNT / "videomae_best.pt"
VIT_LOG_CSV = PROC_MOUNT / "videomae_train_log.csv"

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.40.2",
        "timm==0.9.16",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "tqdm==4.66.4",
        "einops==0.7.0",
    )
)


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class ViTCFG:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "MCG-NJU/videomae-base"
    embed_dim:       int   = 768     # VideoMAE-Base CLS output dimension (mean-pool over patches)
    num_frames:      int   = 16      # must match T_vit in Phase 1
    img_size:        int   = 224

    # ── Fine-tuning ────────────────────────────────────────────────────────────
    batch_size:      int   = 16     # A10G 24 GB VRAM + gradient checkpointing
    epochs:          int   = 50     # LLRD + warmup need more runway
    patience:        int   = 10
    min_ckpt_epoch:  int   = 3

    lr:              float = 1e-4
    lr_min:          float = 1e-6   # cosine decay floor
    layer_decay:     float = 0.75   # LLRD per-layer decay factor
    weight_decay:    float = 0.05
    warmup_epochs:   int   = 5      # linear warmup before cosine decay
    accum_steps:     int   = 4      # effective batch = 16 * 4 = 64
    grad_clip:       float = 1.0
    label_smoothing: float = 0.05
    mixup_alpha:     float = 0.8    # Beta(alpha, alpha) frame-level mixup
    cutmix_alpha:    float = 1.0    # Beta(alpha, alpha) rectangular CutMix
    dropout:         float = 0.1
    ema_decay:       float = 0.999  # ~1000-step window

    # ── Data ──────────────────────────────────────────────────────────────────
    num_workers:     int   = 0      # in-memory dataset; workers waste CPU on 2-core container
    seed:            int   = 42


vcfg = ViTCFG()


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────
def build_vit_dataset_class():
    """
    Returns RWFVideoDataset.

    Loading strategy: store raw uint8 frames [T, H, W, 3] in memory.
    Normalisation and augmentation are applied per-call in __getitem__ so
    the augmented view differs every epoch.

    Augmentations (train only, applied frame-consistently across all T frames):
      - Pad to [T, 256, 256, 3] then random 224x224 crop (same location all T)
      - RandomHorizontalFlip(p=0.5)
      - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    Val/test: pad to 256 then centre-crop 224x224.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    import numpy as np
    import random as _random

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _color_jitter(frames_chw: torch.Tensor,
                      brightness: float = 0.2,
                      contrast:   float = 0.2,
                      saturation: float = 0.2) -> torch.Tensor:
        """
        Apply identical random color jitter to every frame.
        frames_chw: [T, 3, H, W] float32 in [0, 1].
        Returns same shape. Hue rotation omitted — the RGB-rotation
        approximation does not match HSV hue and introduces colour casts.
        """
        # Sample factors once and apply to all frames uniformly
        b = 1.0 + _random.uniform(-brightness, brightness)
        c = 1.0 + _random.uniform(-contrast, contrast)
        s = 1.0 + _random.uniform(-saturation, saturation)

        # Brightness
        frames_chw = torch.clamp(frames_chw * b, 0.0, 1.0)

        # Contrast: scale around per-frame mean
        mean_lum = frames_chw.mean(dim=(2, 3), keepdim=True)
        frames_chw = torch.clamp((frames_chw - mean_lum) * c + mean_lum, 0.0, 1.0)

        # Saturation: blend with grayscale
        gray = (0.2989 * frames_chw[:, 0:1, :, :]
              + 0.5870 * frames_chw[:, 1:2, :, :]
              + 0.1140 * frames_chw[:, 2:3, :, :])
        frames_chw = torch.clamp(
            gray + s * (frames_chw - gray), 0.0, 1.0)

        return frames_chw

    class RWFVideoDataset(Dataset):
        def __init__(self, npz_paths, labels, augment: bool = False):
            self.augment = augment
            self.samples = []
            from tqdm import tqdm
            print(f"  Loading {len(npz_paths)} videos into memory ...")
            for path, lbl in tqdm(zip(npz_paths, labels),
                                  total=len(npz_paths),
                                  desc="Loading", unit="file"):
                try:
                    d = np.load(str(path), allow_pickle=True)
                    # Store uint8 to halve memory vs float32 (~4.8 GB vs 9.7 GB)
                    frames_uint8 = d["frames_vit"]   # [T, H, W, 3] uint8
                    self.samples.append({
                        "frames":  frames_uint8,     # kept as uint8 numpy array
                        "label":   float(lbl),
                        "path":    str(path),
                    })
                except Exception as e:
                    print(f"  WARN: {path}: {e}")
            print(f"  Loaded {len(self.samples)} samples.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            # [T, H, W, 3] uint8 → [T, 3, H, W] float32 in [0, 1]
            frames = torch.from_numpy(
                sample["frames"].astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2)   # [T, 3, H, W]

            # Pad from 224x224 to 256x256 using reflect padding
            # F.pad expects [..., H, W] and pads in reverse dim order
            frames = F.pad(frames, (16, 16, 16, 16), mode="reflect")  # [T, 3, 256, 256]

            if self.augment:
                # Random 224x224 crop — same spatial location across all T frames
                h_start = random.randint(0, 32)
                w_start = random.randint(0, 32)
                frames = frames[:, :, h_start:h_start + 224, w_start:w_start + 224]

                # Horizontal flip — same decision applied to all T frames
                if random.random() < 0.5:
                    frames = torch.flip(frames, dims=[3])

                # Color jitter — same random factors across all T frames
                frames = _color_jitter(frames)
            else:
                # Centre crop 224x224 from padded 256x256
                frames = frames[:, :, 16:240, 16:240]

            # Normalise to ImageNet stats
            frames = (frames - MEAN) / STD

            return {
                "pixel_values": frames,
                "label": torch.tensor(sample["label"], dtype=torch.float32),
            }

    return RWFVideoDataset


# ── LLRD parameter groups ─────────────────────────────────────────────────────
def build_llrd_param_groups(model, base_lr: float,
                            layer_decay: float, weight_decay: float):
    """
    Layer-wise learning rate decay (LLRD) following the VideoMAE fine-tuning
    protocol. Lower layers receive exponentially smaller LRs.

    layer_id assignment:
      head / top-level norm  → id 0  (lr = base_lr, highest)
      encoder.layer.{i}.*   → id = num_layers - i
      patch embed / pos emb  → id = num_layers + 1  (lowest lr)
      anything else          → id = num_layers + 1
    """
    num_layers = len(model.encoder.encoder.layer)   # 12 for ViT-Base

    def get_layer_id(name: str) -> int:
        if name.startswith("head") or name.startswith("norm"):
            return 0
        if "encoder.layer." in name:
            # extract layer index from name like 'encoder.encoder.layer.5.xxx'
            parts = name.split(".")
            for j, p in enumerate(parts):
                if p == "layer" and j + 1 < len(parts):
                    try:
                        i = int(parts[j + 1])
                        return num_layers - i
                    except ValueError:
                        pass
        # patch embedding and position embedding → lowest LR
        return num_layers + 1

    def is_no_decay(name: str) -> bool:
        return name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name

    # Collect unique (layer_id, no_decay) combinations as separate groups
    groups: dict[tuple, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid     = get_layer_id(name)
        no_wd   = is_no_decay(name)
        lr      = base_lr * (layer_decay ** lid)
        wd      = 0.0 if no_wd else weight_decay
        key     = (lid, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
        groups[key]["params"].append(param)

    return list(groups.values())


# ── VideoMAE wrapper ──────────────────────────────────────────────────────────
def build_videomae_model(model_name: str, embed_dim: int, dropout: float):
    """
    Load pretrained VideoMAE-Base from HuggingFace.
    Replace the classification head with Linear(embed_dim, 1).
    Apply attention dropout to all transformer blocks.
    Returns the model.

    Note: stochastic depth is NOT applied post-hoc on HuggingFace VideoMAELayer
    because hasattr(layer, 'drop_path') always returns False on that class.
    """
    import torch
    import torch.nn as nn

    try:
        from transformers import VideoMAEModel
        print(f"  Loading {model_name} from HuggingFace ...")
        base = VideoMAEModel.from_pretrained(model_name)
        base.gradient_checkpointing_enable()

        # Apply attention dropout by setting the actual nn.Dropout module's p
        for layer in base.encoder.layer:
            if hasattr(layer.attention.attention, 'dropout'):
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
                # VideoMAE-Base has no CLS token; mean-pool all patch tokens
                cls = out.last_hidden_state.mean(dim=1)   # [B, 768]
                cls = self.norm(cls)
                cls = self.drop(cls)
                return self.head(cls).squeeze(-1)          # [B]

            def extract_embedding(self, pixel_values):
                """Return mean-pooled 768-d embedding without classification head."""
                with torch.no_grad():
                    out = self.encoder(pixel_values=pixel_values)
                    return out.last_hidden_state.mean(dim=1)   # [B, 768]

        model = VideoMAEClassifier(base, embed_dim, dropout)
        print(f"  Model ready. "
              f"Params: {sum(p.numel() for p in model.parameters()):,}")
        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load VideoMAE from HuggingFace: {e}\n"
            "Ensure 'transformers' is installed and the checkpoint name is "
            "correct (MCG-NJU/videomae-base)."
        )


# ── Mixup helper ──────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha: float = 0.8):
    """
    Mixup augmentation.  lambda ~ Beta(alpha, alpha).
    Mixes entire frames linearly.
    Returns mixed (x, y) — y is the soft mixed label.
    """
    import torch
    import numpy as np
    if alpha <= 0:
        return x, y
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_m  = lam * x + (1 - lam) * x[perm]
    y_m  = lam * y + (1 - lam) * y[perm]
    return x_m, y_m


# ── CutMix helper ─────────────────────────────────────────────────────────────
def cutmix_batch(x, y, alpha: float = 1.0):
    """
    CutMix augmentation.  lambda ~ Beta(alpha, alpha).
    x: [B, T, C, H, W].  A random rectangular region is replaced across all T.
    Returns mixed (x, y) — y is the area-proportional soft label.
    """
    import torch
    import numpy as np
    if alpha <= 0:
        return x, y
    lam  = float(np.random.beta(alpha, alpha))
    B, T, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)

    # Sample box dimensions proportional to sqrt(1 - lam)
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx    = random.randint(0, W)
    cy    = random.randint(0, H)
    x1    = max(0, cx - cut_w // 2)
    y1    = max(0, cy - cut_h // 2)
    x2    = min(W, cx + cut_w // 2)
    y2    = min(H, cy + cut_h // 2)

    x_m = x.clone()
    x_m[:, :, :, y1:y2, x1:x2] = x[perm, :, :, y1:y2, x1:x2]

    # Recalculate lambda from actual box area
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    y_m = lam_actual * y + (1 - lam_actual) * y[perm]
    return x_m, y_m


# ── EMA helper ────────────────────────────────────────────────────────────────
class ModelEMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay: float = 0.999):
        import copy
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        import torch
        for ema_p, p in zip(self.model.parameters(),
                            model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ── Main Modal function ───────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=20480,           # 20 GB — dataset + model + EMA + optimizer states
    timeout=25200,          # 7 hours — covers resumed run (20 more epochs) + Stage B
    volumes={str(PROC_MOUNT): vol_proc},
)
def run_videomae(skip_finetune: bool = False) -> None:
    """
    Stage A: fine-tune VideoMAE-Base on RWF-2000
             (2,000 videos: 1,600 train + 400 val).
    Stage B: extract mean-pooled embeddings → update every NPZ with vit_embedding.
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    from tqdm import tqdm
    import math

    vol_proc.reload()
    seed_everything(vcfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # ── Load split ────────────────────────────────────────────────────────
    if not SPLIT_CSV.exists():
        raise RuntimeError(
            f"split_v9.csv not found at {SPLIT_CSV}. "
            "Run Phase 1 (preprocess) first."
        )
    split_df = pd.read_csv(str(SPLIT_CSV))

    def get_split(name: str):
        sub    = split_df[split_df["split"] == name]
        paths  = [CACHE_DIR / f"{vid}.npz"
                  for vid in sub["video_id"].tolist()]
        labels = sub["label"].tolist()
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  WARN: {len(missing)} NPZ files missing for split={name}")
            print(f"  WARN: first missing -> {missing[0]}")
        present_paths  = [p for p in paths if p.exists()]
        present_labels = [l for p, l in zip(paths, labels) if p.exists()]
        return present_paths, present_labels

    train_paths, train_labels = get_split("train")
    val_paths,   val_labels   = get_split("val")

    print(f"Train={len(train_paths)}  Val={len(val_paths)}")

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "label":        torch.stack([b["label"]        for b in batch]),
        }

    # ═════════════════════════════════════════════════════════════════════
    # Stage A — Fine-tuning loop  (skipped when skip_finetune=True)
    # ═════════════════════════════════════════════════════════════════════
    if skip_finetune:
        # Assert checkpoint exists before paying the cost of model load
        if not VIT_CKPT.exists():
            raise RuntimeError(
                f"--skip-finetune requested but no checkpoint found at {VIT_CKPT}.\n"
                "Run Stage A first (without --skip-finetune)."
            )
        print(f"\n--skip-finetune: loading checkpoint {VIT_CKPT}")
        ck = torch.load(str(VIT_CKPT), map_location=DEVICE)
        model = build_videomae_model(
            vcfg.model_name, vcfg.embed_dim, vcfg.dropout,
        ).to(DEVICE)
        ema = ModelEMA(model, decay=vcfg.ema_decay)
        ema.model.load_state_dict(ck["ema_state_dict"])
        print(f"  epoch={ck['epoch']}  "
              f"val_macro_f1={ck['val_macro_f1']:.4f}  "
              f"val_roc_auc={ck.get('val_roc_auc', float('nan')):.4f}  "
              f"threshold={ck['threshold']:.4f}")
        print("Skipping Stage A — proceeding directly to Stage B.")
    else:
        # ── Datasets (only needed for Stage A) ────────────────────────────
        RWFVideoDataset = build_vit_dataset_class()
        train_ds = RWFVideoDataset(train_paths, train_labels, augment=True)
        val_ds   = RWFVideoDataset(val_paths,   val_labels,   augment=False)

        train_loader = DataLoader(train_ds, batch_size=vcfg.batch_size,
                                  shuffle=True,  num_workers=vcfg.num_workers,
                                  pin_memory=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=vcfg.batch_size,
                                  shuffle=False, num_workers=vcfg.num_workers,
                                  pin_memory=True, collate_fn=collate_fn)

        # ── Model ─────────────────────────────────────────────────────────
        model = build_videomae_model(
            vcfg.model_name, vcfg.embed_dim, vcfg.dropout,
        ).to(DEVICE)

        ema = ModelEMA(model, decay=vcfg.ema_decay)

        # Layer-wise LR decay: lower ViT layers train with smaller LRs
        param_groups = build_llrd_param_groups(
            model, vcfg.lr, vcfg.layer_decay, vcfg.weight_decay)
        optimizer = AdamW(param_groups, betas=(0.9, 0.95))

        # Warmup + cosine LR schedule implemented as LambdaLR
        total_steps   = vcfg.epochs * len(train_loader) // vcfg.accum_steps
        warmup_steps  = vcfg.warmup_epochs * len(train_loader) // vcfg.accum_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress  = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine    = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_scale = vcfg.lr_min / vcfg.lr
            return max(min_scale, cosine)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # ── Resume from checkpoint if available ───────────────────────────
        start_epoch  = 1
        best_val_f1  = -1.0
        patience_cnt = 0
        global_step  = 0

        if VIT_CKPT.exists():
            print(f"\nResuming from checkpoint: {VIT_CKPT}")
            ck = torch.load(str(VIT_CKPT), map_location=DEVICE)
            model.load_state_dict(ck["ema_state_dict"])
            ema.model.load_state_dict(ck["ema_state_dict"])
            start_epoch  = ck["epoch"] + 1
            best_val_f1  = ck["val_macro_f1"]
            patience_cnt = 0
            vcfg.epochs  = ck["epoch"] + 20
            print(f"  Resumed from epoch {ck['epoch']}  "
                  f"(val_f1={ck['val_macro_f1']:.4f})  "
                  f"  Training will stop at epoch {vcfg.epochs}.")
            for _ in range(start_epoch - 1):
                scheduler.step()
        else:
            print("No checkpoint found — starting from pretrained Kinetics-400 weights.")

        # ── Loss: label-smoothed BCE ──────────────────────────────────────
        def smooth_bce(logits, targets, alpha: float = 0.05):
            smooth_t = targets * (1.0 - alpha) + (1.0 - targets) * alpha
            return F.binary_cross_entropy_with_logits(logits, smooth_t)

        # ── Evaluate helper ───────────────────────────────────────────────
        @torch.no_grad()
        def evaluate(loader, use_ema: bool = True):
            m = ema.model if use_ema else model
            m.eval()
            y_true, y_prob = [], []
            for batch in loader:
                pv  = batch["pixel_values"].to(DEVICE)
                lbl = batch["label"].cpu().numpy().astype(int)
                prb = torch.sigmoid(m(pv)).cpu().numpy()
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
        log_rows = []

        print(f"\n{'='*60}")
        print(f"Stage A: VideoMAE fine-tuning — {vcfg.epochs} epochs, "
              f"lr={vcfg.lr:.1e}, layer_decay={vcfg.layer_decay}, "
              f"warmup={vcfg.warmup_epochs} epochs")
        print(f"  mixup={vcfg.mixup_alpha}  cutmix={vcfg.cutmix_alpha}  "
              f"label_smooth={vcfg.label_smoothing}  ema_decay={vcfg.ema_decay}  "
              f"accum_steps={vcfg.accum_steps}")
        print(f"{'='*60}")

        for epoch in range(start_epoch, vcfg.epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"Epoch {epoch:03d}", leave=False, unit="batch")

            for step, batch in pbar:
                pv  = batch["pixel_values"].to(DEVICE)   # [B, T, 3, H, W]
                lbl = batch["label"].to(DEVICE)

                # Zero gradients only at start of accumulation window
                if step % vcfg.accum_steps == 0:
                    optimizer.zero_grad()

                # Apply mixup or cutmix with equal probability when both are active
                use_mix    = vcfg.mixup_alpha > 0.0
                use_cutmix = vcfg.cutmix_alpha > 0.0
                if use_mix and use_cutmix:
                    if random.random() < 0.5:
                        pv_m, lbl_m = mixup_batch(pv, lbl, alpha=vcfg.mixup_alpha)
                    else:
                        pv_m, lbl_m = cutmix_batch(pv, lbl, alpha=vcfg.cutmix_alpha)
                elif use_mix:
                    pv_m, lbl_m = mixup_batch(pv, lbl, alpha=vcfg.mixup_alpha)
                elif use_cutmix:
                    pv_m, lbl_m = cutmix_batch(pv, lbl, alpha=vcfg.cutmix_alpha)
                else:
                    pv_m, lbl_m = pv, lbl

                logits = model(pv_m)
                # Divide loss by accum_steps so gradients scale correctly
                loss   = smooth_bce(logits, lbl_m, vcfg.label_smoothing) / vcfg.accum_steps
                loss.backward()
                epoch_loss += loss.item() * vcfg.accum_steps   # log unscaled loss

                is_last_batch = (step == len(train_loader) - 1)
                if (step + 1) % vcfg.accum_steps == 0 or is_last_batch:
                    nn.utils.clip_grad_norm_(model.parameters(), vcfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    ema.update(model)
                    global_step += 1

                pbar.set_postfix(loss=f"{loss.item() * vcfg.accum_steps:.4f}")

            avg_loss = epoch_loss / max(len(train_loader), 1)

            # ── Validate ──────────────────────────────────────────────────────
            _, _, val_thr, val_f1, val_auc = evaluate(val_loader, use_ema=True)

            log_rows.append({
                "epoch": epoch, "train_loss": avg_loss,
                "val_macro_f1": val_f1, "val_roc_auc": val_auc,
                "val_threshold": val_thr,
            })
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
                  f"Val F1 {val_f1:.4f} | AUC {val_auc:.4f} | "
                  f"Thr {val_thr:.2f}")

            # ── Checkpoint ────────────────────────────────────────────────────
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
                print(f"  Saved checkpoint (val_f1={val_f1:.4f})")
            elif epoch >= vcfg.min_ckpt_epoch:
                patience_cnt += 1
                if patience_cnt >= vcfg.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Save training log
        pd.DataFrame(log_rows).to_csv(str(VIT_LOG_CSV), index=False)

        # ── Val evaluation with best checkpoint ───────────────────────────────
        print("\nLoading best checkpoint for final val evaluation ...")
        ck = torch.load(str(VIT_CKPT), map_location=DEVICE)
        ema.model.load_state_dict(ck["ema_state_dict"])
        best_thr = float(ck["threshold"])

        y_vt, y_pv, _, _, _ = evaluate(val_loader, use_ema=True)
        if len(y_vt) == 0:
            print("  WARN: val loader yielded 0 samples.")
            val_results = {
                "accuracy":  None,
                "macro_f1":  None,
                "roc_auc":   None,
                "threshold": best_thr,
                "note":      "val loader empty — evaluation skipped",
            }
        else:
            preds = (y_pv >= best_thr).astype(int)
            try:
                auc = float(roc_auc_score(y_vt, y_pv))
            except Exception as e:
                print(f"  WARN: roc_auc_score failed: {e}")
                auc = None
            val_results = {
                "accuracy":  float(accuracy_score(y_vt, preds)),
                "macro_f1":  float(f1_score(y_vt, preds, average="macro",
                                            zero_division=0)),
                "roc_auc":   auc,
                "threshold": best_thr,
            }

        print(f"\n{'='*60}")
        print("Stage A complete — VideoMAE standalone (Ablation C)")
        for k, v in val_results.items():
            print(f"  {k:<15}: {v}")
        print(f"{'='*60}")

        import json
        (PROC_MOUNT / "videomae_test_results.json").write_text(
            json.dumps(val_results, indent=2))
        vol_proc.commit()

    # ═════════════════════════════════════════════════════════════════════
    # Stage B — Embedding extraction
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Stage B: Extracting 768-d embeddings for all 2,000 videos")
    print(f"{'='*60}")

    ema.model.eval()

    # Process all splits together
    all_paths  = train_paths + val_paths
    all_labels = train_labels + val_labels

    # Build dataset over all splits with augment=False for deterministic extraction
    RWFVideoDataset2 = build_vit_dataset_class()
    full_ds = RWFVideoDataset2(all_paths, all_labels, augment=False)
    full_loader = DataLoader(full_ds, batch_size=vcfg.batch_size,
                             shuffle=False, num_workers=vcfg.num_workers,
                             pin_memory=True, collate_fn=collate_fn)

    embeddings_map = {}   # npz_path_str -> np.ndarray [768]
    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Extracting embeddings",
                          unit="batch"):
            pv   = batch["pixel_values"].to(DEVICE)   # [B, T, 3, H, W]
            # Mean-pool all patch tokens; VideoMAE-Base has no CLS token
            embs = ema.model.encoder(pixel_values=pv)\
                       .last_hidden_state.mean(dim=1).cpu().numpy()  # [B, 768]
            for i, emb in enumerate(embs):
                path_str = full_ds.samples[sample_idx]["path"]
                embeddings_map[path_str] = emb.astype(np.float32)
                sample_idx += 1

    # Write vit_embedding into each NPZ using atomic tmp->rename pattern
    print(f"Writing embeddings into {len(embeddings_map)} NPZ files ...")
    write_errors = 0
    for npz_path_str, emb in tqdm(embeddings_map.items(),
                                   desc="Writing NPZ", unit="file"):
        npz_path = Path(npz_path_str)
        try:
            # Load existing arrays
            existing = dict(np.load(str(npz_path), allow_pickle=True))
            existing["vit_embedding"] = emb   # [768] float32 (VideoMAE-Base mean-pool)
            # Atomic write: save to .tmp then rename to avoid partial writes
            tmp = npz_path.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), **existing)
            tmp.rename(npz_path)
        except Exception as e:
            print(f"  ERROR updating {npz_path}: {e}")
            write_errors += 1

    print(f"\nEmbedding extraction complete.")
    print(f"  Wrote:  {len(embeddings_map) - write_errors}")
    print(f"  Errors: {write_errors}")

    vol_proc.commit()
    print("Volume committed. Phase 2 complete.")
    print("The VideoMAE model will not be loaded again during graph training.")


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    print("Guardian Eye V9 — Phase 2: VideoMAE")
    print("Run: modal run trigraph_v9_rwf_videomae.py::run_videomae")
