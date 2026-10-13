"""
extract_features.py — extract and save L2-normalised features per encoder.

Usage:
  python extract_features.py --dataset bloodmnist --encoders resnet50 dinov2_s clip_b16
                              --size 224 --device cuda --pool-per-class 500

  Outputs to features/:
    {dataset}_{encoder}_pool_X.npy
    {dataset}_{encoder}_pool_Y.npy
    {dataset}_{encoder}_val_X.npy
    {dataset}_{encoder}_val_Y.npy
    {dataset}_{encoder}_test_X.npy
    {dataset}_{encoder}_test_Y.npy
    {dataset}_pool_idx.npy          (shared across encoders, fixed seed)
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR = Path("features")

# Preprocessing constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD = [0.2686, 0.2613, 0.2758]


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class MedMNISTFeatureDataset(Dataset):
    """Wraps a MedMNIST numpy array with a transform."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        # images: (N, H, W) or (N, H, W, C) uint8
        self.images = images
        self.labels = labels.squeeze().astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        # Convert to PIL-like uint8 tensor then apply torchvision transform
        from PIL import Image
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor = self.transform(pil_img)
        return tensor, int(self.labels[idx])


# ---------------------------------------------------------------------------
# Encoder loaders
# ---------------------------------------------------------------------------

def load_encoder(encoder_key: str, device: torch.device):
    """Return (model, transform, feature_dim)."""
    if encoder_key == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        # Remove the classification head; keep avgpool
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval().to(device)
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return model, tfm, 2048

    elif encoder_key == "dinov2_s":
        logger.info("Loading DINOv2 ViT-S/14 from torch.hub ...")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        model.eval().to(device)
        tfm = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return model, tfm, 384

    elif encoder_key == "clip_b16":
        logger.info("Loading CLIP ViT-B/16 via open_clip ...")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        model.eval().to(device)
        # Use CLIP's own preprocessing (includes CLIP mean/std)
        return model, preprocess, 512

    elif encoder_key == "biomedclip":
        logger.info("Loading BiomedCLIP via open_clip ...")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-224-vit-b-16"
        )
        model.eval().to(device)
        return model, preprocess, 512

    else:
        raise ValueError(f"Unknown encoder: {encoder_key}")


@torch.no_grad()
def extract_features(
    model,
    encoder_key: str,
    dataset_obj: MedMNISTFeatureDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract and L2-normalise features."""
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats: list[np.ndarray] = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        if encoder_key == "resnet50":
            feats = model(imgs).squeeze(-1).squeeze(-1)
        elif encoder_key in ("dinov2_s",):
            feats = model(imgs)
        elif encoder_key in ("clip_b16", "biomedclip"):
            feats = model.encode_image(imgs)
        else:
            feats = model(imgs)
        feats = F.normalize(feats.float(), dim=1)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pool index selection
# ---------------------------------------------------------------------------

def build_pool_indices(
    labels: np.ndarray,
    n_classes: int,
    pool_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample up to pool_per_class indices per class from train set."""
    rng = np.random.default_rng(seed)
    idx_list: list[int] = []
    labels = labels.squeeze().astype(np.int64)
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) < pool_per_class:
            logger.warning(
                "Class %d: only %d train images, fewer than %d requested.",
                c, len(class_idx), pool_per_class,
            )
            chosen = class_idx
        else:
            chosen = rng.choice(class_idx, pool_per_class, replace=False)
        idx_list.extend(chosen.tolist())
    return np.array(idx_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen encoder features.")
    parser.add_argument("--dataset", default="bloodmnist")
    parser.add_argument(
        "--encoders", nargs="+", default=["resnet50", "dinov2_s", "clip_b16"]
    )
    parser.add_argument("--size", type=int, default=224, choices=[28, 64, 128, 224])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool-per-class", type=int, default=500)
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-extract even if files exist.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny synthetic features.")
    args = parser.parse_args()

    FEATURE_DIR.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    if args.smoke:
        _run_smoke(args)
        return

    # --- Load MedMNIST ---
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        logger.error("medmnist not installed. Run: pip install medmnist")
        raise

    info = INFO[args.dataset]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    logger.info("Dataset: %s, classes: %d", args.dataset, n_classes)

    # Check download size hint
    logger.info("Loading %s size=%d (train split) ...", args.dataset, args.size)
    train_data = DataClass(split="train", size=args.size, download=True, as_rgb=True)
    val_data = DataClass(split="val", size=args.size, download=True, as_rgb=True)
    test_data = DataClass(split="test", size=args.size, download=True, as_rgb=True)

    train_imgs = train_data.imgs  # (N, H, W, C) uint8
    train_labels = train_data.labels
    val_imgs = val_data.imgs
    val_labels = val_data.labels
    test_imgs = test_data.imgs
    test_labels = test_data.labels

    logger.info(
        "Splits: train=%d, val=%d, test=%d", len(train_imgs), len(val_imgs), len(test_imgs)
    )

    # --- Pool indices (shared across encoders) ---
    pool_idx_path = FEATURE_DIR / f"{args.dataset}_pool_idx.npy"
    if pool_idx_path.exists() and not args.force:
        pool_idx = np.load(pool_idx_path)
        logger.info("Loaded existing pool indices: %s", pool_idx_path)
    else:
        pool_idx = build_pool_indices(train_labels, n_classes, args.pool_per_class, args.pool_seed)
        np.save(pool_idx_path, pool_idx)
        logger.info("Saved pool indices: %d total (%d classes)", len(pool_idx), n_classes)

    pool_imgs = train_imgs[pool_idx]
    pool_labels = train_labels[pool_idx]

    # --- Extract per encoder ---
    for enc_key in args.encoders:
        prefix = f"{args.dataset}_{enc_key}"
        paths = {
            "pool_X": FEATURE_DIR / f"{prefix}_pool_X.npy",
            "pool_Y": FEATURE_DIR / f"{prefix}_pool_Y.npy",
            "val_X": FEATURE_DIR / f"{prefix}_val_X.npy",
            "val_Y": FEATURE_DIR / f"{prefix}_val_Y.npy",
            "test_X": FEATURE_DIR / f"{prefix}_test_X.npy",
            "test_Y": FEATURE_DIR / f"{prefix}_test_Y.npy",
        }
        if all(p.exists() for p in paths.values()) and not args.force:
            logger.info("[%s] All feature files exist, skipping (use --force to re-extract).", enc_key)
            continue

        logger.info("[%s] Loading encoder ...", enc_key)
        model, tfm, feat_dim = load_encoder(enc_key, device)
        logger.info("[%s] Feature dim: %d", enc_key, feat_dim)

        t0 = time.time()
        for split_name, imgs, labels in [
            ("pool", pool_imgs, pool_labels),
            ("val", val_imgs, val_labels),
            ("test", test_imgs, test_labels),
        ]:
            x_path = FEATURE_DIR / f"{prefix}_{split_name}_X.npy"
            y_path = FEATURE_DIR / f"{prefix}_{split_name}_Y.npy"
            if x_path.exists() and not args.force:
                logger.info("[%s] %s already extracted, skipping.", enc_key, split_name)
                continue
            dset = MedMNISTFeatureDataset(imgs, labels, tfm)
            feats = extract_features(model, enc_key, dset, args.batch_size, device)
            lbls = labels.squeeze().astype(np.int64)
            np.save(x_path, feats)
            np.save(y_path, lbls)
            logger.info(
                "[%s] %s: shape=%s, norm_mean=%.4f",
                enc_key, split_name, feats.shape,
                np.linalg.norm(feats, axis=1).mean(),
            )
        elapsed = time.time() - t0
        logger.info("[%s] Done in %.1f s.", enc_key, elapsed)
        del model


def _run_smoke(args: argparse.Namespace) -> None:
    """Write tiny synthetic feature files for testing."""
    logger.info("[smoke] Writing synthetic feature files ...")
    FEATURE_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    n_classes = 8
    D = 64
    for enc_key in args.encoders:
        prefix = f"{args.dataset}_{enc_key}"
        for split_name, n_per_class in [("pool", 50), ("val", 30), ("test", 30)]:
            n = n_per_class * n_classes
            X = rng.standard_normal((n, D)).astype(np.float32)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)
            np.save(FEATURE_DIR / f"{prefix}_{split_name}_X.npy", X)
            np.save(FEATURE_DIR / f"{prefix}_{split_name}_Y.npy", Y)
    pool_idx = np.arange(50 * n_classes, dtype=np.int64)
    np.save(FEATURE_DIR / f"{args.dataset}_pool_idx.npy", pool_idx)
    logger.info("[smoke] Done.")


if __name__ == "__main__":
    main()
