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
import gc
import logging
import os
import shutil
import time
import urllib.request
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
    logger.info("Dataset: %s, classes: %d, size: %d", args.dataset, n_classes, args.size)
    check_download(info, args.dataset, args.size, medmnist.info.DEFAULT_ROOT)

    # Which (split, encoder) files are still missing? Only those splits get loaded.
    todo = {
        split: [
            enc for enc in args.encoders
            if args.force or not (FEATURE_DIR / f"{args.dataset}_{enc}_{split}_X.npy").exists()
        ]
        for split in ("pool", "val", "test")
    }

    # One split in memory at a time; from train only the pool rows are kept.
    for split_name, encoders in todo.items():
        if not encoders:
            logger.info("[%s] all requested encoders already extracted, skipping.", split_name)
            continue
        if split_name == "pool":
            imgs, labels = load_pool(DataClass, args, n_classes)
        else:
            imgs, labels = load_split(DataClass, split_name, args.size)
        logger.info(
            "[%s] %d images, class counts %s", split_name, len(labels),
            np.bincount(labels, minlength=n_classes).tolist(),
        )
        for enc_key in encoders:
            prefix = FEATURE_DIR / f"{args.dataset}_{enc_key}_{split_name}"
            model, tfm, feat_dim = load_encoder(enc_key, device)
            t0 = time.time()
            dset = MedMNISTFeatureDataset(imgs, labels, tfm)
            feats = extract_features(model, enc_key, dset, args.batch_size, device)
            norms = np.linalg.norm(feats, axis=1)
            if feats.shape[1] != feat_dim or not np.allclose(norms, 1.0, atol=1e-3):
                raise RuntimeError(
                    f"[{enc_key}] {split_name}: bad features shape={feats.shape} "
                    f"norm range=({norms.min():.4f}, {norms.max():.4f})"
                )
            np.save(f"{prefix}_X.npy", feats)
            np.save(f"{prefix}_Y.npy", labels)
            logger.info(
                "[%s] %s: shape=%s, norm min/max=%.4f/%.4f, %.1f s (%.1f img/s)",
                enc_key, split_name, feats.shape, norms.min(), norms.max(),
                time.time() - t0, len(labels) / max(time.time() - t0, 1e-9),
            )
            del model
        del imgs, labels
        gc.collect()


def check_download(info: dict, dataset: str, size: int, root: str) -> None:
    """Log the download size and refuse to start if free disk is too small."""
    size_flag = "" if size == 28 else f"_{size}"
    local = Path(root) / f"{dataset}{size_flag}.npz"
    if local.exists():
        logger.info("Dataset file already present: %s (%.0f MB)", local, local.stat().st_size / 1e6)
        return
    url = info["url" if size == 28 else f"url_{size}"]
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            n_bytes = int(resp.headers.get("Content-Length", 0))
    except OSError as e:
        logger.warning("Could not read download size (%s); continuing.", e)
        return
    free = shutil.disk_usage(root).free
    logger.info("Download size: %.0f MB; free disk at %s: %.0f MB", n_bytes / 1e6, root, free / 1e6)
    if n_bytes and free < n_bytes * 1.5:
        raise SystemExit(
            f"Not enough disk for {dataset} size={size}: need ~{n_bytes * 1.5 / 1e6:.0f} MB. "
            "Free space or use --size 128."
        )


def load_split(DataClass, split: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one MedMNIST split as (uint8 images, int64 labels)."""
    data = DataClass(split=split, size=size, download=True, as_rgb=True, mmap_mode="r")
    return data.imgs, data.labels.squeeze().astype(np.int64)


def load_pool(DataClass, args: argparse.Namespace, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Load train, keep only the fixed-seed pool rows, and free the rest."""
    train_imgs, train_labels = load_split(DataClass, "train", args.size)
    logger.info("[train] %d images (only the pool is kept)", len(train_labels))
    pool_idx_path = FEATURE_DIR / f"{args.dataset}_pool_idx.npy"
    if pool_idx_path.exists() and not args.force:
        pool_idx = np.load(pool_idx_path)
        logger.info("Loaded existing pool indices: %s", pool_idx_path)
    else:
        pool_idx = build_pool_indices(train_labels, n_classes, args.pool_per_class, args.pool_seed)
        np.save(pool_idx_path, pool_idx)
        logger.info("Saved pool indices: %d total (%d classes)", len(pool_idx), n_classes)
    pool_imgs = np.ascontiguousarray(train_imgs[pool_idx])
    pool_labels = train_labels[pool_idx]
    del train_imgs, train_labels
    gc.collect()
    return pool_imgs, pool_labels


def _run_smoke(args: argparse.Namespace) -> None:
    """Write tiny synthetic feature files for testing."""
    smoke_dir = Path("smoke") / "features"  # never the real features/ dir
    logger.info("[smoke] Writing synthetic feature files to %s ...", smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
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
            np.save(smoke_dir / f"{prefix}_{split_name}_X.npy", X)
            np.save(smoke_dir / f"{prefix}_{split_name}_Y.npy", Y)
    pool_idx = np.arange(50 * n_classes, dtype=np.int64)
    np.save(smoke_dir / f"{args.dataset}_pool_idx.npy", pool_idx)
    logger.info("[smoke] Done.")


if __name__ == "__main__":
    main()
