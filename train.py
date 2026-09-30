"""
TerrainSense - train.py
=======================
Fine-tunes a MobileNetV2 (ImageNet pre-trained) on a local terrain image dataset
and saves the trained weights for the Streamlit web app (app.py).

Usage
-----
    python train.py                          # train with default settings
    python train.py --epochs 15              # more training rounds
    python train.py --batch-size 16          # smaller batches if RAM is tight
    python train.py --unfreeze-last-block    # unfreeze last block for more accuracy

Expected dataset layout (class names are read from the folder names):
    dataset/
    ├── forest/   <- forest photos
    ├── hills/    <- hills photos
    ├── rocky/    <- rocky photos
    ├── grass/    <- grass photos
    └── sand/     <- sand photos
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: works without a display (saves PNGs)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ImageNet statistics - required because our backbone was pre-trained with them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Base folders (relative to this script).
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TerrainSense MobileNetV2 model.")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR),
                        help="Path to dataset folder containing one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Number of training epochs (default: 12).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32).")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data used for validation (default: 0.2).")
    parser.add_argument("--unfreeze-last-block", action="store_true",
                        help="Also train the last feature block (slower, often more accurate).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build train and evaluation transform pipelines.

    Train: light augmentation (RandomHorizontalFlip) + resize to 224x224 + ImageNet normalize.
    Eval : deterministic resize to 224x224 + ImageNet normalize.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),   # MobileNetV2 expects 224x224 RGB input
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),           # HWC [0,255] -> CHW float [0,1]
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def load_datasets(data_dir: Path, train_tf, eval_tf, val_split: float, seed: int):
    """Load images with ImageFolder and split them into train/validation sets."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    # Every immediate subfolder of dataset/ becomes a class.
    class_folders = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if not class_folders:
        raise FileNotFoundError(
            f"No class subfolders found in '{data_dir}'.\n"
            "Create one folder per terrain class and put images inside, e.g.:\n"
            "  dataset/forest/  dataset/hills/  dataset/rocky/  dataset/grass/  dataset/sand/"
        )

    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_tf)
    num_classes = len(full_dataset.classes)

    # Sanity checks: need at least 1 image per class and enough for an 80/20 split.
    if len(full_dataset) < 2:
        raise RuntimeError("Dataset contains fewer than 2 images - nothing to train on.")
    counts = np.bincount([label for _, label in full_dataset.samples])
    empty = [full_dataset.classes[i] for i, c in enumerate(counts) if c == 0]
    if empty:
        raise RuntimeError(f"These class folders are empty: {', '.join(empty)}")
    min_count = int(counts.min())
    if min_count < 2:
        raise RuntimeError(
            "At least 2 images per class are required (the split needs a validation sample). "
            f"Class with fewest images has only {min_count}."
        )

    # Split deterministically: first N images -> train, the rest -> validation.
    val_size = max(1, int(len(full_dataset) * val_split))
    val_size = min(val_size, len(full_dataset) - 1)  # keep at least 1 train sample
    train_size = len(full_dataset) - val_size

    indices = list(range(len(full_dataset)))
    random.Random(seed).shuffle(indices)  # shuffle BEFORE splitting so classes are spread evenly
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # Two ImageFolder instances (same files, different transforms) share the split indices,
    # so training samples get augmentation while validation samples stay deterministic.
    train_dataset = Subset(datasets.ImageFolder(str(data_dir), transform=train_tf), train_idx)
    val_dataset = Subset(datasets.ImageFolder(str(data_dir), transform=eval_tf), val_idx)

    class_names = full_dataset.classes
    print(f"Found {len(train_dataset)} training and {len(val_dataset)} validation images "
          f"across {num_classes} classes: {', '.join(class_names)}")
    return train_dataset, val_dataset, class_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(num_classes: int, unfreeze_last_block: bool = False) -> nn.Module:
    """Create MobileNetV2 with frozen feature layers and a new classification head."""
    weights = models.MobileNet_V2_Weights.DEFAULT  # ImageNet pre-trained backbone
    model = models.mobilenet_v2(weights=weights)

    # Freeze every backbone parameter (transfer learning: keep learned features intact).
    for param in model.features.parameters():
        param.requires_grad = False

    # Optionally unfreeze the last feature block for a bit more accuracy.
    if unfreeze_last_block:
        for param in model.features[-1].parameters():
            param.requires_grad = True
        print("Unfroze the last MobileNetV2 feature block.")

    # Replace the top linear layer so it outputs logits for our terrain classes.
    in_features = model.classifier[1].in_features  # 1280 for MobileNetV2
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Run a single training pass. Returns (average loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # reset gradients
        outputs = model(images)          # forward pass
        loss = criterion(outputs, labels)  # CrossEntropyLoss
        loss.backward()                  # backpropagation
        optimizer.step()                 # update weights

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Run inference over the loader. Returns (average loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes: int) -> np.ndarray:
    """Compute a num_classes x num_classes confusion matrix over the loader."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            matrix[t, p] += 1
    return matrix


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, path: Path) -> None:
    """Save loss and accuracy curves for train/validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val loss")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val accuracy")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves -> {path}")


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    """Save a labelled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (validation set)")

    # Annotate each cell with its count.
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Auto-detect hardware: use CUDA when available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_tf, eval_tf = get_transforms()
    train_dataset, val_dataset, class_names = load_datasets(
        data_dir, train_tf, eval_tf, args.val_split, args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # Build model and move it to the target device.
    model = create_model(len(class_names), args.unfreeze_last_block).to(device)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    # Adam only updates parameters that require gradients (i.e. the new head,
    # plus the last block when --unfreeze-last-block is set).
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    print(f"\nStarting training for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Keep a copy of the best weights so we never save an overfit model.
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
              f"val loss {val_loss:.4f} acc {val_acc:.2%} | "
              f"{elapsed:.1f}s")

    # Restore the best validation checkpoint before saving.
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Save artifacts ---------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "terrain_mobilenetv2.pth"
    torch.save(
        {
            "model_state": model.state_dict(),   # weights
            "classes": class_names,              # label order for inference
            "epochs": args.epochs,
            "val_acc": best_val_acc,
        },
        model_path,
    )
    print(f"Saved model -> {model_path} (best val accuracy: {best_val_acc:.2%})")

    plot_training_curves(history, MODELS_DIR / "training_curves.png")

    val_matrix = confusion_matrix(model, val_loader, device, len(class_names))
    plot_confusion_matrix(val_matrix, class_names, MODELS_DIR / "confusion_matrix.png")

    print("\nDone. Next step: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        # Friendly message instead of a long traceback (e.g. dataset not set up yet).
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
