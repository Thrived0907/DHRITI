"""Fine-tune EfficientNet-B0 on RealWaste (Stage 2).

Expected data layout (one sub-folder per class - this is how RealWaste ships):
    data/realwaste/Cardboard/*.jpg
    data/realwaste/Food Organics/*.jpg
    ...

Example:
    python -m src.train_stage2 --data data/realwaste --epochs 25 --batch-size 32

Writes models/stage2_efficientnet_b0.pt (weights + class list) and outputs/stage2_metrics.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    STAGE2_WEIGHTS, OUTPUTS_DIR, CLS_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


def stratified_split(targets, val_frac: float, test_frac: float, seed: int):
    """Per-class shuffled split -> (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, val, test = [], [], []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n_test, n_val = int(round(len(idx) * test_frac)), int(round(len(idx) * val_frac))
        test += idx[:n_test].tolist()
        val += idx[n_test:n_test + n_val].tolist()
        train += idx[n_test + n_val:].tolist()
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 condition classifier (Stage 2)")
    p.add_argument("--data", default=str(ROOT / "data" / "realwaste"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Head LR; backbone uses lr * 0.1")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet initialisation")
    p.add_argument("--out", default=str(STAGE2_WEIGHTS))
    args = p.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms as T
    except ImportError:
        sys.exit("PyTorch/torchvision not installed. Run: pip install torch torchvision")

    from src.stage2_classifier import build_efficientnet_b0

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        sys.exit(f"Dataset folder not found: {data_dir}\nSee data/README.md for RealWaste setup.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = T.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD))
    train_tf = T.Compose([
        T.RandomResizedCrop(CLS_IMAGE_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(), norm,
    ])
    eval_tf = T.Compose([T.Resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)), T.ToTensor(), norm])

    full_train = datasets.ImageFolder(str(data_dir), transform=train_tf)
    full_eval = datasets.ImageFolder(str(data_dir), transform=eval_tf)
    classes = full_train.classes
    print(f"Found {len(full_train)} images in {len(classes)} classes: {classes}")

    tr_idx, va_idx, te_idx = stratified_split(full_train.targets, args.val_split, args.test_split, args.seed)
    print(f"Split -> train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")

    loader = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=device.type == "cuda")
    train_loader = loader(Subset(full_train, tr_idx), True)
    val_loader = loader(Subset(full_eval, va_idx), False)
    test_loader = loader(Subset(full_eval, te_idx), False)

    model = build_efficientnet_b0(len(classes), pretrained=not args.no_pretrained).to(device)
    head_params = list(model.classifier.parameters())
    head_ids = {id(q) for q in head_params}
    backbone_params = [q for q in model.parameters() if id(q) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * 0.1}, {"params": head_params, "lr": args.lr}],
        weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def evaluate(dl):
        model.eval()
        preds, labels = [], []
        with torch.inference_mode():
            for x, y in dl:
                preds.append(model(x.to(device)).argmax(1).cpu())
                labels.append(y)
        return torch.cat(preds).numpy(), torch.cat(labels).numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        vp, vl = evaluate(val_loader)
        val_acc = float((vp == vl).mean())
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_acc": val_acc})
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "state_dict": model.state_dict(), "classes": classes, "arch": "efficientnet_b0",
                "img_size": CLS_IMAGE_SIZE, "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                "val_acc": val_acc,
            }, out_path)
            flag = "  <- saved"
        print(f"Epoch {epoch:02d}/{args.epochs}  loss {running / max(seen, 1):.4f}  val_acc {val_acc:.4f}{flag}")

    # ---- Final test evaluation with the best checkpoint --------------------------------
    ckpt = torch.load(str(out_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    tp, tl = evaluate(test_loader)
    test_acc = float((tp == tl).mean())

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p_ in zip(tl, tp):
        cm[t, p_] += 1
    per_class = {}
    for i, name in enumerate(classes):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        per_class[name] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                           "support": int(cm[i].sum())}

    ckpt["test_acc"] = test_acc
    torch.save(ckpt, out_path)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "stage2_metrics.json"
    metrics_path.write_text(json.dumps({
        "best_val_acc": best_val, "test_acc": test_acc, "classes": classes,
        "per_class": per_class, "confusion_matrix": cm.tolist(), "history": history,
    }, indent=2), encoding="utf-8")
    print(f"\nTest accuracy: {test_acc:.4f}\nWeights: {out_path}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
