"""Train the Stage 1 YOLOv8 detector on a merged TACO + KNR-PET-7 dataset (YOLO format).

Example:
    python -m src.train_stage1 --data configs/waste_detection.yaml --model yolov8s.pt --epochs 100

The best checkpoint is copied to models/stage1_yolov8.pt, which the pipeline loads by default.
See data/README.md for how to prepare the dataset.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import STAGE1_WEIGHTS, OUTPUTS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv8 waste detector (Stage 1)")
    p.add_argument("--data", default=str(ROOT / "configs" / "waste_detection.yaml"), help="YOLO data.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="Base weights: yolov8n/s/m/l/x.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 for first GPU, or cpu")
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=0, help="Number of dataloader workers")
    p.add_argument("--name", default="stage1_yolov8")
    p.add_argument("--out", default=str(STAGE1_WEIGHTS))
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install ultralytics")

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, patience=args.patience,
        project=str(OUTPUTS_DIR / "runs"), name=args.name, exist_ok=True,
        workers=args.workers,
    )

    best = Path(model.trainer.best)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nStage 1 weights saved to: {out}")

    metrics = model.val(data=args.data, split="test" if _has_test_split(args.data) else "val")
    print(f"mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")


def _has_test_split(data_yaml: str) -> bool:
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as fh:
            return bool(yaml.safe_load(fh).get("test"))
    except Exception:
        return False


if __name__ == "__main__":
    main()
