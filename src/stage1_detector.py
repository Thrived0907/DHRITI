"""Stage 1 - YOLOv8 waste object detector (trained on TACO + KNR-PET-7)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

from PIL import Image

from src.config import (
    STAGE1_WEIGHTS, DET_CONF_THRESHOLD, DET_IOU_THRESHOLD, DET_IMAGE_SIZE, DET_MAX_OBJECTS,
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a stage checkpoint is missing."""


@dataclass
class Detection:
    """One Stage 1 result: class, bounding box, and confidence score."""

    class_id: int
    class_name: str
    confidence: float                       # 0-1
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original-image pixels

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        d["box"] = [round(v, 1) for v in self.box]
        return d


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model."""

    def __init__(
        self,
        weights: Union[str, Path] = STAGE1_WEIGHTS,
        conf: float = DET_CONF_THRESHOLD,
        iou: float = DET_IOU_THRESHOLD,
        imgsz: int = DET_IMAGE_SIZE,
        max_det: int = DET_MAX_OBJECTS,
        device: Optional[str] = None,
    ):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise ModelNotFoundError(
                f"Stage 1 YOLOv8 weights not found at {self.weights}.\n"
                "Train them with `python -m src.train_stage1 ...` (see README) or set STAGE1_WEIGHTS."
            )
        try:
            from ultralytics import YOLO
        except ImportError as err:  # pragma: no cover
            raise ImportError("Stage 1 needs the 'ultralytics' package: pip install ultralytics") from err

        self.model = YOLO(str(self.weights))
        self.conf, self.iou, self.imgsz, self.max_det, self.device = conf, iou, imgsz, max_det, device

    @property
    def class_names(self) -> List[str]:
        names = self.model.names
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def detect(
        self,
        image: Image.Image,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Detection]:
        """Detect waste objects. Returns detections sorted by confidence (highest first)."""
        results = self.model.predict(
            source=image.convert("RGB"),  # PIL input is treated as RGB by Ultralytics
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        names = result.names
        detections: List[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(cls_id, str(names[cls_id]), float(box.conf[0]), (x1, y1, x2, y2))
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
