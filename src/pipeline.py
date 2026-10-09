"""Two-stage pipeline: YOLOv8 detect -> crop -> EfficientNet-B0 classify -> decision logic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src import config
from src.decision_logic import Recommendation, decide, dhrti_listing
from src.stage1_detector import Detection, ModelNotFoundError, YoloDetector
from src.stage2_classifier import ClassPrediction, EfficientNetClassifier

Box = Tuple[int, int, int, int]


def expand_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float = config.CROP_PADDING,
    min_size: int = config.MIN_CROP_PIXELS,
) -> Optional[Box]:
    """Pad a box by `padding` (fraction of its size), clip to the image, reject degenerate boxes."""
    x1, y1, x2, y2 = box
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    x1, x2 = x1 - bw * padding, x2 + bw * padding
    y1, y2 = y1 - bh * padding, y2 + bh * padding
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if ix2 - ix1 < min_size or iy2 - iy1 < min_size:
        return None
    return ix1, iy1, ix2, iy2


@dataclass
class ItemResult:
    """Everything known about one object: Stage 1, Stage 2, and the final decision."""

    index: int
    stage1: Optional[Detection]
    stage2: Optional[ClassPrediction]
    recommendation: Recommendation
    crop_box: Box
    from_full_image_fallback: bool = False
    crop: Optional[Image.Image] = field(default=None, repr=False)  # not serialised

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "stage1": self.stage1.to_dict() if self.stage1 else None,
            "stage2": self.stage2.to_dict() if self.stage2 else None,
            "final": self.recommendation.to_dict(),
            "marketplace": dhrti_listing(self.recommendation, self.stage1.class_name if self.stage1 else None),
            "crop_box": list(self.crop_box),
            "from_full_image_fallback": self.from_full_image_fallback,
        }


@dataclass
class AnalysisResult:
    image_size: Tuple[int, int]
    items: List[ItemResult]
    timings_ms: Dict[str, float]
    notes: List[str]
    stages_active: Dict[str, bool]

    @property
    def summary(self) -> Dict[str, Any]:
        actions: Dict[str, int] = {}
        for it in self.items:
            actions[it.recommendation.action] = actions.get(it.recommendation.action, 0) + 1
        n = len(self.items)
        return {
            "objects_found": n,
            "recyclable": sum(1 for i in self.items if i.recommendation.recyclable is True),
            "contamination": sum(1 for i in self.items if i.recommendation.is_contamination),
            "needs_review": sum(1 for i in self.items if i.recommendation.needs_review),
            "actions": actions,
        }

    def primary_item(self) -> Optional[ItemResult]:
        """The dominant item (largest crop among decidable items) - used for single-answer APIs."""
        candidates = [i for i in self.items if i.recommendation.final_material != "unknown"] or self.items
        if not candidates:
            return None

        def area(i: ItemResult) -> int:
            x1, y1, x2, y2 = i.crop_box
            return (x2 - x1) * (y2 - y1)

        return max(candidates, key=area)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "stages_active": self.stages_active,
            "summary": self.summary,
            "items": [i.to_dict() for i in self.items],
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "notes": self.notes,
        }


class TwoStagePipeline:
    """Runs Stage 1 -> Stage 2 -> decision logic. Either stage may be absent (degraded mode)."""

    def __init__(
        self,
        detector: Optional[YoloDetector] = None,
        classifier: Optional[EfficientNetClassifier] = None,
        load_errors: Optional[Dict[str, str]] = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.load_errors = load_errors or {}

    @property
    def ready(self) -> bool:
        return self.detector is not None or self.classifier is not None

    def status(self) -> Dict[str, Any]:
        return {
            "stage1_yolov8": {
                "loaded": self.detector is not None,
                "weights": str(getattr(self.detector, "weights", config.STAGE1_WEIGHTS)),
                "classes": self.detector.class_names if self.detector else None,
                "error": self.load_errors.get("stage1"),
            },
            "stage2_efficientnet_b0": {
                "loaded": self.classifier is not None,
                "weights": str(getattr(self.classifier, "weights", config.STAGE2_WEIGHTS)),
                "classes": self.classifier.classes if self.classifier else None,
                "error": self.load_errors.get("stage2"),
            },
        }

    def analyze(
        self,
        image: Image.Image,
        det_conf: Optional[float] = None,
        det_iou: Optional[float] = None,
        full_image_fallback: bool = config.FULL_IMAGE_FALLBACK,
    ) -> AnalysisResult:
        if not self.ready:
            raise RuntimeError("No models loaded. " + " ".join(self.load_errors.values()))

        image = image.convert("RGB")
        width, height = image.size
        notes: List[str] = []
        timings: Dict[str, float] = {}

        # ---- Stage 1: detection --------------------------------------------------------
        detections: List[Detection] = []
        t0 = time.perf_counter()
        if self.detector is not None:
            detections = self.detector.detect(image, conf=det_conf, iou=det_iou)
        else:
            notes.append("Stage 1 (YOLOv8) is not loaded - classifying the whole image only.")
        timings["stage1_detect"] = (time.perf_counter() - t0) * 1000

        # ---- Build crops ---------------------------------------------------------------
        targets: List[Tuple[Optional[Detection], Box, bool]] = []
        for det in detections:
            box = expand_box(det.box, width, height)
            if box is not None:
                targets.append((det, box, False))
        if not targets and full_image_fallback and self.classifier is not None:
            if self.detector is not None:
                notes.append("YOLOv8 found no objects - Stage 2 classified the whole image instead.")
            targets.append((None, (0, 0, width, height), True))
        elif not targets:
            notes.append("No objects detected.")

        crops = [image.crop(box) for _, box, _ in targets]

        # ---- Stage 2: classification (single batched pass) ------------------------------
        preds: List[Optional[ClassPrediction]] = [None] * len(crops)
        t1 = time.perf_counter()
        if self.classifier is not None and crops:
            preds = list(self.classifier.classify_batch(crops))
        elif self.classifier is None and self.detector is not None:
            notes.append("Stage 2 (EfficientNet-B0) is not loaded - decisions use Stage 1 only.")
        timings["stage2_classify"] = (time.perf_counter() - t1) * 1000

        # ---- Decision logic -------------------------------------------------------------
        items: List[ItemResult] = []
        for i, ((det, box, fallback), crop, pred) in enumerate(zip(targets, crops, preds), start=1):
            rec = decide(
                stage1_class=det.class_name if det else None,
                stage1_conf=det.confidence if det else None,
                stage2_label=pred.label if pred else None,
                stage2_conf=pred.confidence if pred else None,
            )
            items.append(ItemResult(i, det, pred, rec, box, fallback, crop))

        timings["total"] = sum(timings.values())
        return AnalysisResult(
            image_size=(width, height),
            items=items,
            timings_ms=timings,
            notes=notes,
            stages_active={"stage1": self.detector is not None, "stage2": self.classifier is not None},
        )


def load_default_pipeline(allow_partial: bool = True) -> TwoStagePipeline:
    """Load both stages from the default checkpoint paths.

    With allow_partial=True a missing/broken stage is recorded in `load_errors` and the pipeline
    runs in degraded mode; with False the first failure is raised.
    """
    detector = classifier = None
    errors: Dict[str, str] = {}

    try:
        detector = YoloDetector()
    except Exception as err:
        if not allow_partial:
            raise
        errors["stage1"] = str(err)

    try:
        classifier = EfficientNetClassifier()
    except Exception as err:
        if not allow_partial:
            raise
        errors["stage2"] = str(err)

    return TwoStagePipeline(detector, classifier, errors)
