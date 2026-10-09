"""Stage 2 - EfficientNet-B0 classifier for cropped waste objects (trained on RealWaste)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict, Any

from PIL import Image, ImageOps

from src.config import (
    STAGE2_WEIGHTS, CLS_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, PAD_COLOR, REALWASTE_CLASSES,
)
from src.stage1_detector import ModelNotFoundError


def _require_torch():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError as err:  # pragma: no cover
        raise ImportError("Stage 2 needs PyTorch + torchvision: pip install torch torchvision") from err


def build_efficientnet_b0(num_classes: int, pretrained: bool = False):
    """EfficientNet-B0 with a fresh classification head. `pretrained` loads ImageNet weights."""
    torch, torchvision = _require_torch()
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def make_eval_transform(size: int = CLS_IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Aspect-preserving letterbox -> tensor -> ImageNet normalisation (used at inference)."""
    _, torchvision = _require_torch()
    from torchvision import transforms as T

    return T.Compose([
        T.Lambda(lambda im: ImageOps.pad(im.convert("RGB"), (size, size), color=PAD_COLOR)),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ])


@dataclass
class ClassPrediction:
    """One Stage 2 result: condition/quality class with full probability distribution."""

    label: str
    confidence: float                 # 0-1
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        d["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        return d


class EfficientNetClassifier:
    """Loads a checkpoint written by `src/train_stage2.py` and classifies image crops."""

    def __init__(self, weights: Union[str, Path] = STAGE2_WEIGHTS, device: Optional[str] = None):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise ModelNotFoundError(
                f"Stage 2 EfficientNet-B0 weights not found at {self.weights}.\n"
                "Train them with `python -m src.train_stage2 ...` (see README) or set STAGE2_WEIGHTS."
            )
        torch, _ = _require_torch()
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(str(self.weights), map_location=self.device, weights_only=True)
        self.classes: List[str] = list(ckpt.get("classes") or REALWASTE_CLASSES)
        self.img_size: int = int(ckpt.get("img_size", CLS_IMAGE_SIZE))

        self.model = build_efficientnet_b0(len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = make_eval_transform(
            self.img_size, ckpt.get("mean", IMAGENET_MEAN), ckpt.get("std", IMAGENET_STD)
        )

    def classify_batch(self, crops: Sequence[Image.Image]) -> List[ClassPrediction]:
        """Classify several crops in a single forward pass."""
        if not crops:
            return []
        torch = self.torch
        batch = torch.stack([self.transform(c) for c in crops]).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        out: List[ClassPrediction] = []
        for row in probs:
            idx = int(row.argmax())
            out.append(ClassPrediction(
                label=self.classes[idx],
                confidence=float(row[idx]),
                probabilities={c: float(p) for c, p in zip(self.classes, row)},
            ))
        return out

    def classify(self, crop: Image.Image) -> ClassPrediction:
        return self.classify_batch([crop])[0]
