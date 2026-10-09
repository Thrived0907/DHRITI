"""Stub models shared by the pipeline and API tests."""

from src.stage2_classifier import ClassPrediction


class StubDetector:
    weights = "stub"
    class_names = ["Clear plastic bottle", "Drink can"]

    def __init__(self, detections):
        self._d = detections
        self.last_kwargs = None

    def detect(self, image, conf=None, iou=None):
        self.last_kwargs = {"conf": conf, "iou": iou}
        return list(self._d)


class StubClassifier:
    weights = "stub"
    classes = ["Metal", "Plastic"]

    def __init__(self, labels):
        self._labels = labels
        self.crop_sizes = []

    def classify_batch(self, crops):
        self.crop_sizes = [c.size for c in crops]
        out = []
        for i, _ in enumerate(crops):
            label, conf = self._labels[i % len(self._labels)]
            out.append(ClassPrediction(label, conf, {label: conf}))
        return out
