"""API tests - skipped automatically if FastAPI/httpx/python-multipart are not installed."""

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import api  # noqa: E402
from src.pipeline import TwoStagePipeline  # noqa: E402
from src.stage1_detector import Detection  # noqa: E402
from stubs import StubClassifier, StubDetector  # noqa: E402


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (100, 110, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _client(detections, labels, monkeypatch):
    pipe = TwoStagePipeline(StubDetector(detections), StubClassifier(labels))
    monkeypatch.setattr(api, "get_pipeline", lambda reload=False: pipe)
    return TestClient(api.app)


def test_predict_returns_both_stages_and_legacy_fields(monkeypatch):
    client = _client([Detection(0, "Clear plastic bottle", 0.9, (10, 10, 150, 200))],
                     [("Plastic", 0.8)], monkeypatch)
    r = client.post("/api/predict?annotated=true", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["items"][0]["stage1"]["class_name"] == "Clear plastic bottle"
    assert body["items"][0]["stage2"]["label"] == "Plastic"
    assert body["predicted_material"] == "plastic" and body["is_recyclable"] is True
    assert body["suggested_category"] == "recyclable_plastic"
    assert body["annotated_image_base64"]


def test_rejects_non_image(monkeypatch):
    client = _client([], [("Metal", 0.9)], monkeypatch)
    r = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_503_when_no_models(monkeypatch):
    pipe = TwoStagePipeline(None, None, {"stage1": "missing"})
    monkeypatch.setattr(api, "get_pipeline", lambda reload=False: pipe)
    r = TestClient(api.app).post("/api/predict", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 503
