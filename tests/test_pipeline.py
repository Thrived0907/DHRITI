"""Pipeline tests using stub models (no weights or ML packages required)."""

from PIL import Image

from src.pipeline import TwoStagePipeline, expand_box
from src.stage1_detector import Detection
from src.visualize import draw_results
from stubs import StubDetector, StubClassifier


def _img(w=400, h=300):
    return Image.new("RGB", (w, h), (120, 130, 140))


def test_expand_box_pads_clips_and_rejects_tiny():
    assert expand_box((100, 100, 200, 200), 400, 300, padding=0.1) == (90, 90, 210, 210)
    assert expand_box((-50, -50, 20, 20), 400, 300, padding=0.0) == (0, 0, 20, 20)
    assert expand_box((10, 10, 15, 15), 400, 300) is None


def test_two_stage_flow_combines_results():
    det = StubDetector([
        Detection(0, "Clear plastic bottle", 0.9, (10, 10, 110, 210)),
        Detection(1, "Drink can", 0.8, (200, 50, 300, 150)),
    ])
    cls = StubClassifier([("Plastic", 0.85), ("Metal", 0.75)])
    result = TwoStagePipeline(det, cls).analyze(_img())

    assert [i.recommendation.final_material for i in result.items] == ["plastic", "metal"]
    assert all(i.recommendation.action == "Recycle" for i in result.items)
    assert result.summary["objects_found"] == 2 and result.summary["recyclable"] == 2
    assert len(cls.crop_sizes) == 2            # batched into a single call
    assert result.stages_active == {"stage1": True, "stage2": True}
    d = result.to_dict()                       # must be JSON-serialisable
    assert d["items"][0]["stage1"]["class_name"] == "Clear plastic bottle"
    assert d["items"][0]["stage2"]["label"] == "Plastic"
    assert d["items"][0]["marketplace"]["suggested_category"] == "recyclable_plastic"
    import json; json.dumps(d)


def test_thresholds_are_forwarded_to_detector():
    det = StubDetector([])
    TwoStagePipeline(det, StubClassifier([("Metal", 0.9)])).analyze(_img(), det_conf=0.5, det_iou=0.3)
    assert det.last_kwargs == {"conf": 0.5, "iou": 0.3}


def test_full_image_fallback_when_nothing_detected():
    cls = StubClassifier([("Metal", 0.9)])
    result = TwoStagePipeline(StubDetector([]), cls).analyze(_img(400, 300))
    assert len(result.items) == 1 and result.items[0].from_full_image_fallback
    assert cls.crop_sizes == [(400, 300)]
    assert result.items[0].recommendation.final_material == "metal"
    assert any("whole image" in n for n in result.notes)


def test_fallback_can_be_disabled():
    result = TwoStagePipeline(StubDetector([]), StubClassifier([("Metal", 0.9)])).analyze(
        _img(), full_image_fallback=False)
    assert result.items == []
    assert result.primary_item() is None


def test_degraded_modes():
    det = StubDetector([Detection(1, "Drink can", 0.9, (10, 10, 110, 110))])
    only1 = TwoStagePipeline(det, None).analyze(_img())
    assert only1.items[0].stage2 is None and only1.items[0].recommendation.final_material == "metal"
    assert any("Stage 2" in n for n in only1.notes)

    only2 = TwoStagePipeline(None, StubClassifier([("Glass", 0.9)])).analyze(_img())
    assert only2.items[0].stage1 is None and only2.items[0].recommendation.final_material == "glass"


def test_no_models_raises():
    import pytest
    with pytest.raises(RuntimeError):
        TwoStagePipeline(None, None, {"stage1": "missing"}).analyze(_img())


def test_conflict_flows_through_to_review_and_primary_item_is_largest():
    det = StubDetector([
        Detection(1, "Drink can", 0.9, (0, 0, 40, 40)),
        Detection(0, "Clear plastic bottle", 0.85, (100, 20, 300, 280)),
    ])
    cls = StubClassifier([("Glass", 0.9), ("Plastic", 0.9)])   # 1st item conflicts with can
    result = TwoStagePipeline(det, cls).analyze(_img())
    assert result.items[0].recommendation.action == "Manual review"
    assert result.summary["needs_review"] == 1
    assert result.primary_item().index == 2                       # bigger, decidable bottle


def test_visualisation_returns_same_size_image():
    det = StubDetector([Detection(0, "Clear plastic bottle", 0.9, (10, 10, 110, 210))])
    img = _img()
    result = TwoStagePipeline(det, StubClassifier([("Plastic", 0.85)])).analyze(img)
    out = draw_results(img, result)
    assert out.size == img.size and out.tobytes() != img.tobytes()
