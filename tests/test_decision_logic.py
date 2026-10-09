import pytest

from src.decision_logic import (
    decide, dhrti_listing, stage1_family, stage2_family, is_nonrecyclable_plastic,
)


@pytest.mark.parametrize("name,family", [
    ("Clear plastic bottle", "plastic"), ("Glass bottle", "glass"), ("Drink can", "metal"),
    ("Pop tab", "metal"), ("Paper cup", "paper"), ("Disposable plastic cup", "plastic"),
    ("Food waste", "organic"), ("Cigarette", "other"), ("PET_clear", "plastic"),
    ("Drink carton", "paper"), ("Unlabeled litter", "unknown"), (None, "unknown"),
])
def test_stage1_family_mapping(name, family):
    assert stage1_family(name) == family


@pytest.mark.parametrize("label,family", [
    ("Cardboard", "paper"), ("Paper", "paper"), ("Food Organics", "organic"),
    ("Vegetation", "organic"), ("Textile Trash", "textile"),
    ("Miscellaneous Trash", "other"), ("textile_trash", "textile"), ("bogus", "unknown"),
])
def test_stage2_family_mapping(label, family):
    assert stage2_family(label) == family


def test_soft_plastic_detection():
    assert is_nonrecyclable_plastic("Plastic bag & wrapper")
    assert is_nonrecyclable_plastic("Styrofoam piece")
    assert not is_nonrecyclable_plastic("Clear plastic bottle")


def test_agreement_confirms_and_recycles():
    r = decide("Clear plastic bottle", 0.9, "Plastic", 0.8)
    assert r.agreement is True and r.source == "both agree"
    assert r.action == "Recycle" and r.recyclable is True and not r.needs_review
    assert r.confidence == pytest.approx((0.9 * 0.8) ** 0.5)


def test_soft_plastic_is_not_recyclable_even_when_stage2_says_plastic():
    r = decide("Plastic bag & wrapper", 0.8, "Plastic", 0.9)
    assert r.action == "Landfill" and r.recyclable is False and r.is_contamination


def test_conflict_stage1_trusted():
    r = decide("Drink can", 0.92, "Plastic", 0.40)
    assert r.final_material == "metal" and r.source == "stage 1" and r.agreement is False


def test_conflict_stage2_trusted():
    r = decide("Drink can", 0.30, "Glass", 0.85)
    assert r.final_material == "glass" and r.source == "stage 2"


@pytest.mark.parametrize("c1,c2", [(0.9, 0.9), (0.3, 0.3)])
def test_unresolvable_conflict_goes_to_manual_review(c1, c2):
    r = decide("Drink can", c1, "Glass", c2)
    assert r.action == "Manual review" and r.recyclable is None and r.needs_review


def test_stage2_only_and_stage1_only():
    assert decide(stage2_label="Cardboard", stage2_conf=0.9).final_material == "paper"
    only1 = decide(stage1_class="Glass bottle", stage1_conf=0.7)
    assert only1.final_material == "glass" and only1.agreement is None


def test_plastic_without_stage1_carries_verification_warning():
    r = decide(stage2_label="Plastic", stage2_conf=0.9)
    assert "not verified" in r.reason


def test_organic_and_textile_actions():
    assert decide("Food waste", 0.8, "Food Organics", 0.8).action == "Compost"
    assert decide(stage2_label="Textile Trash", stage2_conf=0.9).action == "Reuse / Drop-off"


def test_low_confidence_is_flagged():
    r = decide("Drink can", 0.3, "Metal", 0.3)
    assert r.action == "Recycle" and r.needs_review


def test_nothing_usable():
    r = decide()
    assert r.action == "Manual review" and r.final_material == "unknown"


def test_dhrti_mapping():
    assert dhrti_listing(decide("Clear plastic bottle", 0.9, "Plastic", 0.9))["suggested_category"] == "recyclable_plastic"
    assert dhrti_listing(decide("Drink can", 0.9, "Metal", 0.9))["suggested_category"] == "metal"
    assert dhrti_listing(decide("Pizza box", 0.9, "Cardboard", 0.9))["suggested_category"] == "paper"
    film = dhrti_listing(decide("Plastic film", 0.9, "Plastic", 0.9))
    assert film["suggested_category"] is None and film["is_contamination"] and film["warning"]
    glass = dhrti_listing(decide("Glass bottle", 0.9, "Glass", 0.9))
    assert glass["suggested_category"] is None and not glass["is_contamination"]
