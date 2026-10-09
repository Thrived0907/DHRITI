"""Decision logic: fuse Stage 1 (YOLOv8) and Stage 2 (EfficientNet-B0) into a recycling recommendation.

Pure Python - no ML dependencies - so it can be unit-tested and tuned without model weights.

How it works
------------
1. Both stages' class names are mapped to a common *material family*
   (paper, glass, metal, plastic, organic, textile, other).
2. If the stages agree, the family is confirmed (confidence = geometric mean of both).
3. If they disagree, the stage that is confident (>= STRONG_CONFIDENCE) wins when the
   other one is not. If neither or both are confident, the item is sent to manual review.
4. The final family is translated into an action (recycle / compost / reuse / landfill),
   a bin, and a marketplace category for the Dhrti B2B platform.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from src.config import STRONG_CONFIDENCE, MIN_FINAL_CONFIDENCE

# ---------------------------------------------------------------------------
# Material families
# ---------------------------------------------------------------------------
PAPER, GLASS, METAL, PLASTIC = "paper", "glass", "metal", "plastic"
ORGANIC, TEXTILE, OTHER, UNKNOWN = "organic", "textile", "other", "unknown"

# Stage 2 (RealWaste) label -> family
STAGE2_TO_FAMILY: Dict[str, str] = {
    "cardboard": PAPER,
    "paper": PAPER,
    "glass": GLASS,
    "metal": METAL,
    "plastic": PLASTIC,
    "food organics": ORGANIC,
    "vegetation": ORGANIC,
    "textile trash": TEXTILE,
    "miscellaneous trash": OTHER,
}

# Stage 1 class names differ per dataset (TACO has ~60 classes, KNR-PET-7 has PET sub-types),
# so instead of hard-coding them we map by keywords. Rules are checked in order.
_GLASS_WORDS = {"glass"}
_METAL_WORDS = {"metal", "aluminium", "aluminum", "can", "cans", "tin", "foil", "tab", "aerosol", "scrap"}
_PLASTIC_EXPLICIT = {
    "plastic", "pet", "hdpe", "pp", "ps", "pvc", "polystyrene", "styrofoam", "foam",
    "polypropylene", "polyethylene",
}
_PAPER_WORDS = {
    "paper", "carton", "cardboard", "newspaper", "tetra", "napkin", "tissue",
    "magazine", "box", "corrugated", "envelope",
}
_PLASTIC_IMPLICIT = {
    "bottle", "bottles", "bag", "wrapper", "film", "straw", "lid", "cup", "container",
    "packaging", "tub", "blister", "rings", "cutlery", "utensils", "sachet", "pouch",
}
_ORGANIC_WORDS = {"food", "fruit", "peel", "vegetable", "organic", "leaf", "leaves", "banana", "vegetation"}
_TEXTILE_WORDS = {"textile", "cloth", "clothes", "clothing", "fabric", "shoe", "shoes", "rag"}
_OTHER_WORDS = {"cigarette", "cigarettes", "butt", "diaper", "battery", "misc", "miscellaneous", "trash"}

# Plastic sub-types that curbside programs generally do NOT accept
_NONRECYCLABLE_PLASTIC_WORDS = {
    "film", "bag", "bags", "wrapper", "wrappers", "styrofoam", "foam", "polystyrene", "ps",
    "straw", "straws", "cutlery", "utensils", "sachet", "pouch", "rings", "blister", "packet",
}


def _words(name: str) -> set:
    return set(re.findall(r"[a-z0-9]+", name.lower()))


def stage1_family(class_name: Optional[str]) -> str:
    """Map a Stage 1 detector class name (TACO / KNR-PET-7 / custom) to a material family."""
    if not class_name:
        return UNKNOWN
    w = _words(class_name)
    if w & _GLASS_WORDS:
        return GLASS
    if w & _METAL_WORDS:
        return METAL
    if w & _PLASTIC_EXPLICIT:
        return PLASTIC
    if w & _PAPER_WORDS:
        return PAPER
    if w & _PLASTIC_IMPLICIT:
        return PLASTIC
    if w & _ORGANIC_WORDS:
        return ORGANIC
    if w & _TEXTILE_WORDS:
        return TEXTILE
    if w & _OTHER_WORDS:
        return OTHER
    return UNKNOWN


def stage2_family(label: Optional[str]) -> str:
    """Map a Stage 2 (RealWaste) label to a material family."""
    if not label:
        return UNKNOWN
    key = re.sub(r"[_\-]+", " ", label.strip().lower())
    return STAGE2_TO_FAMILY.get(key, UNKNOWN)


def is_nonrecyclable_plastic(class_name: Optional[str]) -> bool:
    """True when the Stage 1 name indicates soft/foam plastic (bags, film, wrappers, styrofoam...)."""
    return bool(class_name and (_words(class_name) & _NONRECYCLABLE_PLASTIC_WORDS))


# ---------------------------------------------------------------------------
# Family -> action table
# ---------------------------------------------------------------------------
# action, bin, recyclable, is_contamination (for a recycling stream), reason
_ACTIONS = {
    PAPER:   ("Recycle", "Paper & cardboard recycling", True, False,
              "Clean, dry paper and cardboard are widely recyclable."),
    GLASS:   ("Recycle", "Glass recycling", True, False,
              "Glass containers are recyclable; keep them separate from other materials."),
    METAL:   ("Recycle", "Metal recycling", True, False,
              "Metal cans, foil and scrap are highly recyclable."),
    PLASTIC: ("Recycle", "Plastic recycling", True, False,
              "Rigid plastic is recyclable; check the resin code with your local program."),
    ORGANIC: ("Compost", "Organic / compost", False, True,
              "Food and plant matter should be composted, not put in dry recycling."),
    TEXTILE: ("Reuse / Drop-off", "Textile collection point", False, True,
              "Textiles jam sorting machinery; donate or use a textile collection."),
    OTHER:   ("Landfill", "General waste", False, True,
              "This item is not accepted in standard recycling streams."),
}
_SOFT_PLASTIC = ("Landfill", "General waste (soft-plastic drop-off if available)", False, True,
                 "Soft plastic, film and foam are not accepted in curbside recycling.")


@dataclass
class Recommendation:
    """Final decision for one detected object."""

    action: str                    # Recycle | Compost | Reuse / Drop-off | Landfill | Manual review
    bin: str
    final_material: str            # material family the decision is based on
    recyclable: Optional[bool]     # None when undecidable (manual review)
    is_contamination: bool         # would contaminate a recycling stream
    confidence: float              # 0-1 combined confidence
    agreement: Optional[bool]      # True/False if both stages ran, None if only one did
    source: str                    # "both agree" | "stage 1" | "stage 2" | "conflict"
    needs_review: bool
    reason: str
    stage1_family: str = UNKNOWN
    stage2_family: str = UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        return d


def decide(
    stage1_class: Optional[str] = None,
    stage1_conf: Optional[float] = None,
    stage2_label: Optional[str] = None,
    stage2_conf: Optional[float] = None,
) -> Recommendation:
    """Fuse the two stage outputs (confidences as 0-1 floats) into one Recommendation."""
    f1 = stage1_family(stage1_class) if stage1_class else UNKNOWN
    f2 = stage2_family(stage2_label) if stage2_label else UNKNOWN
    c1 = float(stage1_conf) if stage1_conf is not None else 0.0
    c2 = float(stage2_conf) if stage2_conf is not None else 0.0

    have1 = f1 != UNKNOWN
    have2 = f2 != UNKNOWN

    family = UNKNOWN
    confidence = 0.0
    agreement: Optional[bool] = None
    source = "conflict"
    conflict_note = ""

    if have1 and have2:
        if f1 == f2:
            family, agreement, source = f1, True, "both agree"
            confidence = math.sqrt(c1 * c2)
        else:
            agreement = False
            if c1 >= STRONG_CONFIDENCE > c2:
                family, source, confidence = f1, "stage 1", c1
            elif c2 >= STRONG_CONFIDENCE > c1:
                family, source, confidence = f2, "stage 2", c2
            else:
                conflict_note = (
                    f"Stage 1 says '{stage1_class}' ({f1}, {c1:.0%}) but Stage 2 says "
                    f"'{stage2_label}' ({f2}, {c2:.0%}); neither is clearly trustworthy."
                )
    elif have1:
        family, source, confidence = f1, "stage 1", c1
    elif have2:
        family, source, confidence = f2, "stage 2", c2

    # -- Undecidable -> manual review
    if family == UNKNOWN:
        reason = conflict_note or "Neither stage produced a usable material label."
        return Recommendation(
            action="Manual review", bin="Set aside for manual check", final_material=UNKNOWN,
            recyclable=None, is_contamination=False, confidence=confidence, agreement=agreement,
            source="conflict", needs_review=True, reason=reason,
            stage1_family=f1, stage2_family=f2,
        )

    # -- Plastic sub-type: soft plastic/foam is not curbside-recyclable (needs Stage 1 name)
    if family == PLASTIC and is_nonrecyclable_plastic(stage1_class):
        action, bin_, recyclable, contamination, reason = _SOFT_PLASTIC
    else:
        action, bin_, recyclable, contamination, reason = _ACTIONS[family]
        if family == PLASTIC and not have1:
            reason += " (Plastic type not verified by Stage 1 - confirm it is not film or foam.)"

    if source == "stage 1" and agreement is False:
        reason += f" Stage 2 disagreed ('{stage2_label}', {c2:.0%}) but was low confidence."
    elif source == "stage 2" and agreement is False:
        reason += f" Stage 1 disagreed ('{stage1_class}', {c1:.0%}) but was low confidence."

    needs_review = confidence < MIN_FINAL_CONFIDENCE
    if needs_review:
        reason += " Overall confidence is low - please verify manually."

    return Recommendation(
        action=action, bin=bin_, final_material=family, recyclable=recyclable,
        is_contamination=contamination, confidence=confidence, agreement=agreement,
        source=source, needs_review=needs_review, reason=reason,
        stage1_family=f1, stage2_family=f2,
    )


# ---------------------------------------------------------------------------
# Dhrti B2B marketplace mapping (kept compatible with the original IsItRecyc-CNN API)
# ---------------------------------------------------------------------------
def dhrti_listing(rec: Recommendation, stage1_class: Optional[str] = None) -> Dict[str, Any]:
    """Translate a Recommendation into marketplace category / title / warning."""
    fam = rec.final_material
    category: Optional[str] = None
    title = f"Bulk {fam.capitalize()} Scrap"
    warning: Optional[str] = None
    contamination = rec.is_contamination

    if rec.needs_review and rec.action == "Manual review":
        warning = "The two AI stages could not agree on this item. Please verify it manually."
        contamination = False
    elif fam == PLASTIC:
        if rec.recyclable:
            category, title = "recyclable_plastic", "Sorted Recyclable Plastic Scrap"
        else:
            warning = "Non-recyclable plastic or film detected. It cannot be listed as recyclable plastic."
    elif fam == PAPER:
        category, title = "paper", "Baled Paper & Cardboard Waste"
    elif fam == METAL:
        category, title = "metal", "Scrap Metal & Cans Batch"
    elif fam == GLASS:
        warning = "Glass detected. Dhrti marketplace currently only trades Recyclable Plastic, Paper, and Metal."
    elif fam == ORGANIC:
        warning = "Organic waste detected. It should be composted and cannot be listed as recyclable scrap."
    elif fam == TEXTILE:
        warning = "Textile waste detected. It is not accepted in the recycling categories."
    else:
        warning = "Unclassified or non-recyclable item detected."

    return {
        "suggested_category": category,
        "suggested_title": title,
        "is_contamination": contamination,
        "warning": warning,
    }
