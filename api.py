"""FastAPI microservice for the two-stage waste analysis pipeline.

POST /api/predict  - upload an image; returns Stage 1 (YOLOv8) + Stage 2 (EfficientNet-B0) + final decision
GET  /health       - model status

The top-level fields (predicted_material, is_recyclable, suggested_category, warning, ...) keep the
same names as the original IsItRecyc-CNN API so the Dhrti marketplace client keeps working; they
describe the dominant object in the image. Full per-object detail is in `items`.
"""

import base64
import io
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_logic import dhrti_listing
from src.pipeline import TwoStagePipeline, load_default_pipeline
from src.visualize import draw_results

app = FastAPI(
    title="Two-Stage Waste Analysis API",
    description="YOLOv8 detection + EfficientNet-B0 classification + recycling decision logic",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev server (:5173), Express (:5000), etc. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[TwoStagePipeline] = None
_lock = threading.Lock()


def get_pipeline(reload: bool = False) -> TwoStagePipeline:
    global _pipeline
    with _lock:
        if _pipeline is None or reload:
            _pipeline = load_default_pipeline(allow_partial=True)
        return _pipeline


@app.get("/")
@app.get("/health")
def health() -> Dict[str, Any]:
    pipe = get_pipeline()
    return {
        "status": "online" if pipe.ready else "no_models_loaded",
        "service": "two-stage-waste-analysis",
        "models": pipe.status(),
        "dhrti_categories": ["recyclable_plastic", "paper", "metal"],
    }


@app.post("/api/reload")
def reload_models() -> Dict[str, Any]:
    """Re-read checkpoints from disk (e.g. after training finishes) without restarting."""
    pipe = get_pipeline(reload=True)
    return {"status": "online" if pipe.ready else "no_models_loaded", "models": pipe.status()}


@app.post("/api/predict")
def predict(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None, ge=0.0, le=1.0, description="Stage 1 confidence threshold"),
    iou: Optional[float] = Query(None, ge=0.0, le=1.0, description="Stage 1 NMS IoU threshold"),
    annotated: bool = Query(False, description="Include annotated image as base64 JPEG"),
) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Invalid file type '{file.content_type}'. Upload an image (JPEG, PNG, WEBP).")

    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception as err:
        raise HTTPException(400, f"Unable to decode image: {err}")

    pipe = get_pipeline()
    if not pipe.ready:
        raise HTTPException(503, {"message": "No model weights loaded.", "models": pipe.status()})

    result = pipe.analyze(image, det_conf=conf, det_iou=iou)
    payload: Dict[str, Any] = {"success": True, "filename": file.filename, **result.to_dict()}

    # ---- Backward-compatible single-answer fields (dominant object) ---------------------
    primary = result.primary_item()
    payload["primary_index"] = primary.index if primary else None
    if primary is not None:
        rec = primary.recommendation
        listing = dhrti_listing(rec, primary.stage1.class_name if primary.stage1 else None)
        s2_probs = primary.stage2.probabilities if primary.stage2 else {}
        payload.update({
            "predicted_material": rec.final_material,
            "material_confidence": round(rec.confidence * 100.0, 1),
            "is_recyclable": rec.recyclable,
            "recyclable_label": {True: "Yes", False: "No", None: "Unknown"}[rec.recyclable],
            "recyclability_confidence": round(rec.confidence * 100.0, 1),
            "recommendation": rec.action,
            "material_probabilities": {k: round(v * 100.0, 1) for k, v in s2_probs.items()},
            **listing,
        })
    else:
        payload.update({
            "predicted_material": None, "is_recyclable": None, "recommendation": None,
            "suggested_category": None, "is_contamination": False,
            "warning": "No waste objects found in the image.",
        })

    if annotated:
        buf = io.BytesIO()
        draw_results(image, result).save(buf, format="JPEG", quality=90)
        payload["annotated_image_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse(content=payload)


if __name__ == "__main__":
    print("Starting two-stage waste analysis API on http://localhost:8000 (docs at /docs) ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
