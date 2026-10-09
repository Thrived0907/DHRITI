# Two-Stage Waste Analysis: YOLOv8 + EfficientNet-B0

An image of waste goes through two models, and a rule-based decision layer combines their outputs into a
recycling recommendation for each object.

```
 Image ──► Stage 1: YOLOv8 ──► class, box, confidence   (TACO + KNR-PET-7)
                 │
                 ▼  crop each detected object (+10 % padding, letterboxed to 224×224)
           Stage 2: EfficientNet-B0 ──► condition / quality class   (RealWaste)
                 │
                 ▼
           Decision logic ──► Recycle / Compost / Reuse / Landfill / Manual review + bin + reason
```

## Models used

| Stage | Model | Purpose | Dataset |
|---|---|---|---|
| Stage 1 | YOLOv8 | Waste object detection and localization | TACO + KNR-PET-7 |
| Stage 2 | EfficientNet-B0 | Waste condition/quality classification | RealWaste |

## Dataset → Model mapping

| Stage | Dataset | Model | Purpose |
|---|---|---|---|
| Stage 1 | TACO | YOLOv8 | General waste detection |
| Stage 1 | KNR-PET-7 | YOLOv8 | Fine-grained PET detection |
| Stage 2 | RealWaste | EfficientNet-B0 | Condition/quality assessment |
| Final | Outputs from both stages | Decision Logic | Recycling recommendation |

## Important notes

* **No trained weights are included.** Both models must be trained (or supplied) before the app produces
  real predictions. Training scripts are provided (below). Until weights exist, the app and API start and
  report which stage is missing; if only one stage is present the pipeline runs in a degraded single-stage mode.
* **Tested with stubs, not real weights.** The decision logic, cropping, pipeline orchestration and result
  serialisation are unit-tested with stub models. The YOLOv8/EfficientNet code paths, the training scripts,
  the Streamlit app and the API have **not** been executed against real weights or datasets in the environment
  this was written in - expect to fix small integration issues on first run.
* **What RealWaste actually labels.** RealWaste's nine classes (Cardboard, Food Organics, Glass, Metal,
  Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation) are *material categories* of real
  landfill-stream waste, not a graded condition scale such as clean/dirty/damaged. Stage 2 is therefore used as
  an independent second opinion on the material, which is what lets the decision logic cross-check Stage 1.
  If you need true condition grades, you would need a dataset labelled that way and a new class map.
* **Confidences from a softmax are not calibrated probabilities.** Thresholds in `src/config.py` should be tuned on
  your own validation data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
# Stage 2 - EfficientNet-B0 on RealWaste (see data/README.md for layout)
python -m src.train_stage2 --data data/realwaste --epochs 25 --batch-size 32
#   -> models/stage2_efficientnet_b0.pt  and  outputs/stage2_metrics.json (accuracy, per-class P/R, confusion matrix)

# Stage 1 - YOLOv8 on merged TACO + KNR-PET-7 (see data/README.md)
python -m src.train_stage1 --data configs/waste_detection.yaml --model yolov8s.pt --epochs 100
#   -> models/stage1_yolov8.pt
```

## Running

```bash
streamlit run app/app.py     # web UI  (or: python run_app.py)
python api.py                # REST API on :8000, docs at http://localhost:8000/docs  (or: python run_api.py)
pytest -v                    # tests
```

The web UI shows the annotated image, a per-object table (Stage 1 class + confidence, Stage 2 class + confidence,
agreement, final material, recommendation, bin), the cropped object, Stage 2 class probabilities, and the reason
behind each decision. Sidebar sliders control the Stage 1 confidence and NMS thresholds.

### API

`POST /api/predict` (multipart field `file`; optional query `conf`, `iou`, `annotated=true`)

Each entry in `items` carries `stage1`, `stage2`, `final`, and `marketplace`:

```jsonc
{
  "success": true,
  "summary": {"objects_found": 1, "recyclable": 1, "contamination": 0, "needs_review": 0, "actions": {"Recycle": 1}},
  "items": [{
    "index": 1,
    "stage1": {"class_name": "Clear plastic bottle", "confidence": 0.91, "box": [10.5, 20.2, 110.0, 210.0]},
    "stage2": {"label": "Plastic", "confidence": 0.84, "probabilities": {"...": 0.0}},
    "final":  {"action": "Recycle", "bin": "Plastic recycling", "final_material": "plastic",
               "agreement": true, "source": "both agree", "confidence": 0.874, "reason": "..."},
    "marketplace": {"suggested_category": "recyclable_plastic", "is_contamination": false, "warning": null}
  }],
  // Backward-compatible fields for the dominant (largest) object:
  "predicted_material": "plastic", "is_recyclable": true, "suggested_category": "recyclable_plastic"
}
```
(Illustrative shape - values depend on your trained models.)

`GET /health` reports which stages loaded and their class lists; `POST /api/reload` re-reads checkpoints.

## Decision logic (`src/decision_logic.py`)

Both stages' class names are mapped to a **material family** (paper, glass, metal, plastic, organic, textile, other).
Stage 1 names are mapped by keyword, so they work with TACO/KNR-PET-7 names without hard-coding.

| Situation | Result |
|---|---|
| Both stages agree | Family confirmed; confidence = √(c₁·c₂) |
| Disagree, one stage ≥ 60 % and the other below | The confident stage wins; the reason notes the disagreement |
| Disagree, both or neither confident | **Manual review** |
| Only one stage available | That stage is used (plastic is flagged "type not verified") |
| Final confidence < 35 % | Action kept but flagged **needs review** |

| Family | Action | Bin |
|---|---|---|
| Paper / cardboard | Recycle | Paper & cardboard recycling |
| Glass | Recycle | Glass recycling |
| Metal | Recycle | Metal recycling |
| Plastic (rigid, e.g. PET bottle) | Recycle | Plastic recycling |
| Plastic (bag, wrapper, film, foam - from Stage 1 name) | Landfill | General waste (soft-plastic drop-off) |
| Organic (food, vegetation) | Compost | Organic / compost |
| Textile | Reuse / Drop-off | Textile collection point |
| Other / miscellaneous | Landfill | General waste |

Bin rules reflect common curbside programs and vary by municipality - adjust `_ACTIONS` for your locale.
The Dhrti marketplace mapping (`recyclable_plastic`, `paper`, `metal`) is in `dhrti_listing()`.

## Project layout

```
api.py  run_api.py  run_app.py  requirements.txt
app/app.py                     Streamlit UI
configs/waste_detection.yaml   YOLO data config template
data/README.md                 dataset preparation
models/                        put stage1_yolov8.pt and stage2_efficientnet_b0.pt here
src/config.py                  paths, thresholds, class lists
src/stage1_detector.py         YOLOv8 wrapper
src/stage2_classifier.py       EfficientNet-B0 wrapper
src/decision_logic.py          fusion + recycling recommendation + Dhrti mapping
src/pipeline.py                Stage 1 -> crop -> Stage 2 -> decision
src/visualize.py               annotated image
src/train_stage1.py            YOLOv8 training
src/train_stage2.py            EfficientNet-B0 training
tests/                         unit tests (stub models)
```

## Differences from the uploaded `islt-main` repo

`islt-main` contained a single-stage **VGG16 (TensorFlow/Keras)** classifier and no YOLOv8 code; its
`best_mod.h5` was a Git-LFS pointer rather than real weights. This project replaces it with the
YOLOv8 → EfficientNet-B0 pipeline (PyTorch throughout, so no TensorFlow/PyTorch conflict) and keeps its
FastAPI + Streamlit structure and the Dhrti marketplace response fields. The VGG16 code, Grad-CAM,
legacy notebooks and the machine-specific `start_all.bat` were intentionally not carried over.
