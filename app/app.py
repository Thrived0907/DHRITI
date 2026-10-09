"""Streamlit web app: upload an image -> Stage 1 detection -> Stage 2 classification -> recommendation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.pipeline import load_default_pipeline
from src.visualize import draw_results

st.set_page_config(page_title="Two-Stage Waste Analyzer", page_icon="♻️", layout="wide")


@st.cache_resource(show_spinner="Loading YOLOv8 and EfficientNet-B0 ...")
def get_pipeline():
    return load_default_pipeline(allow_partial=True)


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def main() -> None:
    st.title("♻️ Two-Stage Waste Analyzer")
    st.caption("Stage 1: YOLOv8 finds and localises waste  ->  Stage 2: EfficientNet-B0 classifies each crop  "
               "->  Decision logic recommends what to do with it.")

    pipe = get_pipeline()
    status = pipe.status()

    # ------------------------------------------------------------------ sidebar
    st.sidebar.header("⚙️ Settings")
    det_conf = st.sidebar.slider("Stage 1 confidence threshold", 0.05, 0.95, config.DET_CONF_THRESHOLD, 0.05)
    det_iou = st.sidebar.slider("Stage 1 NMS IoU threshold", 0.10, 0.90, config.DET_IOU_THRESHOLD, 0.05)
    fallback = st.sidebar.checkbox("Classify whole image if nothing is detected", value=config.FULL_IMAGE_FALLBACK)

    st.sidebar.header("🧩 Model status")
    for key, title in (("stage1_yolov8", "Stage 1 - YOLOv8"), ("stage2_efficientnet_b0", "Stage 2 - EfficientNet-B0")):
        s = status[key]
        if s["loaded"]:
            st.sidebar.success(f"{title}: loaded")
        else:
            st.sidebar.error(f"{title}: not loaded")
            if s["error"]:
                st.sidebar.caption(s["error"])
    if st.sidebar.button("🔄 Reload models"):
        st.cache_resource.clear()
        st.rerun()

    with st.expander("ℹ️ Models used & dataset mapping"):
        st.markdown("**Models Used**")
        st.dataframe(pd.DataFrame([
            {"Stage": "Stage 1", "Model": "YOLOv8", "Purpose": "Waste object detection and localization",
             "Dataset": "TACO + KNR-PET-7"},
            {"Stage": "Stage 2", "Model": "EfficientNet-B0", "Purpose": "Waste condition/quality classification",
             "Dataset": "RealWaste"},
        ]), hide_index=True)
        st.markdown("**Dataset → Model mapping**")
        st.dataframe(pd.DataFrame([
            {"Stage": "Stage 1", "Dataset": "TACO", "Model": "YOLOv8", "Purpose": "General waste detection"},
            {"Stage": "Stage 1", "Dataset": "KNR-PET-7", "Model": "YOLOv8", "Purpose": "Fine-grained PET detection"},
            {"Stage": "Stage 2", "Dataset": "RealWaste", "Model": "EfficientNet-B0", "Purpose": "Condition/quality assessment"},
            {"Stage": "Final", "Dataset": "Outputs from both stages", "Model": "Decision Logic",
             "Purpose": "Recycling recommendation"},
        ]), hide_index=True)

    if not pipe.ready:
        st.error("No model weights found. Train the models (see README) and place them in `models/`, "
                 "then click **Reload models**.")
        return

    # ------------------------------------------------------------------ input
    uploaded = st.file_uploader("Upload a photo of waste", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded is None:
        st.info("👆 Upload an image to start.")
        return
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as err:
        st.error(f"Could not decode image: {err}")
        return

    with st.spinner("Running Stage 1 -> Stage 2 -> decision logic ..."):
        try:
            result = pipe.analyze(image, det_conf=det_conf, det_iou=det_iou, full_image_fallback=fallback)
        except Exception as err:
            st.error(f"Analysis failed: {err}")
            return

    for note in result.notes:
        st.warning(note)

    # ------------------------------------------------------------------ summary
    s = result.summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects found", s["objects_found"])
    m2.metric("Recyclable", s["recyclable"])
    m3.metric("Contamination", s["contamination"])
    m4.metric("Needs review", s["needs_review"])

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("📷 Input")
        st.image(image, caption=uploaded.name)
    with right:
        st.subheader("🎯 Detections and decisions")
        st.image(draw_results(image, result))

    if not result.items:
        return

    # ------------------------------------------------------------------ results table
    st.subheader("📋 Results by object")
    rows = []
    for it in result.items:
        rec = it.recommendation
        rows.append({
            "#": it.index,
            "Stage 1 class": it.stage1.class_name if it.stage1 else "-",
            "Stage 1 conf.": pct(it.stage1.confidence if it.stage1 else None),
            "Stage 2 class": it.stage2.label if it.stage2 else "-",
            "Stage 2 conf.": pct(it.stage2.confidence if it.stage2 else None),
            "Agree": {True: "✅", False: "⚠️", None: "-"}[rec.agreement],
            "Final material": rec.final_material.capitalize(),
            "Recommendation": rec.action,
            "Bin": rec.bin,
            "Confidence": pct(rec.confidence),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # ------------------------------------------------------------------ per-object detail
    st.subheader("🔍 Object details")
    for it in result.items:
        rec = it.recommendation
        title = f"#{it.index} - {rec.action} ({rec.final_material})"
        with st.expander(title, expanded=len(result.items) == 1):
            c1, c2, c3 = st.columns([1, 1, 1.4], gap="large")
            with c1:
                if it.crop is not None:
                    st.image(it.crop, caption="Cropped object sent to Stage 2")
            with c2:
                st.markdown("**Stage 1 - YOLOv8**")
                if it.stage1:
                    x1, y1, x2, y2 = (int(v) for v in it.stage1.box)
                    st.write(f"Class: **{it.stage1.class_name}**")
                    st.write(f"Confidence: **{pct(it.stage1.confidence)}**")
                    st.write(f"Box: ({x1}, {y1}) → ({x2}, {y2})")
                else:
                    st.write("No detection (whole image used).")
                st.markdown("**Stage 2 - EfficientNet-B0**")
                if it.stage2:
                    st.write(f"Class: **{it.stage2.label}**")
                    st.write(f"Confidence: **{pct(it.stage2.confidence)}**")
                else:
                    st.write("Stage 2 not available.")
            with c3:
                st.markdown("**Final recommendation**")
                (st.success if rec.recyclable else st.error if rec.recyclable is False else st.warning)(
                    f"{rec.action} → {rec.bin}")
                st.write(rec.reason)
                if it.stage2:
                    probs = pd.Series(it.stage2.probabilities).sort_values(ascending=False)
                    st.markdown("**Stage 2 class probabilities**")
                    st.bar_chart(probs)

    st.caption(
        f"Timing: Stage 1 {result.timings_ms['stage1_detect']:.0f} ms · "
        f"Stage 2 {result.timings_ms['stage2_classify']:.0f} ms"
    )


main()
