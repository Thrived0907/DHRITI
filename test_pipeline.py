from pathlib import Path
from PIL import Image
from src.pipeline import load_default_pipeline


IMAGE_PATH = Path("test_scrap.jpg")


def main():
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        return

    image_path = IMAGE_PATH

    print("=" * 60)
    print("DHRti TWO-STAGE PIPELINE TEST")
    print("=" * 60)
    print(f"Image: {image_path}")

    pipeline = load_default_pipeline(allow_partial=False)

    print("\nPipeline status:")
    print(pipeline.status())

    image = Image.open(image_path)

    result = pipeline.analyze(
        image,
        det_conf=0.25,
        det_iou=0.7,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)

    print(f"Image size: {result.image_size}")
    print(f"Objects found: {result.summary['objects_found']}")

    print("\nObjects:")

    for item in result.items:
        print("\n" + "-" * 50)
        print(f"Object {item.index}")

        if item.stage1:
            print(f"YOLO class      : {item.stage1.class_name}")
            print(f"YOLO confidence : {item.stage1.confidence:.3f}")
            print(f"Bounding box    : {item.stage1.box}")
        else:
            print("YOLO detection  : None")

        if item.stage2:
            print(f"Material        : {item.stage2.label}")
            print(f"Material conf.  : {item.stage2.confidence:.3f}")
        else:
            print("Material        : None")

        print(f"Recommendation  : {item.recommendation.action}")
        print(f"Recyclable      : {item.recommendation.recyclable}")
        print(f"Final material  : {item.recommendation.final_material}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result.summary)

    print("\nTimings:")
    print(result.timings_ms)

    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print("-", note)


if __name__ == "__main__":
    main()