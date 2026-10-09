import json
import random
import shutil
from pathlib import Path

TACO_ROOT = Path("data/TACO/data")
OUTPUT_ROOT = Path("data/stage1")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

CLASS_MAP = {
    0: 4,
    1: 6,
    2: 6,
    3: 6,
    4: 0,
    5: 0,
    6: 3,
    7: 0,
    8: 4,
    9: 3,
    10: 4,
    11: 6,
    12: 4,
    13: 6,
    14: 2,
    15: 2,
    16: 2,
    17: 2,
    18: 2,
    19: 2,
    20: 1,
    21: 0,
    22: 6,
    23: 3,
    24: 0,
    25: 5,
    26: 3,
    27: 0,
    28: 4,
    29: 0,
    30: 1,
    31: 1,
    32: 1,
    33: 1,
    34: 1,
    35: 1,
    36: 0,
    37: 0,
    38: 6,
    39: 0,
    40: 0,
    41: 0,
    42: 0,
    43: 0,
    44: 0,
    45: 0,
    46: 6,
    47: 0,
    48: 0,
    49: 0,
    50: 4,
    51: 6,
    52: 4,
    53: 6,
    54: 0,
    55: 0,
    56: 1,
    57: 6,
    58: 6,
    59: 6,
}

CLASS_NAMES = [
    "Plastic",
    "Paper",
    "Cardboard",
    "Glass",
    "Metal",
    "Food Organics",
    "Other",
]


def find_image(image_info):
    file_name = image_info["file_name"]

    candidates = [
        TACO_ROOT / file_name,
        TACO_ROOT / Path(file_name).name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(TACO_ROOT.glob(f"**/{Path(file_name).name}"))

    if matches:
        return matches[0]

    return None


def main():
    annotation_file = TACO_ROOT / "annotations.json"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Could not find annotations file: {annotation_file}"
        )

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    annotations_by_image = {}

    for annotation in annotations:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if category_id not in CLASS_MAP:
            continue

        annotations_by_image.setdefault(image_id, []).append(annotation)

    usable_images = [
        image
        for image in images
        if image["id"] in annotations_by_image
    ]

    random.seed(SEED)
    random.shuffle(usable_images)

    total = len(usable_images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": usable_images[:train_end],
        "val": usable_images[train_end:val_end],
        "test": usable_images[val_end:],
    }

    for split in splits:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    copied = 0
    skipped = 0
    labels_written = 0

    for split, split_images in splits.items():

        for image_info in split_images:

            image_id = image_info["id"]
            source = find_image(image_info)

            if source is None:
                skipped += 1
                continue

            width = image_info["width"]
            height = image_info["height"]

            destination_name = f"{image_id:06d}.jpg"

            destination = (
                OUTPUT_ROOT
                / "images"
                / split
                / destination_name
            )

            shutil.copy2(source, destination)

            label_lines = []

            for annotation in annotations_by_image[image_id]:

                category_id = annotation["category_id"]

                if category_id not in CLASS_MAP:
                    continue

                class_id = CLASS_MAP[category_id]

                x, y, box_width, box_height = annotation["bbox"]

                if box_width <= 0 or box_height <= 0:
                    continue

                x_center = (x + box_width / 2) / width
                y_center = (y + box_height / 2) / height

                normalized_width = box_width / width
                normalized_height = box_height / height

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                normalized_width = max(
                    0,
                    min(1, normalized_width)
                )
                normalized_height = max(
                    0,
                    min(1, normalized_height)
                )

                label_lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{normalized_width:.6f} "
                    f"{normalized_height:.6f}"
                )

            label_file = (
                OUTPUT_ROOT
                / "labels"
                / split
                / f"{image_id:06d}.txt"
            )

            label_file.write_text(
                "\n".join(label_lines),
                encoding="utf-8"
            )

            labels_written += 1
            copied += 1

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""

    for index, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {index}: {name}\n"

    yaml_file = OUTPUT_ROOT / "data.yaml"

    yaml_file.write_text(
        yaml_content,
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("       TACO CONVERSION COMPLETED")
    print("========================================")
    print(f"Total usable images : {total}")
    print(f"Train images        : {len(splits['train'])}")
    print(f"Validation images   : {len(splits['val'])}")
    print(f"Test images         : {len(splits['test'])}")
    print(f"Images copied       : {copied}")
    print(f"Images skipped      : {skipped}")
    print(f"Labels written      : {labels_written}")
    print()
    print("YOLO classes:")
    
    for index, name in enumerate(CLASS_NAMES):
        print(f"{index}: {name}")

    print()
    print(f"YOLO dataset: {OUTPUT_ROOT}")
    print(f"YAML file: {yaml_file}")
    print("========================================")


if __name__ == "__main__":
    main()