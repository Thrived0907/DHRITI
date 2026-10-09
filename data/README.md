# Datasets

Datasets are **not** included (size and licensing). Download them from their official sources.

## Stage 1 - YOLOv8 (TACO + KNR-PET-7)

YOLOv8 needs one merged dataset in YOLO format:

```
data/stage1/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt      # one line per box:  class_id x_center y_center width height  (normalised 0-1)
```

Steps:
1. Download **TACO** (COCO-format annotations) and **KNR-PET-7**.
2. Convert each to YOLO format. TACO ships COCO JSON; Ultralytics' `convert_coco` helper can do this.
   Check KNR-PET-7's own documentation for its annotation format and class list - I have not verified them.
3. Build **one unified class list** (e.g. keep TACO's categories and add the PET sub-types from KNR-PET-7),
   remap every label file to those ids, and write the list into `configs/waste_detection.yaml`.
4. Split 70/15/15 (or similar) **by image**, not by box.

The decision logic maps detector class names to material families by keywords
(`glass`, `can`, `plastic`, `pet`, `paper`, `carton`, ...), so your class names need to be readable
words, not codes like `c17`. Unknown names are reported as "unknown" and sent to manual review.

Train: `python -m src.train_stage1 --data configs/waste_detection.yaml --model yolov8s.pt --epochs 100`

## Stage 2 - EfficientNet-B0 (RealWaste)

Download **RealWaste** (UCI Machine Learning Repository) and place the class folders here:

```
data/realwaste/
  Cardboard/  Food Organics/  Glass/  Metal/  Miscellaneous Trash/
  Paper/  Plastic/  Textile Trash/  Vegetation/
```

If your download has an extra wrapper folder (e.g. `RealWaste/RealWaste/...`), point `--data` at the folder
that directly contains the nine class folders.

Train: `python -m src.train_stage2 --data data/realwaste --epochs 25`
