from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="yolo_dataset/malaria.yaml",
    epochs=150,
    imgsz=224,
    batch=4,
    device="cpu",
    workers=2,

    lr0=0.001,
    optimizer="AdamW",

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=10,
    translate=0.1,
    scale=0.4,
    shear=2,

    flipud=0.0,
    fliplr=0.5,

    mosaic=0.5,
    mixup=0.2,

    project="runs/detect",
    name="MalariaNet",

    patience=30,
    save=True
)