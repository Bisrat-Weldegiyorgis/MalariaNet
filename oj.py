from ultralytics import YOLO

model = YOLO("runs/detect/runs/detect/MalariaNet/weights/best.pt")

results = model.predict(
    "Malaria dataset/Ovale/img/Ring/1708161076-0004-R.png",
    conf=0.05,
    save=True,
    verbose=True
)

print(results[0].boxes)