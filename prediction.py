from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# ============================================
# MODEL
# ============================================

MODEL = "models/best.pt"

model = YOLO("models/best.pt")

RESULT_FOLDER = "static/results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================
# CLASS NAMES
# ============================================

CLASS_NAMES = {
    0: "Falciparum_Ring",
    1: "Falciparum_Trophozoite",
    2: "Falciparum_Schizont",

    3: "Vivax_Ring",
    4: "Vivax_Trophozoite",
    5: "Vivax_Schizont",
    6: "Vivax_Gametocyte",

    7: "Malariae_Ring",
    8: "Malariae_Trophozoite",
    9: "Malariae_Schizont",
    10: "Malariae_Gametocyte",

    11: "Ovale_Ring",
    12: "Ovale_Trophozoite",
    13: "Ovale_Schizont",
    14: "Ovale_Gametocyte",
}

# ============================================
# PREDICTION
# ============================================

def run_prediction(image_path):

    image = cv2.imread(image_path)

    results = model.predict(
        source=image_path,
        conf=0.05,
        iou=0.30,
        imgsz=224,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = CLASS_NAMES.get(cls, "Unknown_Unknown")

            species, stage = label.split("_")

            detections.append({
                "species": species,
                "stage": stage,
                "confidence": conf
            })

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                image,
                f"{species} {stage} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    output_path = os.path.join(
        RESULT_FOLDER,
        f"result_{datetime.now().timestamp()}.png"
    )

    cv2.imwrite(output_path, image)

    if len(detections):

        best = max(
            detections,
            key=lambda x: x["confidence"]
        )

        return {

            "diagnosis": "MALARIA POSITIVE",

            "species": best["species"],

            "stage": best["stage"],

            "confidence": round(
                best["confidence"] * 100,
                2
            ),

            "infected": len(detections),

            "image": output_path
        }

    return {

        "diagnosis": "MALARIA NEGATIVE",

        "species": "Unknown",

        "stage": "Unknown",

        "confidence": 0,

        "infected": 0,

        "image": output_path
    }

# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    image = "Malaria dataset/Ovale/img/Ring/1708161076-0004-R.png"

    result = run_prediction(image)

    print(result)
