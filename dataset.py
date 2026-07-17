import os
import cv2
import random
from pathlib import Path

# ====================================================
# SETTINGS
# ====================================================

BASE_DIR = "Malaria dataset"
OUT_DIR = "yolo_dataset"

VAL_SPLIT = 0.2
random.seed(42)

# ====================================================
# CLASS MAP
# ====================================================

CLASS_MAP = {

    ("falciparum","Ring"):0,
    ("falciparum","Trophozoite"):1,
    ("falciparum","Schizont"):2,

    ("vivax","Ring"):3,
    ("vivax","Trophozoite"):4,
    ("vivax","Schizont"):5,
    ("vivax","Gametocyte"):6,

    ("malariae","Ring"):7,
    ("malariae","Trophozoite"):8,
    ("malariae","Schizont"):9,
    ("malariae","Gametocyte"):10,

    ("ovale","Ring"):11,
    ("ovale","Trophozoite"):12,
    ("ovale","Schizont"):13,
    ("ovale","Gametocyte"):14,
}

CLASS_NAMES = [
    "Falciparum_Ring",
    "Falciparum_Trophozoite",
    "Falciparum_Schizont",

    "Vivax_Ring",
    "Vivax_Trophozoite",
    "Vivax_Schizont",
    "Vivax_Gametocyte",

    "Malariae_Ring",
    "Malariae_Trophozoite",
    "Malariae_Schizont",
    "Malariae_Gametocyte",

    "Ovale_Ring",
    "Ovale_Trophozoite",
    "Ovale_Schizont",
    "Ovale_Gametocyte",
]

# ====================================================
# CREATE OUTPUT FOLDERS
# ====================================================

for folder in [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]:
    os.makedirs(os.path.join(OUT_DIR, folder), exist_ok=True)

# ====================================================
# CONVERT MASK TO YOLO BOX
# ====================================================

def mask_to_yolo(mask):

    _, binary = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)

    H, W = mask.shape

    return (
        (x + w / 2) / W,
        (y + h / 2) / H,
        w / W,
        h / H,
    )

# ====================================================
# COLLECT DATA
# ====================================================
samples = []

species_list = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
]

for species in species_list:

    species_lower = species.lower()

    img_root = os.path.join(BASE_DIR, species, "img")
    gt_root = os.path.join(BASE_DIR, species, "gt")

    if not os.path.isdir(img_root):
        continue

    if not os.path.isdir(gt_root):
        continue

    # Every stage folder
    for stage in os.listdir(img_root):

        img_dir = os.path.join(img_root, stage)
        gt_dir = os.path.join(gt_root, stage)

        if not os.path.isdir(img_dir):
            continue

        if not os.path.isdir(gt_dir):
            continue

        key = (species_lower, stage)

        if key not in CLASS_MAP:
            print("Skipping:", key)
            continue

        class_id = CLASS_MAP[key]

        for file in os.listdir(img_dir):

            if not file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                continue

            img_path = os.path.join(img_dir, file)

            stem = Path(file).stem

            # Try every possible mask extension
            mask_path = None

            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:

                candidate = os.path.join(gt_dir, stem + ext)

                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                continue

            samples.append(
                (
                    img_path,
                    mask_path,
                    class_id
                )
            )

print(f"\nFound {len(samples)} samples")

# ====================================================
# TRAIN / VAL SPLIT
# ====================================================

random.shuffle(samples)

split = int(len(samples) * (1 - VAL_SPLIT))

train_samples = samples[:split]
val_samples = samples[split:]

print("Train:", len(train_samples))
print("Validation:", len(val_samples))

# ====================================================
# SAVE DATA
# ====================================================

def save_split(samples, split_name):

    for img_path, mask_path, class_id in samples:

        image = cv2.imread(img_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        box = mask_to_yolo(mask)

        if box is None:
            continue

        x, y, w, h = box

        stem = Path(img_path).stem

        out_image = os.path.join(
            OUT_DIR,
            "images",
            split_name,
            stem + ".png"
        )

        out_label = os.path.join(
            OUT_DIR,
            "labels",
            split_name,
            stem + ".txt"
        )

        cv2.imwrite(out_image, image)

        with open(out_label, "w") as f:
            f.write(
                f"{class_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )

save_split(train_samples, "train")
save_split(val_samples, "val")

# ====================================================
# CREATE YAML
# ====================================================

yaml_path = os.path.join(OUT_DIR, "malaria.yaml")

with open(yaml_path, "w") as f:

    f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")

    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {name}\n")

print("\n===================================")
print("YOLO dataset created successfully")
print("===================================")
print("Classes :", len(CLASS_NAMES))
print("Train   :", len(train_samples))
print("Val     :", len(val_samples))
print("Output  :", OUT_DIR)