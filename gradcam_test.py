from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ====================================================
# LOAD MODEL
# ====================================================

species_model = YOLO(
    "runs/classify/train-3/weights/best.pt"
)

RESULT_FOLDER = "static/results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ====================================================
# WRAPPER (IMPORTANT FIX)
# ====================================================

class WrappedClassifier(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        out = self.model(x)

        # FIX: Ultralytics returns tuple sometimes
        if isinstance(out, tuple):
            out = out[0]

        return out

# ====================================================
# GRADCAM FUNCTION
# ====================================================

def generate_gradcam(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))

    rgb_float = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(rgb_float)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()

    # IMPORTANT
    tensor.requires_grad_(True)

    # ====================================================
    # WRAP MODEL
    # ====================================================

    model = WrappedClassifier(species_model.model)
    model.eval()

    # ====================================================
    # FIXED TARGET LAYER (CRITICAL)
    # ====================================================

    target_layers = [
        model.model.model[9].conv.conv
    ]

    # ====================================================
    # GRADCAM SETUP
    # ====================================================

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    # ====================================================
    # SAVE RESULT
    # ====================================================

    gradcam_name = f"gradcam_{datetime.now().timestamp()}.png"
    gradcam_path = os.path.join(RESULT_FOLDER, gradcam_name)

    cv2.imwrite(
        gradcam_path,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    print("\n✅ GRADCAM SAVED:", gradcam_path)

    return gradcam_path

# ====================================================
# TEST RUN
# ====================================================

if __name__ == "__main__":

    image_path = "Malaria dataset/Ovale/gt/1708161076-0004-R.png"

    path = generate_gradcam(image_path)

