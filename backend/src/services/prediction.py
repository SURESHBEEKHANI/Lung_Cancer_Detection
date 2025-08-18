from fastapi import HTTPException
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3
from PIL import Image
import numpy as np
from ultralytics.nn.modules import Conv
from torch import nn

# ---- PATCH FOR MISSING C3k2 ----
class C3k2(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(Conv(c1, c_, 1, 1), Conv(c_, c2, 5, 1, p=2), nn.Conv2d(c2, c2, 1, 1, bias=False), nn.BatchNorm2d(c2))
# -------------------------------

# Load YOLO model safely
model_path = "weights/Lung Cancer Detection.pt"
model = None
try:
    model = YOLO(model_path)
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Failed to load YOLO model: {e}")


def predict_tumor(image: Image.Image) -> Image.Image:
    """Run YOLO prediction and return image with segmentation mask"""
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded. Check server logs.")

    try:
        image_np = np.array(image)
        results = model.predict(image_np)
        output_image = results[0].plot()  # Annotated image
        return Image.fromarray(output_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")
