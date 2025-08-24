import numpy as np
from PIL import Image
from fastapi import HTTPException
from utils.model_loader import ModelLoader

loader = ModelLoader()
model = loader.load_model("brain_tumor")

def run_brain_tumor_inference(image: Image.Image) -> Image.Image:
    if model is None:
        raise HTTPException(status_code=500, detail="Brain Tumor model not loaded.")

    try:
        image_np = np.array(image)
        results = model.predict(image_np)
        output_image = results[0].plot()
        return Image.fromarray(output_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")
