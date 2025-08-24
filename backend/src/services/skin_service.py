import numpy as np
from PIL import Image
from fastapi import HTTPException
from utils.model_loader import ModelLoader

# Load the Skin Disease model
loader = ModelLoader()
model = loader.load_model("skin_disease")

def run_skin_disease_inference(image: Image.Image) -> Image.Image:
    if model is None:
        raise HTTPException(status_code=500, detail="Skin Disease model not loaded.")

    try:
        # Convert PIL image to NumPy array
        image_np = np.array(image)

        # Run prediction
        results = model.predict(image_np)

        # Convert results to image (assuming results[0].plot() returns an array)
        output_image = results[0].plot()
        return Image.fromarray(output_image)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")
