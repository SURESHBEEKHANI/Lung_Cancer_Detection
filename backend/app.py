# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

# ---- PATCH FOR MISSING C3k2 ----
# Define C3k2 to avoid YOLO model load error
from ultralytics.nn.modules.block import C3
class C3k2(C3):
    pass
# -------------------------------

from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(title="Lung Cancer Detection.pt")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model safely
model_path = "weights\Lung Cancer Detection.pt"
model = None
try:
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load YOLO model: {e}")
    # For demo purposes, we'll simulate results when model is not available

def predict_tumor(image: Image.Image) -> Image.Image:
    """Predict tumor and return image with overlayed segmentation mask"""
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded. Check server logs.")
    try:
        # Convert PIL to NumPy array for YOLO
        image_np = np.array(image)
        results = model.predict(image_np)
        output_image = results[0].plot()  # Get annotated image
        return Image.fromarray(output_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an MRI image and get Lung Cancer Detection"""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        segmented_image = predict_tumor(image)

        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
