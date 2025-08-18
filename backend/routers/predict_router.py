from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from src.services.prediction import predict_tumor

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/")
async def predict(file: UploadFile = File(...)):
    """Upload MRI image and get Lung Cancer Detection result"""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        segmented_image = predict_tumor(image)

        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
