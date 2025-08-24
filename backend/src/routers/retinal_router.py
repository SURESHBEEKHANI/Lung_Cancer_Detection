from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import uuid
from datetime import datetime
from src.services.retinal_service import run_retinal_disease_inference

retinal_router = APIRouter(prefix="/retinal", tags=["Retinal Disease"])


@retinal_router.post("/predict")
async def predict_retinal_disease(file: UploadFile = File(...)):
    contents = await file.read()
    input_info = {
        "filename": file.filename,
        "content_type": getattr(file, 'content_type', None),
        "size": len(contents)
    }

    image = Image.open(io.BytesIO(contents))
    output = run_retinal_disease_inference(image)

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    buf.seek(0)
    img_bytes = buf.getvalue()
    data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode('utf-8')

    input_data_url = "data:" + (input_info.get('content_type') or 'image/png') + ";base64," + base64.b64encode(contents).decode('utf-8')

    prediction = {
        "id": str(uuid.uuid4()),
    "diagnosis": "",
        "confidence": 0.0,
        "findings": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "image_url": data_url,
        "input_image_url": input_data_url
    }

    return JSONResponse({
        "prediction": prediction,
        "message": "ok",
        "input": input_info
    })
