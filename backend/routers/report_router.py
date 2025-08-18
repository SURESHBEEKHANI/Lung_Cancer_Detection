from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path

from src.services.report import ModelBuilder  # Replace with actual import path

router = APIRouter(prefix="/report", tags=["Report"])

# Directory to temporarily store uploaded images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize your model
model = ModelBuilder()

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Generate the medical report
        report = model.generate_report(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Delete the uploaded file after processing
        if file_path.exists():
            os.remove(file_path)

    return JSONResponse(content={"report": report})
