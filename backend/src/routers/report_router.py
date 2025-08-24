from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from src.services.report_service import ReportService
import shutil
import os
import logging
from src.dependencies import get_current_user

logger = logging.getLogger("report_router")

# Create router
report_router = APIRouter(
    prefix="/report",
    tags=["report"]
)

# Initialize service
service = ReportService()

# Temporary folder for uploaded images
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@report_router.post("/generate", summary="Generate medical report from image", response_model=dict)
async def generate_report(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    """
    Upload an image file to generate a medical report.
    """
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Generate report using service
        report_text = service.generate_report_from_image(file_path)

        # Remove temp file after processing
        os.remove(file_path)

        return {
            "filename": file.filename,
            "report": report_text
        }

    except Exception as e:
        logger.exception("Failed to generate report")
        raise HTTPException(status_code=500, detail="Failed to generate report")
