from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Database
from database.database import Base, engine

# Routers
from src.routers.auth_router import auth_router       # Authentication
from src.routers.brain_router import brain_router
from src.routers.lung_router import lung_router
from src.routers.retinal_router import retinal_router
from src.routers.skin_router import skin_router
from src.routers.report_router import report_router

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DeepMediDetect API",
    description="API for brain, lung, retinal, and skin detection plus report generation using LLM",
    version="1.2.0",  # Updated version
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)       # <-- Auth router first
app.include_router(brain_router)
app.include_router(lung_router)
app.include_router(retinal_router)
app.include_router(skin_router)
app.include_router(report_router)

@app.get("/")
def root():
    return {"message": "DeepMediDetect API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
