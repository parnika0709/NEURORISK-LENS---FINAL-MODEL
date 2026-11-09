"""Brain Tumor MRI Analysis API entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import cases_router, predictions_router, segmentation_router

app = FastAPI(
    title=settings.title,
    description=settings.description,
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cases_router)
app.include_router(predictions_router)
app.include_router(segmentation_router)

@app.get("/api")
async def root():
    return {"message": "Welcome to the Brain Tumor MRI Analysis API", "docs": "/docs"}


# Serve the frontend (apps/api/static/index.html) at "/".
# Mounted last so it never shadows the /cases API routes above.
_static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
