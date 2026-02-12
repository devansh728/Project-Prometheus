"""
SentinEV Core - Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sentinev_core.api import router as core_router
from serviceops_ai.api import router as serviceops_router
from shared.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    await init_db()
    yield


app = FastAPI(
    title="SentinEV API",
    description="EV Predictive Maintenance Platform - Core + ServiceOps",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(core_router, prefix="/api/v1")
app.include_router(serviceops_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sentinev"}


@app.get("/")
async def root():
    return {
        "name": "SentinEV API",
        "version": "2.0.0",
        "endpoints": {
            "core": "/api/v1/status",
            "telemetry": "/api/v1/telemetry",
            "agent": "/api/v1/agent",
            "serviceops": "/api/v1/serviceops",
        },
    }
