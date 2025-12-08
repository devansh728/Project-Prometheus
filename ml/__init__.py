"""
SentinelEY - Machine Learning Model Training Package

This package provides:
- VehicleSimulator: Physics-based telemetry data generation
- DigitalTwinModel: Personalized ML model for anomaly detection
- MLKnowledgeBase: RAG system using ChromaDB
- FeedbackEngine: Gamified LLM feedback with Gemini

Usage:
    from ml import VehicleSimulator, DigitalTwinModel, MLKnowledgeBase, FeedbackEngine
"""

from .physics import VehicleSimulator, VehicleSpecs, DriverProfile, WeatherCondition
from .digital_twin_model import (
    DigitalTwinModel,
    FleetDigitalTwinManager,
    PredictionResult,
)
from .rag_knowledge import MLKnowledgeBase
from .feedback import FeedbackEngine, FleetFeedbackManager, FeedbackResult


__all__ = [
    # Physics
    "VehicleSimulator",
    "VehicleSpecs",
    "DriverProfile",
    "WeatherCondition",
    # ML Model
    "DigitalTwinModel",
    "FleetDigitalTwinManager",
    "PredictionResult",
    # RAG
    "MLKnowledgeBase",
    # Feedback
    "FeedbackEngine",
    "FleetFeedbackManager",
    "FeedbackResult",
]

__version__ = "1.0.0"
