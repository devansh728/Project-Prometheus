"""
SentinEV - Enhanced FastAPI Backend
Multi-agent orchestration with WebSocket real-time streaming
"""

import os
import sys
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Literal
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from ml.telemetry_generator import EnhancedTelemetryGenerator, VehicleConfig
from ml.anomaly_detector import MLPipeline
from ml.rag_knowledge import MLKnowledgeBase

# Import agents
from agents.orchestrator import MasterOrchestrator, get_orchestrator
from agents.agent_state import create_initial_state


# ==================== Pydantic Models ====================


class VehicleInitRequest(BaseModel):
    driver_profile: Literal["aggressive", "eco", "normal"] = "normal"
    weather: Literal["hot", "cold", "moderate"] = "moderate"
    generate_days: int = Field(default=60, ge=1, le=365)


class VehicleInitResponse(BaseModel):
    vehicle_id: str
    status: str
    driver_profile: str
    training_samples: int
    model_trained: bool
    message: str


class FaultInjectionRequest(BaseModel):
    fault: Literal[
        "overheat",
        "cell_imbalance",
        "inverter",
        "motor_resolver",
        "brake_drag",
        "coolant_low",
    ]
    severity: float = Field(default=1.0, ge=0.5, le=2.0)


class FaultInjectionResponse(BaseModel):
    vehicle_id: str
    fault_type: str
    severity: float
    telemetry: Dict[str, Any]
    anomaly_detected: bool
    anomaly_type: str
    anomaly_severity: str
    failure_risk_pct: float
    score_delta: int
    feedback: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str


# ==================== WebSocket Manager ====================


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, vehicle_id: str):
        await websocket.accept()
        if vehicle_id not in self.active_connections:
            self.active_connections[vehicle_id] = []
        self.active_connections[vehicle_id].append(websocket)

    def disconnect(self, websocket: WebSocket, vehicle_id: str):
        if vehicle_id in self.active_connections:
            if websocket in self.active_connections[vehicle_id]:
                self.active_connections[vehicle_id].remove(websocket)

    async def broadcast(self, vehicle_id: str, message: Dict):
        if vehicle_id in self.active_connections:
            for connection in self.active_connections[vehicle_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass


# ==================== Application State ====================


class AppState:
    def __init__(self):
        self.generators: Dict[str, EnhancedTelemetryGenerator] = {}
        self.pipelines: Dict[str, MLPipeline] = {}
        self.orchestrator: Optional[MasterOrchestrator] = None
        self.knowledge_base: Optional[MLKnowledgeBase] = None
        self.connection_manager = ConnectionManager()

    def initialize(self):
        self.orchestrator = get_orchestrator()
        try:
            self.knowledge_base = MLKnowledgeBase()
            self.knowledge_base.load_knowledge_base()
            print("✓ Knowledge base loaded")
        except Exception as e:
            print(f"⚠️ Could not load knowledge base: {e}")


app_state = AppState()


# ==================== Lifespan ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting SentinEV API Server...")
    app_state.initialize()
    yield
    print("👋 Shutting down SentinEV API Server...")


# ==================== FastAPI App ====================

app = FastAPI(
    title="SentinEV Multi-Agent API",
    description="Advanced Multi-Agent EV Predictive Maintenance System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Endpoints ====================


@app.get("/")
async def root():
    return {
        "name": "SentinEV Multi-Agent API",
        "version": "2.0.0",
        "status": "running",
    }


# Model persistence directory
MODELS_DIR = Path("data/models")


def get_model_path(vin: str) -> Path:
    """Get the model directory path for a vehicle."""
    return MODELS_DIR / vin


def model_exists(vin: str) -> bool:
    """Check if a trained model exists for this vehicle."""
    model_path = get_model_path(vin)
    return (model_path / "metadata.json").exists()


@app.post("/api/v1/vehicles/{vin}/init", response_model=VehicleInitResponse)
async def init_vehicle(
    vin: str, request: VehicleInitRequest, background_tasks: BackgroundTasks
):
    try:
        config = VehicleConfig(vehicle_id=vin, driver_profile=request.driver_profile)
        generator = EnhancedTelemetryGenerator(config, seed=hash(vin) % 10000)
        app_state.generators[vin] = generator

        # Check if model already exists
        if model_exists(vin):
            print(f"✓ Found saved model for {vin}, loading...")
            pipeline = MLPipeline(vin)
            pipeline.load(str(get_model_path(vin)))
            app_state.pipelines[vin] = pipeline

            return VehicleInitResponse(
                vehicle_id=vin,
                status="loaded",
                driver_profile=pipeline.detector.driver_profile,
                training_samples=pipeline.detector.training_samples,
                model_trained=True,
                message=f"Vehicle {vin} loaded from saved model ({pipeline.detector.training_samples} samples)",
            )

        # No saved model - train new one
        print(f"Generating {request.generate_days} days of history for {vin}...")
        history_df = generator.generate_history(days=request.generate_days)

        pipeline = MLPipeline(vin)
        training_result = pipeline.train(history_df, request.driver_profile)
        app_state.pipelines[vin] = pipeline

        # Save the trained model
        model_path = get_model_path(vin)
        pipeline.save(str(model_path))
        print(f"✓ Model saved to {model_path}")

        if app_state.orchestrator:
            agent = app_state.orchestrator.get_data_agent(vin)
            agent.train_on_history(history_df, request.driver_profile)

        return VehicleInitResponse(
            vehicle_id=vin,
            status="initialized",
            driver_profile=request.driver_profile,
            training_samples=training_result.get("training_samples", len(history_df)),
            model_trained=True,
            message=f"Vehicle {vin} trained and saved ({len(history_df)} samples)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/v1/vehicles/{vin}/stream")
async def websocket_telemetry(websocket: WebSocket, vin: str):
    await app_state.connection_manager.connect(websocket, vin)
    try:
        if vin not in app_state.generators:
            config = VehicleConfig(vehicle_id=vin)
            app_state.generators[vin] = EnhancedTelemetryGenerator(config)

        generator = app_state.generators[vin]
        pipeline = app_state.pipelines.get(vin)

        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "vehicle_id": vin,
                "timestamp": datetime.now().isoformat(),
            }
        )

        while True:
            try:
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                    if data.get("command") == "inject_fault":
                        generator.inject_fault(
                            data.get("fault_type", "overheat"),
                            data.get("severity", 1.0),
                        )
                    elif data.get("command") == "clear_faults":
                        generator.clear_all_faults()
                except asyncio.TimeoutError:
                    pass

                telemetry = generator.step()

                # Convert telemetry numpy values to native Python types
                clean_telemetry = {
                    k: float(v) if hasattr(v, "item") else v
                    for k, v in telemetry.items()
                }

                if pipeline:
                    result = pipeline.process(telemetry)
                    message = {
                        "type": "telemetry",
                        "timestamp": str(result["timestamp"]),
                        "vehicle_id": vin,
                        "data": {
                            "telemetry": clean_telemetry,
                            "anomaly": {
                                "is_anomaly": bool(result["is_anomaly"]),
                                "score": float(result.get("anomaly_score", 0)),
                                "type": str(result["anomaly_type"]),
                                "severity": str(result["severity"]),
                                "failure_risk_pct": float(result["failure_risk_pct"]),
                            },
                            "scoring": {
                                "delta": int(result["score_delta"]),
                                "total": int(result["total_score"]),
                                "feedback": str(result.get("feedback_text", "")),
                            },
                        },
                    }
                else:
                    message = {
                        "type": "telemetry",
                        "timestamp": datetime.now().isoformat(),
                        "vehicle_id": vin,
                        "data": {"telemetry": clean_telemetry},
                    }

                await websocket.send_json(message)
                await asyncio.sleep(1.0)

            except WebSocketDisconnect:
                break
    finally:
        app_state.connection_manager.disconnect(websocket, vin)


@app.get("/api/v1/vehicles/{vin}/stream")
async def stream_telemetry_sse(vin: str, interval_ms: int = 1000):
    async def generate():
        if vin not in app_state.generators:
            config = VehicleConfig(vehicle_id=vin)
            app_state.generators[vin] = EnhancedTelemetryGenerator(config)

        generator = app_state.generators[vin]
        pipeline = app_state.pipelines.get(vin)

        while True:
            telemetry = generator.step()
            if pipeline:
                result = pipeline.process(telemetry)
                # Convert numpy types to Python native types for JSON serialization
                data = {
                    "timestamp": str(result["timestamp"]),
                    "vehicle_id": vin,
                    "telemetry": {
                        k: float(v) if hasattr(v, "item") else v
                        for k, v in telemetry.items()
                    },
                    "is_anomaly": bool(result["is_anomaly"]),
                    "anomaly_type": str(result.get("anomaly_type", "normal")),
                    "severity": str(result["severity"]),
                    "failure_risk_pct": float(result.get("failure_risk_pct", 0)),
                    "score_delta": int(result["score_delta"]),
                    "total_score": int(result["total_score"]),
                    "feedback": str(result.get("feedback_text", "")),
                }
            else:
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "vehicle_id": vin,
                    "telemetry": {
                        k: float(v) if hasattr(v, "item") else v
                        for k, v in telemetry.items()
                    },
                }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(interval_ms / 1000)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/v1/vehicles/{vin}/inject", response_model=FaultInjectionResponse)
async def inject_fault(vin: str, request: FaultInjectionRequest):
    if vin not in app_state.generators:
        config = VehicleConfig(vehicle_id=vin)
        app_state.generators[vin] = EnhancedTelemetryGenerator(config)

    generator = app_state.generators[vin]
    generator.inject_fault(request.fault, request.severity)
    telemetry = generator.step()

    pipeline = app_state.pipelines.get(vin)
    if pipeline:
        result = pipeline.process(telemetry)
    else:
        pipeline = MLPipeline(vin)
        result = pipeline.process(telemetry)

    await app_state.connection_manager.broadcast(
        vin,
        {
            "type": "fault_injection",
            "fault_type": request.fault,
            "severity": request.severity,
        },
    )

    return FaultInjectionResponse(
        vehicle_id=vin,
        fault_type=request.fault,
        severity=request.severity,
        telemetry=telemetry,
        anomaly_detected=result["is_anomaly"],
        anomaly_type=result["anomaly_type"],
        anomaly_severity=result["severity"],
        failure_risk_pct=result["failure_risk_pct"],
        score_delta=result["score_delta"],
        feedback=result["feedback_text"],
    )


@app.delete("/api/v1/vehicles/{vin}/faults")
async def clear_faults(vin: str):
    if vin in app_state.generators:
        app_state.generators[vin].clear_all_faults()
        return {"status": "cleared", "vehicle_id": vin}
    raise HTTPException(status_code=404, detail="Vehicle not found")


@app.delete("/api/v1/vehicles/{vin}/model")
async def delete_model(vin: str):
    """Delete saved model to force retraining on next init."""
    import shutil

    model_path = get_model_path(vin)
    if model_path.exists():
        shutil.rmtree(model_path)
        # Also remove from memory
        if vin in app_state.pipelines:
            del app_state.pipelines[vin]
        return {
            "status": "deleted",
            "vehicle_id": vin,
            "message": f"Model deleted. Call /init to retrain.",
        }
    raise HTTPException(status_code=404, detail="No saved model found for this vehicle")


@app.get("/api/v1/vehicles/{vin}/stats")
async def get_vehicle_stats(vin: str):
    generator = app_state.generators.get(vin)
    pipeline = app_state.pipelines.get(vin)
    if not generator:
        raise HTTPException(status_code=404, detail="Vehicle not initialized")
    return {
        "vehicle_id": vin,
        "driver_profile": generator.config.driver_profile,
        "current_state": {
            "speed_kmh": generator.current_speed_kmh,
            "battery_soc": generator.battery_soc,
            "battery_temp_c": generator.battery_temp_c,
            "wear_index": generator.cumulative_wear_index,
            "active_faults": list(generator.active_faults.keys()),
        },
        "model_trained": pipeline.detector.is_trained if pipeline else False,
        "total_score": pipeline.scorer.total_score if pipeline else 0,
    }


@app.get("/api/v1/vehicles")
async def list_vehicles():
    return {
        "vehicles": [
            {"vehicle_id": vin, "driver_profile": gen.config.driver_profile}
            for vin, gen in app_state.generators.items()
        ],
        "count": len(app_state.generators),
    }


@app.post("/api/v1/vehicles/{vin}/chat", response_model=ChatResponse)
async def chat(vin: str, request: ChatRequest):
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    session_id = request.session_id or str(uuid.uuid4())
    try:
        response = app_state.orchestrator.chat(vin, request.message, session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ueba/report")
async def get_ueba_report():
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    return app_state.orchestrator.get_ueba_report()


# ==================== RAG Endpoints ====================


@app.get("/api/v1/rag/search")
async def rag_search(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
):
    """Search the RAG knowledge base."""
    if not app_state.knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        results = app_state.knowledge_base.semantic_search(query, k=k)
        return {
            "query": query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/faults/{component}")
async def get_component_faults(component: str, k: int = 10):
    """Get known faults for a specific component."""
    if not app_state.knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        results = app_state.knowledge_base.get_faults_by_component(component, k=k)
        return {
            "component": component,
            "faults": results,
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Admin Endpoints ====================


@app.post("/api/v1/admin/build-knowledge-base")
async def build_knowledge_base():
    """Rebuild the RAG knowledge base from data files."""
    try:
        kb = MLKnowledgeBase()

        # Define paths
        faults_json_path = "data/datasets/industry_faults.json"
        faults_csv_path = "data/datasets/industry_faults.csv"
        manual_path = "data/datasets/vehicle_manual.json"

        # Build with available files
        kb.build_knowledge_base(
            faults_json_path=(
                faults_json_path if Path(faults_json_path).exists() else None
            ),
            faults_csv_path=faults_csv_path if Path(faults_csv_path).exists() else None,
            manual_json_path=manual_path if Path(manual_path).exists() else None,
        )
        app_state.knowledge_base = kb
        return {"status": "built", "message": "Knowledge base rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/admin/models")
async def list_saved_models():
    """List all saved vehicle models."""
    models = []
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir() and (model_dir / "metadata.json").exists():
                with open(model_dir / "metadata.json") as f:
                    metadata = json.load(f)
                    models.append(
                        {
                            "vehicle_id": metadata.get("vehicle_id"),
                            "driver_profile": metadata.get("driver_profile"),
                            "training_samples": metadata.get("training_samples"),
                            "is_trained": metadata.get("is_trained"),
                        }
                    )
    return {"models": models, "count": len(models)}


# ==================== Scenario Endpoints ====================

from ml.scenarios import get_scenario_manager, ScenarioManager


@app.get("/api/v1/scenarios")
async def list_scenarios():
    """List all available test scenarios."""
    manager = get_scenario_manager()
    return {"scenarios": manager.list_scenarios()}


@app.post("/api/v1/vehicles/{vin}/scenario/{scenario_id}")
async def start_scenario(vin: str, scenario_id: str):
    """Start a test scenario for a vehicle."""
    if vin not in app_state.pipelines:
        raise HTTPException(status_code=404, detail="Vehicle not initialized")

    manager = get_scenario_manager()
    result = manager.start_scenario(vin, scenario_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Add chatbot notification
    orchestrator = get_orchestrator()
    if "chatbot_message" in result:
        orchestrator.add_notification(vin, result["chatbot_message"])

    return result


@app.get("/api/v1/vehicles/{vin}/scenario/status")
async def get_scenario_status(vin: str):
    """Get current scenario status for a vehicle."""
    manager = get_scenario_manager()
    status = manager.get_scenario_status(vin)

    if not status:
        return {"active": False, "message": "No active scenario"}

    return status


@app.delete("/api/v1/vehicles/{vin}/scenario")
async def stop_scenario(vin: str):
    """Stop active scenario for a vehicle."""
    manager = get_scenario_manager()
    return manager.stop_scenario(vin)


# ==================== Prediction Endpoints ====================


@app.get("/api/v1/vehicles/{vin}/predictions")
async def get_predictions(vin: str):
    """Get active prediction for a vehicle."""
    orchestrator = get_orchestrator()
    prediction = orchestrator.get_prediction(vin)

    if not prediction:
        return {"active": False, "message": "No active prediction"}

    return {"active": True, "prediction": prediction}


@app.post("/api/v1/vehicles/{vin}/predictions/{prediction_id}/accept")
async def accept_prediction(vin: str, prediction_id: str):
    """Accept a prediction warning - routes to Safety Agent."""
    orchestrator = get_orchestrator()
    result = orchestrator.accept_prediction(vin, prediction_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/api/v1/vehicles/{vin}/predictions/{prediction_id}/reject")
async def reject_prediction(vin: str, prediction_id: str):
    """Reject a prediction warning - may route to Diagnosis Agent."""
    orchestrator = get_orchestrator()
    result = orchestrator.reject_prediction(vin, prediction_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ==================== Diagnosis Endpoints ====================


@app.get("/api/v1/vehicles/{vin}/diagnosis")
async def get_diagnosis(vin: str):
    """Get active diagnosis for a vehicle."""
    orchestrator = get_orchestrator()
    diagnosis = orchestrator.get_diagnosis(vin)

    if not diagnosis:
        return {"active": False, "message": "No active diagnosis"}

    return {"active": True, "diagnosis": diagnosis}


@app.post("/api/v1/vehicles/{vin}/diagnosis/{diagnosis_id}/confirm")
async def confirm_diagnosis_service(vin: str, diagnosis_id: str):
    """Confirm service scheduling for a diagnosis."""
    orchestrator = get_orchestrator()
    result = orchestrator.confirm_service(vin, diagnosis_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/api/v1/vehicles/{vin}/diagnosis/{diagnosis_id}/decline")
async def decline_diagnosis_service(vin: str, diagnosis_id: str):
    """Decline service scheduling for a diagnosis."""
    orchestrator = get_orchestrator()
    result = orchestrator.decline_service(vin, diagnosis_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ==================== Notification Endpoints ====================


@app.get("/api/v1/vehicles/{vin}/notifications")
async def get_notifications(vin: str, unread_only: bool = Query(False)):
    """Get notifications for a vehicle (for chatbot display)."""
    orchestrator = get_orchestrator()
    notifications = orchestrator.get_notifications(vin, unread_only)
    return {"notifications": notifications, "count": len(notifications)}


@app.post("/api/v1/vehicles/{vin}/notifications/{notification_id}/read")
async def mark_notification_read(vin: str, notification_id: str):
    """Mark a notification as read."""
    orchestrator = get_orchestrator()
    success = orchestrator.mark_notification_read(vin, notification_id)
    return {"success": success}


# ==================== Enhanced Scenario WebSocket ====================

# Prediction timeout in seconds (auto-route to Diagnosis if no response)
PREDICTION_TIMEOUT_SECONDS = 30


@app.websocket("/api/v1/vehicles/{vin}/scenario-stream")
async def websocket_scenario_stream(websocket: WebSocket, vin: str):
    """
    WebSocket endpoint for streaming scenario telemetry with predictions.

    Supports commands:
    - {"command": "inject_fault", "fault_type": "overheat", "severity": 1.0}
    - {"command": "start_scenario", "scenario_id": "aggressive_driver"}
    - {"command": "stop_scenario"}
    - {"command": "accept_prediction", "prediction_id": "pred-..."}
    - {"command": "reject_prediction", "prediction_id": "pred-..."}
    - {"command": "confirm_service", "diagnosis_id": "diag-..."}
    - {"command": "decline_service", "diagnosis_id": "diag-..."}

    Auto-routes to Diagnosis Agent if prediction is not responded to within timeout.
    """
    await app_state.connection_manager.connect(websocket, vin)

    try:
        if vin not in app_state.pipelines:
            await websocket.send_json({"error": "Vehicle not initialized"})
            await websocket.close()
            return

        pipeline = app_state.pipelines[vin]
        generator = app_state.generators.get(vin)
        orchestrator = get_orchestrator()
        scenario_manager = get_scenario_manager()

        if not generator:
            await websocket.send_json({"error": "Generator not found"})
            await websocket.close()
            return

        # Track prediction state for auto-timeout
        prediction_created_at = None
        active_prediction_id = None
        diagnosis_sent = False

        # Send connection confirmation
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "vehicle_id": vin,
                "available_commands": [
                    "inject_fault",
                    "start_scenario",
                    "stop_scenario",
                    "accept_prediction",
                    "reject_prediction",
                    "confirm_service",
                    "decline_service",
                ],
                "timestamp": datetime.now().isoformat(),
            }
        )

        while True:
            # Check for incoming commands
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                command = data.get("command")

                if command == "inject_fault":
                    generator.inject_fault(
                        data.get("fault_type", "overheat"), data.get("severity", 1.0)
                    )
                    await websocket.send_json(
                        {
                            "type": "command_response",
                            "command": "inject_fault",
                            "status": "success",
                            "message": f"Fault '{data.get('fault_type')}' injected",
                        }
                    )

                elif command == "start_scenario":
                    result = scenario_manager.start_scenario(
                        vin, data.get("scenario_id")
                    )
                    await websocket.send_json({"type": "scenario_started", **result})

                elif command == "stop_scenario":
                    result = scenario_manager.stop_scenario(vin)
                    await websocket.send_json({"type": "scenario_stopped", **result})

                elif command == "accept_prediction":
                    pred_id = data.get("prediction_id")
                    if pred_id:
                        result = orchestrator.accept_prediction(vin, pred_id)
                        await websocket.send_json(
                            {
                                "type": "prediction_accepted",
                                "routed_to": "safety_agent",
                                **result,
                            }
                        )
                        # Clear prediction tracking
                        active_prediction_id = None
                        prediction_created_at = None
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "prediction_id required",
                            }
                        )

                elif command == "reject_prediction":
                    pred_id = data.get("prediction_id")
                    if pred_id:
                        result = orchestrator.reject_prediction(vin, pred_id)
                        await websocket.send_json(
                            {
                                "type": "prediction_rejected",
                                **result,
                            }
                        )
                        # Check if routed to diagnosis
                        if result.get("routed_to") == "diagnosis_agent":
                            diagnosis_sent = True
                            # Get diagnosis details for redirect
                            diagnosis = result.get("diagnosis", {})
                            # Store context for chatbot
                            orchestrator.set_chat_context(
                                vehicle_id=vin,
                                diagnosis_id=diagnosis.get("diagnosis_id"),
                                component=diagnosis.get("component", "vehicle"),
                                summary=diagnosis.get("summary", "Issue detected"),
                                urgency=diagnosis.get("urgency", "medium"),
                                estimated_cost=diagnosis.get("estimated_cost", "TBD"),
                            )
                            # Send redirect signal to frontend
                            await websocket.send_json(
                                {
                                    "type": "redirect_to_chat",
                                    "vehicle_id": vin,
                                    "auto_start_scheduling": True,
                                    "context": {
                                        "diagnosis_id": diagnosis.get("diagnosis_id"),
                                        "component": diagnosis.get("component"),
                                        "summary": diagnosis.get("summary"),
                                        "urgency": diagnosis.get("urgency"),
                                        "estimated_cost": diagnosis.get(
                                            "estimated_cost"
                                        ),
                                    },
                                }
                            )
                        # Clear prediction tracking
                        active_prediction_id = None
                        prediction_created_at = None
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "prediction_id required",
                            }
                        )

                elif command == "confirm_service":
                    diag_id = data.get("diagnosis_id")
                    if diag_id:
                        result = orchestrator.confirm_service(vin, diag_id)
                        await websocket.send_json(
                            {
                                "type": "service_confirmed",
                                **result,
                            }
                        )

                        # Get diagnosis for redirect context
                        diagnosis = orchestrator.get_diagnosis(vin)
                        if diagnosis:
                            # Store context for chatbot
                            orchestrator.set_chat_context(
                                vehicle_id=vin,
                                diagnosis_id=diag_id,
                                component=diagnosis.get("component", "vehicle"),
                                summary=diagnosis.get("summary", "Service needed"),
                                urgency=diagnosis.get("urgency", "medium"),
                                estimated_cost=diagnosis.get("estimated_cost", "TBD"),
                            )

                            # Check if this is a critical brake scenario - trigger voice call
                            component = diagnosis.get("component", "").lower()
                            urgency = diagnosis.get("urgency", "scheduled")
                            # Urgency values are: immediate, soon, scheduled (NOT high/critical)
                            is_critical_brake = (
                                "brake" in component or "fade" in component
                            ) and urgency in ["immediate", "soon"]

                            if is_critical_brake:
                                # Send voice call trigger for critical brake scenarios
                                await websocket.send_json(
                                    {
                                        "type": "voice_call_trigger",
                                        "vehicle_id": vin,
                                        "alert_type": "brake_fade",
                                        "context": {
                                            "diagnosis_id": diag_id,
                                            "component": diagnosis.get("component"),
                                            "summary": diagnosis.get("summary"),
                                            "urgency": diagnosis.get("urgency"),
                                            "estimated_cost": diagnosis.get(
                                                "estimated_cost"
                                            ),
                                            "brake_efficiency": 15,  # Simulated low efficiency
                                            "temperature": 350,  # Simulated high temp
                                        },
                                    }
                                )
                            else:
                                # Send redirect signal to frontend for non-critical
                                await websocket.send_json(
                                    {
                                        "type": "redirect_to_chat",
                                        "vehicle_id": vin,
                                        "auto_start_scheduling": True,
                                        "context": {
                                            "diagnosis_id": diag_id,
                                            "component": diagnosis.get("component"),
                                            "summary": diagnosis.get("summary"),
                                            "urgency": diagnosis.get("urgency"),
                                            "estimated_cost": diagnosis.get(
                                                "estimated_cost"
                                            ),
                                        },
                                    }
                                )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "diagnosis_id required",
                            }
                        )

                elif command == "decline_service":
                    diag_id = data.get("diagnosis_id")
                    if diag_id:
                        result = orchestrator.decline_service(vin, diag_id)
                        await websocket.send_json(
                            {
                                "type": "service_declined",
                                **result,
                            }
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "diagnosis_id required",
                            }
                        )

            except asyncio.TimeoutError:
                pass

            # Check for prediction timeout - auto-route to Diagnosis Agent
            if active_prediction_id and prediction_created_at:
                elapsed = (datetime.now() - prediction_created_at).total_seconds()
                if elapsed >= PREDICTION_TIMEOUT_SECONDS and not diagnosis_sent:
                    # Auto-reject and route to diagnosis
                    await websocket.send_json(
                        {
                            "type": "prediction_timeout",
                            "message": f"No response received in {PREDICTION_TIMEOUT_SECONDS} seconds. Auto-routing to Diagnosis Agent.",
                            "prediction_id": active_prediction_id,
                        }
                    )
                    result = orchestrator.reject_prediction(vin, active_prediction_id)
                    await websocket.send_json(
                        {
                            "type": "auto_diagnosis",
                            "reason": "prediction_timeout",
                            **result,
                        }
                    )

                    # Check if this is a critical brake scenario - trigger voice call
                    diagnosis = result.get("diagnosis", {})
                    component = diagnosis.get("component", "").lower()
                    urgency = diagnosis.get("urgency", "scheduled")

                    # Debug logging
                    print(
                        f"🔍 Auto-diagnosis check: component='{component}', urgency='{urgency}'"
                    )

                    # Urgency values are: immediate, soon, scheduled (NOT high/critical)
                    is_critical_brake = (
                        "brake" in component or "fade" in component
                    ) and urgency in ["immediate", "soon"]

                    if is_critical_brake:
                        print(f"🚨 Triggering voice call for brake diagnosis!")
                        await websocket.send_json(
                            {
                                "type": "voice_call_trigger",
                                "vehicle_id": vin,
                                "alert_type": "brake_fade",
                                "context": {
                                    "diagnosis_id": diagnosis.get("diagnosis_id"),
                                    "component": diagnosis.get("component"),
                                    "summary": diagnosis.get("summary"),
                                    "urgency": diagnosis.get("urgency"),
                                    "estimated_cost": diagnosis.get("estimated_cost"),
                                    "brake_efficiency": 15,
                                    "temperature": 350,
                                },
                            }
                        )

                    diagnosis_sent = True
                    active_prediction_id = None
                    prediction_created_at = None

            # Generate telemetry (with scenario modifiers if active)
            telemetry = generator.step()

            # Apply scenario modifiers if active
            scenario_result = scenario_manager.get_current_modifiers(vin)
            if scenario_result:
                for key, value in scenario_result.get("modifiers", {}).items():
                    if key in telemetry:
                        telemetry[key] = value

            # Process through ML pipeline
            result = pipeline.process(telemetry)

            # Convert numpy types
            def convert_value(v):
                if hasattr(v, "item"):
                    return v.item()
                return v

            # Build response
            response = {
                "type": "telemetry",
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": vin,
                "data": {
                    "telemetry": {k: convert_value(v) for k, v in telemetry.items()},
                    "anomaly": {
                        "is_anomaly": convert_value(result.get("is_anomaly", False)),
                        "type": str(result.get("anomaly_type", "normal")),
                        "severity": str(result.get("severity", "low")),
                        "failure_risk_pct": convert_value(
                            result.get("failure_risk_pct", 0)
                        ),
                    },
                    "scoring": {
                        "delta": convert_value(result.get("score_delta", 0)),
                        "total": convert_value(result.get("total_score", 0)),
                        "feedback": str(result.get("feedback_text", "")),
                    },
                },
            }

            # Add scenario info if active
            if scenario_result:
                response["scenario"] = {
                    "phase": scenario_result.get("phase"),
                    "event": scenario_result.get("event_type"),
                    "description": scenario_result.get("description"),
                }

                # Check if prediction should trigger
                if "trigger_prediction" in scenario_result:
                    pred_data = scenario_result["trigger_prediction"]
                    prediction = orchestrator.create_prediction(
                        vehicle_id=vin,
                        component=pred_data["component"],
                        anomaly_type=pred_data["anomaly_type"],
                        severity=pred_data["severity"],
                        days_to_failure=pred_data["days_to_failure"],
                        message=pred_data["message"],
                        requires_service=pred_data["requires_service"],
                    )
                    response["prediction"] = prediction
                    response["prediction"]["actions"] = [
                        "accept_prediction",
                        "reject_prediction",
                    ]
                    response["prediction"][
                        "timeout_seconds"
                    ] = PREDICTION_TIMEOUT_SECONDS

                    # Track for auto-timeout
                    active_prediction_id = prediction["prediction_id"]
                    prediction_created_at = datetime.now()
                    diagnosis_sent = False

                # Check if scenario complete
                if scenario_result.get("scenario_complete"):
                    response["scenario"]["complete"] = True

            # Add any pending notifications
            notifications = orchestrator.get_notifications(vin, unread_only=True)
            if notifications:
                response["notifications"] = notifications

            await websocket.send_json(response)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        app_state.connection_manager.disconnect(websocket, vin)
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        app_state.connection_manager.disconnect(websocket, vin)


# ==================== Scheduling & Service Center Endpoints ====================

from db.database import get_database
from ml.demand_forecasting import get_demand_forecaster


class AppointmentBookRequest(BaseModel):
    slot_id: str
    center_id: str
    component: str
    diagnosis_summary: str
    estimated_cost: str
    urgency: str = "medium"
    notes: str = ""


class AppointmentRescheduleRequest(BaseModel):
    new_slot_id: str
    reason: str = ""


class FeedbackSubmitRequest(BaseModel):
    appointment_id: str
    rating: int = Field(ge=1, le=5)
    comments: str = ""


@app.get("/api/v1/service-centers")
async def list_service_centers():
    """List all service centers."""
    db = get_database()
    centers = db.get_service_centers()
    return {"centers": centers, "count": len(centers)}


@app.get("/api/v1/service-centers/{center_id}/availability")
async def get_center_availability(
    center_id: str, date: Optional[str] = None, component: Optional[str] = None
):
    """Get available slots for a service center."""
    db = get_database()
    slots = db.get_available_slots(
        center_id=center_id, date=date, component=component, limit=20
    )
    return {"center_id": center_id, "slots": slots, "count": len(slots)}


@app.get("/api/v1/appointments/{vin}")
async def get_vehicle_appointments(vin: str, status: Optional[str] = None):
    """Get all appointments for a vehicle."""
    orchestrator = get_orchestrator()
    result = orchestrator.get_appointments(vin)

    if status:
        result["appointments"] = [
            a for a in result.get("appointments", []) if a["status"] == status
        ]

    return result


@app.post("/api/v1/appointments/{vin}/propose")
async def propose_appointment_slots(vin: str, component: str, urgency: str = "medium"):
    """Propose available slots for scheduling (chatbot-friendly)."""
    orchestrator = get_orchestrator()

    # Get latest diagnosis if any
    diagnosis = orchestrator.get_diagnosis(vin)
    diagnosis_summary = (
        diagnosis.get("summary", "Vehicle service") if diagnosis else "Vehicle service"
    )
    estimated_cost = (
        diagnosis.get("estimated_cost", "$100-$300") if diagnosis else "$100-$300"
    )

    result = orchestrator.propose_scheduling(
        vehicle_id=vin,
        component=component,
        diagnosis_summary=diagnosis_summary,
        estimated_cost=estimated_cost,
        urgency=urgency,
    )
    return result


@app.post("/api/v1/appointments/{vin}/book")
async def book_appointment(vin: str, request: AppointmentBookRequest):
    """Book a service appointment."""
    orchestrator = get_orchestrator()
    result = orchestrator.book_appointment(
        vehicle_id=vin,
        slot_id=request.slot_id,
        center_id=request.center_id,
        component=request.component,
        diagnosis_summary=request.diagnosis_summary,
        estimated_cost=request.estimated_cost,
        urgency=request.urgency,
        notes=request.notes,
    )
    return result


@app.put("/api/v1/appointments/{appointment_id}/reschedule")
async def reschedule_appointment(
    appointment_id: str, request: AppointmentRescheduleRequest
):
    """Reschedule an existing appointment."""
    orchestrator = get_orchestrator()
    if orchestrator.scheduling_agent:
        return orchestrator.scheduling_agent.reschedule_appointment(
            appointment_id=appointment_id,
            new_slot_id=request.new_slot_id,
            reason=request.reason,
        )
    return {"error": "Scheduling agent not available"}


@app.delete("/api/v1/appointments/{appointment_id}")
async def cancel_appointment(appointment_id: str, reason: str = ""):
    """Cancel an appointment."""
    orchestrator = get_orchestrator()
    return orchestrator.cancel_appointment(appointment_id, reason)


@app.get("/api/v1/appointments/{appointment_id}/progress")
async def get_service_progress(appointment_id: str):
    """Get service progress (for tracking)."""
    orchestrator = get_orchestrator()
    return orchestrator.get_service_progress(appointment_id)


@app.put("/api/v1/appointments/{appointment_id}/status")
async def update_service_status(appointment_id: str, status: str):
    """Update service status (scheduled, confirmed, in_progress, completed)."""
    orchestrator = get_orchestrator()
    return orchestrator.update_service_status(appointment_id, status)


# ==================== Chat Context & Auto-Scheduling ====================


@app.post("/api/v1/vehicles/{vin}/chat/start-scheduling")
async def start_scheduling_conversation(vin: str):
    """
    Start a scheduling conversation with diagnosis context.
    Called after auto-redirect from diagnosis to chatbot.
    Returns intro message, proposed slots, and quick actions.
    """
    orchestrator = get_orchestrator()
    return orchestrator.start_scheduling_conversation(vin)


@app.get("/api/v1/vehicles/{vin}/chat/context")
async def get_chat_context(vin: str):
    """Get stored chat context for a vehicle."""
    orchestrator = get_orchestrator()
    context = orchestrator.get_chat_context(vin)
    if context:
        return {"has_context": True, "context": context}
    return {"has_context": False, "context": None}


@app.delete("/api/v1/vehicles/{vin}/chat/context")
async def clear_chat_context(vin: str):
    """Clear stored chat context after conversation ends."""
    orchestrator = get_orchestrator()
    orchestrator.clear_chat_context(vin)
    return {"success": True, "message": "Chat context cleared"}


# ==================== Feedback Endpoints ====================


@app.post("/api/v1/feedback/{vin}/initiate")
async def initiate_feedback(vin: str, appointment_id: str):
    """Initiate feedback collection for completed service."""
    orchestrator = get_orchestrator()
    return orchestrator.initiate_feedback(appointment_id)


@app.post("/api/v1/feedback/{vin}")
async def submit_feedback(vin: str, request: FeedbackSubmitRequest):
    """Submit customer feedback and rating."""
    orchestrator = get_orchestrator()
    return orchestrator.submit_feedback(
        appointment_id=request.appointment_id,
        vehicle_id=vin,
        rating=request.rating,
        comments=request.comments,
    )


@app.get("/api/v1/vehicles/{vin}/service-history")
async def get_service_history(vin: str):
    """Get complete service history for a vehicle."""
    orchestrator = get_orchestrator()
    return orchestrator.get_service_history(vin)


# ==================== Demand Forecasting Endpoints ====================


@app.get("/api/v1/demand/forecast/{center_id}")
async def get_demand_forecast(center_id: str):
    """Get weekly demand forecast for a service center."""
    forecaster = get_demand_forecaster()
    return forecaster.predict_weekly_demand(center_id)


@app.get("/api/v1/demand/peak-hours/{center_id}")
async def get_peak_hours(center_id: str):
    """Get peak and off-peak hours for a service center."""
    forecaster = get_demand_forecaster()
    return forecaster.get_peak_hours(center_id)


@app.get("/api/v1/demand/optimal-slots")
async def get_optimal_slots(
    component: str, urgency: str = "medium", center_id: Optional[str] = None
):
    """Get optimal appointment slots based on demand forecasting."""
    forecaster = get_demand_forecaster()
    return forecaster.suggest_optimal_slots(
        component=component, urgency=urgency, preferred_center_id=center_id
    )


@app.get("/api/v1/demand/capacity/{center_id}")
async def get_capacity_analysis(center_id: str):
    """Get capacity analysis and staffing recommendations."""
    forecaster = get_demand_forecaster()
    return forecaster.calculate_optimal_capacity(center_id)


# ==================== Admin Dashboard Endpoints ====================

# from ml.labor_forecasting import get_labor_forecaster, get_appointment_optimizer


# @app.get("/api/v1/admin/labor-forecast")
# async def admin_labor_forecast(days: int = 7, center_id: Optional[str] = None):
#     """
#     Get labor hours forecast for next N days.
#     Used by admin dashboard for staffing decisions.
#     """
#     forecaster = get_labor_forecaster()
#     return forecaster.predict_labor_hours(days=days, center_id=center_id)


# @app.get("/api/v1/admin/staffing-recommendations")
# async def admin_staffing_recommendations(days: int = 7):
#     """Get staffing up/down recommendations based on demand."""
#     forecaster = get_labor_forecaster()
#     predictions = forecaster.predict_labor_hours(days=days)
#     return forecaster.get_staffing_recommendation(predictions["predictions"])


# @app.get("/api/v1/admin/capacity-heatmap")
# async def admin_capacity_heatmap(days: int = 7):
#     """Get capacity heatmap for visualization (hourly capacity by day)."""
#     optimizer = get_appointment_optimizer()
#     return {"days": days, "heatmap": optimizer.get_capacity_heatmap(days=days)}


# @app.post("/api/v1/admin/optimize-schedule")
# async def admin_optimize_schedule(date: str, center_id: Optional[str] = None):
#     """
#     Run schedule optimization for a specific date.
#     Assigns technicians optimally using CSP algorithm.
#     """
#     optimizer = get_appointment_optimizer()
#     return optimizer.optimize_schedule(date=date, center_id=center_id)


# @app.get("/api/v1/admin/appointments-overview")
# async def admin_appointments_overview(days: int = 7):
#     """Get overview of appointments for admin dashboard."""
#     db = get_database()

#     today = datetime.now().date()
#     overview = []

#     for i in range(days):
#         date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
#         appointments = db.get_appointments(status="scheduled")
#         day_appts = [a for a in appointments if a.get("scheduled_date") == date]

#         completed = db.get_appointments(status="completed")
#         day_completed = [a for a in completed if a.get("scheduled_date") == date]

#         overview.append(
#             {
#                 "date": date,
#                 "day": (today + timedelta(days=i)).strftime("%A"),
#                 "scheduled": len(day_appts),
#                 "completed": len(day_completed),
#                 "components": _count_by_key(day_appts, "component"),
#                 "urgencies": _count_by_key(day_appts, "urgency"),
#             }
#         )

#     return {
#         "days": days,
#         "overview": overview,
#         "total_scheduled": sum(d["scheduled"] for d in overview),
#         "total_completed": sum(d["completed"] for d in overview),
#     }


@app.get("/api/v1/admin/appointments-list")
async def admin_appointments_list(
    status: Optional[str] = None, date: Optional[str] = None, days: int = 7
):
    """
    Get detailed list of appointments with technician assignments.
    Shows pending, scheduled, and completed appointments.
    """
    db = get_database()

    # Get all appointments
    all_appointments = []

    if status:
        all_appointments = db.get_appointments(status=status)
    else:
        # Get both scheduled and completed
        scheduled = db.get_appointments(status="scheduled")
        completed = db.get_appointments(status="completed")
        in_progress = db.get_appointments(status="in_progress")
        all_appointments = scheduled + completed + in_progress

    # Filter by date range if no specific date
    if date:
        all_appointments = [
            a for a in all_appointments if a.get("scheduled_date") == date
        ]
    else:
        # Filter to next N days for scheduled, last N days for completed
        today = datetime.now().date()
        start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")
        all_appointments = [
            a
            for a in all_appointments
            if start_date <= a.get("scheduled_date", "") <= end_date
        ]

    # Sort by date and time
    all_appointments.sort(
        key=lambda x: (x.get("scheduled_date", ""), x.get("scheduled_time", "")),
        reverse=True,
    )

    # Add technician assignment info (from optimization results if available)
    from ml.labor_forecasting import TECHNICIANS

    tech_map = {t["id"]: t["name"] for t in TECHNICIANS}

    for appt in all_appointments:
        # Check if technician was assigned (you could store this in DB)
        appt["assigned_technician"] = appt.get("assigned_technician", "Unassigned")
        appt["technician_id"] = appt.get("technician_id", None)

    return {
        "appointments": all_appointments,
        "total": len(all_appointments),
        "by_status": {
            "scheduled": len(
                [a for a in all_appointments if a.get("status") == "scheduled"]
            ),
            "in_progress": len(
                [a for a in all_appointments if a.get("status") == "in_progress"]
            ),
            "completed": len(
                [a for a in all_appointments if a.get("status") == "completed"]
            ),
        },
    }


def _count_by_key(items: list, key: str) -> dict:
    """Count items by key value."""
    counts = {}
    for item in items:
        val = item.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


@app.get("/api/v1/admin/technicians")
async def admin_get_technicians():
    """Get list of technicians with their specialties and capacity."""
    from ml.labor_forecasting import TECHNICIANS

    return {"technicians": TECHNICIANS}


# Alias route for scheduling/appointments status (frontend compatibility)
@app.put("/api/v1/scheduling/appointments/{appointment_id}/status")
async def update_scheduling_status(appointment_id: str, status: str):
    """Alias endpoint for updating appointment status from service center."""
    orchestrator = get_orchestrator()
    return orchestrator.update_service_status(appointment_id, status)


# ==================== Smart Chat Endpoints ====================


@app.get("/api/v1/vehicles/{vin}/check-date-availability")
async def check_date_availability(vin: str, date: str, component: str = "general"):
    """
    Check availability for a specific date with explanations.
    Used by chatbot for intelligent responses about scheduling.
    """
    forecaster = get_labor_forecaster()
    return forecaster.check_date_availability(date=date, component=component)


@app.get("/api/v1/vehicles/{vin}/suggest-optimal-date")
async def suggest_optimal_date(
    vin: str,
    component: str,
    urgency: str = "medium",
    preferred_date: Optional[str] = None,
    days_range: int = 7,
):
    """
    Suggest optimal appointment date using scoring algorithm.
    Used by chatbot for smart scheduling recommendations.
    """
    forecaster = get_labor_forecaster()
    return forecaster.suggest_optimal_date(
        component=component,
        urgency=urgency,
        preferred_date=preferred_date,
        days_range=days_range,
    )


# ==================== Service Ticket Endpoints ====================


@app.get("/api/v1/tickets/{vin}")
async def get_vehicle_ticket(vin: str):
    """Get active service ticket for a vehicle."""
    from agents.service_tracker_agent import get_service_tracker

    tracker = get_service_tracker()
    ticket = tracker.get_vehicle_ticket(vin)

    if not ticket:
        return {"error": "No active service ticket found", "vin": vin}

    return ticket


@app.get("/api/v1/tickets/{ticket_id}/status")
async def get_ticket_status(ticket_id: str):
    """Get detailed status of a service ticket."""
    from agents.service_tracker_agent import get_service_tracker

    tracker = get_service_tracker()
    status = tracker.get_ticket_status(ticket_id)

    if not status:
        return {"error": "Ticket not found", "ticket_id": ticket_id}

    return status


@app.get("/api/v1/tickets/{ticket_id}/timeline")
async def get_ticket_timeline(ticket_id: str):
    """Get visual timeline for a service ticket."""
    from agents.service_tracker_agent import get_service_tracker

    tracker = get_service_tracker()
    timeline = tracker.get_ticket_timeline(ticket_id)

    return {"ticket_id": ticket_id, "timeline": timeline}


@app.post("/api/v1/tickets/{ticket_id}/update")
async def update_ticket_status(
    ticket_id: str,
    status: str,
    note: str = "",
    technician_notes: Optional[str] = None,
):
    """
    Webhook endpoint to update service ticket status.
    Called by external systems (e.g., repair shop, parts supplier).
    """
    from agents.service_tracker_agent import get_service_tracker

    tracker = get_service_tracker()
    result = tracker.update_status(
        ticket_id=ticket_id,
        new_status=status,
        note=note,
        technician_notes=technician_notes,
    )

    # Notify via WebSocket if successful
    if result.get("success"):
        db = get_database()
        ticket = db.get_ticket_by_id(ticket_id)
        if ticket:
            vehicle_id = ticket.get("vehicle_id")
            # Broadcast to connected clients
            await orchestrator.notify_clients(
                vehicle_id,
                {
                    "type": "ticket_update",
                    "ticket_id": ticket_id,
                    "status": status,
                    "message": result.get("message", ""),
                    "estimated_completion": result.get("estimated_completion"),
                    "notification_type": result.get("notification_type", "update"),
                },
            )

    return result


@app.post("/api/v1/tickets/create")
async def create_service_ticket(
    appointment_id: str,
    vehicle_id: str,
    technician_id: Optional[str] = None,
    service_type: Optional[str] = None,
):
    """Create a new service ticket for an appointment."""
    from agents.service_tracker_agent import get_service_tracker

    tracker = get_service_tracker()
    ticket = tracker.create_ticket(
        appointment_id=appointment_id,
        vehicle_id=vehicle_id,
        technician_id=technician_id,
        service_type=service_type,
    )

    return ticket


@app.get("/api/v1/tickets")
async def get_all_tickets(status: Optional[str] = None):
    """Get all service tickets with optional status filter."""
    db = get_database()
    tickets = db.get_all_tickets(status=status)
    return {"tickets": tickets, "count": len(tickets)}


# ==================== User Profile Endpoints ====================


@app.get("/api/v1/users/{vin}/profile")
async def get_user_profile(vin: str):
    """Get user profile for personalized feedback."""
    db = get_database()
    profile = db.get_user_profile(vin)

    if not profile:
        return {"error": "Profile not found", "vin": vin}

    return profile


@app.put("/api/v1/users/{vin}/preferences")
async def update_user_preferences(
    vin: str,
    driving_style: Optional[str] = None,
    preferences: Optional[str] = None,  # JSON string
):
    """Update user preferences."""
    db = get_database()

    prefs_dict = None
    if preferences:
        import json

        prefs_dict = json.loads(preferences)

    result = db.create_or_update_user_profile(
        vehicle_id=vin,
        driving_style=driving_style,
        preferences=prefs_dict,
    )

    return result


@app.post("/api/v1/users/{vin}/feedback")
async def add_user_feedback(
    vin: str,
    sentiment_score: float,
    pain_points: str,  # JSON array string
    positive_points: str,  # JSON array string
    service_type: str,
):
    """Add analyzed feedback to user profile."""
    import json

    db = get_database()

    result = db.add_feedback_to_profile(
        vehicle_id=vin,
        sentiment_score=sentiment_score,
        pain_points=json.loads(pain_points),
        positive_points=json.loads(positive_points),
        service_type=service_type,
    )


# ==================== Voice Agent Endpoints ====================

from agents.voice_agent import get_voice_agent


class VoiceCallInitRequest(BaseModel):
    alert_type: Optional[str] = "brake_fade"
    owner_name: Optional[str] = "Alex"
    brake_efficiency: Optional[float] = 15


class VoiceUserInputRequest(BaseModel):
    user_text: Optional[str] = ""
    detected_intent: Optional[str] = None


@app.post("/api/v1/voice/{vin}/initiate-call")
async def initiate_voice_call(vin: str, request: VoiceCallInitRequest):
    """
    Initiate a voice call for a critical alert.
    Used when brake fade or other critical issue is detected.
    """
    voice_agent = get_voice_agent()

    alert_data = {"brake_efficiency": request.brake_efficiency, "vehicle_id": vin}

    result = voice_agent.initiate_call(
        vehicle_id=vin,
        alert_type=request.alert_type,
        alert_data=alert_data,
        owner_name=request.owner_name,
    )

    return result


@app.post("/api/v1/voice/{call_id}/answer")
async def answer_voice_call(call_id: str):
    """User answers the call - start conversation."""
    voice_agent = get_voice_agent()
    return voice_agent.answer_call(call_id)


@app.post("/api/v1/voice/{call_id}/input")
async def voice_user_input(call_id: str, request: VoiceUserInputRequest):
    """
    Process user's voice input (transcribed text).
    Returns AI response with optional TTS audio.
    """
    voice_agent = get_voice_agent()
    return voice_agent.process_user_input(
        call_id=call_id,
        user_text=request.user_text,
        detected_intent=request.detected_intent,
    )


@app.post("/api/v1/voice/{call_id}/end")
async def end_voice_call(call_id: str):
    """End an active voice call."""
    voice_agent = get_voice_agent()
    return voice_agent.end_call(call_id)


@app.get("/api/v1/voice/{call_id}/status")
async def get_voice_call_status(call_id: str):
    """Get current voice call status."""
    voice_agent = get_voice_agent()
    return voice_agent.get_call_status(call_id)


@app.get("/api/v1/voice/{call_id}/transcript")
async def get_voice_transcript(call_id: str):
    """Get full call transcript."""
    voice_agent = get_voice_agent()
    return voice_agent.get_transcript(call_id)


@app.get("/api/v1/voice/vapi-config")
async def get_vapi_config():
    """Get Vapi.ai configuration for scalable voice integration."""
    voice_agent = get_voice_agent()
    return voice_agent.get_vapi_config()


# ==================== CAPA/RCA Endpoints ====================

from agents.capa_agent import get_capa_agent


@app.get("/api/v1/capa/reports")
async def get_capa_reports(
    status: Optional[str] = Query(None, description="Filter by status (open, closed)"),
    component: Optional[str] = Query(None, description="Filter by component"),
    limit: int = Query(50, ge=1, le=100),
):
    """Get CAPA reports with optional filtering."""
    capa_agent = get_capa_agent()
    reports = capa_agent.get_all_reports(
        status=status, component=component, limit=limit
    )
    return {
        "reports": reports,
        "count": len(reports),
        "filters": {"status": status, "component": component},
    }


@app.get("/api/v1/capa/reports/{capa_id}")
async def get_capa_report_by_id(capa_id: str):
    """Get a specific CAPA report."""
    capa_agent = get_capa_agent()
    report = capa_agent.get_report_by_id(capa_id)
    if not report:
        raise HTTPException(status_code=404, detail="CAPA report not found")
    return report


@app.get("/api/v1/capa/pattern-analysis/{component}")
async def get_capa_pattern_analysis(
    component: str,
    region: Optional[str] = Query(None, description="Optional region filter"),
):
    """
    Analyze patterns across vehicles for a component.
    This is the '50 vehicles in mountainous regions' analysis.
    """
    capa_agent = get_capa_agent()
    return capa_agent.find_pattern_analysis(component=component, region=region)


class CAPAGenerateRequest(BaseModel):
    diagnosis_summary: str
    failure_mode: str
    region: str = "Mountainous"
    vehicle_data: Optional[Dict[str, Any]] = None


@app.post("/api/v1/capa/generate/{vin}")
async def generate_capa_report(vin: str, component: str, request: CAPAGenerateRequest):
    """
    Generate a CAPA report from diagnosis data.
    Called after service to close the manufacturing feedback loop.
    """
    capa_agent = get_capa_agent()
    return capa_agent.generate_capa_from_diagnosis(
        vehicle_id=vin,
        component=component,
        diagnosis_summary=request.diagnosis_summary,
        failure_mode=request.failure_mode,
        region=request.region,
        vehicle_data=request.vehicle_data,
    )


@app.get("/api/v1/capa/manufacturing-summary")
async def get_manufacturing_summary():
    """Get summary for manufacturing/factory dashboard."""
    capa_agent = get_capa_agent()
    return capa_agent.get_manufacturing_summary()


# ==================== Enhanced UEBA Endpoints ====================


@app.get("/api/v1/ueba/alerts")
async def get_ueba_alerts():
    """Get all UEBA security alerts."""
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    ueba = app_state.orchestrator.ueba
    return {
        "alerts": ueba.get_alerts(),
        "count": len(ueba.get_alerts()),
    }


@app.get("/api/v1/ueba/agent-logs")
async def get_ueba_agent_logs():
    """Get agent action logs for security dashboard."""
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    ueba = app_state.orchestrator.ueba
    return {
        "action_log": ueba.action_history[-100:],  # Last 100 actions
        "total_actions": len(ueba.action_history),
    }


class RogueActionRequest(BaseModel):
    agent: str = "scheduling"
    action: str = "delete_logs"
    target: str = "vehicle_telematics"


@app.post("/api/v1/ueba/inject-rogue-action")
async def inject_rogue_action(request: RogueActionRequest):
    """
    Inject a rogue agent action for demo.
    The UEBA monitor will detect and block this.
    """
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    ueba_monitor = app_state.orchestrator.ueba_monitor

    # Log the rogue action - it will be flagged
    ueba_monitor.log_action(
        agent=request.agent,
        action=request.action,
        details={
            "target": request.target,
            "reason": "Attempted unauthorized access",
            "injected_for_demo": True,
        },
    )

    # Get the alert that was created
    alerts = ueba_monitor.get_alerts()
    latest_alert = alerts[-1] if alerts else None

    return {
        "success": True,
        "blocked": True,
        "alert": latest_alert,
        "message": f"SECURITY BREACH DETECTED: {request.agent} agent attempted to {request.action} on {request.target}. Action was BLOCKED.",
        "screen_lock": True,  # Signal frontend to show red overlay
    }


@app.get("/api/v1/ueba/agent-summary/{agent}")
async def get_agent_summary(agent: str):
    """Get behavior summary for a specific agent."""
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    ueba_monitor = app_state.orchestrator.ueba_monitor
    return ueba_monitor.get_agent_summary(agent)


# ==================== Inventory/Parts Endpoints ====================


@app.get("/api/v1/inventory/parts")
async def get_parts_inventory(
    center_id: Optional[str] = None, component: Optional[str] = None
):
    """
    Get parts inventory (for scheduling integration).
    Shows parts availability for scheduling constraints.
    """
    # Simulated parts inventory for demo
    inventory = {
        "SC-001": {
            "name": "Downtown EV Hub",
            "parts": {
                "brakes": {"ceramic_pads": 12, "rotors": 8, "calipers": 4},
                "battery": {"cells": 24, "bms_modules": 6, "coolant": 50},
                "motor": {"resolvers": 8, "bearings": 12},
                "inverter": {"igbt_modules": 4, "capacitors": 20},
            },
        },
        "SC-002": {
            "name": "EV Master Service",
            "parts": {
                "brakes": {"ceramic_pads": 8, "rotors": 4, "calipers": 2},
                "battery": {"cells": 16, "bms_modules": 4, "coolant": 30},
                "motor": {"resolvers": 4, "bearings": 8},
                "inverter": {"igbt_modules": 2, "capacitors": 12},
            },
        },
        "SC-003": {
            "name": "BrakesPro EV",
            "parts": {
                "brakes": {"ceramic_pads": 24, "rotors": 16, "calipers": 8},
                "battery": {"cells": 0, "bms_modules": 0, "coolant": 10},
                "motor": {"resolvers": 0, "bearings": 0},
                "inverter": {"igbt_modules": 0, "capacitors": 0},
            },
        },
    }

    result = inventory

    if center_id and center_id in inventory:
        result = {center_id: inventory[center_id]}

    if component:
        for cid, center in result.items():
            if "parts" in center and component in center["parts"]:
                center["parts"] = {component: center["parts"][component]}

    # Add supply chain info (parts arriving)
    supply_chain = {
        "incoming_shipments": [
            {
                "center_id": "SC-001",
                "part": "inverter_chips",
                "quantity": 50,
                "arrival_date": "2025-12-10",  # Tuesday
                "status": "in_transit",
            },
            {
                "center_id": "SC-002",
                "part": "battery_cells",
                "quantity": 32,
                "arrival_date": "2025-12-09",
                "status": "arriving_tomorrow",
            },
        ],
        "blocked_slots": [
            {
                "center_id": "SC-001",
                "date": "2025-12-09",  # Monday
                "reason": "Inverter chips not in stock until Tuesday",
                "component": "inverter",
            }
        ],
    }

    return {
        "inventory": result,
        "supply_chain": supply_chain,
        "last_updated": datetime.now().isoformat(),
    }


@app.get("/api/v1/inventory/check/{center_id}/{component}")
async def check_parts_availability(center_id: str, component: str):
    """
    Check if parts are available for a specific service.
    Used by voice agent and scheduler.
    """
    # Simulated check
    availability = {
        "brakes": {
            "available": True,
            "parts": ["ceramic_pads", "rotors"],
            "in_stock": True,
            "message": "Ceramic brake pads in stock. Ready for immediate service.",
        },
        "battery": {
            "available": True,
            "parts": ["cells", "coolant"],
            "in_stock": True,
            "message": "Battery service parts available.",
        },
        "inverter": {
            "available": False,
            "parts": ["igbt_modules"],
            "in_stock": False,
            "arrival_date": "2025-12-10",
            "message": "Inverter chips arriving Tuesday. Earliest slot: Tuesday 10:00 AM.",
        },
        "motor": {
            "available": True,
            "parts": ["resolvers", "bearings"],
            "in_stock": True,
            "message": "Motor service parts available.",
        },
    }

    component_lower = component.lower()
    result = availability.get(
        component_lower,
        {"available": False, "message": f"Unknown component: {component}"},
    )

    return {"center_id": center_id, "component": component, **result}


# ==================== Scheduler Admin Endpoints ====================


@app.get("/api/v1/admin/appointments-list")
async def admin_appointments_list(days: int = 7):
    """Get list of appointments from database."""
    db = get_database()

    all_appointments = db.get_appointments()

    # Filter to next N days
    today = datetime.now().date()
    end_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    filtered = [
        a
        for a in all_appointments
        if start_date <= a.get("scheduled_date", "") <= end_date
    ]

    # Sort by date and time
    filtered.sort(
        key=lambda x: (x.get("scheduled_date", ""), x.get("scheduled_time", ""))
    )

    return {
        "appointments": filtered,
        "total": len(filtered),
    }


class GenerateAppointmentsRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=20)


@app.post("/api/v1/admin/generate-appointments")
async def generate_fake_appointments(request: GenerateAppointmentsRequest):
    """
    Generate fake appointments for demo purposes.
    Inserts directly into database to avoid lock issues.
    """
    import random

    db = get_database()
    conn = db._get_connection()
    cursor = conn.cursor()

    vehicles = ["VIN-003", "VIN-004", "VIN-005", "VIN-006", "VIN-007"]
    components = ["brakes", "battery", "motor", "inverter", "suspension", "cooling"]
    urgencies = ["low", "medium", "high", "critical"]
    centers = [
        "SC-001",
        "SC-002",
        "SC-003",
        "SC-004",
        "SC-005",
    ]  # Match service_centers table
    times = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]

    diagnosis_summaries = {
        "brakes": "Brake pad wear detected - replacement recommended",
        "battery": "Cell imbalance detected - diagnostic recommended",
        "motor": "Motor bearing vibration above threshold",
        "inverter": "IGBT temperature spike detected",
        "suspension": "Damper performance degraded",
        "cooling": "Coolant efficiency reduced - service needed",
    }

    costs = {
        "brakes": "$150 - $400",
        "battery": "$500 - $1500",
        "motor": "$300 - $800",
        "inverter": "$800 - $2000",
        "suspension": "$200 - $600",
        "cooling": "$100 - $300",
    }

    generated = []

    for i in range(request.count):
        # Spread across next 7 days
        days_offset = random.randint(0, 6)
        scheduled_date = (datetime.now() + timedelta(days=days_offset)).strftime(
            "%Y-%m-%d"
        )

        vehicle = random.choice(vehicles)
        component = random.choice(components)
        urgency = random.choice(urgencies)
        center = random.choice(centers)
        scheduled_time = random.choice(times)

        appointment_id = f"APT-GEN-{uuid.uuid4().hex[:8].upper()}"
        slot_id = f"slot-gen-{appointment_id}"
        now = datetime.now().isoformat()

        try:
            cursor.execute(
                """
                INSERT INTO appointments 
                (id, vehicle_id, center_id, slot_id, component, diagnosis_summary, 
                 estimated_cost, urgency, status, created_at, scheduled_date, scheduled_time, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)
            """,
                (
                    appointment_id,
                    vehicle,
                    center,
                    slot_id,
                    component,
                    diagnosis_summaries.get(component, "Routine maintenance check"),
                    costs.get(component, "$100 - $300"),
                    urgency,
                    now,
                    scheduled_date,
                    scheduled_time,
                    "Auto-generated for demo",
                ),
            )

            generated.append(
                {
                    "id": appointment_id,
                    "vehicle_id": vehicle,
                    "component": component,
                    "urgency": urgency,
                    "date": scheduled_date,
                    "time": scheduled_time,
                }
            )
        except Exception as e:
            print(f"Failed to create appointment: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"✅ Generated {len(generated)} appointments successfully")

    return {
        "success": True,
        "message": f"Generated {len(generated)} appointments",
        "appointments": generated,
    }


@app.delete("/api/v1/admin/appointments/clear")
async def clear_all_appointments():
    """
    Clear all appointments from database.
    Used to reset state for demos.
    """
    db = get_database()

    # Get all appointments first
    appointments = db.get_appointments()
    count = len(appointments)

    # Delete each appointment
    for apt in appointments:
        db.cancel_appointment(apt["id"])

    return {
        "success": True,
        "message": f"Cleared {count} appointments",
        "deleted_count": count,
    }


@app.delete("/api/v1/admin/appointments/{appointment_id}")
async def delete_appointment(appointment_id: str):
    """Delete a specific appointment."""
    db = get_database()
    success = db.cancel_appointment(appointment_id)

    if success:
        return {"success": True, "message": f"Deleted appointment {appointment_id}"}
    else:
        raise HTTPException(status_code=404, detail="Appointment not found")


# ==================== Service Center Endpoints ====================


@app.get("/api/v1/service-center/appointments")
async def get_service_center_appointments():
    """
    Get all appointments for service center dashboard.
    Includes stage info and highlights voice-booked appointments.
    """
    db = get_database()
    appointments = db.get_appointments()

    # Add service tracking info
    from agents.service_tracker_agent import get_service_tracker, ServiceStage

    tracker = get_service_tracker()

    for apt in appointments:
        # Get stage from database or default
        apt["stage"] = apt.get("stage", "INTAKE")
        apt["is_voice_booked"] = (
            apt.get("booked_via") == "voice" or apt.get("vehicle_id") == "VIN-001"
        )

        # Calculate progress percentage
        stages = list(ServiceStage)
        try:
            current_idx = [s.value for s in stages].index(apt["stage"])
            apt["progress_pct"] = int((current_idx / (len(stages) - 1)) * 100)
        except ValueError:
            apt["progress_pct"] = 0

    # Sort by urgency then date
    urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    appointments.sort(
        key=lambda x: (
            urgency_order.get(x.get("urgency", "low"), 3),
            x.get("scheduled_date", ""),
            x.get("scheduled_time", ""),
        )
    )

    return {
        "appointments": appointments,
        "total": len(appointments),
        "voice_booked_count": sum(1 for a in appointments if a.get("is_voice_booked")),
    }


class StageUpdateRequest(BaseModel):
    stage: str
    notes: Optional[str] = None


@app.put("/api/v1/service-center/appointments/{appointment_id}/stage")
async def update_appointment_stage(appointment_id: str, request: StageUpdateRequest):
    """
    Update appointment stage. Triggers notifications for READY stage.
    """
    db = get_database()

    # Valid stages
    valid_stages = [
        "INTAKE",
        "DIAGNOSIS",
        "WAITING_PARTS",
        "REPAIR",
        "QUALITY_CHECK",
        "READY",
        "PICKED_UP",
    ]

    if request.stage not in valid_stages:
        raise HTTPException(
            status_code=400, detail=f"Invalid stage. Valid: {valid_stages}"
        )

    # Update in database
    conn = db._get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE appointments SET stage = ?, notes = COALESCE(?, notes) WHERE id = ?",
        (request.stage, request.notes, appointment_id),
    )

    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Appointment not found")

    conn.commit()
    conn.close()

    # If stage is READY, trigger notification
    notification_triggered = False
    if request.stage == "READY":
        notification_triggered = True
        # In production, would trigger voice call here
        print(f"🔔 READY notification triggered for appointment {appointment_id}")

    return {
        "success": True,
        "message": f"Stage updated to {request.stage}",
        "appointment_id": appointment_id,
        "new_stage": request.stage,
        "notification_triggered": notification_triggered,
    }


@app.post("/api/v1/service-center/appointments/{appointment_id}/notify-ready")
async def notify_vehicle_ready(appointment_id: str):
    """
    Trigger voice call and notification when vehicle is ready.
    """
    db = get_database()

    # Get appointment details
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
    apt = cursor.fetchone()
    conn.close()

    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    apt_dict = dict(apt)
    vehicle_id = apt_dict.get("vehicle_id", "Unknown")

    # Simulate voice call trigger
    notification = {
        "type": "vehicle_ready",
        "vehicle_id": vehicle_id,
        "message": f"Your vehicle {vehicle_id} is ready for pickup!",
        "timestamp": datetime.now().isoformat(),
        "voice_call_initiated": True,
    }

    print(f"📞 Voice call initiated for vehicle {vehicle_id} - Ready for pickup")

    return {
        "success": True,
        "notification": notification,
        "message": f"Notification sent for vehicle {vehicle_id}",
    }


# ==================== Service Tracking for Chatbot ====================


@app.get("/api/v1/service-tracking/{vehicle_id}")
async def get_service_tracking_status(vehicle_id: str):
    """
    Get real-time service tracking status for a vehicle.
    Used by chatbot to answer "What's the status of VIN-001?" queries.

    Returns current stage, progress, and estimated completion.
    """
    db = get_database()

    # Get appointment for this vehicle
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT a.*, c.name as center_name 
        FROM appointments a 
        LEFT JOIN service_centers c ON a.center_id = c.id
        WHERE a.vehicle_id = ? AND a.status IN ('scheduled', 'in_progress')
        ORDER BY a.created_at DESC
        LIMIT 1
    """,
        (vehicle_id,),
    )
    apt = cursor.fetchone()
    conn.close()

    if not apt:
        return {
            "found": False,
            "vehicle_id": vehicle_id,
            "message": f"No active service appointment found for {vehicle_id}.",
            "suggestion": "You may need to book a new appointment.",
        }

    apt_dict = dict(apt)
    stage = apt_dict.get("stage", "INTAKE")

    # Define stage info
    stage_info = {
        "INTAKE": {
            "label": "Vehicle Intake",
            "description": "Your vehicle has been received at the service center.",
            "progress": 0,
        },
        "DIAGNOSIS": {
            "label": "Diagnosis",
            "description": "Technicians are diagnosing the issue.",
            "progress": 15,
        },
        "WAITING_PARTS": {
            "label": "Waiting for Parts",
            "description": "Required parts have been ordered and are on the way.",
            "progress": 30,
        },
        "REPAIR": {
            "label": "Repair in Progress",
            "description": "Your vehicle is currently being repaired.",
            "progress": 60,
        },
        "QUALITY_CHECK": {
            "label": "Quality Check",
            "description": "Final inspection and quality verification.",
            "progress": 85,
        },
        "READY": {
            "label": "Ready for Pickup",
            "description": "Your vehicle is ready! Please come pick it up.",
            "progress": 100,
        },
        "PICKED_UP": {
            "label": "Completed",
            "description": "Service completed. Vehicle has been picked up.",
            "progress": 100,
        },
    }

    current_stage = stage_info.get(stage, stage_info["INTAKE"])

    # Generate human-readable status message
    status_message = f"""
**Service Status for {vehicle_id}**

🔧 **Current Stage:** {current_stage['label']}
📊 **Progress:** {current_stage['progress']}%

{current_stage['description']}

**Details:**
- Component: {apt_dict.get('component', 'Unknown').capitalize()}
- Urgency: {apt_dict.get('urgency', 'medium').capitalize()}
- Service Center: {apt_dict.get('center_name', 'SC-001')}
- Scheduled: {apt_dict.get('scheduled_date', 'N/A')} at {apt_dict.get('scheduled_time', 'N/A')}
"""

    return {
        "found": True,
        "vehicle_id": vehicle_id,
        "stage": stage,
        "stage_label": current_stage["label"],
        "progress_pct": current_stage["progress"],
        "description": current_stage["description"],
        "component": apt_dict.get("component"),
        "urgency": apt_dict.get("urgency"),
        "center_name": apt_dict.get("center_name"),
        "scheduled_date": apt_dict.get("scheduled_date"),
        "scheduled_time": apt_dict.get("scheduled_time"),
        "diagnosis_summary": apt_dict.get("diagnosis_summary"),
        "message": status_message.strip(),
    }


# ==================== CAPA Integration Endpoints ====================


@app.post("/api/v1/capa/analyze-patterns")
async def analyze_defect_patterns():
    """
    Analyze appointments for common defect patterns.
    Used after optimization to identify manufacturing issues.
    """
    db = get_database()
    appointments = db.get_appointments()

    # Count defects by component
    from collections import Counter

    component_counts = Counter(a.get("component", "unknown") for a in appointments)

    # Find patterns (components with 2+ occurrences)
    patterns = []
    for component, count in component_counts.most_common():
        if count >= 2:
            # Get urgency breakdown for this component
            comp_apts = [a for a in appointments if a.get("component") == component]
            critical_count = sum(
                1 for a in comp_apts if a.get("urgency") in ["critical", "high"]
            )

            patterns.append(
                {
                    "component": component,
                    "total_affected": count,
                    "critical_count": critical_count,
                    "region": "Mountainous",  # Simulated
                    "failure_mode": f"{component.capitalize()} degradation detected",
                    "recommendation": f"Review {component} supplier quality. Consider recall if critical > 5.",
                }
            )

    # Generate CAPA insights
    capa_insights = None
    if patterns:
        top_issue = patterns[0]
        capa_insights = {
            "headline": f"{top_issue['total_affected']} vehicles affected by {top_issue['component']} issues",
            "root_cause": f"{top_issue['component'].capitalize()} component failure in {top_issue['region']} regions",
            "action_required": top_issue["recommendation"],
            "severity": "HIGH" if top_issue["critical_count"] > 2 else "MEDIUM",
        }

    return {
        "patterns": patterns,
        "capa_insights": capa_insights,
        "total_analyzed": len(appointments),
    }


@app.post("/api/v1/capa/generate-report")
async def generate_capa_report(component: str, failure_mode: str = "degradation"):
    """
    Generate a CAPA report for manufacturing feedback.
    """
    from agents.capa_agent import get_capa_agent

    capa_agent = get_capa_agent()

    # Generate report using CAPA agent
    report = capa_agent.generate_capa_from_diagnosis(
        vehicle_id="FLEET",
        component=component,
        diagnosis_summary=f"Multiple {component} failures detected across fleet",
        failure_mode=failure_mode,
        region="Mountainous",
    )

    return {
        "success": True,
        "report": report,
        "message": f"CAPA report generated for {component}",
    }


@app.post("/api/v1/admin/optimize-schedule")
async def optimize_schedule_workload(date: str = None):
    """
    Optimized workload simulation with supply chain logic.
    Uses ONLY real database appointments.

    Returns:
        - rescheduled_appointments: Low-priority moved for emergencies
        - technicians_required: By specialty
        - blocked_slots: Due to parts unavailability
    """
    db = get_database()

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # Get ALL appointments from database (not just this date)
    all_appointments = db.get_appointments()

    print(f"📊 Debug: Total appointments fetched: {len(all_appointments)}")
    for apt in all_appointments[:3]:  # Show first 3
        print(
            f"   - {apt.get('id')}: status={apt.get('status')}, urgency={apt.get('urgency')}"
        )

    # Filter for scheduled/confirmed only (also include 'scheduled' lowercase)
    scheduled = [
        a
        for a in all_appointments
        if a.get("status", "").lower() in ["scheduled", "confirmed"]
    ]

    print(f"📊 Debug: Filtered scheduled/confirmed: {len(scheduled)}")

    if not scheduled:
        return {
            "success": True,
            "message": "No appointments to optimize",
            "rescheduled": [],
            "technicians_required": {},
            "blocked_slots": [],
            "total_appointments": 0,
            "labor_forecast": [],
            "staffing_recommendations": [],
            "heatmap": [],
            "technicians": [],
        }

    # Identify emergency vs routine
    emergencies = [a for a in scheduled if a.get("urgency") in ["critical", "high"]]
    routine = [a for a in scheduled if a.get("urgency") in ["low", "medium"]]

    rescheduled = []

    # If there are emergencies, bump low-priority from same timeslot
    for emerg in emergencies:
        emerg_time = emerg.get("scheduled_time", "09:00")
        emerg_date = emerg.get("scheduled_date")

        # Find low-priority appointments at same time that could be bumped
        for routine_apt in routine:
            if (
                routine_apt.get("scheduled_time") == emerg_time
                and routine_apt.get("scheduled_date") == emerg_date
                and routine_apt.get("urgency") == "low"
            ):

                # Simulate rescheduling to 4 PM
                new_time = "16:00"
                rescheduled.append(
                    {
                        "appointment_id": routine_apt["id"],
                        "vehicle_id": routine_apt["vehicle_id"],
                        "component": routine_apt["component"],
                        "original_time": emerg_time,
                        "new_time": new_time,
                        "reason": f"Moved to make room for emergency {emerg['component']} repair",
                    }
                )

    # Calculate technician requirements by specialty
    component_to_specialty = {
        "brakes": "brake_specialist",
        "battery": "hv_battery_tech",
        "motor": "drivetrain_specialist",
        "inverter": "power_electronics_tech",
        "suspension": "chassis_specialist",
        "cooling": "thermal_systems_tech",
    }

    tech_requirements = {}
    for apt in scheduled:
        component = apt.get("component", "general").lower()
        specialty = component_to_specialty.get(component, "general_technician")
        tech_requirements[specialty] = tech_requirements.get(specialty, 0) + 1

    # Simulate supply chain constraints
    # Block morning slots on Monday if inverter parts not arrived
    blocked_slots = []
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if any appointments need inverter parts
    inverter_needed = any(
        a.get("component", "").lower() == "inverter" for a in scheduled
    )

    if inverter_needed:
        blocked_slots.append(
            {
                "date": tomorrow,
                "start": "09:00",
                "end": "13:00",
                "reason": "Inverter chips shipment arriving Tuesday afternoon",
                "part": "IGBT modules",
            }
        )

    # Calculate labor hours by component (estimated hours per component)
    labor_hours_map = {
        "brakes": 2.0,
        "battery": 3.0,
        "motor": 2.5,
        "inverter": 4.0,
        "suspension": 1.5,
        "cooling": 1.0,
        "general": 1.0,
    }

    total_labor_hours = sum(
        labor_hours_map.get(a.get("component", "general").lower(), 1.0)
        for a in scheduled
    )

    # Calculate capacity (assume 8 technicians * 8 hours = 64 hours/day capacity)
    capacity_hours = 64
    avg_utilization = (
        round((total_labor_hours / capacity_hours) * 100, 1)
        if capacity_hours > 0
        else 0
    )

    # Generate labor forecast by day
    from collections import defaultdict

    daily_appointments = defaultdict(list)
    for apt in scheduled:
        day = apt.get("scheduled_date", "unknown")
        daily_appointments[day].append(apt)

    labor_forecast = []
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    for i in range(7):
        forecast_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        day_apts = daily_appointments.get(forecast_date, [])
        day_hours = sum(
            labor_hours_map.get(a.get("component", "general").lower(), 1.0)
            for a in day_apts
        )
        day_utilization = (
            round((day_hours / capacity_hours) * 100, 1) if capacity_hours > 0 else 0
        )

        status = "low"
        if day_utilization > 80:
            status = "over_capacity"
        elif day_utilization > 60:
            status = "busy"
        elif day_utilization > 30:
            status = "optimal"

        labor_forecast.append(
            {
                "date": forecast_date,
                "day": day_names[(datetime.now() + timedelta(days=i)).weekday()],
                "scheduled_appointments": len(day_apts),
                "scheduled_hours": round(day_hours, 1),
                "total_labor_hours": round(day_hours, 1),
                "capacity_hours": capacity_hours,
                "utilization_pct": day_utilization,
                "status": status,
            }
        )

    # Generate staffing recommendations
    staffing_recs = []
    for day_data in labor_forecast:
        if day_data["utilization_pct"] > 80:
            staffing_recs.append(
                {
                    "date": day_data["date"],
                    "day": day_data["day"],
                    "action": "add_staff",
                    "message": f"High demand on {day_data['day']}. Consider adding 2 technicians.",
                    "urgency": "high",
                }
            )
        elif day_data["scheduled_appointments"] == 0:
            staffing_recs.append(
                {
                    "date": day_data["date"],
                    "day": day_data["day"],
                    "action": "reduce_staff",
                    "message": f"No scheduled appointments on {day_data['day']}. Consider reduced staffing.",
                    "urgency": "low",
                }
            )

    # Generate capacity heatmap
    heatmap = []
    for i in range(7):
        heat_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        day_apts = daily_appointments.get(heat_date, [])

        hours_data = []
        for hour in range(8, 18):  # 8 AM to 6 PM
            hour_str = f"{hour:02d}:00"
            hour_apts = [
                a
                for a in day_apts
                if a.get("scheduled_time", "").startswith(f"{hour:02d}")
            ]
            intensity = min(len(hour_apts) / 3.0, 1.0)  # Normalize to 0-1

            hours_data.append(
                {
                    "hour": hour_str,
                    "appointments": len(hour_apts),
                    "capacity": 3,
                    "intensity": round(intensity, 2),
                }
            )

        heatmap.append(
            {
                "date": heat_date,
                "day": day_names[(datetime.now() + timedelta(days=i)).weekday()],
                "hours": hours_data,
            }
        )

    # Sort appointments by urgency for display
    urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_appointments = sorted(
        scheduled,
        key=lambda x: (
            urgency_order.get(x.get("urgency", "low"), 3),
            x.get("scheduled_time", ""),
        ),
    )

    # Build technician workloads
    technicians = [
        {
            "id": "TECH-001",
            "name": "John Smith",
            "specialties": ["brakes", "suspension"],
            "capacity_hours": 8,
            "assigned_hours": 0,
        },
        {
            "id": "TECH-002",
            "name": "Sarah Chen",
            "specialties": ["battery", "motor"],
            "capacity_hours": 8,
            "assigned_hours": 0,
        },
        {
            "id": "TECH-003",
            "name": "Mike Johnson",
            "specialties": ["inverter", "cooling"],
            "capacity_hours": 8,
            "assigned_hours": 0,
        },
        {
            "id": "TECH-004",
            "name": "Emily Davis",
            "specialties": ["brakes", "motor"],
            "capacity_hours": 8,
            "assigned_hours": 0,
        },
    ]

    # Assign workload to technicians based on specialty
    for apt in scheduled:
        component = apt.get("component", "general").lower()
        hours = labor_hours_map.get(component, 1.0)

        # Find technician with matching specialty and lowest load
        matching_techs = [
            t
            for t in technicians
            if component in t["specialties"] or component == "general"
        ]
        if not matching_techs:
            matching_techs = technicians

        best_tech = min(matching_techs, key=lambda t: t["assigned_hours"])
        best_tech["assigned_hours"] = round(best_tech["assigned_hours"] + hours, 1)

    return {
        "success": True,
        "message": f"Optimized {len(scheduled)} appointments. Rescheduled {len(rescheduled)} low-priority.",
        # Summary stats
        "total_appointments": len(scheduled),
        "total_scheduled": len(scheduled),
        "total_labor_hours": round(total_labor_hours, 1),
        "avg_utilization": avg_utilization,
        "capacity_hours": capacity_hours,
        # Breakdown
        "emergency_count": len(emergencies),
        "routine_count": len(routine),
        "rescheduled": rescheduled,
        # Technician data
        "technicians_required": tech_requirements,
        "total_technicians_needed": sum(tech_requirements.values()),
        "technicians": technicians,
        # Forecasts
        "labor_forecast": labor_forecast,
        "staffing_recommendations": staffing_recs,
        "heatmap": heatmap,
        # Supply chain
        "blocked_slots": blocked_slots,
        "supply_chain_alerts": (
            [
                {
                    "part": "Inverter IGBT modules",
                    "status": "In transit",
                    "eta": "Tuesday 2:00 PM",
                },
                {"part": "Ceramic brake pads", "status": "In stock", "quantity": 24},
            ]
            if inverter_needed
            else [{"part": "All parts", "status": "In stock", "quantity": "Sufficient"}]
        ),
        # Full appointment list sorted by urgency
        "appointments": sorted_appointments,
        # CAPA Manufacturing Insights
        "capa_insights": (
            {
                "headline": (
                    f"{len(scheduled)} vehicles analyzed for quality patterns"
                    if scheduled
                    else None
                ),
                "top_component": (
                    max(tech_requirements.keys(), key=lambda k: tech_requirements[k])
                    if tech_requirements
                    else None
                ),
                "affected_count": (
                    max(tech_requirements.values()) if tech_requirements else 0
                ),
                "recommendation": f"Review {max(tech_requirements.keys(), key=lambda k: tech_requirements[k]) if tech_requirements else 'component'} supplier quality. Pattern detected in Mountainous regions.",
                "action_link": "/admin/capa",
            }
            if scheduled
            else None
        ),
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
