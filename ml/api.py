"""
SentinelEY - ML Model Training API
FastAPI endpoints for ML model serving and real-time predictions
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, List, Any, Literal
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .physics import VehicleSimulator, DRIVER_PROFILES, WEATHER_CONDITIONS
from .digital_twin_model import DigitalTwinModel, FleetDigitalTwinManager
from .rag_knowledge import MLKnowledgeBase
from .feedback import FeedbackEngine, FleetFeedbackManager


# ==================== Pydantic Models ====================


class VehicleInitRequest(BaseModel):
    """Request to initialize a new vehicle."""

    driver_profile: Literal["aggressive", "eco", "normal"] = "normal"
    weather: Literal["hot", "cold", "moderate"] = "moderate"
    generate_days: int = Field(default=60, ge=1, le=365)


class VehicleInitResponse(BaseModel):
    """Response after vehicle initialization."""

    vehicle_id: str
    status: str
    driver_profile: str
    weather: str
    training_samples: int
    model_trained: bool
    message: str


class FaultInjectionRequest(BaseModel):
    """Request to inject a fault for testing."""

    fault: Literal[
        "overheat",
        "cell_imbalance",
        "inverter",
        "motor_resolver",
        "brake_drag",
        "coolant_low",
    ]
    severity: float = Field(default=1.0, ge=0.5, le=2.0)


class VehicleStatsResponse(BaseModel):
    """Vehicle statistics response."""

    vehicle_id: str
    model_trained: bool
    training_samples: int
    total_score: int
    anomaly_count: int
    badges: List[str]
    recent_predictions: List[Dict[str, Any]]


class RAGSearchRequest(BaseModel):
    """RAG search request."""

    query: str
    k: int = Field(default=5, ge=1, le=20)
    filter_type: Optional[str] = None


class RAGSearchResponse(BaseModel):
    """RAG search response."""

    query: str
    results: List[Dict[str, Any]]
    count: int


# ==================== Global State ====================


class AppState:
    """Application state container."""

    def __init__(self):
        self.simulators: Dict[str, VehicleSimulator] = {}
        self.model_manager = FleetDigitalTwinManager()
        self.feedback_manager = FleetFeedbackManager()
        self.knowledge_base = MLKnowledgeBase()
        self.prediction_history: Dict[str, List[Dict]] = {}

        # Try to load existing knowledge base
        self.knowledge_base.load_knowledge_base()


app_state = AppState()


# ==================== Lifespan ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 SentinelEY ML API Starting...")
    print(f"   Models Directory: {app_state.model_manager.models_dir}")
    yield
    print("🛑 SentinelEY ML API Shutting down...")


# ==================== FastAPI App ====================

app = FastAPI(
    title="SentinelEY ML Training API",
    description="""
    Personalized Digital Twin Engine for Electric Vehicles.
    
    Features:
    - Physics-based telemetry data generation
    - Personalized ML anomaly detection
    - RAG-enhanced knowledge retrieval
    - Gamified driver feedback with LLM
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Endpoints ====================


@app.get("/")
async def root():
    """API health check and info."""
    return {
        "name": "SentinelEY ML Training API",
        "version": "1.0.0",
        "status": "running",
        "vehicles_active": len(app_state.simulators),
        "models_trained": len(
            [v for v in app_state.model_manager.models.values() if v.is_trained]
        ),
    }


@app.post("/init_vehicle/{vin}", response_model=VehicleInitResponse)
async def init_vehicle(
    vin: str, request: VehicleInitRequest, background_tasks: BackgroundTasks
):
    """
    Initialize a vehicle, generate history, and train the ML model.

    This endpoint:
    1. Creates a physics-based simulator for the vehicle
    2. Generates historical telemetry data
    3. Trains a personalized anomaly detection model
    """
    try:
        # Create simulator
        simulator = VehicleSimulator(
            vehicle_id=vin,
            driver_profile=request.driver_profile,
            weather_condition=request.weather,
        )

        # Generate historical data
        print(f"📊 Generating {request.generate_days} days of history for {vin}...")
        history_df = simulator.generate_history(days=request.generate_days)

        # Train model
        print(f"🎯 Training model for {vin}...")
        app_state.model_manager.train_model(
            vehicle_id=vin,
            historical_data=history_df,
            driver_profile=request.driver_profile,
            save=True,
        )

        # Store simulator
        app_state.simulators[vin] = simulator
        simulator.reset()  # Reset for real-time use

        # Initialize feedback engine
        app_state.feedback_manager.get_engine(vin, f"Driver-{vin}")

        # Initialize prediction history
        app_state.prediction_history[vin] = []

        return VehicleInitResponse(
            vehicle_id=vin,
            status="initialized",
            driver_profile=request.driver_profile,
            weather=request.weather,
            training_samples=len(history_df),
            model_trained=True,
            message=f"Vehicle {vin} initialized successfully with {len(history_df)} training samples",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/{vin}")
async def stream_telemetry(vin: str):
    """
    Stream real-time physics data + ML predictions + gamified feedback.

    Returns Server-Sent Events (SSE) stream with:
    - Telemetry data from physics simulator
    - Anomaly detection results
    - Gamified feedback
    """
    if vin not in app_state.simulators:
        raise HTTPException(
            status_code=404,
            detail=f"Vehicle {vin} not initialized. Call POST /init_vehicle/{vin} first.",
        )

    async def generate_stream():
        simulator = app_state.simulators[vin]
        model = app_state.model_manager.get_model(vin)
        feedback_engine = app_state.feedback_manager.get_engine(vin)

        try:
            while True:
                # Generate telemetry
                telemetry = simulator.step(dt_seconds=1)

                # Get ML prediction
                prediction = model.predict_realtime(telemetry)

                # Get feedback
                feedback_result = feedback_engine.process_telemetry(telemetry)

                # Combine results
                result = {
                    "timestamp": telemetry["timestamp"],
                    "telemetry": {
                        "speed_kmh": telemetry["speed_kmh"],
                        "battery_temp_c": telemetry["battery_temp_c"],
                        "battery_soc_percent": telemetry["battery_soc_percent"],
                        "power_draw_kw": telemetry["power_draw_kw"],
                        "regen_efficiency": telemetry["regen_efficiency"],
                        "wear_index": telemetry["wear_index"],
                    },
                    "prediction": {
                        "anomaly_score": prediction.anomaly_score,
                        "is_anomaly": prediction.is_anomaly,
                        "failure_risk_percent": prediction.failure_risk_percent,
                        "anomaly_type": prediction.anomaly_type,
                        "severity": prediction.severity,
                        "suggested_action": prediction.suggested_action,
                    },
                    "feedback": {
                        "score_delta": feedback_result.score_delta,
                        "total_score": feedback_result.total_score,
                        "message": feedback_result.feedback_message,
                    },
                }

                # Store in history
                if vin in app_state.prediction_history:
                    app_state.prediction_history[vin].append(result)
                    # Keep only last 1000
                    app_state.prediction_history[vin] = app_state.prediction_history[
                        vin
                    ][-1000:]

                # Send SSE event
                yield f"data: {json.dumps(result)}\n\n"

                # Wait before next update
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            print(f"Stream cancelled for {vin}")

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/inject_fault/{vin}")
async def inject_fault(vin: str, request: FaultInjectionRequest):
    """
    Artificially inject a fault to test detection.

    Available faults:
    - overheat: Battery overheating
    - cell_imbalance: Cell voltage imbalance
    - inverter: Inverter power issues
    - motor_resolver: Motor resolver drift
    - brake_drag: Stuck brake caliper
    - coolant_low: Low coolant level
    """
    if vin not in app_state.simulators:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not initialized")

    simulator = app_state.simulators[vin]
    model = app_state.model_manager.get_model(vin)
    feedback_engine = app_state.feedback_manager.get_engine(vin)

    # Inject fault and get telemetry
    telemetry = simulator.inject_fault(request.fault, request.severity)

    # Get prediction on faulted data
    prediction = model.predict_realtime(telemetry)

    # Get feedback
    feedback_result = feedback_engine.process_telemetry(telemetry)

    return {
        "vehicle_id": vin,
        "fault_injected": request.fault,
        "severity": request.severity,
        "telemetry": telemetry,
        "detection": {
            "anomaly_detected": prediction.is_anomaly,
            "anomaly_score": prediction.anomaly_score,
            "anomaly_type": prediction.anomaly_type,
            "severity": prediction.severity,
            "failure_risk_percent": prediction.failure_risk_percent,
            "suggested_action": prediction.suggested_action,
        },
        "feedback": {
            "score_delta": feedback_result.score_delta,
            "message": feedback_result.feedback_message,
        },
    }


@app.get("/stats/{vin}", response_model=VehicleStatsResponse)
async def get_vehicle_stats(vin: str):
    """
    Returns vehicle stats: anomaly history, wear index, driver score.
    """
    model = app_state.model_manager.get_model(vin)
    feedback_engine = app_state.feedback_manager.get_engine(vin)

    # Get model info
    model_info = model.get_model_info()

    # Get feedback summary
    feedback_summary = feedback_engine.get_score_summary()

    # Get recent predictions
    recent = app_state.prediction_history.get(vin, [])[-10:]

    # Count anomalies
    anomaly_count = sum(
        1
        for p in app_state.prediction_history.get(vin, [])
        if p.get("prediction", {}).get("is_anomaly", False)
    )

    return VehicleStatsResponse(
        vehicle_id=vin,
        model_trained=model.is_trained,
        training_samples=model.training_samples,
        total_score=feedback_summary["total_score"],
        anomaly_count=anomaly_count,
        badges=[b["name"] for b in feedback_summary["badges_earned"]],
        recent_predictions=recent,
    )


@app.get("/rag_search", response_model=RAGSearchResponse)
async def rag_search(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results", ge=1, le=20),
    filter_type: Optional[str] = Query(
        None, description="Filter by type (fault, specification, etc.)"
    ),
):
    """
    Search the RAG knowledge base for faults/manuals.

    Useful for debugging and exploring the knowledge base.
    """
    kb = app_state.knowledge_base

    if not kb.vectorstore:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not initialized. Run: python -m ml.rag_knowledge --build",
        )

    results = kb.semantic_search(query, k=k)

    # Filter by type if specified
    if filter_type:
        results = [
            r for r in results if r.get("metadata", {}).get("type") == filter_type
        ]

    return RAGSearchResponse(query=query, results=results, count=len(results))


@app.get("/vehicles")
async def list_vehicles():
    """List all initialized vehicles."""
    vehicles = []
    for vin, simulator in app_state.simulators.items():
        model = app_state.model_manager.get_model(vin)
        feedback = app_state.feedback_manager.get_engine(vin)

        vehicles.append(
            {
                "vehicle_id": vin,
                "driver_profile": simulator.driver.name,
                "weather": simulator.weather.name,
                "model_trained": model.is_trained,
                "total_score": feedback.total_score,
            }
        )

    return {"count": len(vehicles), "vehicles": vehicles}


@app.post("/build_knowledge_base")
async def build_knowledge_base():
    """
    Build the RAG knowledge base from data files.

    This indexes:
    - Industry faults from data/datasets/industry_faults.csv
    - Vehicle manual from data/datasets/vehicle_manual.json
    """
    from pathlib import Path

    base_path = Path(__file__).parent.parent / "data" / "datasets"
    faults_path = base_path / "industry_faults.csv"
    manual_path = base_path / "vehicle_manual.json"

    if not faults_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Faults file not found: {faults_path}"
        )

    try:
        app_state.knowledge_base.build_knowledge_base(
            faults_csv_path=str(faults_path),
            manual_json_path=str(manual_path) if manual_path.exists() else None,
        )

        return {"status": "success", "message": "Knowledge base built successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info/{vin}")
async def get_model_info(vin: str):
    """Get detailed model information for a vehicle."""
    model = app_state.model_manager.get_model(vin)
    return model.get_model_info()


@app.delete("/vehicle/{vin}")
async def delete_vehicle(vin: str):
    """Remove a vehicle from the system."""
    if vin in app_state.simulators:
        del app_state.simulators[vin]

    if vin in app_state.model_manager.models:
        del app_state.model_manager.models[vin]

    if vin in app_state.feedback_manager.engines:
        del app_state.feedback_manager.engines[vin]

    if vin in app_state.prediction_history:
        del app_state.prediction_history[vin]

    return {"status": "deleted", "vehicle_id": vin}


# ==================== WebSocket Endpoints ====================


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, vin: str):
        await websocket.accept()
        if vin not in self.active_connections:
            self.active_connections[vin] = []
        self.active_connections[vin].append(websocket)

    def disconnect(self, websocket: WebSocket, vin: str):
        if vin in self.active_connections:
            if websocket in self.active_connections[vin]:
                self.active_connections[vin].remove(websocket)

    async def broadcast(self, vin: str, message: dict):
        if vin in self.active_connections:
            dead = []
            for connection in self.active_connections[vin]:
                try:
                    await connection.send_json(message)
                except:
                    dead.append(connection)
            for d in dead:
                self.disconnect(d, vin)


ws_manager = ConnectionManager()


@app.websocket("/ws/stream/{vin}")
async def websocket_stream(websocket: WebSocket, vin: str):
    """
    WebSocket endpoint for real-time bidirectional streaming.

    Sends: telemetry + predictions + feedback
    Receives: commands (e.g., inject_fault, set_speed)
    """
    await ws_manager.connect(websocket, vin)

    if vin not in app_state.simulators:
        await websocket.send_json(
            {
                "error": f"Vehicle {vin} not initialized. Call POST /init_vehicle/{vin} first."
            }
        )
        await websocket.close()
        return

    try:
        simulator = app_state.simulators[vin]
        model = app_state.model_manager.get_model(vin)
        feedback_engine = app_state.feedback_manager.get_engine(vin)

        # Start streaming task
        async def stream_data():
            while True:
                telemetry = simulator.step(dt_seconds=1)
                prediction = model.predict_realtime(telemetry)
                feedback_result = feedback_engine.process_telemetry(telemetry)

                message = {
                    "type": "telemetry",
                    "timestamp": telemetry["timestamp"],
                    "telemetry": {
                        "speed_kmh": telemetry["speed_kmh"],
                        "battery_temp_c": telemetry["battery_temp_c"],
                        "battery_soc_percent": telemetry["battery_soc_percent"],
                        "power_draw_kw": telemetry["power_draw_kw"],
                        "regen_efficiency": telemetry["regen_efficiency"],
                    },
                    "prediction": {
                        "anomaly_score": prediction.anomaly_score,
                        "is_anomaly": prediction.is_anomaly,
                        "failure_risk_percent": prediction.failure_risk_percent,
                        "suggested_action": prediction.suggested_action,
                    },
                    "feedback": {
                        "score_delta": feedback_result.score_delta,
                        "total_score": feedback_result.total_score,
                        "message": feedback_result.feedback_message,
                    },
                }

                await websocket.send_json(message)
                await asyncio.sleep(1)

        # Start streaming in background
        stream_task = asyncio.create_task(stream_data())

        # Handle incoming commands
        try:
            while True:
                data = await websocket.receive_json()
                command = data.get("command")

                if command == "inject_fault":
                    fault = data.get("fault", "overheat")
                    severity = data.get("severity", 1.0)
                    telemetry = simulator.inject_fault(fault, severity)
                    prediction = model.predict_realtime(telemetry)

                    await websocket.send_json(
                        {
                            "type": "fault_response",
                            "fault": fault,
                            "severity": severity,
                            "anomaly_detected": prediction.is_anomaly,
                            "anomaly_score": prediction.anomaly_score,
                        }
                    )

                elif command == "get_stats":
                    summary = feedback_engine.get_score_summary()
                    await websocket.send_json(
                        {
                            "type": "stats",
                            "total_score": summary["total_score"],
                            "badges": [b["name"] for b in summary["badges_earned"]],
                            "streak": summary["streak"],
                        }
                    )

        except WebSocketDisconnect:
            pass
        finally:
            stream_task.cancel()

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        ws_manager.disconnect(websocket, vin)


@app.websocket("/ws/chatbot/{vin}")
async def websocket_chatbot(websocket: WebSocket, vin: str):
    """
    WebSocket endpoint for interactive chatbot.

    Sends: gamified feedback messages, predictions, warnings
    Receives: user acknowledgments, questions
    """
    await ws_manager.connect(websocket, vin)

    try:
        feedback_engine = app_state.feedback_manager.get_engine(vin, f"Driver-{vin}")

        # Send welcome message
        await websocket.send_json(
            {
                "type": "message",
                "role": "assistant",
                "content": f"🚗 Welcome! Your digital twin for {vin} is ready. I'll provide real-time insights on your driving.",
                "timestamp": datetime.now().isoformat(),
            }
        )

        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            # Simulate chatbot response (in production, use LLM)
            if "score" in user_message.lower():
                summary = feedback_engine.get_score_summary()
                response = f"📊 Your current score is {summary['total_score']} points! "
                if summary["badges_earned"]:
                    response += f"Badges: {', '.join(b['name'] for b in summary['badges_earned'])}"
            elif "status" in user_message.lower() or "how" in user_message.lower():
                if vin in app_state.simulators:
                    response = (
                        "✅ Your vehicle is running normally. No anomalies detected."
                    )
                else:
                    response = "⚠️ Vehicle not initialized. Please initialize first."
            else:
                response = "👍 Got it! I'm monitoring your vehicle in real-time."

            await websocket.send_json(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, vin)


@app.post("/diagnosis_escalation/{vin}")
async def diagnosis_escalation(vin: str, request: dict = None):
    """
    Mock Master Agent escalation endpoint.

    Called when Data Analysis Agent detects critical anomalies (score > 0.9).
    In production, this would trigger the full LangGraph workflow.
    """
    if request is None:
        request = {}

    # Get current vehicle state
    model = app_state.model_manager.get_model(vin)

    if not model.is_trained:
        return {
            "action": "pending",
            "reason": "model_not_trained",
            "severity": "low",
            "message": "Vehicle model not yet trained. Cannot evaluate escalation.",
        }

    # Get recent prediction if available
    history = app_state.prediction_history.get(vin, [])
    recent = history[-1] if history else None

    anomaly_score = request.get("anomaly_score", 0.5)
    if recent:
        anomaly_score = recent.get("prediction", {}).get("anomaly_score", anomaly_score)

    # Determine escalation response
    if anomaly_score > 0.9:
        return {
            "action": "escalate",
            "reason": "critical_anomaly",
            "severity": "critical",
            "anomaly_score": anomaly_score,
            "message": "🚨 CRITICAL: Immediate service required. Master Agent has been notified.",
            "recommended_action": "Contact service center immediately",
            "master_agent_notified": True,
        }
    elif anomaly_score > 0.7:
        return {
            "action": "schedule",
            "reason": "high_anomaly",
            "severity": "high",
            "anomaly_score": anomaly_score,
            "message": "⚠️ HIGH: Schedule service within 24 hours recommended.",
            "recommended_action": "Schedule preventive maintenance",
            "master_agent_notified": True,
        }
    elif anomaly_score > 0.5:
        return {
            "action": "monitor",
            "reason": "medium_anomaly",
            "severity": "medium",
            "anomaly_score": anomaly_score,
            "message": "📊 MEDIUM: Continued monitoring. Consider service if pattern persists.",
            "recommended_action": "Continue monitoring",
            "master_agent_notified": False,
        }
    else:
        return {
            "action": "none",
            "reason": "normal_operation",
            "severity": "low",
            "anomaly_score": anomaly_score,
            "message": "✅ Normal operation. No escalation needed.",
            "master_agent_notified": False,
        }


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn

    print("🚗 Starting SentinelEY ML API Server...")
    uvicorn.run("ml.api:app", host="0.0.0.0", port=8001, reload=True)
