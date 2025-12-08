"""
SentinEV - Data Analysis Agent
LangGraph agent for real-time telemetry analysis using ML pipeline
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_state import AgentState, AgentMode

# ML Pipeline imports
from ml.anomaly_detector import MLPipeline, AdvancedAnomalyDetector, ScoringEngine

# LangGraph imports
try:
    from langgraph.graph import StateGraph
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not installed. Run: pip install langgraph langchain-core")

# Gemini for chatbot
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# RAG imports
try:
    from ml.rag_knowledge import MLKnowledgeBase

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class DataAnalysisAgent:
    """
    Data Analysis Agent using LangGraph.

    Responsibilities:
    1. Analyze streaming telemetry using ML pipeline
    2. Detect anomalies and predict failures
    3. Generate positive/negative scoring
    4. Retrieve relevant context from RAG
    5. Trigger alerts for critical events
    """

    def __init__(self, vehicle_id: str):
        """Initialize the Data Analysis Agent."""
        self.vehicle_id = vehicle_id
        self.pipeline: Optional[MLPipeline] = None
        self.knowledge_base: Optional[Any] = None
        self.llm = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize ML pipeline and knowledge base."""
        # Initialize ML pipeline
        self.pipeline = MLPipeline(self.vehicle_id)

        # Initialize RAG knowledge base
        if RAG_AVAILABLE:
            try:
                self.knowledge_base = MLKnowledgeBase()
                self.knowledge_base.load_knowledge_base()
                print(f"✓ RAG knowledge base loaded for {self.vehicle_id}")
            except Exception as e:
                print(f"⚠️ Could not load knowledge base: {e}")

        # Initialize Gemini LLM for enhanced responses
        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )
            if api_key:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-lite",
                        google_api_key=api_key,
                        temperature=0.7,
                    )
                    print(f"✓ Gemini LLM initialized for {self.vehicle_id}")
                except Exception as e:
                    print(f"⚠️ Could not initialize Gemini: {e}")

    def train_on_history(self, historical_data, driver_profile: str = "normal") -> Dict:
        """Train the ML pipeline on historical data."""
        if self.pipeline:
            return self.pipeline.train(historical_data, driver_profile)
        return {"error": "Pipeline not initialized"}

    def _retrieve_rag_context(
        self, anomaly_type: str, affected_components: List[str]
    ) -> List[Dict]:
        """Retrieve relevant context from RAG for the anomaly."""
        context = []

        if not self.knowledge_base:
            return context

        try:
            # Search for similar faults
            if anomaly_type != "normal":
                results = self.knowledge_base.retrieve_similar_faults(
                    f"{anomaly_type} in EV vehicle", k=3
                )
                context.extend(results)

            # Get component specs
            for component in affected_components:
                specs = self.knowledge_base.get_component_specs(component)
                if specs:
                    context.append(specs)

            # Get maintenance recommendations
            maintenance = self.knowledge_base.get_maintenance_for_anomaly(anomaly_type)
            context.extend(maintenance)

        except Exception as e:
            print(f"RAG retrieval error: {e}")

        return context

    def _generate_enhanced_feedback(self, state: AgentState) -> str:
        """Generate enhanced feedback using Gemini."""
        if not self.llm:
            return state.get("feedback_text", "")

        try:
            # Build context for LLM
            context = f"""
You are an EV vehicle assistant for vehicle {state['vehicle_id']}.
Current situation:
- Anomaly detected: {state['is_anomaly']}
- Type: {state['anomaly_type']}
- Severity: {state['severity']}
- Failure risk: {state['failure_risk_pct']}%
- Score change: {state['score_delta']} points
- Affected components: {', '.join(state['affected_components'])}

Generate a brief, helpful response (under 50 words) that:
1. Acknowledges the current state
2. Provides actionable advice
3. Is friendly but professional
"""

            response = self.llm.invoke([SystemMessage(content=context)])
            return response.content

        except Exception as e:
            print(f"LLM feedback generation error: {e}")
            return state.get("feedback_text", "")

    def _create_alert(self, state: AgentState) -> Optional[Dict]:
        """Create an alert if conditions warrant."""
        if not state["is_anomaly"]:
            return None

        if state["severity"] not in ["high", "critical"]:
            return None

        import uuid

        alert = {
            "alert_id": f"ALERT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(),
            "vehicle_id": state["vehicle_id"],
            "anomaly_type": state["anomaly_type"],
            "severity": state["severity"],
            "failure_risk_pct": state["failure_risk_pct"],
            "time_to_failure_hours": state["time_to_failure_hours"],
            "affected_components": state["affected_components"],
            "contributing_factors": state["contributing_factors"][:3],  # Top 3
            "recommended_action": self._get_recommended_action(state),
            "acknowledged": False,
            "resolved": False,
        }

        return alert

    def _get_recommended_action(self, state: AgentState) -> str:
        """Get recommended action based on anomaly type."""
        actions = {
            "thermal_battery": "Reduce power demand and allow battery to cool. Consider stopping safely.",
            "thermal_motor": "Reduce acceleration and allow motor to cool. Check cooling system.",
            "thermal_brake": "Stop driving and allow brakes to cool. Check for stuck caliper.",
            "power_anomaly": "Check regen system and motor efficiency. Schedule diagnostic.",
            "driving_behavior": "Adopt smoother driving style to reduce wear.",
            "wear_degradation": "Schedule preventive maintenance inspection.",
            "soc_anomaly": "Check battery management system. May need recalibration.",
        }

        return actions.get(
            state["anomaly_type"], "Monitor situation and schedule service if persists."
        )

    def _log_agent_action(self, action_type: str, details: Dict) -> Dict:
        """Log agent action for UEBA monitoring."""
        return {
            "timestamp": datetime.now().isoformat(),
            "agent": "data_analysis",
            "vehicle_id": self.vehicle_id,
            "action_type": action_type,
            "details": details,
        }

    def analyze(self, state: AgentState) -> AgentState:
        """
        Main analysis function - processes telemetry through ML pipeline.

        This is the primary node in the LangGraph workflow.
        """
        telemetry = state.get("current_telemetry", {})

        if not telemetry:
            state["should_continue"] = False
            return state

        # Run ML pipeline
        if self.pipeline:
            result = self.pipeline.process(telemetry)

            # Update state with results
            state["is_anomaly"] = result["is_anomaly"]
            state["anomaly_score"] = result["anomaly_score"]
            state["anomaly_type"] = result["anomaly_type"]
            state["severity"] = result["severity"]
            state["failure_risk_pct"] = result["failure_risk_pct"]
            state["time_to_failure_hours"] = result["time_to_failure_hours"]
            state["affected_components"] = result["affected_components"]
            state["contributing_factors"] = result["contributing_factors"]

            state["score_delta"] = result["score_delta"]
            state["total_score"] = result["total_score"]
            state["scoring_events"] = result["scoring_events"]
            state["feedback_text"] = result["feedback_text"]
            state["badges_earned"] = result["badges_earned"]

        # Retrieve RAG context
        rag_context = self._retrieve_rag_context(
            state["anomaly_type"], state["affected_components"]
        )
        state["rag_context"] = rag_context

        # Generate enhanced feedback if anomaly
        if state["is_anomaly"] and self.llm:
            state["feedback_text"] = self._generate_enhanced_feedback(state)

        # Create alert if needed
        alert = self._create_alert(state)
        if alert:
            state["active_alerts"] = [alert]
            state["mode"] = AgentMode.ALERT.value
        else:
            state["active_alerts"] = []

        # Update mode based on severity
        if state["severity"] == "critical":
            state["mode"] = AgentMode.EMERGENCY.value
        elif state["is_anomaly"]:
            state["mode"] = AgentMode.ALERT.value
        else:
            state["mode"] = AgentMode.MONITORING.value

        # Log action for UEBA
        action_log = self._log_agent_action(
            "analyze_telemetry",
            {
                "is_anomaly": state["is_anomaly"],
                "severity": state["severity"],
                "alert_created": alert is not None,
            },
        )
        state["agent_actions_log"] = [action_log]

        # Determine next agent
        if state["is_anomaly"]:
            state["next_agent"] = "orchestrator"  # Escalate to orchestrator
        else:
            state["next_agent"] = None  # Stay in monitoring

        state["current_agent"] = "data_analysis"
        state["should_continue"] = True

        return state

    def get_analysis_summary(self, state: AgentState) -> Dict:
        """Get a summary of the current analysis state."""
        return {
            "vehicle_id": state["vehicle_id"],
            "timestamp": datetime.now().isoformat(),
            "mode": state["mode"],
            # Anomaly summary
            "anomaly_status": {
                "is_anomaly": state["is_anomaly"],
                "type": state["anomaly_type"],
                "severity": state["severity"],
                "failure_risk": f"{state['failure_risk_pct']}%",
                "time_to_failure": (
                    f"{state['time_to_failure_hours']}h"
                    if state["time_to_failure_hours"]
                    else None
                ),
            },
            # Scoring summary
            "scoring": {
                "delta": state["score_delta"],
                "total": state["total_score"],
                "feedback": state["feedback_text"],
                "badges": state["badges_earned"],
            },
            # Alerts
            "alerts": state["active_alerts"],
            # RAG insights
            "rag_context_count": len(state.get("rag_context", [])),
        }


def create_data_analysis_graph(agent: DataAnalysisAgent) -> Optional[Any]:
    """
    Create LangGraph workflow for Data Analysis Agent.

    Returns a compiled graph that can be invoked with state.
    """
    if not LANGGRAPH_AVAILABLE:
        print("⚠️ LangGraph not available")
        return None

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", agent.analyze)

    # Simple linear flow for now
    workflow.set_entry_point("analyze")
    workflow.set_finish_point("analyze")

    # Compile
    return workflow.compile()


if __name__ == "__main__":
    # Test the Data Analysis Agent
    print("🔍 Testing Data Analysis Agent")
    print("=" * 50)

    from agents.agent_state import create_initial_state

    # Create agent
    agent = DataAnalysisAgent("VIN_TEST_001")

    # Create test state
    state = create_initial_state("VIN_TEST_001")
    state["current_telemetry"] = {
        "speed_kmh": 95.0,
        "acceleration_ms2": 2.5,
        "jerk_ms3": 0.5,
        "power_draw_kw": 55.0,
        "regen_efficiency": 0.75,
        "battery_soc_pct": 68.0,
        "battery_temp_c": 42.0,
        "motor_temp_c": 78.0,
        "inverter_temp_c": 72.0,
        "brake_temp_c": 95.0,
        "coolant_temp_c": 45.0,
        "wear_index": 0.12,
    }

    # Run analysis
    result_state = agent.analyze(state)

    # Print results
    summary = agent.get_analysis_summary(result_state)
    print(f"\nAnalysis Summary:")
    print(f"  Mode: {summary['mode']}")
    print(f"  Anomaly: {summary['anomaly_status']['is_anomaly']}")
    print(f"  Severity: {summary['anomaly_status']['severity']}")
    print(f"  Score: {summary['scoring']['total']} ({summary['scoring']['delta']:+d})")
    print(f"  Feedback: {summary['scoring']['feedback']}")

    print("\n✅ Data Analysis Agent test complete")
