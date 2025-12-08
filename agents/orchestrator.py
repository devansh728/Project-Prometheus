"""
SentinEV - Master Orchestrator Agent
LangGraph-based multi-agent orchestration with voice support
"""

import os
import sys
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_state import AgentState, AgentMode, create_initial_state
from agents.data_analysis_agent import DataAnalysisAgent
from agents.safety_agent import get_safety_agent, SafetyAgent
from agents.diagnosis_agent import get_diagnosis_agent, DiagnosisAgent
from agents.scheduling_agent import get_scheduling_agent, SchedulingAgent
from agents.feedback_agent import get_feedback_agent, FeedbackAgent

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not installed. Run: pip install langgraph")

# LangChain imports
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print(
        "⚠️ LangChain not installed. Run: pip install langchain-core langchain-google-genai"
    )

# Voice support (optional - requires PyAudio which needs system dependencies)
VOICE_AVAILABLE = False
try:
    import speech_recognition as sr
    from gtts import gTTS
    import io

    VOICE_AVAILABLE = True
except ImportError:
    # Voice is optional, don't print warning
    pass


class VoiceHandler:
    """
    Voice input/output handler for voice-based agent interaction.
    Uses SpeechRecognition for input and gTTS for output.
    """

    def __init__(self):
        """Initialize voice handler."""
        self.recognizer = None
        self.microphone = None

        if VOICE_AVAILABLE:
            self.recognizer = sr.Recognizer()
            try:
                self.microphone = sr.Microphone()
            except Exception as e:
                # Microphone requires PyAudio - optional for core functionality
                self.microphone = None

    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for voice input and transcribe to text.

        Args:
            timeout: Seconds to wait for speech

        Returns:
            Transcribed text or None if failed
        """
        if not self.recognizer or not self.microphone:
            return None

        try:
            with self.microphone as source:
                print("🎤 Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=10
                )

            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

    def speak(self, text: str, save_path: Optional[str] = None) -> Optional[bytes]:
        """
        Convert text to speech.

        Args:
            text: Text to speak
            save_path: Optional path to save audio file

        Returns:
            Audio bytes or None
        """
        if not VOICE_AVAILABLE:
            return None

        try:
            tts = gTTS(text=text, lang="en", slow=False)

            if save_path:
                tts.save(save_path)
                return None
            else:
                # Return as bytes
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.read()

        except Exception as e:
            print(f"TTS error: {e}")
            return None


class UEBAMonitor:
    """
    User and Entity Behavior Analytics for agent monitoring.
    Detects anomalous agent behavior and unauthorized actions.
    """

    ALLOWED_ACTIONS = {
        "data_analysis": [
            "analyze_telemetry",
            "retrieve_rag_context",
            "generate_feedback",
            "create_alert",
        ],
        "orchestrator": [
            "route_to_agent",
            "process_user_input",
            "generate_response",
            "escalate_alert",
            "acknowledge_alert",
        ],
    }

    BLOCKED_PATTERNS = [
        "access_other_vehicle",
        "modify_training_data",
        "bypass_safety_check",
        "external_api_unauthorized",
    ]

    def __init__(self):
        """Initialize UEBA monitor."""
        self.action_history: List[Dict] = []
        self.alerts: List[Dict] = []
        self.agent_baselines: Dict[str, Dict] = {}

    @property
    def action_log(self) -> List[Dict]:
        """Alias for action_history for API compatibility."""
        return self.action_history

    def log_action(self, agent: str, action: str, details: Dict) -> None:
        """Log an agent action."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "details": details,
        }
        self.action_history.append(entry)

        # Check for anomalies
        self._check_for_anomalies(entry)

    def _check_for_anomalies(self, entry: Dict) -> None:
        """Check if action is anomalous."""
        agent = entry["agent"]
        action = entry["action"]

        # Check if action is allowed for agent
        allowed = self.ALLOWED_ACTIONS.get(agent, [])
        if action not in allowed and action not in ["log_action"]:
            self.alerts.append(
                {
                    "timestamp": entry["timestamp"],
                    "type": "unauthorized_action",
                    "agent": agent,
                    "action": action,
                    "severity": "high",
                    "message": f"Agent '{agent}' attempted unauthorized action: {action}",
                }
            )

        # Check for blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in action.lower():
                self.alerts.append(
                    {
                        "timestamp": entry["timestamp"],
                        "type": "blocked_pattern",
                        "agent": agent,
                        "action": action,
                        "severity": "critical",
                        "message": f"Blocked pattern detected: {pattern}",
                    }
                )

    def get_alerts(self) -> List[Dict]:
        """Get all UEBA alerts."""
        return self.alerts

    def get_agent_summary(self, agent: str) -> Dict:
        """Get summary of agent's actions."""
        agent_actions = [a for a in self.action_history if a["agent"] == agent]

        return {
            "agent": agent,
            "total_actions": len(agent_actions),
            "action_types": list(set(a["action"] for a in agent_actions)),
            "alerts": len([a for a in self.alerts if a["agent"] == agent]),
        }


class MasterOrchestrator:
    """
    Master Orchestrator Agent using LangGraph.

    Responsibilities:
    1. Coordinate all worker agents
    2. Route tasks based on state
    3. Handle user interactions (text and voice)
    4. Monitor agent behavior via UEBA
    5. Manage conversation flow
    """

    SYSTEM_PROMPT = """You are the SentinEV Master Orchestrator, an AI assistant for EV predictive maintenance.

Your responsibilities:
1. Monitor vehicle health through telemetry analysis
2. Explain vehicle issues to owners in simple terms
3. Recommend maintenance actions
4. Help schedule service appointments
5. Provide driving tips to improve efficiency and reduce wear

Current vehicle context will be provided. Respond helpfully and concisely.
Be friendly but professional. If there's an urgent issue, clearly communicate the severity.

IMPORTANT: Keep responses under 100 words unless asked for details."""

    def __init__(self):
        """Initialize the Master Orchestrator."""
        self.data_agents: Dict[str, DataAnalysisAgent] = {}
        self.voice_handler = VoiceHandler()
        self.ueba = UEBAMonitor()
        self.llm = None
        self.workflow = None

        # New: Safety and Diagnosis agents
        self.safety_agent: Optional[SafetyAgent] = None
        self.diagnosis_agent: Optional[DiagnosisAgent] = None

        # New: Prediction tracking per vehicle
        self.predictions: Dict[str, Dict] = {}  # vehicle_id -> prediction state

        # New: Notification queue per vehicle (for chatbot)
        self.notification_queue: Dict[str, List[Dict]] = (
            {}
        )  # vehicle_id -> [notifications]

        # New: User behavior tracking (for repeated rejection detection)
        self.rejection_counts: Dict[str, int] = {}  # vehicle_id -> count

        # New: Scheduling and Feedback agents
        self.scheduling_agent: Optional[SchedulingAgent] = None
        self.feedback_agent: Optional[FeedbackAgent] = None

        # New: Chat context store (for diagnosis -> chat redirect)
        self.chat_contexts: Dict[str, Dict] = {}  # vehicle_id -> diagnosis context

        self._initialize_llm()
        self._initialize_agents()
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()

    def _initialize_agents(self):
        """Initialize Safety, Diagnosis, Scheduling, and Feedback agents."""
        self.safety_agent = get_safety_agent()
        self.diagnosis_agent = get_diagnosis_agent()
        self.scheduling_agent = get_scheduling_agent()
        self.feedback_agent = get_feedback_agent()
        print("✓ Safety, Diagnosis, Scheduling, and Feedback agents initialized")

    def _initialize_llm(self):
        """Initialize the LLM for conversation."""
        if LANGCHAIN_AVAILABLE:
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
                    print("✓ Orchestrator LLM initialized")
                except Exception as e:
                    print(f"⚠️ Could not initialize LLM: {e}")

    def get_data_agent(self, vehicle_id: str) -> DataAnalysisAgent:
        """Get or create a Data Analysis Agent for a vehicle."""
        if vehicle_id not in self.data_agents:
            self.data_agents[vehicle_id] = DataAnalysisAgent(vehicle_id)
        return self.data_agents[vehicle_id]

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        if not LANGGRAPH_AVAILABLE:
            return

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze", self._node_analyze)
        workflow.add_node("respond", self._node_respond)
        workflow.add_node("alert_handler", self._node_alert_handler)

        # Add edges
        workflow.set_entry_point("analyze")

        # Conditional routing after analysis
        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analysis,
            {
                "alert": "alert_handler",
                "respond": "respond",
                "end": END,
            },
        )

        workflow.add_edge("alert_handler", "respond")
        workflow.add_edge("respond", END)

        # Compile with checkpointer for state persistence
        self.workflow = workflow.compile(checkpointer=MemorySaver())

    def _route_after_analysis(self, state: AgentState) -> str:
        """Route to next node based on analysis results."""
        if state.get("severity") in ["high", "critical"]:
            return "alert"
        elif state.get("messages"):
            return "respond"
        else:
            return "end"

    def _node_analyze(self, state: AgentState) -> AgentState:
        """Analysis node - delegates to Data Analysis Agent."""
        vehicle_id = state.get("vehicle_id")
        if not vehicle_id:
            return state

        # Get data agent
        agent = self.get_data_agent(vehicle_id)

        # Run analysis
        state = agent.analyze(state)

        # Log for UEBA
        self.ueba.log_action(
            "orchestrator",
            "route_to_agent",
            {
                "target_agent": "data_analysis",
                "vehicle_id": vehicle_id,
            },
        )

        return state

    def _node_respond(self, state: AgentState) -> AgentState:
        """Response generation node."""
        if not self.llm:
            return state

        # Build context
        messages = state.get("messages", [])
        if not messages:
            return state

        # Create conversation context
        context = f"""
Vehicle: {state.get('vehicle_id')}
Driver Profile: {state.get('driver_profile', 'unknown')}
Mode: {state.get('mode')}

Current Status:
- Anomaly Detected: {state.get('is_anomaly', False)}
- Type: {state.get('anomaly_type', 'normal')}
- Severity: {state.get('severity', 'low')}
- Failure Risk: {state.get('failure_risk_pct', 0)}%
- Score: {state.get('total_score', 0)} ({state.get('score_delta', 0):+d} this reading)

Alerts: {len(state.get('active_alerts', []))} active
"""

        # Get last user message
        last_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content")
                break

        if not last_message:
            return state

        try:
            # Generate response
            response = self.llm.invoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    SystemMessage(content=f"Context:\n{context}"),
                    HumanMessage(content=last_message),
                ]
            )

            # Add response to messages
            new_messages = [
                {
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.now().isoformat(),
                }
            ]
            state["messages"] = new_messages

            # Log for UEBA
            self.ueba.log_action(
                "orchestrator",
                "generate_response",
                {
                    "response_length": len(response.content),
                },
            )

        except Exception as e:
            print(f"Response generation error: {e}")

        return state

    def _node_alert_handler(self, state: AgentState) -> AgentState:
        """Alert handling node."""
        alerts = state.get("active_alerts", [])

        if alerts:
            # Log alert handling
            self.ueba.log_action(
                "orchestrator",
                "escalate_alert",
                {
                    "alert_count": len(alerts),
                    "severity": state.get("severity"),
                },
            )

            # Generate alert message
            alert = alerts[0] if alerts else None
            if alert:
                alert_msg = f"""
⚠️ **{alert.get('severity', 'unknown').upper()} ALERT**
Type: {alert.get('anomaly_type', 'unknown')}
Risk: {alert.get('failure_risk_pct', 0)}%
Components: {', '.join(alert.get('affected_components', []))}
Action: {alert.get('recommended_action', 'Contact service')}
"""
                state["messages"] = state.get("messages", []) + [
                    {
                        "role": "system",
                        "content": alert_msg,
                        "timestamp": datetime.now().isoformat(),
                    }
                ]

        return state

    def process_telemetry(
        self,
        vehicle_id: str,
        telemetry: Dict[str, float],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process telemetry through the full workflow.

        Args:
            vehicle_id: Vehicle identifier
            telemetry: Current telemetry readings
            session_id: Optional session ID for state persistence

        Returns:
            Processing result with anomaly detection and scoring
        """
        # Create or update state
        state = create_initial_state(vehicle_id, session_id)
        state["current_telemetry"] = telemetry

        if self.workflow:
            # Run through LangGraph workflow
            config = {"configurable": {"thread_id": session_id or vehicle_id}}
            result = self.workflow.invoke(state, config)
            return result
        else:
            # Fallback to direct agent call
            agent = self.get_data_agent(vehicle_id)
            return agent.analyze(state)

    def chat(
        self, vehicle_id: str, user_message: str, session_id: Optional[str] = None
    ) -> str:
        """
        Process a chat message.

        Args:
            vehicle_id: Vehicle identifier
            user_message: User's message
            session_id: Optional session ID

        Returns:
            Assistant's response
        """
        # Create state with message
        state = create_initial_state(vehicle_id, session_id)
        state["messages"] = [
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        ]

        # Get current vehicle telemetry if available
        if vehicle_id in self.data_agents:
            agent = self.data_agents[vehicle_id]
            # If pipeline has last analysis, we could use it here

        # Check for SERVICE TRACKING queries FIRST (before LLM)
        message_lower = user_message.lower()
        service_tracking_keywords = [
            "status",
            "tracking",
            "where is my",
            "update on",
            "progress",
            "stage",
            "repair status",
            "service status",
            "what stage",
            "how is my",
            "when will",
            "is it ready",
            "pickup",
        ]
        is_service_query = any(kw in message_lower for kw in service_tracking_keywords)

        if is_service_query:
            # Call service tracking endpoint
            try:
                from db.database import get_database

                db = get_database()
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

                if apt:
                    apt_dict = dict(apt)
                    stage = apt_dict.get("stage", "INTAKE")
                    stage_labels = {
                        "INTAKE": (
                            "Vehicle Intake",
                            "Your vehicle has been received at the service center.",
                            0,
                        ),
                        "DIAGNOSIS": (
                            "Diagnosis",
                            "Technicians are diagnosing the issue.",
                            15,
                        ),
                        "WAITING_PARTS": (
                            "Waiting for Parts",
                            "Required parts are on the way.",
                            30,
                        ),
                        "REPAIR": (
                            "Repair in Progress",
                            "Your vehicle is currently being repaired.",
                            60,
                        ),
                        "QUALITY_CHECK": (
                            "Quality Check",
                            "Final inspection and verification.",
                            85,
                        ),
                        "READY": (
                            "Ready for Pickup",
                            "Your vehicle is ready! Please come pick it up.",
                            100,
                        ),
                        "PICKED_UP": (
                            "Completed",
                            "Service completed. Vehicle picked up.",
                            100,
                        ),
                    }
                    label, desc, progress = stage_labels.get(
                        stage, ("Unknown", "Status unknown", 0)
                    )

                    return f"""**Service Status for {vehicle_id}** 🔧

**Current Stage:** {label}
**Progress:** {progress}%

{desc}

**Details:**
- Component: {apt_dict.get('component', 'Unknown').capitalize()}
- Urgency: {apt_dict.get('urgency', 'medium').capitalize()}
- Service Center: {apt_dict.get('center_name', 'SC-001')}
- Scheduled: {apt_dict.get('scheduled_date', 'N/A')} at {apt_dict.get('scheduled_time', 'N/A')}

Need more info? Ask me anything about your service!"""
            except Exception as e:
                print(f"Service tracking error: {e}")
                # Fall through to LLM if tracking fails

        # Generate response
        if self.llm:
            try:
                # Check for stored diagnosis context
                context = self.chat_contexts.get(vehicle_id)
                context_prompt = ""
                scheduling_info = ""

                if context:
                    context_prompt = f"""

IMPORTANT CONTEXT - The user was just redirected here after a diagnosis:
- Component: {context.get('component', 'Unknown')}
- Issue: {context.get('summary', 'Vehicle issue detected')}
- Urgency: {context.get('urgency', 'medium')}
- Estimated Cost: {context.get('estimated_cost', 'TBD')}

Your goal:
1. Summarize the problem clearly and empathetically
2. Propose scheduling options (mention you can show available slots)
3. Be collaborative - ask about their preferences
4. If they want to schedule, guide them through booking
"""

                # Check if user wants to CONFIRM a booking (not just ask about dates)
                booking_confirm_keywords = [
                    "book",
                    "confirm",
                    "yes",
                    "proceed",
                    "schedule that",
                    "sounds good",
                    "let's do it",
                    "book it",
                    "8:00",
                    "8 am",
                    "12:00",
                    "4:00",
                    "8am",
                    "12pm",
                    "4pm",
                ]
                message_lower = user_message.lower()
                wants_to_book = any(
                    kw in message_lower for kw in booking_confirm_keywords
                )

                # If user wants to book AND we have context with a component
                if wants_to_book and context:
                    # Try to extract date and time from message
                    extracted_date = self._extract_date_from_message(user_message)
                    extracted_time = self._extract_time_from_message(user_message)

                    component = context.get("component", "general")

                    # Find available slot matching their request
                    try:
                        from agents.scheduling_agent import get_scheduling_agent

                        scheduling_agent = get_scheduling_agent()

                        availability = scheduling_agent.check_availability(
                            component=component, preferred_date=extracted_date, limit=5
                        )

                        if availability.get("available") and availability.get("slots"):
                            # Find best matching slot
                            matching_slot = None
                            for slot in availability["slots"]:
                                # Match by date if specified
                                if (
                                    extracted_date
                                    and slot.get("date") != extracted_date
                                ):
                                    continue
                                # Match by time if specified
                                if extracted_time:
                                    slot_hour = int(
                                        slot.get("start_time", "00:00").split(":")[0]
                                    )
                                    if slot_hour != extracted_time:
                                        continue
                                matching_slot = slot
                                break

                            # If no exact match, take first available
                            if not matching_slot:
                                matching_slot = availability["slots"][0]

                            # Execute the booking!
                            booking_result = scheduling_agent.book_appointment(
                                vehicle_id=vehicle_id,
                                slot_id=matching_slot["id"],
                                center_id=matching_slot["center_id"],
                                component=component,
                                diagnosis_summary=context.get(
                                    "summary", "Preventive service"
                                ),
                                estimated_cost=context.get(
                                    "estimated_cost", "$100-$200"
                                ),
                                urgency=context.get("urgency", "medium"),
                            )

                            if booking_result.get("success"):
                                # Clear context after successful booking
                                self.chat_contexts.pop(vehicle_id, None)
                                return booking_result.get(
                                    "message", "✅ Appointment booked!"
                                )
                            else:
                                # Booking failed, let LLM handle error
                                context_prompt += f"\n\nBOOKING ATTEMPT FAILED: {booking_result.get('message', 'Unknown error')}"
                    except Exception as e:
                        print(f"Error during booking: {e}")
                        context_prompt += (
                            f"\n\nBOOKING ERROR: Could not complete booking - {str(e)}"
                        )

                # Check if user is asking about scheduling/dates
                scheduling_keywords = [
                    "schedule",
                    "book",
                    "appointment",
                    "date",
                    "december",
                    "january",
                    "tomorrow",
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "available",
                    "slot",
                    "when",
                    "time",
                ]
                is_scheduling_query = any(
                    kw in user_message.lower() for kw in scheduling_keywords
                )

                if is_scheduling_query:
                    # Try to extract date from message
                    extracted_date = self._extract_date_from_message(user_message)
                    component = (
                        context.get("component", "general") if context else "general"
                    )

                    # Get availability info
                    try:
                        from ml.labor_forecasting import get_labor_forecaster

                        forecaster = get_labor_forecaster()

                        if extracted_date:
                            # Check specific date
                            availability = forecaster.check_date_availability(
                                extracted_date, component
                            )
                            scheduling_info = f"""

AVAILABILITY DATA for {extracted_date}:
- Available: {availability.get('available')}
- Slots: {availability.get('slots_count')} slots available
- Utilization: {availability.get('utilization_pct')}%
- Specialists: {availability.get('specialists_available')} ({', '.join(availability.get('specialist_names', []))})
- Reasons: {'; '.join(availability.get('reasons', []))}
- Cost Warning: {availability.get('cost_warning', 'None')}

ALTERNATIVE DATES (if this date is busy):
"""
                            for alt in availability.get("alternatives", [])[:3]:
                                scheduling_info += f"- {alt['date']}: {alt['slots_available']} slots ({alt['reason']})\n"

                            scheduling_info += """
RESPONSE GUIDANCE:
- If date is busy (utilization > 70%), explain why and suggest alternatives
- If no specialists, explain and offer similar options
- Mention cost implications if delaying important repairs
- Be collaborative: "Would you prefer..." or "I can also check..."
"""
                        else:
                            # Suggest optimal dates
                            suggestion = forecaster.suggest_optimal_date(
                                component=component,
                                urgency=(
                                    context.get("urgency", "medium")
                                    if context
                                    else "medium"
                                ),
                            )
                            optimal = suggestion.get("optimal_date")
                            if optimal:
                                scheduling_info = f"""

OPTIMAL DATE SUGGESTION:
- Best Date: {optimal.get('date')} (Score: {optimal.get('score')})
- Utilization: {optimal.get('utilization_pct')}%
- Specialists: {optimal.get('specialists_available')} available

ALTERNATIVES:
"""
                                for alt in suggestion.get("alternatives", [])[:3]:
                                    scheduling_info += f"- {alt.get('date')}: Score {alt.get('score')}, {alt.get('slots_count')} slots\n"
                    except Exception as e:
                        print(f"Error checking availability: {e}")

                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=self.SYSTEM_PROMPT
                            + context_prompt
                            + scheduling_info
                        ),
                        HumanMessage(content=user_message),
                    ]
                )
                return response.content
            except Exception as e:
                return f"I'm having trouble responding right now. Error: {e}"
        else:
            return "LLM not available. Please check your API configuration."

    def _extract_date_from_message(self, message: str) -> Optional[str]:
        """Extract date from user message."""
        import re
        from datetime import datetime, timedelta

        message_lower = message.lower()
        today = datetime.now().date()

        # Check for relative dates
        if "tomorrow" in message_lower:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if "today" in message_lower:
            return today.strftime("%Y-%m-%d")

        # Check for day names
        days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        for i, day in enumerate(days):
            if day in message_lower:
                days_ahead = (i - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # Next week if today
                return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        # Check for explicit dates like "December 9th" or "9th December"
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        for month_name, month_num in months.items():
            if month_name in message_lower:
                # Extract day number
                day_match = re.search(r"(\d{1,2})(st|nd|rd|th)?", message_lower)
                if day_match:
                    day = int(day_match.group(1))
                    year = today.year
                    # If month is in the past, assume next year
                    if month_num < today.month or (
                        month_num == today.month and day < today.day
                    ):
                        year += 1
                    try:
                        return f"{year}-{month_num:02d}-{day:02d}"
                    except:
                        pass

        return None

    def _extract_time_from_message(self, message: str) -> Optional[int]:
        """Extract hour from user message (returns hour as int, e.g., 8 for 8:00 AM)."""
        import re

        message_lower = message.lower()

        # Check for time patterns like "8:00", "8am", "4 pm", "12:00"
        time_patterns = [
            r"(\d{1,2}):(\d{2})\s*(am|pm)?",  # 8:00, 8:00 am
            r"(\d{1,2})\s*(am|pm)",  # 8am, 4 pm
        ]

        for pattern in time_patterns:
            match = re.search(pattern, message_lower)
            if match:
                hour = int(match.group(1))
                ampm = match.group(len(match.groups()))  # Last group is am/pm

                if ampm == "pm" and hour < 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0

                return hour

        # Common time references
        if "morning" in message_lower:
            return 8
        if "noon" in message_lower or "lunch" in message_lower:
            return 12
        if "afternoon" in message_lower:
            return 14
        if "evening" in message_lower:
            return 17

        return None

    def voice_interaction(self, vehicle_id: str) -> Optional[str]:
        """
        Handle voice-based interaction.

        Returns:
            Response text (also available as audio)
        """
        # Listen for input
        user_speech = self.voice_handler.listen()

        if not user_speech:
            return None

        print(f"You said: {user_speech}")

        # Process as chat
        response = self.chat(vehicle_id, user_speech)

        # Generate speech output
        audio = self.voice_handler.speak(response)

        return response

    def inject_fault(
        self, vehicle_id: str, fault_type: str, severity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Inject a fault for testing anomaly detection.

        Args:
            vehicle_id: Vehicle identifier
            fault_type: Type of fault to inject
            severity: Fault severity (0.5-2.0)

        Returns:
            Result of processing with injected fault
        """
        # Get data agent
        agent = self.get_data_agent(vehicle_id)

        # We need a telemetry generator to inject fault
        from ml.telemetry_generator import EnhancedTelemetryGenerator, VehicleConfig

        config = VehicleConfig(vehicle_id)
        generator = EnhancedTelemetryGenerator(config)

        # Inject the fault
        generator.inject_fault(fault_type, severity)

        # Generate telemetry with fault
        telemetry = generator.step()

        # Process through workflow
        result = self.process_telemetry(vehicle_id, telemetry)

        # Log for UEBA
        self.ueba.log_action(
            "orchestrator",
            "fault_injection",
            {
                "vehicle_id": vehicle_id,
                "fault_type": fault_type,
                "severity": severity,
            },
        )

        return result

    def get_ueba_report(self) -> Dict[str, Any]:
        """Get UEBA monitoring report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_actions": len(self.ueba.action_history),
            "alerts": self.ueba.get_alerts(),
            "agent_summaries": {
                "orchestrator": self.ueba.get_agent_summary("orchestrator"),
                "data_analysis": self.ueba.get_agent_summary("data_analysis"),
            },
        }

    # ==================== Prediction & Agent Routing ====================

    def add_notification(self, vehicle_id: str, notification: Dict[str, Any]):
        """Add a notification to the queue for a vehicle."""
        if vehicle_id not in self.notification_queue:
            self.notification_queue[vehicle_id] = []
        notification["id"] = (
            f"notif-{len(self.notification_queue[vehicle_id])}-{datetime.now().strftime('%H%M%S')}"
        )
        notification["read"] = False
        self.notification_queue[vehicle_id].append(notification)

    def get_notifications(
        self, vehicle_id: str, unread_only: bool = False
    ) -> List[Dict]:
        """Get notifications for a vehicle."""
        notifications = self.notification_queue.get(vehicle_id, [])
        if unread_only:
            return [n for n in notifications if not n.get("read", False)]
        return notifications

    def mark_notification_read(self, vehicle_id: str, notification_id: str) -> bool:
        """Mark a notification as read."""
        notifications = self.notification_queue.get(vehicle_id, [])
        for n in notifications:
            if n.get("id") == notification_id:
                n["read"] = True
                return True
        return False

    def create_prediction(
        self,
        vehicle_id: str,
        component: str,
        anomaly_type: str,
        severity: str,
        days_to_failure: int,
        message: str,
        requires_service: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a prediction alert for a vehicle.

        Args:
            vehicle_id: Vehicle identifier
            component: Affected component
            anomaly_type: Type of anomaly
            severity: Severity level
            days_to_failure: Estimated days until failure
            message: Prediction message
            requires_service: Whether service is required

        Returns:
            Prediction object
        """
        prediction_id = f"pred-{vehicle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        prediction = {
            "prediction_id": prediction_id,
            "vehicle_id": vehicle_id,
            "component": component,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "days_to_failure": days_to_failure,
            "message": message,
            "requires_service": requires_service,
            "status": "pending",  # pending, accepted, rejected
            "created_at": datetime.now().isoformat(),
        }

        # Store prediction
        self.predictions[vehicle_id] = prediction

        # Create chatbot notification
        notification = {
            "type": "prediction_alert",
            "prediction_id": prediction_id,
            "severity": severity,
            "component": component,
            "days_to_failure": days_to_failure,
            "message": message,
            "actions": ["accept", "reject"],
            "timestamp": datetime.now().isoformat(),
        }
        self.add_notification(vehicle_id, notification)

        # Log for UEBA
        self.ueba.log_action(
            "orchestrator",
            "create_prediction",
            {
                "vehicle_id": vehicle_id,
                "prediction_id": prediction_id,
                "severity": severity,
            },
        )

        return prediction

    def get_prediction(self, vehicle_id: str) -> Optional[Dict]:
        """Get active prediction for a vehicle."""
        return self.predictions.get(vehicle_id)

    def accept_prediction(self, vehicle_id: str, prediction_id: str) -> Dict[str, Any]:
        """
        User accepts a prediction warning.
        Routes to Safety Agent for precautions.
        """
        prediction = self.predictions.get(vehicle_id)
        if not prediction or prediction.get("prediction_id") != prediction_id:
            return {"error": "Prediction not found"}

        # Update prediction status
        prediction["status"] = "accepted"
        prediction["accepted_at"] = datetime.now().isoformat()

        # Reset rejection count
        self.rejection_counts[vehicle_id] = 0

        # Route to Safety Agent
        if self.safety_agent:
            advice = self.safety_agent.process_accepted_prediction(
                prediction_id=prediction_id,
                anomaly_type=prediction["anomaly_type"],
                component=prediction["component"],
                severity=prediction["severity"],
                days_to_failure=prediction["days_to_failure"],
            )

            # Convert to chatbot message and add to notifications
            chatbot_msg = self.safety_agent.to_chatbot_message(advice)
            self.add_notification(vehicle_id, chatbot_msg)

            # Add toast notification for points
            toast = {
                "type": "toast",
                "variant": "positive",
                "points": advice.points_awarded,
                "message": f"🌟 +{advice.points_awarded} points for accepting the safety warning!",
                "timestamp": datetime.now().isoformat(),
            }
            self.add_notification(vehicle_id, toast)

            # Log for UEBA
            self.ueba.log_action(
                "orchestrator",
                "route_to_safety_agent",
                {"vehicle_id": vehicle_id, "prediction_id": prediction_id},
            )

            return {
                "status": "accepted",
                "routed_to": "safety_agent",
                "advice": chatbot_msg,
                "points_awarded": advice.points_awarded,
            }

        return {"status": "accepted", "error": "Safety agent not available"}

    def reject_prediction(self, vehicle_id: str, prediction_id: str) -> Dict[str, Any]:
        """
        User rejects/ignores a prediction warning.
        Tracks rejections and routes to Diagnosis Agent if repeated.
        """
        prediction = self.predictions.get(vehicle_id)
        if not prediction or prediction.get("prediction_id") != prediction_id:
            return {"error": "Prediction not found"}

        # Update prediction status
        prediction["status"] = "rejected"
        prediction["rejected_at"] = datetime.now().isoformat()

        # Increment rejection count
        self.rejection_counts[vehicle_id] = self.rejection_counts.get(vehicle_id, 0) + 1
        rejection_count = self.rejection_counts[vehicle_id]

        # If requires_service or repeated rejections (3+), route to Diagnosis Agent
        if prediction.get("requires_service") or rejection_count >= 3:
            return self._route_to_diagnosis(vehicle_id, prediction)
        else:
            # Add warning notification
            notification = {
                "type": "rejection_warning",
                "message": f"⚠️ You've ignored {rejection_count} warning(s) for {prediction['component']}. "
                f"Continued behavior may cause damage requiring service.",
                "rejection_count": rejection_count,
                "max_rejections": 3,
                "timestamp": datetime.now().isoformat(),
            }
            self.add_notification(vehicle_id, notification)

            # Add negative points toast
            toast = {
                "type": "toast",
                "variant": "negative",
                "points": -15,
                "message": f"⚠️ -15 points for ignoring safety warning",
                "timestamp": datetime.now().isoformat(),
            }
            self.add_notification(vehicle_id, toast)

            return {
                "status": "rejected",
                "rejection_count": rejection_count,
                "warning": f"Ignoring {3 - rejection_count} more warnings will trigger mandatory diagnosis.",
            }

    def _route_to_diagnosis(self, vehicle_id: str, prediction: Dict) -> Dict[str, Any]:
        """Route to Diagnosis Agent for major issues."""
        if not self.diagnosis_agent:
            return {"error": "Diagnosis agent not available"}

        diagnosis_id = f"diag-{vehicle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Get current telemetry if available
        telemetry_data = {}
        if vehicle_id in self.data_agents:
            agent = self.data_agents[vehicle_id]
            if hasattr(agent, "last_telemetry"):
                telemetry_data = agent.last_telemetry or {}

        # Start diagnosis
        result = self.diagnosis_agent.start_diagnosis(
            diagnosis_id=diagnosis_id,
            vehicle_id=vehicle_id,
            component=prediction["component"],
            issue_type=prediction["anomaly_type"],
            severity=prediction["severity"],
            days_to_failure=prediction["days_to_failure"],
            telemetry_data=telemetry_data,
        )

        # Convert to chatbot message and add to notifications
        chatbot_msg = self.diagnosis_agent.to_chatbot_message(result)
        self.add_notification(vehicle_id, chatbot_msg)

        # Log for UEBA
        self.ueba.log_action(
            "orchestrator",
            "route_to_diagnosis_agent",
            {"vehicle_id": vehicle_id, "diagnosis_id": diagnosis_id},
        )

        return {
            "status": "diagnosis_started",
            "routed_to": "diagnosis_agent",
            "diagnosis": chatbot_msg,
        }

    def confirm_service(self, vehicle_id: str, diagnosis_id: str) -> Dict[str, Any]:
        """Confirm service scheduling from diagnosis."""
        if not self.diagnosis_agent:
            return {"error": "Diagnosis agent not available"}

        result = self.diagnosis_agent.confirm_service(diagnosis_id)

        # Add confirmation notification
        notification = {
            "type": "service_confirmed",
            "diagnosis_id": diagnosis_id,
            "message": result.get("message", "Service scheduled"),
            "component": result.get("component"),
            "timestamp": datetime.now().isoformat(),
        }
        self.add_notification(vehicle_id, notification)

        # Log for UEBA
        self.ueba.log_action(
            "orchestrator",
            "confirm_service",
            {"vehicle_id": vehicle_id, "diagnosis_id": diagnosis_id},
        )

        return result

    def decline_service(self, vehicle_id: str, diagnosis_id: str) -> Dict[str, Any]:
        """Decline service scheduling from diagnosis."""
        if not self.diagnosis_agent:
            return {"error": "Diagnosis agent not available"}

        result = self.diagnosis_agent.decline_service(diagnosis_id)

        # Add warning notification
        notification = {
            "type": "service_declined_warning",
            "diagnosis_id": diagnosis_id,
            "message": result.get("warning", "Service declined"),
            "monitoring": result.get("monitoring"),
            "timestamp": datetime.now().isoformat(),
        }
        self.add_notification(vehicle_id, notification)

        return result

    def get_diagnosis(self, vehicle_id: str) -> Optional[Dict]:
        """Get active diagnosis for a vehicle."""
        if not self.diagnosis_agent:
            return None

        # Find diagnosis for this vehicle
        for diag_id, diagnosis in self.diagnosis_agent.active_diagnoses.items():
            if diagnosis.vehicle_id == vehicle_id:
                return self.diagnosis_agent.to_chatbot_message(diagnosis)
        return None

    # ==================== Scheduling Flow ====================

    def propose_scheduling(
        self,
        vehicle_id: str,
        component: str,
        diagnosis_summary: str,
        estimated_cost: str,
        urgency: str = "medium",
    ) -> Dict[str, Any]:
        """
        Propose appointment slots after diagnosis.
        Called after DiagnosisAgent completes.
        """
        if not self.scheduling_agent:
            return {"error": "Scheduling agent not initialized"}

        result = self.scheduling_agent.propose_slots(
            vehicle_id=vehicle_id,
            component=component,
            diagnosis_summary=diagnosis_summary,
            estimated_cost=estimated_cost,
            urgency=urgency,
        )

        # Add to notification queue
        if result["success"]:
            self.add_notification(
                vehicle_id,
                "scheduling_proposed",
                "Schedule Your Service",
                f"We have {len(result['slots'])} available slots for your {component} service.",
            )

        return result

    def book_appointment(
        self,
        vehicle_id: str,
        slot_id: str,
        center_id: str,
        component: str,
        diagnosis_summary: str,
        estimated_cost: str,
        urgency: str = "medium",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Book a service appointment."""
        if not self.scheduling_agent:
            return {"error": "Scheduling agent not initialized"}

        result = self.scheduling_agent.book_appointment(
            vehicle_id=vehicle_id,
            slot_id=slot_id,
            center_id=center_id,
            component=component,
            diagnosis_summary=diagnosis_summary,
            estimated_cost=estimated_cost,
            urgency=urgency,
            notes=notes,
        )

        if result["success"] and result.get("notification"):
            notif = result["notification"]
            self.add_notification(
                vehicle_id, notif["type"], notif["title"], notif["body"]
            )

        self.ueba.log_action(
            "scheduling_agent",
            "book_appointment",
            {
                "vehicle_id": vehicle_id,
                "result": "success" if result["success"] else "failed",
            },
        )

        return result

    def get_appointments(self, vehicle_id: str) -> Dict[str, Any]:
        """Get appointments for a vehicle."""
        if not self.scheduling_agent:
            return {"error": "Scheduling agent not initialized"}
        return self.scheduling_agent.get_appointments(vehicle_id)

    def update_service_status(
        self, appointment_id: str, new_status: str
    ) -> Dict[str, Any]:
        """Update service status and notify customer."""
        if not self.scheduling_agent:
            return {"error": "Scheduling agent not initialized"}

        result = self.scheduling_agent.update_service_status(appointment_id, new_status)

        # If completed, trigger feedback flow
        if new_status == "completed" and result["success"]:
            # Get vehicle_id from appointment
            appointments = self.scheduling_agent.db.get_appointments()
            appt = next((a for a in appointments if a["id"] == appointment_id), None)
            if appt:
                self.add_notification(
                    appt["vehicle_id"],
                    "service_completed",
                    "✅ Service Complete!",
                    "Your vehicle is ready. Please share your feedback!",
                )

        return result

    def cancel_appointment(
        self, appointment_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Cancel an appointment."""
        if not self.scheduling_agent:
            return {"error": "Scheduling agent not initialized"}
        return self.scheduling_agent.cancel_appointment(appointment_id, reason)

    def get_service_progress(self, appointment_id: str) -> Dict[str, Any]:
        """Get service progress for tracking."""
        if not self.scheduling_agent:
            return {"error": "Scheduling agent not initialized"}
        return self.scheduling_agent.get_service_progress(appointment_id)

    # ==================== Feedback Flow ====================

    def initiate_feedback(self, appointment_id: str) -> Dict[str, Any]:
        """Initiate feedback collection for completed service."""
        if not self.feedback_agent:
            return {"error": "Feedback agent not initialized"}

        result = self.feedback_agent.initiate_followup(appointment_id)

        if result["success"] and result.get("notification"):
            notif = result["notification"]
            self.add_notification(
                result["vehicle_id"], notif["type"], notif["title"], notif["body"]
            )

        return result

    def submit_feedback(
        self, appointment_id: str, vehicle_id: str, rating: int, comments: str = ""
    ) -> Dict[str, Any]:
        """Submit customer feedback."""
        if not self.feedback_agent:
            return {"error": "Feedback agent not initialized"}

        result = self.feedback_agent.process_feedback(
            appointment_id=appointment_id,
            vehicle_id=vehicle_id,
            rating=rating,
            comments=comments,
        )

        if result["success"]:
            self.ueba.log_action(
                "feedback_agent",
                "submit_feedback",
                {"vehicle_id": vehicle_id, "rating": rating},
            )
        return result

    def get_service_history(self, vehicle_id: str) -> Dict[str, Any]:
        """Get complete service history for a vehicle."""
        if not self.feedback_agent:
            return {"error": "Feedback agent not initialized"}
        return self.feedback_agent.get_service_history(vehicle_id)

    # ==================== Chat Context Management ====================

    def set_chat_context(
        self,
        vehicle_id: str,
        diagnosis_id: str = None,
        component: str = None,
        summary: str = None,
        urgency: str = None,
        estimated_cost: str = None,
        **kwargs,
    ):
        """
        Store diagnosis context for chat session.
        Called after diagnosis to enable context-aware chatbot.
        """
        self.chat_contexts[vehicle_id] = {
            "diagnosis_id": diagnosis_id,
            "component": component,
            "summary": summary,
            "urgency": urgency,
            "estimated_cost": estimated_cost,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

    def get_chat_context(self, vehicle_id: str) -> Optional[Dict]:
        """Retrieve stored chat context for a vehicle."""
        return self.chat_contexts.get(vehicle_id)

    def clear_chat_context(self, vehicle_id: str):
        """Clear stored chat context after conversation ends."""
        if vehicle_id in self.chat_contexts:
            del self.chat_contexts[vehicle_id]

    def start_scheduling_conversation(self, vehicle_id: str) -> Dict[str, Any]:
        """
        Start a scheduling conversation with context.
        Used when user is redirected after diagnosis.

        Returns:
            Dict with intro message, proposed slots, and quick actions
        """
        context = self.get_chat_context(vehicle_id)

        if not context:
            return {
                "success": False,
                "message": "Hello! How can I help you today?",
                "has_context": False,
            }

        component = context.get("component", "general")
        summary = context.get("summary", "Vehicle issue detected")
        urgency = context.get("urgency", "medium")
        estimated_cost = context.get("estimated_cost", "TBD")

        # Generate intro message
        intro_message = self._generate_diagnosis_intro(context)

        # Get scheduling proposals
        if self.scheduling_agent:
            proposal = self.scheduling_agent.propose_slots(
                vehicle_id=vehicle_id,
                component=component,
                diagnosis_summary=summary,
                estimated_cost=estimated_cost,
                urgency=urgency,
            )

            return {
                "success": True,
                "has_context": True,
                "intro_message": intro_message,
                "scheduling_message": proposal.get("message", ""),
                "slots": proposal.get("slots", []),
                "context": context,
                "quick_actions": [
                    {"label": "📅 Show available slots", "action": "show_slots"},
                    {"label": "📞 Call me instead", "action": "call_request"},
                    {"label": "⏰ Remind me later", "action": "remind_later"},
                    {"label": "❓ I have questions", "action": "ask_question"},
                ],
            }
        else:
            return {
                "success": True,
                "has_context": True,
                "intro_message": intro_message,
                "message": "I'd be happy to help you schedule service. Let me check availability...",
                "context": context,
            }

    def _generate_diagnosis_intro(self, context: Dict) -> str:
        """Generate personalized intro message after diagnosis redirect."""
        component = context.get("component", "your vehicle")
        summary = context.get("summary", "an issue")
        urgency = context.get("urgency", "medium")
        cost = context.get("estimated_cost", "varies")

        urgency_phrases = {
            "critical": "⚠️ **Urgent Attention Required**\n\nI need to discuss something important with you.",
            "high": "I wanted to talk to you about something that needs attention soon.",
            "medium": "I have some information about your vehicle that we should discuss.",
            "low": "Just a quick heads up about something we noticed.",
        }

        intro = f"""{urgency_phrases.get(urgency, urgency_phrases['medium'])}

**Diagnosis Summary:**
🔧 **Component:** {component.title()}
📋 **Issue:** {summary}
💰 **Estimated Cost:** {cost}

I can help you schedule a service appointment right now. We have several convenient time slots available.

Would you like me to show you the available options? Or if you have any questions about the diagnosis, I'm here to help!"""

        return intro


# Global orchestrator instance
_orchestrator: Optional[MasterOrchestrator] = None


def get_orchestrator() -> MasterOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MasterOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    # Test the orchestrator
    print("🎭 Testing Master Orchestrator")
    print("=" * 50)

    orchestrator = get_orchestrator()

    # Test telemetry processing
    test_telemetry = {
        "speed_kmh": 85.0,
        "acceleration_ms2": 1.5,
        "jerk_ms3": 0.3,
        "power_draw_kw": 42.0,
        "regen_efficiency": 0.78,
        "battery_soc_pct": 65.0,
        "battery_temp_c": 38.0,
        "motor_temp_c": 72.0,
        "inverter_temp_c": 65.0,
        "brake_temp_c": 85.0,
        "coolant_temp_c": 40.0,
        "wear_index": 0.08,
    }

    print("\n📊 Processing telemetry...")
    result = orchestrator.process_telemetry("VIN_TEST", test_telemetry)
    print(f"Mode: {result.get('mode')}")
    print(f"Anomaly: {result.get('is_anomaly')}")
    print(f"Score: {result.get('total_score')} ({result.get('score_delta'):+d})")

    # Test chat
    print("\n💬 Testing chat...")
    response = orchestrator.chat("VIN_TEST", "What's the status of my vehicle?")
    print(f"Response: {response[:200]}...")

    # Test fault injection
    print("\n🔧 Testing fault injection...")
    fault_result = orchestrator.inject_fault("VIN_TEST", "overheat", 1.5)
    print(f"Fault result - Anomaly: {fault_result.get('is_anomaly')}")
    print(f"Severity: {fault_result.get('severity')}")

    # UEBA report
    print("\n🔐 UEBA Report:")
    ueba_report = orchestrator.get_ueba_report()
    print(f"Total actions logged: {ueba_report['total_actions']}")
    print(f"UEBA alerts: {len(ueba_report['alerts'])}")

    print("\n✅ Master Orchestrator test complete")
