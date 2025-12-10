"""
SentinEV - Voice Agent
Browser-based voice interaction using Web Speech API with Vapi.ai compatibility.
Handles real-time voice conversations for critical alerts and scheduling.
"""

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Import database singleton
from db.database import get_database

# Phase 10: Voice personas and emotion detection
from agents.voice_personas import (
    EmotionState,
    AlertScenario,
    VoicePersona,
    PERSONAS,
    detect_emotion,
    get_persona_for_scenario,
    get_emotion_response_modifier,
    get_scenario_script,
    translate_technical_term,
    get_health_coaching_tip,
    SCENARIO_SCRIPTS,
    EMOTION_RESPONSE_MODIFIERS,
)

# RAG Knowledge Base for Q&A
try:
    from ml.rag_knowledge import MLKnowledgeBase

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Optional: Gemini for conversation
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Optional: gTTS for server-side TTS fallback
try:
    from gtts import gTTS
    import io
    import base64

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class CallState(Enum):
    """Voice call states."""

    IDLE = "idle"
    RINGING = "ringing"
    CONNECTED = "connected"
    ON_HOLD = "on_hold"
    ENDED = "ended"


class ConversationStage(Enum):
    """Stages of the voice conversation flow."""

    GREETING = "greeting"
    ALERT_EXPLANATION = "alert_explanation"
    SAFETY_CHECK = "safety_check"
    SCHEDULING_OFFER = "scheduling_offer"
    SLOT_CONFIRMATION = "slot_confirmation"
    BOOKING_CONFIRMED = "booking_confirmed"
    FAREWELL = "farewell"


@dataclass
class VoiceCall:
    """Represents an active voice call session."""

    call_id: str
    vehicle_id: str
    state: CallState = CallState.IDLE
    stage: ConversationStage = ConversationStage.GREETING
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    transcript: List[Dict[str, str]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    proposed_slot: Optional[Dict] = None
    booked_appointment: Optional[Dict] = None
    # Phase 10: Emotion tracking and persona
    current_emotion: EmotionState = EmotionState.CALM
    emotion_history: List[EmotionState] = field(default_factory=list)
    persona: Optional[VoicePersona] = None
    scenario: AlertScenario = AlertScenario.GENERAL_WARNING


# Conversation scripts for Brake Fade scenario
BRAKE_FADE_SCRIPTS = {
    ConversationStage.GREETING: {
        "ai": "Hi {owner_name}, this is SentinEV. Urgent safety alert regarding your vehicle.",
        "expected_intents": ["acknowledge", "question"],
    },
    ConversationStage.ALERT_EXPLANATION: {
        "ai": "Your brake fluid has boiled due to overheating on this descent. You currently have only {brake_efficiency}% braking efficiency remaining. This is a critical safety concern.",
        "expected_intents": ["understand", "question", "panic"],
    },
    ConversationStage.SAFETY_CHECK: {
        "ai": "For your immediate safety, I recommend reducing speed and using engine braking. Are you currently in a safe location to discuss next steps?",
        "expected_intents": ["yes", "no", "question"],
    },
    ConversationStage.SCHEDULING_OFFER: {
        "ai": "I've checked the Service Center Scheduler. The nearest center, '{center_name}', has the specific ceramic brake pads in stock. I have a bay reserved at {slot_time} today. Shall I confirm this appointment and send navigation to your vehicle?",
        "expected_intents": ["yes", "no", "alternative", "question"],
    },
    ConversationStage.SLOT_CONFIRMATION: {
        "ai": "Just to confirm: I'm booking you at {center_name} for {slot_time} today for brake pad replacement. The estimated cost is {cost}. Should I proceed?",
        "expected_intents": ["yes", "no", "question"],
    },
    ConversationStage.BOOKING_CONFIRMED: {
        "ai": "Your appointment is confirmed at {center_name} for {slot_time}. I'm sending navigation to your vehicle now. The service center has been notified of the urgency. Is there anything else I can help with?",
        "expected_intents": ["no", "question"],
    },
    ConversationStage.FAREWELL: {
        "ai": "Drive safely, {owner_name}. The navigation is now active on your vehicle display. If you need any assistance, just say 'Hey SentinEV'. Take care.",
        "expected_intents": [],
    },
}


class VoiceAgent:
    """
    Voice Agent for real-time conversations.

    Supports:
    - Web Speech API (browser-based, free)
    - Vapi.ai (cloud-based, scalable)
    - Server-side TTS fallback (gTTS)
    """

    def __init__(self):
        """Initialize the Voice Agent with Phase 10 enhancements."""
        self.active_calls: Dict[str, VoiceCall] = {}
        self.llm = None
        self.conversation_chain = None
        self.knowledge_base = None
        self._initialize_llm()
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize RAG knowledge base for Q&A during calls."""
        if not RAG_AVAILABLE:
            print("⚠️ RAG knowledge base not available")
            return

        try:
            self.knowledge_base = MLKnowledgeBase()
            if self.knowledge_base.load_knowledge_base():
                print("✅ Voice Agent RAG initialized")
            else:
                print("⚠️ RAG knowledge base not built yet")
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {e}")

    def _initialize_llm(self):
        """Initialize LLM for dynamic conversation."""
        if not LLM_AVAILABLE:
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                google_api_key=api_key,
                temperature=0.7,
            )

            # Create conversation chain for dynamic responses
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are SentinEV, an AI voice assistant for electric vehicle safety.
You are currently on a phone call with a vehicle owner about a critical safety issue.
Be calm, professional, and reassuring. Keep responses SHORT (1-2 sentences) for voice.
Current stage: {stage}
Vehicle issue: {issue}
Owner name: {owner_name}
Available service center: {center_name}
Available slot: {slot_time}
""",
                    ),
                    ("human", "{user_input}"),
                ]
            )

            self.conversation_chain = prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"⚠️ Voice Agent LLM init failed: {e}")

    # ==================== Phase 10: Emotion-Adaptive Methods ====================

    def _detect_and_track_emotion(
        self, call: VoiceCall, user_text: str
    ) -> EmotionState:
        """
        Detect user's emotional state and track it in call history.

        Args:
            call: Active voice call
            user_text: User's spoken text

        Returns:
            Detected EmotionState
        """
        emotion = detect_emotion(user_text)
        call.current_emotion = emotion
        call.emotion_history.append(emotion)
        return emotion

    def _adapt_message_for_emotion(self, message: str, emotion: EmotionState) -> str:
        """
        Adapt response message based on user's emotional state.

        Args:
            message: Original response message
            emotion: Detected emotional state

        Returns:
            Emotion-adapted message
        """
        modifier = get_emotion_response_modifier(emotion)
        prefix = modifier.get("prefix", "")

        # For panicked users, break into steps
        if modifier.get("break_into_steps") and "." in message:
            sentences = message.split(".")
            if len(sentences) > 2:
                message = ". ".join(sentences[:2]) + "."

        # Simplify technical terms if needed
        if modifier.get("simplify"):
            for term, translation in [
                ("brake fade", "loss of braking power"),
                ("thermal runaway", "dangerous overheating"),
                ("cell imbalance", "uneven battery charge"),
            ]:
                message = message.replace(term, translation)

        return prefix + message

    def _answer_question_with_rag(
        self, call: VoiceCall, question: str
    ) -> Optional[str]:
        """
        Answer customer question using RAG knowledge base.

        Args:
            call: Active voice call with context
            question: Customer's question

        Returns:
            RAG-generated answer or None
        """
        if not self.knowledge_base:
            return None

        try:
            # Get context from call
            alert_type = call.context.get("alert_type", "")
            component = call.context.get("component", "general")

            # Search knowledge base
            results = self.knowledge_base.semantic_search(question, k=3)

            if not results:
                return None

            # Extract relevant info
            context_text = "\n".join([r.page_content for r in results[:2]])

            # Use LLM to synthesize answer if available
            if self.llm:
                try:
                    prompt = f"""Based on this technical knowledge:
{context_text}

Answer this customer question briefly (1-2 sentences) in simple language:
"{question}"

Context: The customer has a {alert_type} issue with their EV."""

                    response = self.llm.invoke(prompt)
                    return (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                except Exception:
                    pass

            # Fallback: return first result summary
            return f"Based on our records: {results[0].page_content[:200]}..."

        except Exception as e:
            print(f"⚠️ RAG query failed: {e}")
            return None

    def _get_scenario_from_alert(self, alert_type: str) -> AlertScenario:
        """Map alert type string to AlertScenario enum."""
        mapping = {
            "brake_fade": AlertScenario.BRAKE_FADE,
            "brake_failure": AlertScenario.BRAKE_FADE,
            "battery_critical": AlertScenario.BATTERY_CRITICAL,
            "battery_degradation": AlertScenario.BATTERY_CRITICAL,
            "motor_overtemp": AlertScenario.MOTOR_OVERTEMP,
            "motor_overheat": AlertScenario.MOTOR_OVERTEMP,
            "coolant_low": AlertScenario.COOLANT_LOW,
            "coolant_leak": AlertScenario.COOLANT_LOW,
            "tire_pressure": AlertScenario.TIRE_PRESSURE,
            "tire_low": AlertScenario.TIRE_PRESSURE,
            "inverter_fault": AlertScenario.INVERTER_FAULT,
        }
        return mapping.get(alert_type.lower(), AlertScenario.GENERAL_WARNING)

    def generate_health_coaching(
        self, vehicle_id: str, category: str = "battery_optimization"
    ) -> Dict[str, Any]:
        """
        Generate proactive health coaching message.

        Args:
            vehicle_id: Vehicle identifier
            category: Coaching category (battery_optimization, driving_efficiency, tire_care)

        Returns:
            Coaching message and metadata
        """
        import random
        from datetime import datetime

        # Get seasonal context
        month = datetime.now().month
        season = (
            "winter"
            if month in [12, 1, 2]
            else "summer" if month in [6, 7, 8] else None
        )

        # Get coaching tip
        if category == "seasonal_prep" and season:
            tip = get_health_coaching_tip("seasonal_prep", season)
        else:
            tip = get_health_coaching_tip(category)

        if not tip:
            tip = "Keep your vehicle well-maintained for optimal performance."

        return {
            "vehicle_id": vehicle_id,
            "category": category,
            "message": tip,
            "generated_at": datetime.utcnow().isoformat(),
            "is_proactive": True,
        }

    def initiate_call(
        self,
        vehicle_id: str,
        alert_type: str,
        alert_data: Dict[str, Any],
        owner_name: str = "Alex",
    ) -> Dict[str, Any]:
        """
        Initiate a voice call for a critical alert.
        Phase 10: Now includes scenario and persona selection.

        Args:
            vehicle_id: Vehicle identifier
            alert_type: Type of alert (brake_fade, battery_critical, etc.)
            alert_data: Alert details
            owner_name: Owner's name for personalization

        Returns:
            Call initiation response with call_id and initial message
        """
        call_id = f"call-{vehicle_id}-{uuid.uuid4().hex[:8]}"

        # Phase 10: Determine scenario and persona
        scenario = self._get_scenario_from_alert(alert_type)
        scenario_script = get_scenario_script(scenario)
        severity = alert_data.get("severity", scenario_script.get("urgency", "medium"))
        persona = get_persona_for_scenario(scenario, severity)

        # Build context from scenario script and alert data
        context = {
            "alert_type": alert_type,
            "alert_data": alert_data,
            "owner_name": owner_name,
            "brake_efficiency": alert_data.get("brake_efficiency", 15),
            "health": alert_data.get("battery_health", 50),
            "temperature": alert_data.get("motor_temp", 100),
            "level": alert_data.get("coolant_level", 30),
            "pressure": alert_data.get("tire_pressure", 28),
            "recommended": alert_data.get("recommended_pressure", 35),
            "tire_position": alert_data.get("tire_position", "front left"),
            "efficiency": alert_data.get("brake_efficiency", 15),
            "issue_description": alert_data.get("description", "a potential issue"),
            "center_name": alert_data.get("center_name", "Downtown EV Hub"),
            "slot_time": alert_data.get("slot_time", "2:30 PM"),
            "cost": scenario_script.get("typical_cost", "$150 - $400"),
            "component": scenario_script.get("component", "general"),
        }

        # Create call session with Phase 10 enhancements
        call = VoiceCall(
            call_id=call_id,
            vehicle_id=vehicle_id,
            state=CallState.RINGING,
            started_at=datetime.utcnow().isoformat(),
            context=context,
            scenario=scenario,
            persona=persona,
        )

        self.active_calls[call_id] = call

        return {
            "success": True,
            "call_id": call_id,
            "state": call.state.value,
            "scenario": scenario.value,
            "persona": persona.name,
            "severity": severity,
            "message": "Call initiated. Waiting for user to answer...",
            "ring_audio_url": "/audio/ring.mp3",
        }

    def answer_call(self, call_id: str) -> Dict[str, Any]:
        """
        User answers the call - start the conversation.

        Returns:
            Initial greeting message for TTS
        """
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}

        call = self.active_calls[call_id]
        call.state = CallState.CONNECTED
        call.stage = ConversationStage.GREETING

        # Get greeting message
        script = BRAKE_FADE_SCRIPTS[ConversationStage.GREETING]
        message = script["ai"].format(**call.context)

        # Add to transcript
        call.transcript.append(
            {
                "speaker": "ai",
                "text": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Generate TTS if available
        audio_base64 = None
        if GTTS_AVAILABLE:
            audio_base64 = self._generate_tts(message)

        return {
            "success": True,
            "call_id": call_id,
            "state": call.state.value,
            "stage": call.stage.value,
            "message": message,
            "audio_base64": audio_base64,
            "expected_intents": script["expected_intents"],
            "use_web_speech_api": True,  # Signal frontend to use Web Speech API
        }

    def process_user_input(
        self, call_id: str, user_text: str, detected_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user's voice input and generate response.
        Phase 10: Now includes emotion detection, RAG Q&A, and adaptive responses.

        Args:
            call_id: Active call ID
            user_text: Transcribed user speech
            detected_intent: Optional detected intent (yes, no, question, etc.)

        Returns:
            AI response for TTS
        """
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}

        call = self.active_calls[call_id]

        # Phase 10: Detect and track emotion
        emotion = self._detect_and_track_emotion(call, user_text)

        # Add user input to transcript
        call.transcript.append(
            {
                "speaker": "user",
                "text": user_text,
                "timestamp": datetime.utcnow().isoformat(),
                "emotion": emotion.value,  # Phase 10: Track emotion
            }
        )

        # Detect intent if not provided
        if not detected_intent:
            detected_intent = self._detect_intent(user_text)

        # Phase 10: Handle questions with RAG
        if detected_intent == "question":
            rag_answer = self._answer_question_with_rag(call, user_text)
            if rag_answer:
                # Adapt answer for emotion
                rag_answer = self._adapt_message_for_emotion(rag_answer, emotion)
                response = {
                    "message": rag_answer,
                    "expected_intents": ["yes", "no", "question", "acknowledge"],
                    "rag_used": True,
                }
            else:
                # Fall back to conversation progression
                response = self._progress_conversation(call, detected_intent, user_text)
        else:
            # Progress conversation based on stage and intent
            response = self._progress_conversation(call, detected_intent, user_text)

        # Phase 10: Adapt message for emotion
        response["message"] = self._adapt_message_for_emotion(
            response["message"], emotion
        )

        # Add AI response to transcript
        call.transcript.append(
            {
                "speaker": "ai",
                "text": response["message"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Generate TTS
        audio_base64 = None
        if GTTS_AVAILABLE:
            audio_base64 = self._generate_tts(response["message"])

        return {
            "success": True,
            "call_id": call_id,
            "state": call.state.value,
            "stage": call.stage.value,
            "message": response["message"],
            "audio_base64": audio_base64,
            "action": response.get("action"),
            "booking": response.get("booking"),
            "expected_intents": response.get("expected_intents", []),
            "call_ended": call.state == CallState.ENDED,
            "emotion_detected": emotion.value,  # Phase 10: Return emotion
            "rag_used": response.get("rag_used", False),
        }

    def _detect_intent(self, text: str) -> str:
        """Simple intent detection from user text."""
        text_lower = text.lower()

        # Callback/later intents (check BEFORE 'no' to handle frustrated users)
        if any(
            phrase in text_lower
            for phrase in [
                "call me later",
                "call back later",
                "call later",
                "busy",
                "not now",
                "not a good time",
                "can't talk",
                "in a hurry",
                "i'm driving",
                "later",
            ]
        ):
            return "callback_later"

        # Positive intents
        if any(
            word in text_lower
            for word in [
                "yes",
                "yeah",
                "sure",
                "okay",
                "ok",
                "please",
                "confirm",
                "go ahead",
            ]
        ):
            return "yes"

        # Negative intents (refusal of service, not timing)
        if any(word in text_lower for word in ["no", "nope", "don't want", "cancel"]):
            return "no"

        # Questions
        if any(
            word in text_lower
            for word in [
                "what",
                "why",
                "how",
                "when",
                "where",
                "can i",
                "could",
                "is this urgent",
            ]
        ):
            return "question"

        # Alternative requests
        if any(
            word in text_lower
            for word in [
                "different",
                "another",
                "other",
                "morning",
                "afternoon",
                "tomorrow",
            ]
        ):
            return "alternative"

        # Panic/concern
        if any(
            word in text_lower
            for word in ["oh no", "help", "scared", "worried", "dangerous"]
        ):
            return "panic"

        return "acknowledge"

    def _progress_conversation(
        self, call: VoiceCall, intent: str, user_text: str
    ) -> Dict[str, Any]:
        """Progress the conversation based on current stage and intent."""

        current_stage = call.stage
        context = call.context

        # Handle callback_later intent at ANY stage - user is frustrated/busy
        if intent == "callback_later":
            # Acknowledge frustration but emphasize urgency if critical
            is_critical = (
                context.get("alert_type") in ["brake_fade", "brake_failure"]
                or context.get("brake_efficiency", 100) < 30
            )

            if is_critical:
                return {
                    "message": "I completely understand you're busy, and I apologize for the interruption. "
                    "However, this IS a critical safety issue with your brakes. "
                    "For your safety, I strongly recommend at least pulling over when possible. "
                    "Would you like me to send you a text with the details and call back in 30 minutes?",
                    "expected_intents": ["yes", "no", "text_me"],
                    "callback_offered": True,
                }
            else:
                return {
                    "message": "I understand you're busy. This isn't an emergency, so I can call back later. "
                    "Would you prefer I call back in 30 minutes, or should I send you a text with all the details?",
                    "expected_intents": ["yes", "no", "text_me"],
                    "callback_offered": True,
                }

        # Stage transitions based on intent
        if current_stage == ConversationStage.GREETING:
            call.stage = ConversationStage.ALERT_EXPLANATION
            script = BRAKE_FADE_SCRIPTS[call.stage]
            return {
                "message": script["ai"].format(**context),
                "expected_intents": script["expected_intents"],
            }

        elif current_stage == ConversationStage.ALERT_EXPLANATION:
            if intent == "panic":
                return {
                    "message": "I understand this is concerning, but please stay calm. Your safety is the priority. Let me help you get this resolved quickly.",
                    "expected_intents": ["acknowledge"],
                }
            call.stage = ConversationStage.SAFETY_CHECK
            script = BRAKE_FADE_SCRIPTS[call.stage]
            return {
                "message": script["ai"].format(**context),
                "expected_intents": script["expected_intents"],
            }

        elif current_stage == ConversationStage.SAFETY_CHECK:
            if intent == "no":
                return {
                    "message": "Please find a safe place to stop as soon as possible. Once you're safely pulled over, let me know and we'll arrange immediate assistance.",
                    "expected_intents": ["yes"],
                }
            call.stage = ConversationStage.SCHEDULING_OFFER
            script = BRAKE_FADE_SCRIPTS[call.stage]
            return {
                "message": script["ai"].format(**context),
                "expected_intents": script["expected_intents"],
            }

        elif current_stage == ConversationStage.SCHEDULING_OFFER:
            if intent == "yes":
                call.stage = ConversationStage.SLOT_CONFIRMATION
                script = BRAKE_FADE_SCRIPTS[call.stage]
                return {
                    "message": script["ai"].format(**context),
                    "expected_intents": script["expected_intents"],
                }
            elif intent == "no" or intent == "alternative":
                return {
                    "message": f"I understand. I also have availability at 4:00 PM today or 9:00 AM tomorrow at the same center. Which would work better for you?",
                    "expected_intents": ["yes", "alternative"],
                }
            elif intent == "question":
                if "drive home" in user_text.lower() or "can i" in user_text.lower():
                    return {
                        "message": "No, that is highly unsafe. With only {brake_efficiency}% braking efficiency, you risk complete brake failure. The service center is the safest option.".format(
                            **context
                        ),
                        "expected_intents": ["yes", "acknowledge"],
                    }
                return {
                    "message": "The ceramic brake pads will restore full braking performance. The service takes about 2 hours. Would you like me to book the 2:30 PM slot?",
                    "expected_intents": ["yes", "no"],
                }

        elif current_stage == ConversationStage.SLOT_CONFIRMATION:
            if intent == "yes":
                call.stage = ConversationStage.BOOKING_CONFIRMED
                script = BRAKE_FADE_SCRIPTS[call.stage]
                call.booked_appointment = {
                    "center_name": context["center_name"],
                    "time": context["slot_time"],
                    "component": "brakes",
                    "cost": context["cost"],
                }

                # Save booking to database
                db_appointment = self._save_booking_to_db(call)
                if db_appointment:
                    call.booked_appointment["appointment_id"] = db_appointment.get("id")

                return {
                    "message": script["ai"].format(**context),
                    "expected_intents": script["expected_intents"],
                    "action": "book_appointment",
                    "booking": call.booked_appointment,
                }
            else:
                call.stage = ConversationStage.SCHEDULING_OFFER
                return {
                    "message": "No problem. Would you prefer a different time? I have slots available throughout the day.",
                    "expected_intents": ["yes", "alternative"],
                }

        elif current_stage == ConversationStage.BOOKING_CONFIRMED:
            call.stage = ConversationStage.FAREWELL
            script = BRAKE_FADE_SCRIPTS[call.stage]
            call.state = CallState.ENDED
            call.ended_at = datetime.utcnow().isoformat()
            return {"message": script["ai"].format(**context), "expected_intents": []}

        elif current_stage == ConversationStage.FAREWELL:
            call.state = CallState.ENDED
            call.ended_at = datetime.utcnow().isoformat()
            return {"message": "Goodbye. Stay safe!", "expected_intents": []}

        # Fallback - use LLM for dynamic response
        if self.conversation_chain:
            try:
                response = self.conversation_chain.invoke(
                    {
                        "stage": current_stage.value,
                        "issue": context.get("alert_type", "brake issue"),
                        "owner_name": context.get("owner_name", "there"),
                        "center_name": context.get("center_name", "the service center"),
                        "slot_time": context.get("slot_time", "soon"),
                        "user_input": user_text,
                    }
                )
                return {
                    "message": response,
                    "expected_intents": ["yes", "no", "question"],
                }
            except Exception:
                pass

        return {
            "message": "I understand. Let me help you with that. Would you like me to book the service appointment?",
            "expected_intents": ["yes", "no"],
        }

    def _generate_tts(self, text: str) -> Optional[str]:
        """Generate TTS audio and return as base64."""
        if not GTTS_AVAILABLE:
            return None

        try:
            # Generate speech
            tts = gTTS(text=text, lang="en", slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Encode as base64
            audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
            return audio_base64
        except Exception as e:
            print(f"TTS generation failed: {e}")
            return None

    def _save_booking_to_db(self, call: VoiceCall) -> Optional[Dict[str, Any]]:
        """
        Save the voice call booking to the database.
        This ensures the appointment shows up in Service Center and Scheduler.
        Database handles its own locking now, so this is simpler.
        """
        import time

        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                db = get_database()
                context = call.context
                booking = call.booked_appointment

                # Calculate scheduled date and time
                scheduled_date = datetime.now().strftime("%Y-%m-%d")

                # Get a slot ID (create one if needed)
                slots = db.get_available_slots(center_id="SC-001", date=scheduled_date)
                slot_id = (
                    slots[0]["id"] if slots else f"slot-voice-{uuid.uuid4().hex[:8]}"
                )

                # Create the appointment with critical urgency for brake repairs
                appointment = db.create_appointment(
                    vehicle_id=call.vehicle_id,
                    center_id="SC-001",
                    slot_id=slot_id,
                    component=booking.get("component", "brakes"),
                    diagnosis_summary=f"Emergency brake repair - Voice booking. Brake efficiency at {context.get('brake_efficiency', 15)}%. Customer: {context.get('owner_name', 'Unknown')}",
                    estimated_cost=booking.get("cost", "$150 - $400"),
                    urgency="critical",
                    notes=f"Booked via Voice Agent call {call.call_id}. Alert type: {context.get('alert_type', 'brake_fade')}. [VOICE]",
                )

                if appointment and appointment.get("id"):
                    print(
                        f"✅ Voice booking saved to database: {appointment.get('id')}"
                    )
                    return appointment
                else:
                    print(f"⚠️ Appointment creation returned error: {appointment}")
                    return None

            except Exception as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    print(
                        f"⚠️ Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"⚠️ Failed to save voice booking to database: {e}")
                    return None

        return None

        return None

    def end_call(self, call_id: str) -> Dict[str, Any]:
        """End an active call."""
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}

        call = self.active_calls[call_id]
        call.state = CallState.ENDED
        call.ended_at = datetime.utcnow().isoformat()

        return {
            "success": True,
            "call_id": call_id,
            "state": call.state.value,
            "transcript": call.transcript,
            "booking": call.booked_appointment,
            "duration_seconds": self._calculate_duration(call),
        }

    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """Get current call status."""
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}

        call = self.active_calls[call_id]
        return {
            "success": True,
            "call_id": call_id,
            "vehicle_id": call.vehicle_id,
            "state": call.state.value,
            "stage": call.stage.value,
            "transcript": call.transcript,
            "booking": call.booked_appointment,
        }

    def get_transcript(self, call_id: str) -> Dict[str, Any]:
        """Get full call transcript."""
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}

        call = self.active_calls[call_id]
        return {
            "success": True,
            "call_id": call_id,
            "transcript": call.transcript,
            "total_turns": len(call.transcript),
        }

    def _calculate_duration(self, call: VoiceCall) -> int:
        """Calculate call duration in seconds."""
        if not call.started_at or not call.ended_at:
            return 0

        try:
            start = datetime.fromisoformat(call.started_at)
            end = datetime.fromisoformat(call.ended_at)
            return int((end - start).total_seconds())
        except Exception:
            return 0

    # Vapi.ai compatibility methods
    def get_vapi_config(self) -> Dict[str, Any]:
        """
        Get configuration for Vapi.ai integration.
        This allows seamless upgrade from Web Speech API to Vapi.
        """
        return {
            "provider": "vapi",
            "model": "gpt-4",  # or custom model
            "voice": {"provider": "11labs", "voiceId": "rachel"},
            "firstMessage": BRAKE_FADE_SCRIPTS[ConversationStage.GREETING]["ai"],
            "context": {
                "scenario": "brake_fade_emergency",
                "conversation_flow": list(ConversationStage.__members__.keys()),
            },
            "functions": [
                {
                    "name": "book_appointment",
                    "description": "Book a service appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "center_id": {"type": "string"},
                            "slot_time": {"type": "string"},
                            "component": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "check_inventory",
                    "description": "Check parts availability",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "part_name": {"type": "string"},
                            "center_id": {"type": "string"},
                        },
                    },
                },
            ],
        }


# Singleton instance
_voice_agent: Optional[VoiceAgent] = None


def get_voice_agent() -> VoiceAgent:
    """Get or create VoiceAgent singleton."""
    global _voice_agent
    if _voice_agent is None:
        _voice_agent = VoiceAgent()
    return _voice_agent
