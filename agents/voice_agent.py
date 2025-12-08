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

# Import database
from db.database import Database

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
        """Initialize the Voice Agent."""
        self.active_calls: Dict[str, VoiceCall] = {}
        self.llm = None
        self.conversation_chain = None
        self.db = Database()
        self._initialize_llm()

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

    def initiate_call(
        self,
        vehicle_id: str,
        alert_type: str,
        alert_data: Dict[str, Any],
        owner_name: str = "Alex",
    ) -> Dict[str, Any]:
        """
        Initiate a voice call for a critical alert.

        Args:
            vehicle_id: Vehicle identifier
            alert_type: Type of alert (brake_fade, battery_critical, etc.)
            alert_data: Alert details
            owner_name: Owner's name for personalization

        Returns:
            Call initiation response with call_id and initial message
        """
        call_id = f"call-{vehicle_id}-{uuid.uuid4().hex[:8]}"

        # Create call session
        call = VoiceCall(
            call_id=call_id,
            vehicle_id=vehicle_id,
            state=CallState.RINGING,
            started_at=datetime.utcnow().isoformat(),
            context={
                "alert_type": alert_type,
                "alert_data": alert_data,
                "owner_name": owner_name,
                "brake_efficiency": alert_data.get("brake_efficiency", 15),
                "center_name": "Downtown EV Hub",
                "slot_time": "2:30 PM",
                "cost": "$150 - $400",
            },
        )

        self.active_calls[call_id] = call

        return {
            "success": True,
            "call_id": call_id,
            "state": call.state.value,
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

        # Add user input to transcript
        call.transcript.append(
            {
                "speaker": "user",
                "text": user_text,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Detect intent if not provided
        if not detected_intent:
            detected_intent = self._detect_intent(user_text)

        # Progress conversation based on stage and intent
        response = self._progress_conversation(call, detected_intent, user_text)

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
        }

    def _detect_intent(self, text: str) -> str:
        """Simple intent detection from user text."""
        text_lower = text.lower()

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

        # Negative intents
        if any(
            word in text_lower
            for word in ["no", "nope", "don't", "can't", "later", "not now"]
        ):
            return "no"

        # Questions
        if any(
            word in text_lower
            for word in ["what", "why", "how", "when", "where", "can i", "could"]
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
        Includes retry logic for handling database locking.
        """
        import time

        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                context = call.context
                booking = call.booked_appointment

                # Calculate scheduled date and time
                scheduled_date = datetime.now().strftime("%Y-%m-%d")
                scheduled_time = booking.get("time", "2:30 PM")

                # Get a slot ID (create one if needed)
                slots = self.db.get_available_slots(
                    center_id="SC-001", date=scheduled_date  # Use correct center ID
                )
                slot_id = (
                    slots[0]["id"] if slots else f"slot-voice-{uuid.uuid4().hex[:8]}"
                )

                # Create the appointment with critical urgency for brake repairs
                appointment = self.db.create_appointment(
                    vehicle_id=call.vehicle_id,
                    center_id="SC-001",  # Use correct center ID
                    slot_id=slot_id,
                    component=booking.get("component", "brakes"),
                    diagnosis_summary=f"Emergency brake repair - Voice booking. Brake efficiency at {context.get('brake_efficiency', 15)}%. Customer: {context.get('owner_name', 'Unknown')}",
                    estimated_cost=booking.get("cost", "$150 - $400"),
                    urgency="critical",  # Voice bookings are typically critical
                    notes=f"Booked via Voice Agent call {call.call_id}. Alert type: {context.get('alert_type', 'brake_fade')}",
                )

                # Mark as voice-booked in database
                if appointment and appointment.get("id"):
                    try:
                        conn = self.db._get_connection()
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE appointments SET booked_via = 'voice', stage = 'INTAKE' WHERE id = ?",
                            (appointment["id"],),
                        )
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print(f"Warning: Could not update booked_via: {e}")

                print(f"✅ Voice booking saved to database: {appointment.get('id')}")
                return appointment

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
