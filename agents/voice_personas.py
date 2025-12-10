"""
SentinEV - Voice Personas and Speech Configuration
Centralized persona and script management for voice-first customer engagement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class EmotionState(Enum):
    """Detected emotional states from user speech."""

    CALM = "calm"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    PANICKED = "panicked"


class AlertScenario(Enum):
    """Alert scenarios with specific conversation scripts."""

    BRAKE_FADE = "brake_fade"
    BATTERY_CRITICAL = "battery_critical"
    MOTOR_OVERTEMP = "motor_overtemp"
    COOLANT_LOW = "coolant_low"
    TIRE_PRESSURE = "tire_pressure"
    INVERTER_FAULT = "inverter_fault"
    GENERAL_WARNING = "general_warning"


@dataclass
class VoicePersona:
    """Voice persona configuration for conversation style."""

    name: str
    tone: str  # professional, friendly, urgent, calm
    vocabulary_level: str  # simple, moderate, technical
    urgency_style: str  # direct, reassuring, measured
    speech_rate: str  # slow, normal, fast
    empathy_level: str  # low, moderate, high

    def get_prompt_modifier(self) -> str:
        """Return LLM prompt modifier for this persona."""
        return f"""Speak in a {self.tone} tone. Use {self.vocabulary_level} vocabulary.
Be {self.urgency_style} about urgency. Speak at a {self.speech_rate} pace.
Show {self.empathy_level} empathy in responses."""


# Pre-defined personas
PERSONAS = {
    "professional": VoicePersona(
        name="Professional",
        tone="professional and courteous",
        vocabulary_level="moderate",
        urgency_style="measured and clear",
        speech_rate="normal",
        empathy_level="moderate",
    ),
    "friendly": VoicePersona(
        name="Friendly",
        tone="warm and friendly",
        vocabulary_level="simple",
        urgency_style="reassuring",
        speech_rate="normal",
        empathy_level="high",
    ),
    "emergency": VoicePersona(
        name="Emergency",
        tone="calm but urgent",
        vocabulary_level="simple",
        urgency_style="direct and action-oriented",
        speech_rate="slow",
        empathy_level="high",
    ),
    "technical": VoicePersona(
        name="Technical",
        tone="professional and precise",
        vocabulary_level="technical",
        urgency_style="fact-based",
        speech_rate="normal",
        empathy_level="moderate",
    ),
}


# Emotion detection keywords and patterns
EMOTION_PATTERNS = {
    EmotionState.PANICKED: [
        "oh my god",
        "help",
        "scared",
        "terrified",
        "emergency",
        "dangerous",
        "crash",
        "accident",
        "die",
        "dying",
        "can't stop",
        "brake not working",
        "out of control",
    ],
    EmotionState.ANXIOUS: [
        "worried",
        "nervous",
        "concerned",
        "not sure",
        "afraid",
        "what if",
        "is it safe",
        "should i",
        "can i drive",
    ],
    EmotionState.FRUSTRATED: [
        "again",
        "always",
        "keep happening",
        "tired of",
        "annoyed",
        "ridiculous",
        "waste of time",
        "not again",
        "seriously",
        "busy",
        "not now",
        "in a hurry",
        "can't talk",
        "call me later",
        "later",
        "another time",
        "not a good time",
        "i'm driving",
    ],
    EmotionState.CONFUSED: [
        "don't understand",
        "what does",
        "what is",
        "meaning",
        "explain",
        "confused",
        "not clear",
        "why",
        "how come",
    ],
}


# Emotion-based response modifiers
EMOTION_RESPONSE_MODIFIERS = {
    EmotionState.PANICKED: {
        "prefix": "I understand this feels urgent. Let me help you step by step. ",
        "tone": "extra calm and reassuring",
        "speech_rate": "slow",
        "simplify": True,
        "break_into_steps": True,
    },
    EmotionState.ANXIOUS: {
        "prefix": "I want to assure you that we're going to take care of this. ",
        "tone": "reassuring and confident",
        "speech_rate": "normal",
        "simplify": False,
        "provide_timeline": True,
    },
    EmotionState.FRUSTRATED: {
        "prefix": "I completely understand your frustration. ",
        "tone": "empathetic and solution-focused",
        "speech_rate": "normal",
        "simplify": False,
        "offer_alternatives": True,
    },
    EmotionState.CONFUSED: {
        "prefix": "Let me explain that more clearly. ",
        "tone": "patient and educational",
        "speech_rate": "slow",
        "simplify": True,
        "use_analogies": True,
    },
    EmotionState.CALM: {
        "prefix": "",
        "tone": "professional",
        "speech_rate": "normal",
        "simplify": False,
    },
}


# Scenario-specific conversation scripts
SCENARIO_SCRIPTS = {
    AlertScenario.BRAKE_FADE: {
        "greeting": "Hi {owner_name}, this is SentinEV with an urgent safety alert about your vehicle's braking system.",
        "explanation": "Our sensors detected that your brake fluid has overheated. You currently have approximately {efficiency}% braking efficiency. This requires immediate attention.",
        "safety_advice": "For your immediate safety, please reduce speed and use engine braking by downshifting. Find a safe place to stop if possible.",
        "scheduling": "I've located {center_name} nearby with the specific brake components in stock. They have a bay available at {slot_time}. Shall I book this for you?",
        "urgency": "critical",
        "component": "brakes",
        "typical_cost": "$150 - $400",
        "typical_duration": "2-3 hours",
    },
    AlertScenario.BATTERY_CRITICAL: {
        "greeting": "Hi {owner_name}, this is SentinEV. I'm calling about an important battery alert for your vehicle.",
        "explanation": "Your battery health has dropped to {health}% and we're detecting cell imbalance. This could affect your range and charging capability.",
        "safety_advice": "Your vehicle is safe to drive, but I recommend limiting trips to essential journeys until the battery is serviced.",
        "scheduling": "I can schedule a battery diagnostic at {center_name} for {slot_time}. This will help us determine if cell balancing or replacement is needed.",
        "urgency": "high",
        "component": "battery",
        "typical_cost": "$100 - $2000",
        "typical_duration": "1-4 hours",
    },
    AlertScenario.MOTOR_OVERTEMP: {
        "greeting": "Hi {owner_name}, this is SentinEV regarding a motor temperature warning.",
        "explanation": "Your electric motor temperature has exceeded safe limits at {temperature}°C. This could indicate a cooling system issue or excessive load.",
        "safety_advice": "Please reduce speed and avoid heavy acceleration. If possible, pull over and let the motor cool for 10-15 minutes.",
        "scheduling": "I recommend a motor and cooling system inspection at {center_name}. The earliest slot is at {slot_time}.",
        "urgency": "high",
        "component": "motor",
        "typical_cost": "$200 - $800",
        "typical_duration": "2-3 hours",
    },
    AlertScenario.COOLANT_LOW: {
        "greeting": "Hi {owner_name}, this is SentinEV with a cooling system notification.",
        "explanation": "Your vehicle's coolant level is low at {level}%. This affects the thermal management of your battery and motor.",
        "safety_advice": "You can continue driving for short distances, but I recommend avoiding highway speeds until topped up.",
        "scheduling": "A quick coolant top-up and inspection is available at {center_name} for {slot_time}. Would you like me to book it?",
        "urgency": "medium",
        "component": "cooling_system",
        "typical_cost": "$50 - $150",
        "typical_duration": "30-60 minutes",
    },
    AlertScenario.TIRE_PRESSURE: {
        "greeting": "Hi {owner_name}, this is SentinEV with a tire pressure alert.",
        "explanation": "We've detected that your {tire_position} tire pressure is at {pressure} PSI, which is below the recommended {recommended} PSI.",
        "safety_advice": "Low tire pressure can affect handling and increase energy consumption. Please inflate at the nearest opportunity.",
        "scheduling": "If you'd like, I can schedule a tire inspection at {center_name} for {slot_time}.",
        "urgency": "low",
        "component": "tires",
        "typical_cost": "$0 - $100",
        "typical_duration": "15-30 minutes",
    },
    AlertScenario.INVERTER_FAULT: {
        "greeting": "Hi {owner_name}, this is SentinEV with an important inverter system alert.",
        "explanation": "Your vehicle's power inverter is showing fault codes. This component converts battery power for the motor.",
        "safety_advice": "While your vehicle may still operate, power delivery may be limited. I recommend going directly to a service center.",
        "scheduling": "Inverter diagnostics require specialized equipment. {center_name} is equipped for this and has availability at {slot_time}.",
        "urgency": "high",
        "component": "inverter",
        "typical_cost": "$300 - $1500",
        "typical_duration": "3-5 hours",
    },
    AlertScenario.GENERAL_WARNING: {
        "greeting": "Hi {owner_name}, this is SentinEV with a maintenance notification.",
        "explanation": "Our monitoring system has detected {issue_description}. This is worth addressing to maintain optimal performance.",
        "safety_advice": "Your vehicle is safe to drive, but scheduling service soon is recommended.",
        "scheduling": "I can help you book an appointment at {center_name} for {slot_time}. Would that work for you?",
        "urgency": "low",
        "component": "general",
        "typical_cost": "Varies",
        "typical_duration": "1-2 hours",
    },
}


# Proactive health coaching templates
HEALTH_COACHING_TEMPLATES = {
    "battery_optimization": [
        "Your battery performs best when kept between 20% and 80% charge for daily use.",
        "Avoid leaving your vehicle at 100% charge for extended periods to preserve battery life.",
        "Preconditioning your battery before fast charging improves charging speed and battery health.",
    ],
    "driving_efficiency": [
        "Smooth acceleration and regenerative braking can extend your range by up to 20%.",
        "Use eco mode for city driving to maximize efficiency.",
        "High speeds above 70 mph significantly reduce range due to aerodynamic drag.",
    ],
    "tire_care": [
        "Check your tire pressure monthly - EV tires experience higher loads than traditional vehicles.",
        "Proper tire pressure improves range and handling.",
        "Rotate your tires every 5,000-7,500 miles for even wear.",
    ],
    "seasonal_prep": {
        "winter": [
            "Cold weather can reduce range by 20-40%. Precondition while plugged in.",
            "Use seat heaters instead of cabin heat to save energy.",
        ],
        "summer": [
            "Park in shade when possible to reduce cooling load on the battery.",
            "Precondition the cabin while charging for comfortable entry.",
        ],
    },
}


# Technical-to-customer-friendly translations
TECH_TRANSLATIONS = {
    "brake fade": "temporary reduction in braking power due to heat",
    "cell imbalance": "uneven charge distribution in battery cells",
    "thermal runaway": "dangerous overheating condition",
    "inverter fault": "issue with the power conversion system",
    "SOC": "state of charge (battery level)",
    "SOH": "state of health (battery condition)",
    "regen": "regenerative braking",
    "HV": "high voltage",
    "BMS": "battery management system",
    "DTC": "diagnostic trouble code",
    "TPMS": "tire pressure monitoring system",
}


def detect_emotion(text: str) -> EmotionState:
    """
    Detect emotional state from user text.

    Args:
        text: User's spoken text

    Returns:
        Detected EmotionState
    """
    text_lower = text.lower()

    # Check patterns in order of urgency
    for emotion, patterns in EMOTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return emotion

    return EmotionState.CALM


def get_persona_for_scenario(scenario: AlertScenario, severity: str) -> VoicePersona:
    """
    Select appropriate persona based on scenario and severity.

    Args:
        scenario: The alert scenario
        severity: Severity level (critical, high, medium, low)

    Returns:
        Appropriate VoicePersona
    """
    if severity == "critical":
        return PERSONAS["emergency"]
    elif severity == "high":
        return PERSONAS["professional"]
    elif scenario == AlertScenario.TIRE_PRESSURE:
        return PERSONAS["friendly"]
    else:
        return PERSONAS["professional"]


def translate_technical_term(term: str) -> str:
    """
    Translate technical term to customer-friendly language.

    Args:
        term: Technical term

    Returns:
        Customer-friendly translation or original term
    """
    return TECH_TRANSLATIONS.get(term.lower(), term)


def get_emotion_response_modifier(emotion: EmotionState) -> Dict[str, Any]:
    """
    Get response modifiers for detected emotion.

    Args:
        emotion: Detected emotional state

    Returns:
        Dictionary of response modifiers
    """
    return EMOTION_RESPONSE_MODIFIERS.get(
        emotion, EMOTION_RESPONSE_MODIFIERS[EmotionState.CALM]
    )


def get_scenario_script(scenario: AlertScenario) -> Dict[str, Any]:
    """
    Get conversation script for a scenario.

    Args:
        scenario: Alert scenario

    Returns:
        Script dictionary
    """
    return SCENARIO_SCRIPTS.get(
        scenario, SCENARIO_SCRIPTS[AlertScenario.GENERAL_WARNING]
    )


def get_health_coaching_tip(category: str, season: Optional[str] = None) -> str:
    """
    Get a health coaching tip for proactive messaging.

    Args:
        category: Coaching category
        season: Optional season for seasonal tips

    Returns:
        Coaching tip string
    """
    import random

    if category == "seasonal_prep" and season:
        tips = HEALTH_COACHING_TEMPLATES.get("seasonal_prep", {}).get(season, [])
    else:
        tips = HEALTH_COACHING_TEMPLATES.get(category, [])

    return random.choice(tips) if tips else ""
