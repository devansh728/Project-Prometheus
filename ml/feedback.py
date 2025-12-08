"""
SentinelEY - Gamified Feedback Engine
Provides real-time, gamified feedback using Gemini API
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(
        "⚠️ LangChain Google GenAI not installed. Run: pip install langchain-google-genai"
    )


@dataclass
class FeedbackResult:
    """Result from feedback generation."""

    score_delta: int  # Points earned/lost
    total_score: int  # Running total
    events: List[Dict[str, Any]]  # List of events detected
    feedback_message: str  # LLM-generated feedback
    badge: Optional[str] = None  # Badge earned
    streak_info: Optional[Dict[str, int]] = None  # Streak information


class FeedbackEngine:
    """
    Gamification and LLM feedback engine.

    Features:
    - Point-based scoring system
    - Badge/achievement system
    - Streak tracking
    - Witty LLM-generated feedback via Gemini
    """

    # Scoring rules
    SCORING_RULES = {
        # Negative events
        "harsh_braking": {
            "points": -10,
            "threshold": {"jerk_ms3": 8.0},
            "condition": "greater",
            "message": "Harsh braking detected",
        },
        "battery_overheating": {
            "points": -20,
            "threshold": {"battery_temp_c": 55},
            "condition": "greater",
            "message": "Battery overheating risk",
        },
        "excessive_speed": {
            "points": -5,
            "threshold": {"speed_kmh": 120},
            "condition": "greater",
            "message": "Excessive speed",
        },
        "low_battery": {
            "points": -15,
            "threshold": {"battery_soc_percent": 10},
            "condition": "less",
            "message": "Critically low battery",
        },
        "high_power_draw": {
            "points": -5,
            "threshold": {"power_draw_kw": 150},
            "condition": "greater",
            "message": "High power consumption",
        },
        # Positive events
        "efficient_regen": {
            "points": 5,
            "threshold": {"regen_efficiency": 0.9},
            "condition": "greater",
            "message": "Excellent regenerative braking",
        },
        "eco_driving": {
            "points": 3,
            "threshold": {"efficiency_wh_km": 120},
            "condition": "less",
            "message": "Eco-efficient driving",
        },
        "smooth_acceleration": {
            "points": 2,
            "threshold": {"jerk_ms3": 2.0},
            "condition": "less",
            "message": "Smooth acceleration",
        },
        "optimal_speed": {
            "points": 2,
            "threshold": {"speed_kmh": 80},
            "condition": "between",
            "range": [60, 90],
            "message": "Optimal highway speed",
        },
    }

    # Badges
    BADGES = {
        "eco_warrior": {
            "name": "🌱 Eco Warrior",
            "condition": "5_efficient_regen_streak",
            "description": "5 consecutive efficient regen braking events",
        },
        "smooth_operator": {
            "name": "🎯 Smooth Operator",
            "condition": "10_smooth_drives",
            "description": "10 drives without harsh braking",
        },
        "range_master": {
            "name": "🔋 Range Master",
            "condition": "efficiency_below_130",
            "description": "Average efficiency below 130 Wh/km",
        },
        "cool_headed": {
            "name": "❄️ Cool Headed",
            "condition": "no_overheating_24h",
            "description": "24 hours without battery overheating",
        },
        "century_club": {
            "name": "💯 Century Club",
            "condition": "score_100",
            "description": "Reach 100 points",
        },
    }

    def __init__(self, driver_name: str = "Driver"):
        """
        Initialize the feedback engine.

        Args:
            driver_name: Name to address the driver
        """
        self.driver_name = driver_name
        self.total_score = 0
        self.event_history: List[Dict[str, Any]] = []
        self.badges_earned: List[str] = []
        self.streaks: Dict[str, int] = {
            "efficient_regen": 0,
            "smooth_drives": 0,
            "no_overheating": 0,
        }

        # Initialize Gemini LLM
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize Gemini LLM."""
        if not GEMINI_AVAILABLE:
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ GOOGLE_API_KEY not set. LLM feedback will use templates.")
            return

        try:
            # Use gemini-pro which is more widely available than gemini-1.5-flash
            # Can be overridden via LLM_MODEL env variable
            model_name = os.getenv("LLM_MODEL", "gemini-pro")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=100,
                convert_system_message_to_human=True,  # For compatibility
            )
            print(f"✅ Gemini LLM initialized with model: {model_name}")
        except Exception as e:
            print(f"⚠️ Could not initialize Gemini: {e}. Using template feedback.")

    def _evaluate_condition(
        self,
        value: float,
        condition: str,
        threshold: float,
        range_values: Optional[List[float]] = None,
    ) -> bool:
        """
        Evaluate if a condition is met.

        Args:
            value: Current value
            condition: Condition type (greater, less, between)
            threshold: Threshold value
            range_values: Range for 'between' condition

        Returns:
            True if condition is met
        """
        if condition == "greater":
            return value > threshold
        elif condition == "less":
            return value < threshold
        elif condition == "between" and range_values:
            return range_values[0] <= value <= range_values[1]
        return False

    def calculate_score(
        self, telemetry: Dict[str, float]
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Calculate driving score based on telemetry.

        Args:
            telemetry: Current telemetry reading

        Returns:
            Tuple of (score_delta, list of events)
        """
        score_delta = 0
        events = []

        for event_name, rule in self.SCORING_RULES.items():
            threshold_key = list(rule["threshold"].keys())[0]
            threshold_value = rule["threshold"][threshold_key]

            if threshold_key not in telemetry:
                continue

            current_value = telemetry[threshold_key]
            condition = rule["condition"]
            range_values = rule.get("range")

            if self._evaluate_condition(
                current_value, condition, threshold_value, range_values
            ):
                points = rule["points"]
                score_delta += points

                event = {
                    "name": event_name,
                    "points": points,
                    "message": rule["message"],
                    "value": current_value,
                    "threshold": threshold_value,
                    "timestamp": datetime.now().isoformat(),
                }
                events.append(event)
                self.event_history.append(event)

                # Update streaks
                self._update_streaks(event_name, points > 0)

        # Update total score
        self.total_score += score_delta

        # Check for new badges
        new_badge = self._check_badges()
        if new_badge:
            events.append(
                {
                    "name": "badge_earned",
                    "badge": new_badge,
                    "message": f"Badge earned: {self.BADGES[new_badge]['name']}",
                }
            )

        return score_delta, events

    def _update_streaks(self, event_name: str, is_positive: bool):
        """Update streak counters."""
        if event_name == "efficient_regen" and is_positive:
            self.streaks["efficient_regen"] += 1
        elif event_name == "harsh_braking":
            self.streaks["smooth_drives"] = 0
        elif event_name == "battery_overheating":
            self.streaks["no_overheating"] = 0

        # Increment positive streaks for good behavior
        if is_positive:
            if event_name == "smooth_acceleration":
                self.streaks["smooth_drives"] += 1

    def _check_badges(self) -> Optional[str]:
        """Check if any new badges should be awarded."""
        for badge_id, badge_info in self.BADGES.items():
            if badge_id in self.badges_earned:
                continue

            condition = badge_info["condition"]
            earned = False

            if condition == "5_efficient_regen_streak":
                earned = self.streaks.get("efficient_regen", 0) >= 5
            elif condition == "10_smooth_drives":
                earned = self.streaks.get("smooth_drives", 0) >= 10
            elif condition == "score_100":
                earned = self.total_score >= 100

            if earned:
                self.badges_earned.append(badge_id)
                return badge_id

        return None

    def generate_feedback(self, events: List[Dict[str, Any]], score_delta: int) -> str:
        """
        Generate witty LLM feedback for driving events.

        Args:
            events: List of events detected
            score_delta: Points earned/lost

        Returns:
            Feedback message string
        """
        if not events:
            return "All systems nominal. Keep up the good driving!"

        # Build event summary
        event_summary = ", ".join([e["message"] for e in events if "message" in e])

        # If LLM is available, generate witty feedback
        if self.llm:
            try:
                prompt = self._build_feedback_prompt(events, score_delta)
                response = self.llm.invoke(prompt)
                return response.content.strip()
            except Exception as e:
                print(f"LLM feedback error: {e}")

        # Fallback to template-based feedback
        return self._template_feedback(events, score_delta)

    def _build_feedback_prompt(
        self, events: List[Dict[str, Any]], score_delta: int
    ) -> str:
        """Build prompt for LLM feedback generation."""
        event_descriptions = []
        for event in events:
            if "message" in event:
                event_descriptions.append(
                    f"- {event['message']} ({event.get('points', 0):+d} points)"
                )

        events_text = "\n".join(event_descriptions)

        tone = "encouraging" if score_delta >= 0 else "concerned but helpful"

        prompt = f"""You are a witty EV driving coach assistant for {self.driver_name}. 
Generate a short, {tone} feedback message (max 25 words) based on these driving events:

{events_text}

Net score change: {score_delta:+d} points
Total score: {self.total_score} points

Be conversational, specific, and if score is negative, give a brief actionable tip.
Do not use emojis in the response."""

        return prompt

    def _template_feedback(self, events: List[Dict[str, Any]], score_delta: int) -> str:
        """Generate template-based feedback when LLM is unavailable."""
        if score_delta > 0:
            templates = [
                f"Great driving, {self.driver_name}! You earned {score_delta} points.",
                f"Nice work! +{score_delta} points for efficient driving.",
                f"Eco-pro mode activated! {score_delta} points earned.",
            ]
        elif score_delta < 0:
            main_event = events[0] if events else {"message": "driving event"}
            templates = [
                f"Watch out! {main_event.get('message', 'Event detected')}. Try smoother inputs.",
                f"Oops! {score_delta} points. {main_event.get('message', 'Check your driving')}.",
                f"Time to adjust! {main_event.get('message', 'Be more careful')}.",
            ]
        else:
            templates = [
                "Steady as she goes. All systems normal.",
                "Cruising along nicely. Keep it up!",
            ]

        import random

        return random.choice(templates)

    def process_telemetry(self, telemetry: Dict[str, float]) -> FeedbackResult:
        """
        Process telemetry and generate complete feedback.

        Args:
            telemetry: Current telemetry reading

        Returns:
            FeedbackResult with score and feedback
        """
        # Calculate score
        score_delta, events = self.calculate_score(telemetry)

        # Generate feedback
        feedback_message = self.generate_feedback(events, score_delta)

        # Check for new badge
        new_badge = None
        for event in events:
            if event.get("name") == "badge_earned":
                new_badge = event.get("badge")

        return FeedbackResult(
            score_delta=score_delta,
            total_score=self.total_score,
            events=events,
            feedback_message=feedback_message,
            badge=new_badge,
            streak_info=self.streaks.copy(),
        )

    def get_score_summary(self) -> Dict[str, Any]:
        """Get summary of current scores and achievements."""
        return {
            "total_score": self.total_score,
            "badges_earned": [
                {
                    "id": b,
                    "name": self.BADGES[b]["name"],
                    "description": self.BADGES[b]["description"],
                }
                for b in self.badges_earned
            ],
            "streaks": self.streaks,
            "recent_events": self.event_history[-10:],
            "total_events": len(self.event_history),
        }

    def reset_session(self):
        """Reset the current driving session."""
        self.total_score = 0
        self.event_history = []
        self.streaks = {k: 0 for k in self.streaks}
        # Keep earned badges


class FleetFeedbackManager:
    """Manages feedback engines for an entire fleet."""

    def __init__(self):
        """Initialize fleet manager."""
        self.engines: Dict[str, FeedbackEngine] = {}

    def get_engine(
        self, vehicle_id: str, driver_name: str = "Driver"
    ) -> FeedbackEngine:
        """
        Get or create a feedback engine for a vehicle.

        Args:
            vehicle_id: Vehicle identifier
            driver_name: Driver's name

        Returns:
            FeedbackEngine instance
        """
        if vehicle_id not in self.engines:
            self.engines[vehicle_id] = FeedbackEngine(driver_name)
        return self.engines[vehicle_id]

    def process_telemetry(
        self, vehicle_id: str, telemetry: Dict[str, float], driver_name: str = "Driver"
    ) -> FeedbackResult:
        """
        Process telemetry for a vehicle.

        Args:
            vehicle_id: Vehicle identifier
            telemetry: Current telemetry
            driver_name: Driver's name

        Returns:
            FeedbackResult
        """
        engine = self.get_engine(vehicle_id, driver_name)
        return engine.process_telemetry(telemetry)

    def get_fleet_leaderboard(self) -> List[Dict[str, Any]]:
        """Get leaderboard sorted by score."""
        leaderboard = []
        for vid, engine in self.engines.items():
            leaderboard.append(
                {
                    "vehicle_id": vid,
                    "driver_name": engine.driver_name,
                    "total_score": engine.total_score,
                    "badges_count": len(engine.badges_earned),
                }
            )

        return sorted(leaderboard, key=lambda x: x["total_score"], reverse=True)


if __name__ == "__main__":
    # Demo: Test feedback engine
    print("🎮 SentinelEY Feedback Engine Demo")
    print("=" * 50)

    engine = FeedbackEngine(driver_name="Demo Driver")

    # Test with various telemetry readings
    test_readings = [
        {
            "speed_kmh": 80,
            "jerk_ms3": 1.5,
            "regen_efficiency": 0.92,
            "battery_temp_c": 35,
            "battery_soc_percent": 75,
            "efficiency_wh_km": 115,
            "power_draw_kw": 45,
        },
        {
            "speed_kmh": 130,
            "jerk_ms3": 10.0,
            "regen_efficiency": 0.55,
            "battery_temp_c": 58,
            "battery_soc_percent": 60,
            "efficiency_wh_km": 220,
            "power_draw_kw": 180,
        },
        {
            "speed_kmh": 75,
            "jerk_ms3": 1.2,
            "regen_efficiency": 0.95,
            "battery_temp_c": 32,
            "battery_soc_percent": 80,
            "efficiency_wh_km": 105,
            "power_draw_kw": 35,
        },
    ]

    for i, reading in enumerate(test_readings, 1):
        print(f"\n📊 Reading {i}:")
        print(
            f"   Speed: {reading['speed_kmh']} km/h | Jerk: {reading['jerk_ms3']} m/s³"
        )
        print(
            f"   Battery: {reading['battery_temp_c']}°C | SoC: {reading['battery_soc_percent']}%"
        )

        result = engine.process_telemetry(reading)

        print(f"\n   🎯 Score Delta: {result.score_delta:+d}")
        print(f"   📊 Total Score: {result.total_score}")
        print(f"   💬 Feedback: {result.feedback_message}")

        if result.events:
            print(f"   📋 Events:")
            for event in result.events:
                if "points" in event:
                    print(f"      - {event['message']}: {event['points']:+d}")

    print("\n" + "=" * 50)
    summary = engine.get_score_summary()
    print(f"📊 Final Summary:")
    print(f"   Total Score: {summary['total_score']}")
    print(f"   Badges: {len(summary['badges_earned'])}")
    print(f"   Total Events: {summary['total_events']}")
