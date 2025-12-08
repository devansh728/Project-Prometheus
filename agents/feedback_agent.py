"""
SentinEV - Feedback Agent
Handles post-service follow-up, customer satisfaction, and maintenance records.
Uses conversational AI for engaging feedback collection.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# LLM imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from db.database import get_database


class FeedbackAgent:
    """
    Feedback Agent for post-service customer engagement.

    Features:
    - Proactive follow-up after service completion
    - Conversational feedback collection
    - Star rating with comments
    - Maintenance history updates
    - Next service recommendations
    """

    def __init__(self):
        self.db = get_database()
        self.llm = None
        self._initialize_llm()

        # Conversation templates
        self.followup_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a friendly customer service representative for SentinEV.
The customer just completed a vehicle service. Your job is to:
1. Thank them warmly
2. Ask about their experience
3. Request a rating (1-5 stars)
4. Encourage them to share feedback

Be warm, professional, and genuinely interested in their experience.
Keep it brief but personal.

Service details:
- Vehicle: {vehicle_id}
- Service: {service_type}
- Component: {component}
- Service Center: {center_name}
- Date: {service_date}""",
                ),
                ("human", "Please follow up with me about my recent service."),
            ]
        )

        self.feedback_response_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are responding to customer feedback. Be empathetic and professional.

If rating is 4-5: Thank them enthusiastically, mention we're glad they're satisfied
If rating is 3: Thank them, acknowledge room for improvement
If rating is 1-2: Apologize sincerely, assure them we'll do better

Always:
- Mention their next recommended service if applicable
- Invite them back
- Keep it warm and genuine""",
                ),
                (
                    "human",
                    """Customer gave {rating} stars.
Comments: {comments}
Service was: {component}
Next service recommended: {next_service}""",
                ),
            ]
        )

        self.satisfaction_prompts = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate follow-up questions based on the rating.

For 5 stars: Ask what made it exceptional, would they recommend us
For 4 stars: Ask what could make it perfect next time
For 3 stars: Ask specifically what could be improved
For 1-2 stars: Express concern, ask what went wrong, offer to make it right

Be conversational and caring.""",
                ),
                ("human", "Customer rated {rating} stars for {component} service."),
            ]
        )

    def _initialize_llm(self):
        """Initialize the LLM for conversation generation."""
        if not LLM_AVAILABLE:
            return

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    google_api_key=api_key,
                    temperature=0.7,
                )
            except Exception as e:
                print(f"Warning: Could not initialize feedback LLM: {e}")

    def initiate_followup(self, appointment_id: str) -> Dict[str, Any]:
        """
        Initiate a follow-up conversation after service completion.

        Args:
            appointment_id: The completed appointment ID

        Returns:
            Dict with follow-up message and prompts
        """
        # Get appointment details
        appointments = self.db.get_appointments()
        appointment = next((a for a in appointments if a["id"] == appointment_id), None)

        if not appointment:
            return {
                "success": False,
                "message": "Could not find the appointment for follow-up.",
            }

        if appointment["status"] != "completed":
            return {
                "success": False,
                "message": "Follow-up can only be initiated for completed services.",
            }

        # Generate personalized follow-up message
        if self.llm:
            try:
                chain = self.followup_template | self.llm | StrOutputParser()
                message = chain.invoke(
                    {
                        "vehicle_id": appointment["vehicle_id"],
                        "service_type": appointment.get(
                            "diagnosis_summary", "Vehicle Service"
                        ),
                        "component": appointment["component"],
                        "center_name": appointment.get("center_name", "Service Center"),
                        "service_date": appointment["scheduled_date"],
                    }
                )
            except Exception:
                message = self._fallback_followup_message(appointment)
        else:
            message = self._fallback_followup_message(appointment)

        return {
            "success": True,
            "message": message,
            "appointment_id": appointment_id,
            "vehicle_id": appointment["vehicle_id"],
            "prompt_type": "initial_followup",
            "quick_actions": [
                {"label": "⭐⭐⭐⭐⭐ Excellent!", "rating": 5},
                {"label": "⭐⭐⭐⭐ Good", "rating": 4},
                {"label": "⭐⭐⭐ Average", "rating": 3},
                {"label": "⭐⭐ Below Average", "rating": 2},
                {"label": "⭐ Poor", "rating": 1},
            ],
            "notification": {
                "type": "feedback_request",
                "title": "How was your service? 🚗",
                "body": f"Please take a moment to rate your {appointment['component']} service.",
            },
        }

    def _fallback_followup_message(self, appointment: Dict) -> str:
        """Generate follow-up message without LLM."""
        return f"""Hi there! 👋

Thank you for choosing SentinEV for your recent **{appointment['component']}** service at {appointment.get('center_name', 'our service center')}.

We hope everything went smoothly! Your feedback helps us improve and serve you better.

**How would you rate your experience?** Please select a rating below:

⭐⭐⭐⭐⭐ - Excellent
⭐⭐⭐⭐ - Good  
⭐⭐⭐ - Average
⭐⭐ - Below Average
⭐ - Poor

Your honest feedback means a lot to us! 🙏"""

    def process_feedback(
        self, appointment_id: str, vehicle_id: str, rating: int, comments: str = ""
    ) -> Dict[str, Any]:
        """
        Process and store customer feedback.

        Args:
            appointment_id: The appointment being rated
            vehicle_id: Vehicle ID
            rating: 1-5 star rating
            comments: Optional feedback comments

        Returns:
            Dict with response message and next steps
        """
        # Validate rating
        if not 1 <= rating <= 5:
            return {
                "success": False,
                "message": "Please provide a rating between 1 and 5 stars.",
            }

        # Store feedback in database
        feedback = self.db.submit_feedback(
            appointment_id=appointment_id,
            vehicle_id=vehicle_id,
            rating=rating,
            comments=comments,
        )

        # Get appointment for context
        appointments = self.db.get_appointments(vehicle_id)
        appointment = next((a for a in appointments if a["id"] == appointment_id), None)
        component = appointment["component"] if appointment else "service"

        # Calculate next service recommendation
        next_service = self._calculate_next_service(component)

        # Generate response based on rating
        if self.llm:
            try:
                chain = self.feedback_response_template | self.llm | StrOutputParser()
                message = chain.invoke(
                    {
                        "rating": rating,
                        "comments": comments if comments else "No additional comments",
                        "component": component,
                        "next_service": next_service["recommendation"],
                    }
                )
            except Exception:
                message = self._fallback_feedback_response(
                    rating, component, next_service
                )
        else:
            message = self._fallback_feedback_response(rating, component, next_service)

        # Update maintenance history
        if appointment:
            self.update_maintenance_log(
                vehicle_id=vehicle_id,
                appointment_id=appointment_id,
                service_type=appointment.get("diagnosis_summary", "Vehicle Service"),
                component=component,
                next_service_due=next_service["due_date"],
            )

        return {
            "success": True,
            "message": message,
            "feedback_id": feedback["id"],
            "rating": rating,
            "next_service": next_service,
            "notification": {
                "type": "feedback_received",
                "title": "Thank You! 🙏",
                "body": f"Your {rating}-star rating has been recorded.",
            },
        }

    def _fallback_feedback_response(
        self, rating: int, component: str, next_service: Dict
    ) -> str:
        """Generate feedback response without LLM."""
        responses = {
            5: f"""🌟 **Wow, thank you so much for the 5-star rating!**

We're thrilled that your {component} service exceeded your expectations! Your satisfaction means everything to us.

Your next recommended service: **{next_service['recommendation']}**
Estimated due date: **{next_service['due_date']}**

We look forward to serving you again! Safe travels! 🚗✨""",
            4: f"""⭐ **Thank you for your 4-star rating!**

We're glad you had a good experience with your {component} service. We're always striving for that perfect 5-star experience!

Your next recommended service: **{next_service['recommendation']}**
Estimated due date: **{next_service['due_date']}**

See you next time! 🚗""",
            3: f"""Thank you for your honest feedback on your {component} service.

We appreciate you letting us know there's room for improvement. We'd love to do better next time!

Is there anything specific we could improve?

Your next recommended service: **{next_service['recommendation']}**
Estimated due date: **{next_service['due_date']}**""",
            2: f"""We're sorry your {component} service didn't meet expectations.

Your feedback is really important, and we want to make this right. Would you like to speak with our customer service team?

We truly value your business and want to improve.

Your next service: **{next_service['recommendation']}** - Due: {next_service['due_date']}""",
            1: f"""We're truly sorry to hear about your experience with the {component} service.

This is not the standard we aim for, and we sincerely apologize. We want to understand what went wrong and make it right.

Please reach out to our customer care team, and we'll prioritize resolving your concerns.

Thank you for giving us the opportunity to improve. 🙏""",
        }

        return responses.get(rating, responses[3])

    def _calculate_next_service(self, component: str) -> Dict[str, str]:
        """Calculate next service recommendation based on component."""
        service_intervals = {
            "brakes": {
                "interval_days": 180,
                "recommendation": "Brake system inspection and pad check",
            },
            "battery": {
                "interval_days": 365,
                "recommendation": "Battery health check and thermal system inspection",
            },
            "motor": {
                "interval_days": 365,
                "recommendation": "Motor and inverter diagnostic check",
            },
            "general": {
                "interval_days": 90,
                "recommendation": "General vehicle inspection",
            },
            "multiple": {
                "interval_days": 90,
                "recommendation": "Comprehensive multi-system check",
            },
        }

        interval = service_intervals.get(component, service_intervals["general"])
        due_date = (
            datetime.now() + timedelta(days=interval["interval_days"])
        ).strftime("%Y-%m-%d")

        return {
            "recommendation": interval["recommendation"],
            "due_date": due_date,
            "interval_days": interval["interval_days"],
        }

    def update_maintenance_log(
        self,
        vehicle_id: str,
        appointment_id: str,
        service_type: str,
        component: str,
        next_service_due: str = None,
        mileage: int = None,
    ) -> Dict[str, Any]:
        """
        Update vehicle maintenance history.

        Args:
            vehicle_id: Vehicle ID
            appointment_id: Related appointment
            service_type: Type of service performed
            component: Component serviced
            next_service_due: When next service is due
            mileage: Current vehicle mileage

        Returns:
            Dict with maintenance record
        """
        # Get appointment for cost info
        appointments = self.db.get_appointments(vehicle_id)
        appointment = next((a for a in appointments if a["id"] == appointment_id), None)
        cost = appointment.get("estimated_cost", "N/A") if appointment else "N/A"

        record = self.db.add_maintenance_record(
            vehicle_id=vehicle_id,
            appointment_id=appointment_id,
            service_type=service_type,
            component=component,
            description=f"Completed {service_type}",
            cost=cost,
            next_service_due=next_service_due,
            mileage=mileage,
        )

        return {
            "success": True,
            "record_id": record["id"],
            "message": f"Maintenance record updated for {vehicle_id}",
        }

    def get_service_history(self, vehicle_id: str) -> Dict[str, Any]:
        """
        Get complete service history for a vehicle.

        Args:
            vehicle_id: Vehicle ID

        Returns:
            Dict with service history and summary
        """
        history = self.db.get_maintenance_history(vehicle_id)
        feedback = self.db.get_feedback(vehicle_id)

        # Calculate summary stats
        total_services = len(history)
        avg_rating = (
            sum(f["rating"] for f in feedback) / len(feedback) if feedback else 0
        )

        # Find upcoming service
        upcoming = None
        for record in history:
            if record.get("next_service_due"):
                due_date = record["next_service_due"]
                if due_date >= datetime.now().strftime("%Y-%m-%d"):
                    upcoming = {
                        "type": record.get("component", "General"),
                        "due_date": due_date,
                    }
                    break

        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "total_services": total_services,
            "average_rating": round(avg_rating, 1),
            "history": history,
            "feedback": feedback,
            "upcoming_service": upcoming,
            "message": f"Found {total_services} service records for {vehicle_id}.",
        }

    def generate_satisfaction_survey(
        self, appointment_id: str, initial_rating: int
    ) -> Dict[str, Any]:
        """
        Generate follow-up survey questions based on initial rating.

        Args:
            appointment_id: The appointment
            initial_rating: The star rating given

        Returns:
            Dict with survey questions
        """
        questions = {
            5: [
                "What made this experience exceptional?",
                "Would you recommend us to friends and family?",
                "Any specific staff member who went above and beyond?",
            ],
            4: [
                "What could we do to make it a 5-star experience?",
                "Was there anything that could have been better?",
                "How was our communication throughout the process?",
            ],
            3: [
                "What aspects of the service could be improved?",
                "Did we meet your expectations for timing and quality?",
                "How was the staff friendliness and professionalism?",
            ],
            2: [
                "We're sorry to hear that. What specifically went wrong?",
                "Was there a communication issue we could address?",
                "What would you like us to do to make this right?",
            ],
            1: [
                "We sincerely apologize. Please tell us what happened.",
                "Would you like us to contact you directly to resolve this?",
                "What could we have done differently?",
            ],
        }

        return {
            "success": True,
            "appointment_id": appointment_id,
            "rating": initial_rating,
            "survey_questions": questions.get(initial_rating, questions[3]),
            "message": "Please help us understand your experience better.",
        }

    def get_pending_feedbacks(self, hours_since_completion: int = 24) -> List[Dict]:
        """
        Get appointments that need feedback follow-up.

        Args:
            hours_since_completion: Hours after completion to request feedback

        Returns:
            List of appointments needing feedback
        """
        # Get completed appointments
        appointments = self.db.get_appointments(status="completed")
        existing_feedback = self.db.get_feedback()
        feedback_apt_ids = {f["appointment_id"] for f in existing_feedback}

        # Filter to those without feedback
        pending = []
        cutoff = (datetime.now() - timedelta(hours=hours_since_completion)).isoformat()

        for appt in appointments:
            if appt["id"] not in feedback_apt_ids:
                if appt.get("completed_at", "") < cutoff:
                    pending.append(appt)

        return pending

    # ==================== Phase 4 Enhancements ====================

    async def generate_personalized_question(
        self, vehicle_id: str, service_type: str
    ) -> Dict[str, Any]:
        """
        Generate personalized follow-up questions based on user profile.

        Uses driving style and past feedback to tailor questions.
        E.g., "Since you drive aggressively, did you notice improved braking response?"
        """
        # Get user profile
        profile = self.db.get_user_profile(vehicle_id)
        driving_style = profile.get("driving_style", "normal") if profile else "normal"
        past_feedback = profile.get("past_feedback", []) if profile else []

        # Find relevant past pain points
        past_pain_points = []
        for fb in past_feedback[-3:]:  # Last 3 feedbacks
            past_pain_points.extend(fb.get("pain_points", []))

        # Style-based question templates
        style_questions = {
            "aggressive": {
                "brakes": "Since you drive with a sporty style, did you notice improved braking response and pedal feel?",
                "battery": "Given your dynamic driving habits, are you satisfied with the battery range improvement?",
                "motor": "With your performance-focused driving, did the motor feel more responsive?",
                "default": "As an enthusiastic driver, did the service meet your performance expectations?",
            },
            "normal": {
                "brakes": "How's the braking smoothness for your daily commute?",
                "battery": "Are you happy with the battery range for your regular trips?",
                "motor": "Is the motor running smoothly for everyday driving?",
                "default": "Did the service meet your expectations for daily use?",
            },
            "conservative": {
                "brakes": "For your careful driving style, do the brakes feel safe and reliable?",
                "battery": "Is the battery performance meeting your efficiency expectations?",
                "motor": "Is the motor running quietly and efficiently?",
                "default": "Did the service enhance your comfortable driving experience?",
            },
        }

        questions = style_questions.get(driving_style, style_questions["normal"])
        primary_question = questions.get(service_type, questions["default"])

        # Add context from past issues
        followup_questions = []
        if past_pain_points:
            if "wait time" in " ".join(past_pain_points).lower():
                followup_questions.append(
                    "We heard you had wait time concerns before - was the timing better this time?"
                )
            if "communication" in " ".join(past_pain_points).lower():
                followup_questions.append(
                    "We've improved our updates - did you feel well-informed throughout?"
                )

        return {
            "success": True,
            "primary_question": primary_question,
            "followup_questions": followup_questions,
            "driving_style": driving_style,
            "personalized": True,
            "context": f"Based on driving style: {driving_style}",
        }

    async def extract_sentiment(
        self, feedback_text: str, rating: int
    ) -> Dict[str, Any]:
        """
        Extract structured sentiment from feedback using LLM.

        Returns:
            - sentiment_score: 0-1 float
            - pain_points: list of issues mentioned
            - positive_points: list of positives mentioned
            - suggestions: list of improvement suggestions
        """
        if not self.llm:
            # Fallback without LLM
            return self._fallback_sentiment(feedback_text, rating)

        prompt = f"""Analyze this customer feedback and extract structured insights.

Feedback: "{feedback_text}"
Rating: {rating}/5 stars

Return a JSON object with:
- sentiment_score: 0.0 to 1.0 (0=very negative, 1=very positive)
- pain_points: list of specific complaints/issues (max 3)
- positive_points: list of specific positives mentioned (max 3)
- suggestions: list of improvement suggestions from the feedback (max 2)
- emotional_tone: one of ["happy", "satisfied", "neutral", "disappointed", "frustrated"]

Return ONLY valid JSON, no markdown."""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a sentiment analysis expert. Return only valid JSON."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            result = json.loads(response.content.strip())
            return {
                "success": True,
                **result,
            }
        except Exception as e:
            return self._fallback_sentiment(feedback_text, rating)

    def _fallback_sentiment(self, text: str, rating: int) -> Dict[str, Any]:
        """Fallback sentiment analysis without LLM."""
        text_lower = text.lower()

        # Simple keyword-based analysis
        negative_keywords = [
            "wait",
            "slow",
            "long",
            "expensive",
            "poor",
            "bad",
            "issue",
            "problem",
        ]
        positive_keywords = [
            "great",
            "excellent",
            "fast",
            "friendly",
            "clean",
            "professional",
            "happy",
        ]

        pain_points = [kw for kw in negative_keywords if kw in text_lower]
        positive_points = [kw for kw in positive_keywords if kw in text_lower]

        # Sentiment from rating
        sentiment_score = rating / 5.0

        return {
            "success": True,
            "sentiment_score": sentiment_score,
            "pain_points": pain_points[:3],
            "positive_points": positive_points[:3],
            "suggestions": [],
            "emotional_tone": self._rating_to_tone(rating),
        }

    def _rating_to_tone(self, rating: int) -> str:
        """Map rating to emotional tone."""
        tones = {
            5: "happy",
            4: "satisfied",
            3: "neutral",
            2: "disappointed",
            1: "frustrated",
        }
        return tones.get(rating, "neutral")

    async def save_feedback_to_profile(
        self,
        vehicle_id: str,
        feedback_text: str,
        rating: int,
        service_type: str,
    ) -> Dict:
        """
        Analyze feedback and save to user profile for future recall.

        Combines extract_sentiment with database storage.
        """
        # Extract sentiment
        sentiment = await self.extract_sentiment(feedback_text, rating)

        if not sentiment.get("success"):
            return {"error": "Failed to analyze sentiment"}

        # Save to profile
        result = self.db.add_feedback_to_profile(
            vehicle_id=vehicle_id,
            sentiment_score=sentiment.get("sentiment_score", rating / 5.0),
            pain_points=sentiment.get("pain_points", []),
            positive_points=sentiment.get("positive_points", []),
            service_type=service_type,
        )

        return {
            **result,
            "sentiment": sentiment,
        }

    def recall_past_feedback(self, vehicle_id: str) -> Dict[str, Any]:
        """
        Recall past feedback for personalized chat interactions.

        Used by chat agent to say things like:
        "Last time you mentioned the wait was long - we've since..."
        """
        profile = self.db.get_user_profile(vehicle_id)

        if not profile:
            return {
                "has_history": False,
                "message": "Welcome! This appears to be your first visit.",
            }

        past_feedback = profile.get("past_feedback", [])
        total_services = profile.get("total_services", 0)
        avg_satisfaction = profile.get("avg_satisfaction", 0)

        if not past_feedback:
            return {
                "has_history": True,
                "total_services": total_services,
                "message": f"Welcome back! You've visited us {total_services} times.",
            }

        # Get most recent feedback
        recent = past_feedback[-1]

        # Build context for chat
        context_items = []

        if recent.get("pain_points"):
            context_items.append(
                f"Last time you mentioned: {', '.join(recent['pain_points'])}"
            )

        if recent.get("positive_points"):
            context_items.append(
                f"You appreciated: {', '.join(recent['positive_points'])}"
            )

        # Check for trends
        if len(past_feedback) >= 2:
            scores = [fb.get("sentiment_score", 0.5) for fb in past_feedback[-3:]]
            if scores[-1] > scores[0]:
                context_items.append("We're glad your satisfaction has been improving!")
            elif scores[-1] < scores[0]:
                context_items.append("We're committed to improving your experience.")

        return {
            "has_history": True,
            "total_services": total_services,
            "avg_satisfaction": round(avg_satisfaction, 2),
            "driving_style": profile.get("driving_style", "normal"),
            "recent_service_type": recent.get("service_type"),
            "context_for_chat": context_items,
            "pain_points_history": list(
                set(pp for fb in past_feedback for pp in fb.get("pain_points", []))
            )[:5],
            "message": f"Welcome back! You've been with us for {total_services} services with {round(avg_satisfaction * 100)}% satisfaction.",
        }

    def get_improvement_insights(self, vehicle_id: str = None) -> Dict:
        """
        Get aggregate insights from all feedback for service improvement.

        If vehicle_id provided, gets individual insights.
        Otherwise, gets system-wide insights.
        """
        if vehicle_id:
            profile = self.db.get_user_profile(vehicle_id)
            feedback_list = profile.get("past_feedback", []) if profile else []
        else:
            # Get all profiles for system-wide analysis
            # For now, return placeholder
            feedback_list = []

        if not feedback_list:
            return {"message": "Not enough feedback for insights"}

        # Aggregate pain points
        all_pain_points = []
        all_positive_points = []

        for fb in feedback_list:
            all_pain_points.extend(fb.get("pain_points", []))
            all_positive_points.extend(fb.get("positive_points", []))

        # Count frequencies
        from collections import Counter

        pain_counts = Counter(all_pain_points)
        positive_counts = Counter(all_positive_points)

        return {
            "success": True,
            "top_pain_points": pain_counts.most_common(5),
            "top_positives": positive_counts.most_common(5),
            "total_feedbacks_analyzed": len(feedback_list),
            "improvement_suggestions": [
                f"Address recurring issue: {issue}"
                for issue, _ in pain_counts.most_common(3)
            ],
        }


# Singleton instance
_feedback_agent: Optional[FeedbackAgent] = None


def get_feedback_agent() -> FeedbackAgent:
    """Get or create FeedbackAgent singleton."""
    global _feedback_agent
    if _feedback_agent is None:
        _feedback_agent = FeedbackAgent()
    return _feedback_agent
