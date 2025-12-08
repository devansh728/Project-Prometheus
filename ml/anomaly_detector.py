"""
SentinEV - Advanced Anomaly Detection Pipeline
Personalized per-vehicle anomaly detection with ML models and scoring
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import joblib

# HuggingFace for text generation
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not installed. Run: pip install transformers")

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "⚠️ Sentence Transformers not installed. Run: pip install sentence-transformers"
    )


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""

    is_anomaly: bool
    anomaly_score: float  # -1 (most anomalous) to 1 (most normal)
    anomaly_type: str
    severity: str  # low, medium, high, critical
    confidence: float
    failure_risk_pct: float
    time_to_failure_hours: Optional[float]
    affected_components: List[str]
    contributing_factors: List[Dict[str, Any]]


@dataclass
class ScoringResult:
    """Result from positive/negative scoring."""

    score_delta: int
    total_score: int
    events: List[Dict[str, Any]]
    feedback_text: str
    badges_earned: List[str]


class AdvancedAnomalyDetector:
    """
    Advanced per-vehicle anomaly detection using personalized ML models.

    Features:
    - Isolation Forest for outlier detection
    - Random Forest for failure prediction
    - DBSCAN for anomaly clustering
    - Personalized thresholds based on driver behavior
    - Time-to-failure estimation
    """

    FEATURE_COLUMNS = [
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "power_draw_kw",
        "regen_efficiency",
        "battery_soc_pct",
        "battery_temp_c",
        "motor_temp_c",
        "inverter_temp_c",
        "brake_temp_c",
        "coolant_temp_c",
        "wear_index",
    ]

    ANOMALY_TYPES = {
        "thermal_battery": ["battery_temp_c", "coolant_temp_c"],
        "thermal_motor": ["motor_temp_c", "inverter_temp_c"],
        "thermal_brake": ["brake_temp_c"],
        "power_anomaly": ["power_draw_kw", "regen_efficiency"],
        "driving_behavior": ["jerk_ms3", "acceleration_ms2", "speed_kmh"],
        "wear_degradation": ["wear_index"],
        "soc_anomaly": ["battery_soc_pct"],
    }

    def __init__(self, vehicle_id: str, contamination: float = 0.05):
        """
        Initialize anomaly detector for a specific vehicle.

        Args:
            vehicle_id: Unique vehicle identifier
            contamination: Expected proportion of outliers
        """
        self.vehicle_id = vehicle_id
        self.contamination = contamination

        # Models
        self.isolation_forest: Optional[IsolationForest] = None
        self.failure_predictor: Optional[RandomForestRegressor] = None
        self.anomaly_classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[RobustScaler] = None
        self.cluster_model: Optional[DBSCAN] = None

        # Learned baselines
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.personalized_thresholds: Dict[str, float] = {}
        self.driver_profile: str = "normal"

        # State
        self.is_trained = False
        self.training_samples = 0

        # Load vehicle manual thresholds
        self._load_thresholds()

    def _load_thresholds(self):
        """Load default thresholds from vehicle manual."""
        manual_path = Path("data/datasets/vehicle_manual.json")
        if manual_path.exists():
            with open(manual_path) as f:
                manual = json.load(f)
                if "thresholds" in manual:
                    self.personalized_thresholds = manual["thresholds"][
                        "anomaly_detection"
                    ]
        else:
            # Default thresholds
            self.personalized_thresholds = {
                "battery_temp_warning_c": 50,
                "battery_temp_critical_c": 60,
                "motor_temp_warning_c": 100,
                "inverter_temp_warning_c": 80,
                "brake_temp_warning_c": 350,
                "jerk_harsh_threshold_ms3": 4.0,
                "regen_efficiency_poor_threshold": 0.5,
                "power_draw_anomaly_factor": 1.5,
            }

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from telemetry data."""
        available_cols = [c for c in self.FEATURE_COLUMNS if c in data.columns]
        return data[available_cols].values

    def _compute_baseline_stats(self, data: pd.DataFrame) -> None:
        """Compute baseline statistics for anomaly detection."""
        for col in self.FEATURE_COLUMNS:
            if col in data.columns:
                self.baseline_stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "q25": float(data[col].quantile(0.25)),
                    "q75": float(data[col].quantile(0.75)),
                    "iqr": float(data[col].quantile(0.75) - data[col].quantile(0.25)),
                }

    def _adjust_thresholds_for_driver(self, driver_profile: str) -> None:
        """Adjust anomaly thresholds based on driver profile."""
        self.driver_profile = driver_profile

        adjustments = {
            "aggressive": {
                "battery_temp_warning_c": 55,  # Higher threshold for aggressive
                "motor_temp_warning_c": 110,
                "jerk_harsh_threshold_ms3": 6.0,
                "power_draw_anomaly_factor": 1.8,
            },
            "eco": {
                "battery_temp_warning_c": 45,  # Lower threshold for eco
                "motor_temp_warning_c": 90,
                "jerk_harsh_threshold_ms3": 3.0,
                "power_draw_anomaly_factor": 1.3,
            },
            "normal": {},  # Use defaults
        }

        if driver_profile in adjustments:
            self.personalized_thresholds.update(adjustments[driver_profile])

    def _create_failure_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create synthetic failure risk labels for supervised learning.
        Based on proximity to critical thresholds.
        """
        risk_scores = np.zeros(len(data))

        # Battery temperature risk
        if "battery_temp_c" in data.columns:
            temp = data["battery_temp_c"].values
            critical = self.personalized_thresholds.get("battery_temp_critical_c", 60)
            warning = self.personalized_thresholds.get("battery_temp_warning_c", 50)
            risk_scores += np.clip((temp - warning) / (critical - warning) * 30, 0, 30)

        # Motor temperature risk
        if "motor_temp_c" in data.columns:
            temp = data["motor_temp_c"].values
            warning = self.personalized_thresholds.get("motor_temp_warning_c", 100)
            risk_scores += np.clip((temp - warning * 0.8) / (warning * 0.2) * 20, 0, 20)

        # Brake temperature risk - critical for brake fade scenarios
        if "brake_temp_c" in data.columns:
            temp = data["brake_temp_c"].values
            warning = self.personalized_thresholds.get("brake_temp_warning_c", 350)
            critical = warning * 1.2  # 420°C critical threshold
            risk_scores += np.clip((temp - warning) / (critical - warning) * 35, 0, 35)

        # Wear index risk
        if "wear_index" in data.columns:
            wear = data["wear_index"].values
            if len(wear) > 0:
                max_wear = max(1, wear.max())
                risk_scores += np.clip(wear / max_wear * 15, 0, 15)

        # Regen efficiency degradation risk
        if "regen_efficiency" in data.columns:
            regen = data["regen_efficiency"].values
            poor_threshold = self.personalized_thresholds.get(
                "regen_efficiency_poor_threshold", 0.5
            )
            risk_scores += np.clip(
                (poor_threshold - regen) / poor_threshold * 20, 0, 20
            )

        # Jerk/harsh driving risk
        if "jerk_ms3" in data.columns:
            jerk = np.abs(data["jerk_ms3"].values)
            threshold = self.personalized_thresholds.get(
                "jerk_harsh_threshold_ms3", 4.0
            )
            risk_scores += np.clip((jerk - threshold) / threshold * 15, 0, 15)

        return np.clip(risk_scores, 0, 100)

    def _label_anomaly_types(self, data: pd.DataFrame) -> np.ndarray:
        """Label anomaly types based on which thresholds are violated."""
        labels = np.zeros(len(data), dtype=int)  # 0 = normal

        for i, row in data.iterrows():
            idx = data.index.get_loc(i)

            # Check each anomaly type
            if row.get("battery_temp_c", 0) > self.personalized_thresholds.get(
                "battery_temp_warning_c", 50
            ):
                labels[idx] = 1  # thermal_battery
            elif row.get("motor_temp_c", 0) > self.personalized_thresholds.get(
                "motor_temp_warning_c", 100
            ):
                labels[idx] = 2  # thermal_motor
            elif row.get("brake_temp_c", 0) > self.personalized_thresholds.get(
                "brake_temp_warning_c", 350
            ):
                labels[idx] = 3  # thermal_brake
            elif abs(row.get("jerk_ms3", 0)) > self.personalized_thresholds.get(
                "jerk_harsh_threshold_ms3", 4.0
            ):
                labels[idx] = 4  # driving_behavior
            elif row.get("regen_efficiency", 1) < self.personalized_thresholds.get(
                "regen_efficiency_poor_threshold", 0.5
            ):
                labels[idx] = 5  # power_anomaly

        return labels

    def train(
        self, historical_data: pd.DataFrame, driver_profile: str = "normal"
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection models on historical data.

        Args:
            historical_data: DataFrame with telemetry history
            driver_profile: Driver behavior profile

        Returns:
            Training metrics
        """
        # Adjust thresholds for driver
        self._adjust_thresholds_for_driver(driver_profile)

        # Compute baseline statistics
        self._compute_baseline_stats(historical_data)

        # Extract features
        X = self._extract_features(historical_data)

        if len(X) < 10:
            return {"error": "Insufficient training data"}

        # Fit scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest for anomaly detection
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)

        # Create labels for supervised learning
        failure_risks = self._create_failure_labels(historical_data)
        anomaly_types = self._label_anomaly_types(historical_data)

        # Train failure risk predictor
        self.failure_predictor = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.failure_predictor.fit(X_scaled, failure_risks)

        # Train anomaly type classifier
        if len(np.unique(anomaly_types)) > 1:
            self.anomaly_classifier = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
            )
            self.anomaly_classifier.fit(X_scaled, anomaly_types)

        # Train clustering for anomaly grouping
        self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
        self.cluster_model.fit(X_scaled)

        self.is_trained = True
        self.training_samples = len(historical_data)

        return {
            "vehicle_id": self.vehicle_id,
            "driver_profile": driver_profile,
            "training_samples": self.training_samples,
            "features_used": len(self.FEATURE_COLUMNS),
            "baseline_computed": list(self.baseline_stats.keys()),
            "models_trained": [
                "isolation_forest",
                "failure_predictor",
                "anomaly_classifier",
            ],
        }

    def _classify_severity(self, anomaly_score: float, failure_risk: float) -> str:
        """Determine severity based on scores."""
        if failure_risk > 80 or anomaly_score < -0.5:
            return "critical"
        elif failure_risk > 50 or anomaly_score < -0.3:
            return "high"
        elif failure_risk > 25 or anomaly_score < -0.1:
            return "medium"
        else:
            return "low"

    def _identify_anomaly_type(
        self, telemetry: Dict[str, float]
    ) -> Tuple[str, List[str]]:
        """Identify the type of anomaly and affected components."""
        anomaly_type = "unknown"
        affected = []

        # Check each category
        if telemetry.get("battery_temp_c", 0) > self.personalized_thresholds.get(
            "battery_temp_warning_c", 50
        ):
            anomaly_type = "thermal_battery"
            affected.extend(["battery", "cooling_system"])

        if telemetry.get("motor_temp_c", 0) > self.personalized_thresholds.get(
            "motor_temp_warning_c", 100
        ):
            anomaly_type = "thermal_motor"
            affected.extend(["motor", "inverter"])

        if telemetry.get("brake_temp_c", 0) > self.personalized_thresholds.get(
            "brake_temp_warning_c", 350
        ):
            anomaly_type = "thermal_brake"
            affected.append("brakes")

        if abs(telemetry.get("jerk_ms3", 0)) > self.personalized_thresholds.get(
            "jerk_harsh_threshold_ms3", 4.0
        ):
            anomaly_type = "driving_behavior"
            affected.extend(["drivetrain", "brakes"])

        if telemetry.get("regen_efficiency", 1) < self.personalized_thresholds.get(
            "regen_efficiency_poor_threshold", 0.5
        ):
            anomaly_type = "power_anomaly"
            affected.extend(["motor", "battery"])

        if not affected:
            anomaly_type = "normal"

        return anomaly_type, list(set(affected))

    def _identify_contributing_factors(
        self, telemetry: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify factors contributing to the anomaly."""
        factors = []

        for feature, stats in self.baseline_stats.items():
            if feature in telemetry:
                value = telemetry[feature]
                mean = stats["mean"]
                std = stats["std"]

                if std > 0:
                    z_score = (value - mean) / std
                    if abs(z_score) > 2:
                        factors.append(
                            {
                                "feature": feature,
                                "value": value,
                                "mean": mean,
                                "z_score": round(z_score, 2),
                                "deviation": "high" if z_score > 0 else "low",
                            }
                        )

        return sorted(factors, key=lambda x: abs(x["z_score"]), reverse=True)

    def _estimate_time_to_failure(
        self, failure_risk: float, telemetry: Dict[str, float]
    ) -> Optional[float]:
        """Estimate time to failure based on current state and trends."""
        if failure_risk < 20:
            return None  # No imminent failure

        # Simple estimation based on risk level
        # Higher risk = shorter time to failure
        if failure_risk > 80:
            return 2.0  # 2 hours
        elif failure_risk > 60:
            return 24.0  # 24 hours
        elif failure_risk > 40:
            return 72.0  # 3 days
        else:
            return 168.0  # 1 week

    def predict(self, telemetry: Dict[str, float]) -> AnomalyResult:
        """
        Predict if current telemetry is anomalous.

        Args:
            telemetry: Current telemetry reading

        Returns:
            AnomalyResult with detection details
        """
        if not self.is_trained:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="untrained",
                severity="unknown",
                confidence=0.0,
                failure_risk_pct=0.0,
                time_to_failure_hours=None,
                affected_components=[],
                contributing_factors=[],
            )

        # Prepare feature vector
        features = []
        for col in self.FEATURE_COLUMNS:
            if col in telemetry:
                features.append(telemetry[col])
            else:
                features.append(self.baseline_stats.get(col, {}).get("mean", 0))

        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get anomaly score from Isolation Forest
        anomaly_score = float(self.isolation_forest.decision_function(X_scaled)[0])
        is_anomaly = self.isolation_forest.predict(X_scaled)[0] == -1

        # Get failure risk prediction
        failure_risk = float(self.failure_predictor.predict(X_scaled)[0])
        failure_risk = np.clip(failure_risk, 0, 100)

        # Identify anomaly type
        anomaly_type, affected = self._identify_anomaly_type(telemetry)

        # Determine severity
        severity = self._classify_severity(anomaly_score, failure_risk)

        # Get contributing factors
        factors = self._identify_contributing_factors(telemetry)

        # Estimate time to failure
        ttf = self._estimate_time_to_failure(failure_risk, telemetry)

        # Compute confidence
        confidence = min(0.95, 0.5 + (self.training_samples / 2000))

        return AnomalyResult(
            is_anomaly=is_anomaly or severity in ["high", "critical"],
            anomaly_score=round(anomaly_score, 4),
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=round(confidence, 3),
            failure_risk_pct=round(failure_risk, 1),
            time_to_failure_hours=ttf,
            affected_components=affected,
            contributing_factors=factors,
        )

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save sklearn models
        if self.isolation_forest:
            joblib.dump(self.isolation_forest, model_dir / "isolation_forest.joblib")
        if self.failure_predictor:
            joblib.dump(self.failure_predictor, model_dir / "failure_predictor.joblib")
        if self.anomaly_classifier:
            joblib.dump(
                self.anomaly_classifier, model_dir / "anomaly_classifier.joblib"
            )
        if self.scaler:
            joblib.dump(self.scaler, model_dir / "scaler.joblib")

        # Save metadata
        metadata = {
            "vehicle_id": self.vehicle_id,
            "driver_profile": self.driver_profile,
            "contamination": self.contamination,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "baseline_stats": self.baseline_stats,
            "personalized_thresholds": self.personalized_thresholds,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        model_dir = Path(path)

        if (model_dir / "isolation_forest.joblib").exists():
            self.isolation_forest = joblib.load(model_dir / "isolation_forest.joblib")
        if (model_dir / "failure_predictor.joblib").exists():
            self.failure_predictor = joblib.load(model_dir / "failure_predictor.joblib")
        if (model_dir / "anomaly_classifier.joblib").exists():
            self.anomaly_classifier = joblib.load(
                model_dir / "anomaly_classifier.joblib"
            )
        if (model_dir / "scaler.joblib").exists():
            self.scaler = joblib.load(model_dir / "scaler.joblib")

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
            self.vehicle_id = metadata["vehicle_id"]
            self.driver_profile = metadata["driver_profile"]
            self.is_trained = metadata["is_trained"]
            self.training_samples = metadata["training_samples"]
            self.baseline_stats = metadata["baseline_stats"]
            self.personalized_thresholds = metadata["personalized_thresholds"]


class ScoringEngine:
    """
    Gamified scoring engine with positive/negative points.
    Uses HuggingFace for text feedback generation.
    """

    SCORING_RULES = {
        # Negative events
        "harsh_braking": {"points": -15, "threshold": ("jerk_ms3", "less", -4.0)},
        "harsh_acceleration": {
            "points": -10,
            "threshold": ("jerk_ms3", "greater", 4.0),
        },
        "battery_overheat": {
            "points": -25,
            "threshold": ("battery_temp_c", "greater", 50),
        },
        "motor_overheat": {
            "points": -20,
            "threshold": ("motor_temp_c", "greater", 100),
        },
        "brake_overheat": {
            "points": -20,
            "threshold": ("brake_temp_c", "greater", 350),
        },
        "low_regen": {"points": -10, "threshold": ("regen_efficiency", "less", 0.5)},
        "excessive_speed": {"points": -15, "threshold": ("speed_kmh", "greater", 140)},
        "critical_battery_temp": {
            "points": -50,
            "threshold": ("battery_temp_c", "greater", 60),
        },
        # Positive events
        "excellent_regen": {
            "points": 10,
            "threshold": ("regen_efficiency", "greater", 0.9),
        },
        "good_regen": {
            "points": 5,
            "threshold": ("regen_efficiency", "between", [0.75, 0.9]),
        },
        "smooth_driving": {
            "points": 5,
            "threshold": ("jerk_ms3", "between", [-2.0, 2.0]),
        },
        "efficient_speed": {
            "points": 5,
            "threshold": ("speed_kmh", "between", [50, 100]),
        },
        "optimal_battery_temp": {
            "points": 5,
            "threshold": ("battery_temp_c", "between", [20, 35]),
        },
    }

    BADGES = {
        "eco_warrior": {
            "condition": "total_score > 500",
            "description": "Achieved 500+ eco points",
        },
        "smooth_operator": {
            "condition": "smooth_streak > 20",
            "description": "20 consecutive smooth readings",
        },
        "regen_master": {
            "condition": "regen_streak > 15",
            "description": "15 consecutive excellent regen",
        },
        "cool_headed": {
            "condition": "no_overheat_streak > 50",
            "description": "50 readings without overheating",
        },
    }

    FEEDBACK_TEMPLATES = {
        "harsh_braking": [
            "Whoa! That was a hard stop. Try anticipating traffic to brake smoother.",
            "Your brakes called - they want a gentler touch! Smoother stops save pads and energy.",
        ],
        "harsh_acceleration": [
            "Easy on the accelerator! Smooth takeoffs improve range by up to 15%.",
            "Jackrabbit starts detected - your battery prefers a gradual approach.",
        ],
        "battery_overheat": [
            "⚠️ Battery running hot! Consider reducing power demand or stopping to cool down.",
            "Temperature alert: Your battery is climbing into the danger zone. Ease off!",
        ],
        "excellent_regen": [
            "🌟 Excellent regeneration! You're putting energy back where it belongs.",
            "Regen master! That smooth braking just added miles to your range.",
        ],
        "good_regen": [
            "Nice regen capture! Keep up the smooth driving.",
            "Good energy recovery - you're driving like a pro!",
        ],
        "smooth_driving": [
            "✨ Silky smooth! Your passengers (and battery) thank you.",
            "Perfect driving dynamics - you're in the eco zone!",
        ],
    }

    def __init__(self, driver_name: str = "Driver"):
        """Initialize scoring engine."""
        self.driver_name = driver_name
        self.total_score = 0
        self.session_events: List[Dict] = []
        self.badges_earned: List[str] = []

        # Streak tracking
        self.streaks = {
            "smooth_streak": 0,
            "regen_streak": 0,
            "no_overheat_streak": 0,
        }

        # Text generator
        self.text_generator = None
        self._init_text_generator()

    def _init_text_generator(self):
        """Initialize HuggingFace text generator."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a small, fast model for feedback generation
                self.text_generator = pipeline(
                    "text2text-generation", model="google/flan-t5-small", max_length=100
                )
                print("✓ HuggingFace text generator initialized (flan-t5-small)")
            except Exception as e:
                print(f"⚠️ Could not load text generator: {e}")
                self.text_generator = None

    def _check_condition(
        self, telemetry: Dict[str, float], feature: str, condition: str, threshold: Any
    ) -> bool:
        """Check if a scoring condition is met."""
        if feature not in telemetry:
            return False

        value = telemetry[feature]

        if condition == "greater":
            return value > threshold
        elif condition == "less":
            return value < threshold
        elif condition == "between":
            return threshold[0] <= value <= threshold[1]
        return False

    def _generate_feedback_text(self, events: List[Dict], score_delta: int) -> str:
        """Generate feedback text using templates or LLM."""
        if not events:
            if score_delta >= 0:
                return "Keep up the good driving!"
            else:
                return "Room for improvement - try smoother inputs."

        # Get primary event
        primary_event = max(events, key=lambda x: abs(x["points"]))
        event_name = primary_event["event"]

        # Try template first
        if event_name in self.FEEDBACK_TEMPLATES:
            import random

            return random.choice(self.FEEDBACK_TEMPLATES[event_name])

        # Try LLM generation
        if self.text_generator:
            try:
                prompt = f"""Generate a brief, witty driving feedback message (under 20 words) for: 
The driver had a {event_name.replace('_', ' ')} event, scoring {score_delta} points.
Make it encouraging but informative."""

                result = self.text_generator(prompt)
                if result and len(result) > 0:
                    return result[0]["generated_text"].strip()
            except Exception as e:
                print(f"LLM generation failed: {e}")

        # Fallback
        if score_delta < 0:
            return f"Detected {event_name.replace('_', ' ')}. Try to improve your driving technique."
        else:
            return f"Great job with {event_name.replace('_', ' ')}! Keep it up."

    def _update_streaks(self, events: List[Dict]) -> None:
        """Update streak counters based on events."""
        event_names = [e["event"] for e in events]

        # Smooth driving streak
        if "smooth_driving" in event_names:
            self.streaks["smooth_streak"] += 1
        elif any(e in event_names for e in ["harsh_braking", "harsh_acceleration"]):
            self.streaks["smooth_streak"] = 0

        # Regen streak
        if "excellent_regen" in event_names or "good_regen" in event_names:
            self.streaks["regen_streak"] += 1
        elif "low_regen" in event_names:
            self.streaks["regen_streak"] = 0

        # No overheat streak
        if not any(
            e in event_names
            for e in [
                "battery_overheat",
                "motor_overheat",
                "brake_overheat",
                "critical_battery_temp",
            ]
        ):
            self.streaks["no_overheat_streak"] += 1
        else:
            self.streaks["no_overheat_streak"] = 0

    def _check_badges(self) -> List[str]:
        """Check if any new badges are earned."""
        new_badges = []

        if self.total_score > 500 and "eco_warrior" not in self.badges_earned:
            self.badges_earned.append("eco_warrior")
            new_badges.append("eco_warrior")

        if (
            self.streaks["smooth_streak"] > 20
            and "smooth_operator" not in self.badges_earned
        ):
            self.badges_earned.append("smooth_operator")
            new_badges.append("smooth_operator")

        if (
            self.streaks["regen_streak"] > 15
            and "regen_master" not in self.badges_earned
        ):
            self.badges_earned.append("regen_master")
            new_badges.append("regen_master")

        if (
            self.streaks["no_overheat_streak"] > 50
            and "cool_headed" not in self.badges_earned
        ):
            self.badges_earned.append("cool_headed")
            new_badges.append("cool_headed")

        return new_badges

    def score(self, telemetry: Dict[str, float]) -> ScoringResult:
        """
        Calculate score for current telemetry.

        Args:
            telemetry: Current telemetry reading

        Returns:
            ScoringResult with points and feedback
        """
        events = []
        score_delta = 0

        # Check each rule
        for event_name, rule in self.SCORING_RULES.items():
            feature, condition, threshold = rule["threshold"]
            if self._check_condition(telemetry, feature, condition, threshold):
                events.append(
                    {
                        "event": event_name,
                        "points": rule["points"],
                        "feature": feature,
                        "value": telemetry.get(feature),
                    }
                )
                score_delta += rule["points"]

        # Update totals and streaks
        self.total_score += score_delta
        self.session_events.extend(events)
        self._update_streaks(events)

        # Check for badges
        new_badges = self._check_badges()

        # Generate feedback
        feedback = self._generate_feedback_text(events, score_delta)

        return ScoringResult(
            score_delta=score_delta,
            total_score=self.total_score,
            events=events,
            feedback_text=feedback,
            badges_earned=new_badges,
        )

    def reset(self) -> None:
        """Reset session state."""
        self.session_events.clear()
        self.streaks = {k: 0 for k in self.streaks}


class MLPipeline:
    """
    Unified ML Pipeline combining anomaly detection and scoring.
    Used by Data Analysis Agent for real-time processing.
    """

    def __init__(self, vehicle_id: str, driver_name: str = "Driver"):
        """Initialize the complete ML pipeline."""
        self.vehicle_id = vehicle_id
        self.detector = AdvancedAnomalyDetector(vehicle_id)
        self.scorer = ScoringEngine(driver_name)

    def train(
        self, historical_data: pd.DataFrame, driver_profile: str = "normal"
    ) -> Dict:
        """Train the pipeline on historical data."""
        return self.detector.train(historical_data, driver_profile)

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Process telemetry through the full pipeline.

        Returns combined anomaly detection and scoring results.
        """
        # Run anomaly detection
        anomaly_result = self.detector.predict(telemetry)

        # Run scoring
        scoring_result = self.scorer.score(telemetry)

        return {
            "timestamp": datetime.now().isoformat(),
            "vehicle_id": self.vehicle_id,
            "telemetry": telemetry,
            # Anomaly detection
            "is_anomaly": anomaly_result.is_anomaly,
            "anomaly_score": anomaly_result.anomaly_score,
            "anomaly_type": anomaly_result.anomaly_type,
            "severity": anomaly_result.severity,
            "failure_risk_pct": anomaly_result.failure_risk_pct,
            "time_to_failure_hours": anomaly_result.time_to_failure_hours,
            "affected_components": anomaly_result.affected_components,
            "contributing_factors": anomaly_result.contributing_factors,
            # Scoring
            "score_delta": scoring_result.score_delta,
            "total_score": scoring_result.total_score,
            "scoring_events": scoring_result.events,
            "feedback_text": scoring_result.feedback_text,
            "badges_earned": scoring_result.badges_earned,
        }

    def save(self, path: str) -> None:
        """Save pipeline state."""
        self.detector.save(path)

    def load(self, path: str) -> None:
        """Load pipeline state."""
        self.detector.load(path)


if __name__ == "__main__":
    # Test the ML pipeline
    print("🔬 Testing Advanced ML Pipeline")
    print("=" * 50)

    # Create test data
    test_telemetry = {
        "speed_kmh": 95.5,
        "acceleration_ms2": 2.1,
        "jerk_ms3": 0.8,
        "power_draw_kw": 45.0,
        "regen_efficiency": 0.82,
        "battery_soc_pct": 72.0,
        "battery_temp_c": 38.0,
        "motor_temp_c": 75.0,
        "inverter_temp_c": 68.0,
        "brake_temp_c": 120.0,
        "coolant_temp_c": 42.0,
        "wear_index": 0.15,
    }

    # Test detector (without training)
    detector = AdvancedAnomalyDetector("TEST_VIN")
    result = detector.predict(test_telemetry)
    print(f"Anomaly result (untrained): {result.anomaly_type}")

    # Test scorer
    scorer = ScoringEngine("TestDriver")
    score_result = scorer.score(test_telemetry)
    print(f"Score delta: {score_result.score_delta}")
    print(f"Feedback: {score_result.feedback_text}")

    print("\n✅ ML Pipeline test complete")
