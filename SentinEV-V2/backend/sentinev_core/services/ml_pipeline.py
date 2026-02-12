"""
Simulated ML Pipeline for SentinEV
Uses statistical methods with realistic logic to simulate:
- Anomaly Detection (Z-score + threshold-based)
- Failure Prediction (Probability estimation based on degradation patterns)
- Severity Classification (Rule-based with component-specific thresholds)

NOTE: This is a high-fidelity simulation. In production, these would be
trained models (LSTM-AE, LightGBM, etc.)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    is_anomaly: bool
    anomaly_score: float  # 0-1
    contributing_sensors: List[str]
    z_scores: Dict[str, float]


@dataclass
class FailurePrediction:
    """Failure prediction result."""

    failure_probability: float  # 0-1
    estimated_rul_days: Optional[int]  # Remaining Useful Life
    failure_type: Optional[str]
    confidence: float


@dataclass
class DiagnosisResult:
    """Full diagnosis result."""

    anomaly: AnomalyResult
    prediction: FailurePrediction
    severity: Severity
    primary_concern: Optional[str]
    recommended_action: str
    explanation: str


class AnomalyDetector:
    """
    Simulates anomaly detection using statistical Z-score analysis.
    In production, this would use LSTM-Autoencoder or similar.
    """

    # Thresholds for anomaly detection per sensor
    SENSOR_THRESHOLDS = {
        "engine_temp": {"warning": 95, "critical": 105, "max": 120},
        "battery_voltage": {"warning": 12.0, "critical": 11.5, "min": 10.0},
        "coolant_level": {"warning": 0.80, "critical": 0.70, "min": 0.50},
        "brake_pressure": {"warning": 40, "critical": 35, "min": 25},
        "vibration_amplitude": {"warning": 0.20, "critical": 0.28, "max": 0.40},
    }

    def detect(
        self, sensors: Dict[str, float], baseline: Dict[str, Dict[str, float]]
    ) -> AnomalyResult:
        """
        Detect anomalies using Z-score analysis.

        Args:
            sensors: Current sensor readings
            baseline: Expected values with mean and std for each sensor
        """
        z_scores = {}
        contributing = []
        total_anomaly = 0.0

        for sensor, value in sensors.items():
            if sensor not in baseline:
                continue

            mean = baseline[sensor].get("mean", value)
            std = baseline[sensor].get("std", 1.0)

            if std == 0:
                std = 0.1

            z = abs(value - mean) / std
            z_scores[sensor] = round(z, 3)

            # Check against absolute thresholds
            threshold = self.SENSOR_THRESHOLDS.get(sensor, {})
            is_high_anomaly = False

            if "max" in threshold and value > threshold["warning"]:
                is_high_anomaly = True
            elif "min" in threshold and value < threshold["warning"]:
                is_high_anomaly = True

            # Z-score > 2 or threshold breach
            if z > 2.0 or is_high_anomaly:
                contributing.append(sensor)
                total_anomaly += min(z / 3.0, 1.0)

        # Normalize anomaly score
        if len(z_scores) > 0:
            anomaly_score = min(total_anomaly / len(z_scores), 1.0)
        else:
            anomaly_score = 0.0

        return AnomalyResult(
            is_anomaly=anomaly_score > 0.3 or len(contributing) > 0,
            anomaly_score=round(anomaly_score, 4),
            contributing_sensors=contributing,
            z_scores=z_scores,
        )


class FailurePredictor:
    """
    Simulates failure prediction based on degradation patterns.
    In production, this would use LightGBM or similar gradient boosting.
    """

    # Component-specific degradation rates and failure thresholds
    COMPONENT_PROFILES = {
        "brake": {
            "sensors": ["brake_pressure", "vibration_amplitude"],
            "failure_threshold": {"brake_pressure": 30, "vibration_amplitude": 0.35},
            "typical_rul_days": 14,
        },
        "battery": {
            "sensors": ["battery_voltage", "engine_temp"],
            "failure_threshold": {"battery_voltage": 11.0, "engine_temp": 110},
            "typical_rul_days": 30,
        },
        "cooling": {
            "sensors": ["coolant_level", "engine_temp"],
            "failure_threshold": {"coolant_level": 0.60, "engine_temp": 108},
            "typical_rul_days": 21,
        },
    }

    def predict(
        self,
        sensors: Dict[str, float],
        degradation_config: Dict[str, Any],
        vehicle_mileage: int = 0,
    ) -> FailurePrediction:
        """
        Predict failure probability and remaining useful life.
        """
        degradation_rate = degradation_config.get("rate", 0.0)
        component = degradation_config.get("component")

        if not component or degradation_rate == 0:
            return FailurePrediction(
                failure_probability=0.05,  # Baseline 5%
                estimated_rul_days=None,
                failure_type=None,
                confidence=0.9,
            )

        profile = self.COMPONENT_PROFILES.get(component, {})
        thresholds = profile.get("failure_threshold", {})
        typical_rul = profile.get("typical_rul_days", 30)

        # Calculate distance to failure threshold
        proximity_scores = []
        for sensor in profile.get("sensors", []):
            if sensor in sensors and sensor in thresholds:
                current = sensors[sensor]
                threshold = thresholds[sensor]

                # Different logic for min vs max thresholds
                if sensor in ["battery_voltage", "brake_pressure", "coolant_level"]:
                    # Lower is worse
                    proximity = (
                        max(0, (current - threshold) / current) if current > 0 else 1.0
                    )
                else:
                    # Higher is worse
                    proximity = (
                        max(0, (threshold - current) / threshold)
                        if threshold > 0
                        else 1.0
                    )

                proximity_scores.append(1 - proximity)

        if proximity_scores:
            failure_probability = sum(proximity_scores) / len(proximity_scores)
            failure_probability = min(max(failure_probability, 0.0), 1.0)
        else:
            failure_probability = 0.1

        # Estimate RUL based on degradation rate
        if degradation_rate > 0 and failure_probability < 1.0:
            estimated_rul = int(
                typical_rul * (1 - failure_probability) / (degradation_rate * 10)
            )
            estimated_rul = max(1, min(estimated_rul, 365))
        else:
            estimated_rul = None

        return FailurePrediction(
            failure_probability=round(failure_probability, 3),
            estimated_rul_days=estimated_rul,
            failure_type=f"{component}_degradation",
            confidence=round(0.7 + 0.2 * (1 - failure_probability), 2),
        )


class SeverityClassifier:
    """
    Classifies severity based on anomaly scores and failure predictions.
    Uses rule-based logic with component-specific considerations.
    """

    def classify(
        self,
        anomaly: AnomalyResult,
        prediction: FailurePrediction,
        sensors: Dict[str, float],
    ) -> Tuple[Severity, str, str]:
        """
        Classify severity and determine primary concern.

        Returns:
            (severity, primary_concern, recommended_action)
        """
        # Check for emergency conditions first
        if sensors.get("engine_temp", 0) > 110:
            return (
                Severity.EMERGENCY,
                "thermal_runaway_risk",
                "Stop vehicle immediately. Do not charge. Contact emergency services if smoke observed.",
            )

        if sensors.get("battery_voltage", 12) < 10.5:
            return (
                Severity.EMERGENCY,
                "battery_critical",
                "Battery critically low. Stop driving and arrange towing to service center.",
            )

        # Critical conditions
        if prediction.failure_probability > 0.7 or anomaly.anomaly_score > 0.7:
            primary = prediction.failure_type or "multiple_systems"
            return (
                Severity.CRITICAL,
                primary,
                f"Schedule service within 48 hours. {primary.replace('_', ' ').title()} requires immediate attention.",
            )

        # Warning conditions
        if prediction.failure_probability > 0.4 or anomaly.anomaly_score > 0.4:
            primary = prediction.failure_type or "general_wear"
            return (
                Severity.WARNING,
                primary,
                f"Schedule service within 1-2 weeks. Monitor {', '.join(anomaly.contributing_sensors) or 'system'}.",
            )

        # Info - everything normal or minor issues
        return (
            Severity.INFO,
            None,
            "Vehicle systems operating normally. Continue regular monitoring.",
        )


class MLPipeline:
    """
    Unified ML Pipeline that orchestrates all prediction components.
    """

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.failure_predictor = FailurePredictor()
        self.severity_classifier = SeverityClassifier()

    def diagnose(
        self,
        sensors: Dict[str, float],
        baseline: Dict[str, Dict[str, float]],
        degradation_config: Dict[str, Any],
        mileage: int = 0,
    ) -> DiagnosisResult:
        """
        Run full diagnosis pipeline.
        """
        # Step 1: Detect anomalies
        anomaly = self.anomaly_detector.detect(sensors, baseline)

        # Step 2: Predict failures
        prediction = self.failure_predictor.predict(
            sensors, degradation_config, mileage
        )

        # Step 3: Classify severity
        severity, primary_concern, action = self.severity_classifier.classify(
            anomaly, prediction, sensors
        )

        # Generate explanation
        explanation = self._generate_explanation(anomaly, prediction, severity)

        return DiagnosisResult(
            anomaly=anomaly,
            prediction=prediction,
            severity=severity,
            primary_concern=primary_concern,
            recommended_action=action,
            explanation=explanation,
        )

    def _generate_explanation(
        self, anomaly: AnomalyResult, prediction: FailurePrediction, severity: Severity
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        if anomaly.contributing_sensors:
            sensors_str = ", ".join(anomaly.contributing_sensors)
            parts.append(f"Unusual readings detected in: {sensors_str}.")

        if prediction.failure_type:
            parts.append(
                f"Predicted issue: {prediction.failure_type.replace('_', ' ')}."
            )
            parts.append(
                f"Failure probability: {prediction.failure_probability*100:.0f}%."
            )

        if prediction.estimated_rul_days:
            parts.append(
                f"Estimated time to failure: {prediction.estimated_rul_days} days."
            )

        if not parts:
            parts = ["All systems operating within normal parameters."]

        return " ".join(parts)


# Singleton instance
ml_pipeline = MLPipeline()
