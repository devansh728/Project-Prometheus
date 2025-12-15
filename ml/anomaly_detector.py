"""
SentinEV - Advanced Anomaly Detection Pipeline v2.0
====================================================
Uses ml_pipeline models (LSTM-AE, LightGBM) for production-grade detection.
Keeps ScoringEngine for gamification.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# ML Pipeline model loader
from ml.model_loader import get_model_loader, ModelLoader

# Streaming window buffer
from streaming import VehicleWindowBuffer, WindowConfig

# HuggingFace for text generation
try:
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""

    is_anomaly: bool
    anomaly_score: float  # Reconstruction error from LSTM-AE
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
    Advanced per-vehicle anomaly detection using ml_pipeline models.

    Features:
    - LSTM Autoencoder for anomaly detection (replaces Isolation Forest)
    - LightGBM for failure prediction (replaces RandomForest)
    - LightGBM for severity classification
    - Window-based aggregate features
    - Digital twin personalization
    """

    # Still keep threshold-based anomaly type identification
    ANOMALY_THRESHOLDS = {
        "battery_temp_c": {"warning": 50, "critical": 60},
        "motor_temp_c": {"warning": 100, "critical": 120},
        "inverter_temp_c": {"warning": 80, "critical": 100},
        "battery_cell_delta_v": {"warning": 0.1, "critical": 0.2},
        "motor_rpm": {"max": 12000, "critical": 14000},
    }

    ANOMALY_TYPES = {
        "battery_thermal": ["battery_temp_c", "battery_cell_delta_v"],
        "motor_thermal": ["motor_temp_c", "inverter_temp_c"],
        "battery_degradation": ["battery_soc_pct", "battery_cell_delta_v"],
        "motor_fault": ["motor_rpm", "motor_temp_c"],
        "inverter_fault": ["inverter_temp_c"],
        "driving_behavior": ["speed_kph", "throttle_pct", "brake_pct"],
    }

    def __init__(self, vehicle_id: str):
        """Initialize detector with ml_pipeline models."""
        self.vehicle_id = vehicle_id

        # Load pre-trained models from ml_pipeline
        self.model_loader: ModelLoader = get_model_loader()

        # Per-vehicle window buffer for aggregate features
        self.window_buffer = VehicleWindowBuffer(vehicle_id)

        # Learned baselines (computed from training data)
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.driver_profile: str = "normal"

        # Digital twin state
        self.digital_twin = {
            "vehicle_id": vehicle_id,
            "odometer_km": 0,
            "battery_cycles": 0,
            "component_health": {
                "battery": 100,
                "motor": 100,
                "inverter": 100,
                "brakes": 100,
            },
            "fault_history": [],
        }

        # Personalized thresholds (can be adjusted per driver)
        self.personalized_thresholds = dict(self.ANOMALY_THRESHOLDS)

        # Training state
        self.is_trained = self.model_loader.failure_predictor is not None
        self.training_samples = 0

        # Inference timing
        self.last_inference_latency_ms = 0

    def update_digital_twin(self, telemetry: Dict) -> None:
        """Update digital twin with new telemetry."""
        if "odometer_km" in telemetry:
            self.digital_twin["odometer_km"] = telemetry["odometer_km"]

        # Track battery cycles (simplified)
        soc = telemetry.get("battery_soc_pct", 100)
        if hasattr(self, "_last_soc"):
            if self._last_soc > 80 and soc < 20:
                self.digital_twin["battery_cycles"] += 1
        self._last_soc = soc

        # Degrade component health based on stress
        if telemetry.get("battery_temp_c", 0) > 50:
            self.digital_twin["component_health"]["battery"] -= 0.01
        if telemetry.get("motor_temp_c", 0) > 100:
            self.digital_twin["component_health"]["motor"] -= 0.01

    def _adjust_thresholds_for_driver(self, driver_profile: str) -> None:
        """Adjust thresholds based on driver profile."""
        self.driver_profile = driver_profile

        adjustments = {
            "aggressive": {
                "battery_temp_c": {"warning": 55, "critical": 65},
                "motor_temp_c": {"warning": 110, "critical": 130},
            },
            "eco": {
                "battery_temp_c": {"warning": 45, "critical": 55},
                "motor_temp_c": {"warning": 90, "critical": 110},
            },
        }

        if driver_profile in adjustments:
            for key, val in adjustments[driver_profile].items():
                self.personalized_thresholds[key] = val

    def _compute_baseline_stats(self, data: pd.DataFrame) -> None:
        """Compute baseline statistics for ALL 66 aggregate columns expected by ML models.

        The LightGBM model expects columns like: speed_kph_mean, speed_kph_std, etc.
        We compute these from the raw telemetry columns in the historical data.
        """
        # Base columns that get aggregated (without suffix)
        base_columns = [
            "speed_kph",
            "motor_rpm",
            "motor_temp_c",
            "inverter_temp_c",
            "battery_soc_pct",
            "battery_voltage_v",
            "battery_current_a",
            "battery_temp_c",
            "battery_cell_delta_v",
            "hvac_power_kw",
            "throttle_pct",
            "brake_pct",
            "regen_pct",
            "accel_x",
            "accel_y",
            "accel_z",
        ]

        # Also store raw column stats
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns and len(data[col].dropna()) > 0:
                self.baseline_stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()) if len(data[col]) > 1 else 0.1,
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                }

        # Generate aggregate column stats (mean, std, min, max variants)
        # These are what the LightGBM model actually expects
        for base_col in base_columns:
            # Find the matching raw column in data (handle naming variations)
            raw_col = None
            if base_col in data.columns:
                raw_col = base_col
            elif base_col.replace("_pct", "_percent") in data.columns:
                raw_col = base_col.replace("_pct", "_percent")

            if raw_col and raw_col in data.columns:
                col_data = data[raw_col].dropna()
                if len(col_data) > 0:
                    col_mean = float(col_data.mean())
                    col_std = (
                        float(col_data.std()) if len(col_data) > 1 else col_mean * 0.1
                    )
                    col_min = float(col_data.min())
                    col_max = float(col_data.max())
                else:
                    # Fallback defaults
                    col_mean, col_std, col_min, col_max = 0, 0.1, 0, 1
            else:
                # Column not found - use sensible defaults based on column type
                defaults = {
                    "speed_kph": (60, 20, 0, 150),
                    "motor_rpm": (3000, 1500, 0, 8000),
                    "motor_temp_c": (50, 15, 25, 100),
                    "inverter_temp_c": (45, 12, 25, 90),
                    "battery_soc_pct": (70, 20, 20, 100),
                    "battery_voltage_v": (380, 20, 350, 420),
                    "battery_current_a": (80, 40, 0, 200),
                    "battery_temp_c": (30, 8, 20, 55),
                    "battery_cell_delta_v": (0.05, 0.03, 0, 0.15),
                    "hvac_power_kw": (2, 1, 0, 5),
                    "throttle_pct": (30, 25, 0, 100),
                    "brake_pct": (10, 15, 0, 100),
                    "regen_pct": (60, 20, 0, 100),
                    "accel_x": (0.1, 0.3, -2, 2),
                    "accel_y": (0, 0.1, -1, 1),
                    "accel_z": (1, 0.1, 0.8, 1.2),
                }
                col_mean, col_std, col_min, col_max = defaults.get(
                    base_col, (0, 0.1, 0, 1)
                )

            # Create all 4 aggregate columns for this base
            self.baseline_stats[f"{base_col}_mean"] = {
                "mean": col_mean,
                "std": col_std * 0.3,
                "min": col_min,
                "max": col_max,
            }
            self.baseline_stats[f"{base_col}_std"] = {
                "mean": col_std,
                "std": col_std * 0.5,
                "min": 0,
                "max": col_std * 3,
            }
            self.baseline_stats[f"{base_col}_min"] = {
                "mean": col_min,
                "std": col_std * 0.5,
                "min": col_min * 0.8,
                "max": col_mean,
            }
            self.baseline_stats[f"{base_col}_max"] = {
                "mean": col_max,
                "std": col_std * 0.5,
                "min": col_mean,
                "max": col_max * 1.2,
            }

        # Add the two extra derived features
        power_mean = self.baseline_stats.get("power_kw_mean", {}).get("mean", 30)
        self.baseline_stats["power_kw_mean"] = {
            "mean": power_mean,
            "std": 15,
            "min": 0,
            "max": 150,
        }
        self.baseline_stats["accel_magnitude_mean"] = {
            "mean": 1.0,
            "std": 0.3,
            "min": 0,
            "max": 3,
        }

    def train(
        self, historical_data: pd.DataFrame, driver_profile: str = "normal"
    ) -> Dict[str, Any]:
        """
        Train baseline stats from historical data.
        The main models are already pre-trained in ml_pipeline.
        """
        self._adjust_thresholds_for_driver(driver_profile)
        self._compute_baseline_stats(historical_data)
        self.training_samples = len(historical_data)
        self.is_trained = True

        return {
            "vehicle_id": self.vehicle_id,
            "driver_profile": driver_profile,
            "training_samples": self.training_samples,
            "models_loaded": {
                "failure_predictor": self.model_loader.failure_predictor is not None,
                "severity_classifier": self.model_loader.severity_classifier
                is not None,
                "lstm_autoencoder": self.model_loader.lstm_autoencoder is not None,
            },
            "baseline_features": list(self.baseline_stats.keys()),
        }

    def _identify_anomaly_type(
        self, telemetry: Dict[str, float]
    ) -> Tuple[str, List[str]]:
        """Identify anomaly type based on threshold violations."""
        anomaly_type = "normal"
        affected = []

        # Battery thermal check
        if telemetry.get("battery_temp_c", 0) > self.personalized_thresholds.get(
            "battery_temp_c", {}
        ).get("warning", 50):
            anomaly_type = "battery_thermal"
            affected.extend(["battery", "cooling_system"])

        # Motor thermal check
        if telemetry.get("motor_temp_c", 0) > self.personalized_thresholds.get(
            "motor_temp_c", {}
        ).get("warning", 100):
            anomaly_type = "motor_thermal"
            affected.extend(["motor", "inverter"])

        # Cell imbalance
        if telemetry.get("battery_cell_delta_v", 0) > self.personalized_thresholds.get(
            "battery_cell_delta_v", {}
        ).get("warning", 0.1):
            anomaly_type = "battery_degradation"
            affected.append("battery")

        return anomaly_type, list(set(affected))

    def _identify_contributing_factors(
        self, telemetry: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify factors contributing to the anomaly."""
        factors = []

        for feature, stats in self.baseline_stats.items():
            if feature in telemetry:
                value = telemetry[feature]
                mean = stats.get("mean", 0)
                std = stats.get("std", 1)

                if std > 0:
                    z_score = (value - mean) / std
                    if abs(z_score) > 2:
                        factors.append(
                            {
                                "feature": feature,
                                "value": round(value, 2),
                                "mean": round(mean, 2),
                                "z_score": round(z_score, 2),
                                "deviation": "high" if z_score > 0 else "low",
                            }
                        )

        return sorted(factors, key=lambda x: abs(x["z_score"]), reverse=True)[:5]

    def _estimate_time_to_failure(self, failure_risk: float) -> Optional[float]:
        """Estimate time to failure based on risk level."""
        if failure_risk < 0.3:
            return None
        elif failure_risk > 0.9:
            return 2.0  # 2 hours
        elif failure_risk > 0.7:
            return 24.0  # 24 hours
        elif failure_risk > 0.5:
            return 72.0  # 3 days
        else:
            return 168.0  # 1 week

    def _prepare_features(self, telemetry: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for model prediction."""
        feature_cols = self.model_loader.feature_columns
        if not feature_cols:
            # Fallback: use available numeric values
            return np.array(list(telemetry.values()))

        features = []
        for col in feature_cols:
            if col in telemetry:
                features.append(telemetry[col])
            else:
                # Use baseline mean or 0
                features.append(self.baseline_stats.get(col, {}).get("mean", 0))

        return np.array(features)

    def predict(self, telemetry: Dict[str, float]) -> AnomalyResult:
        """
        Predict anomaly using ml_pipeline models.

        Uses:
        - LSTM-AE for anomaly score
        - LightGBM for failure probability
        - LightGBM for severity classification

        IMPORTANT: ML models require aggregate features (mean, std, min, max).
        Until aggregate features are available (~10 frames), only telemetry-based
        detection is used (active faults, temperature thresholds).
        """
        start_time = time.time()

        # Update digital twin
        self.update_digital_twin(telemetry)

        # Add frame to window buffer
        aggregate_features = self.window_buffer.add_frame(telemetry)
        frame_count = self.window_buffer.frame_count

        # PRIMARY: Check for actual telemetry anomalies (active faults or temperature thresholds)
        # These work IMMEDIATELY without needing ML models
        active_faults = telemetry.get("active_faults", [])
        has_active_faults = (
            len(active_faults) > 0
            if isinstance(active_faults, list)
            else bool(active_faults)
        )

        # Temperature-based anomaly detection
        brake_temp = telemetry.get("brake_temp_c", 0)
        battery_temp = telemetry.get("battery_temp_c", 0)
        motor_temp = telemetry.get("motor_temp_c", 0)

        temp_anomaly = (
            brake_temp > 200  # Brake overheating
            or battery_temp > 50  # Battery thermal issue
            or motor_temp > 100  # Motor thermal issue
        )

        # ============================================================
        # ML PREDICTIONS - ONLY when aggregate features are available
        # ============================================================
        if aggregate_features is not None:
            # Use aggregate features from window - these have proper mean/std/min/max columns
            feature_dict = aggregate_features.iloc[0].to_dict()
            features = self._prepare_features(feature_dict)
            feature_source = "aggregate"
            use_ml = True

            # Get predictions from ml_pipeline models
            failure_prob, is_failure = self.model_loader.predict_failure(features)
            severity = self.model_loader.predict_severity(features)

            # Compute anomaly score using LSTM-AE (if sequence available and dimensions match)
            sequence = self.window_buffer.get_sequence_for_lstm(seq_len=30)
            expected_dim = (
                self.model_loader.lstm_config.get("input_dim")
                if self.model_loader.lstm_config
                else None
            )

            lstm_used = False
            if (
                sequence is not None
                and expected_dim is not None
                and sequence.shape[1] == expected_dim
            ):
                anomaly_score = self.model_loader.compute_anomaly_score(sequence)
                lstm_threshold = (
                    self.model_loader.lstm_config.get("threshold", 0.5)
                    if self.model_loader.lstm_config
                    else 0.5
                )
                lstm_anomaly = anomaly_score > lstm_threshold
                lstm_used = True
            else:
                # LSTM not available - use LightGBM failure_prob as anomaly score
                anomaly_score = failure_prob
                lstm_anomaly = False
                lstm_threshold = None

            # Determine ML-based anomaly (only trust if using aggregates)
            ml_anomaly = is_failure or lstm_anomaly

        else:
            # NO AGGREGATE FEATURES YET - Skip ML models entirely
            # Raw telemetry doesn't have mean/std/min/max columns that models expect
            feature_source = "raw (ML skipped)"
            use_ml = False

            # Safe defaults - no ML prediction yet
            failure_prob = 0.0
            is_failure = False
            severity = "low"
            anomaly_score = 0.0
            lstm_used = False
            lstm_anomaly = False
            lstm_threshold = None
            ml_anomaly = False
            features = []  # Empty for logging

        # Final anomaly determination:
        # ONLY use telemetry-based detection (faults + temperature thresholds)
        # ML predictions are INFORMATIONAL ONLY - they don't trigger anomaly status
        # (The LightGBM model was trained on hourly historical data and is miscalibrated
        # for real-time inference with 1-second intervals)
        is_anomaly = has_active_faults or temp_anomaly

        # Determine severity based on source
        if has_active_faults:
            severity = "high" if brake_temp > 150 or battery_temp > 45 else "medium"
        elif temp_anomaly:
            if brake_temp > 300 or battery_temp > 55:
                severity = "critical"
            elif brake_temp > 200 or battery_temp > 50 or motor_temp > 100:
                severity = "high"
            else:
                severity = "medium"
        elif not use_ml:
            severity = "low"
        # else: keep ML-predicted severity

        # Identify anomaly type and affected components
        anomaly_type, affected_components = self._identify_anomaly_type(telemetry)

        # Get contributing factors
        contributing_factors = self._identify_contributing_factors(telemetry)

        # Estimate time to failure (only if ML is active)
        ttf = self._estimate_time_to_failure(failure_prob) if use_ml else 168.0

        # Compute confidence based on data availability
        confidence = (
            0.3 if not use_ml else (0.9 if aggregate_features is not None else 0.5)
        )

        self.last_inference_latency_ms = (time.time() - start_time) * 1000

        # ==================== DETAILED ML LOGGING ====================
        # Log every 10 frames or when anomaly detected
        should_log = (frame_count <= 3) or (frame_count % 30 == 0) or is_anomaly

        if should_log:
            print(f"\n{'='*60}")
            print(
                f"🧠 ML MODEL PREDICTIONS [Frame {frame_count}] - Vehicle: {self.vehicle_id}"
            )
            print(f"{'='*60}")
            print(f"📊 Feature Source: {feature_source} | ML Active: {use_ml}")
            print(f"")

            if use_ml:
                print(f"� ML PREDICTIONS (Informational Only - not used for anomaly detection)")
                print(f"   Note: Model trained on hourly data, not calibrated for real-time")
                print(f"")
                print(f"🔹 MODEL 1: LightGBM Failure Predictor")
                print(f"   Probability: {failure_prob*100:.2f}%")
                print(f"")
                print(f"🔹 MODEL 2: LightGBM Severity Classifier")
                print(f"   Predicted Severity: {severity}")
                print(f"")
                print(f"🔹 MODEL 3: LSTM Autoencoder")
                if lstm_used:
                    print(f"   Anomaly Score: {anomaly_score:.4f}")
                else:
                    print(f"   ⚠️ NOT USED - Dimension mismatch")
            else:
                print(
                    f"⏳ ML SKIPPED - Waiting for aggregate features ({frame_count}/10 frames)"
                )

            print(f"")
            print(f"📍 TELEMETRY CHECK:")
            print(f"   Active Faults: {active_faults}")
            print(f"   Brake Temp: {brake_temp:.1f}°C (threshold: 200°C)")
            print(f"   Battery Temp: {battery_temp:.1f}°C (threshold: 50°C)")
            print(f"   Motor Temp: {motor_temp:.1f}°C (threshold: 100°C)")
            print(f"")
            print(f"🎯 FINAL RESULT:")
            print(f"   Is Anomaly: {'✅ YES' if is_anomaly else '❌ NO'}")
            print(f"   Anomaly Type: {anomaly_type}")
            print(f"   Severity: {severity}")
            print(f"   Failure Risk: {failure_prob*100:.1f}%")
            reason = (
                "Active Faults"
                if has_active_faults
                else (
                    "Temp Threshold"
                    if temp_anomaly
                    else ("Model Prediction" if (is_anomaly and use_ml) else "Normal")
                )
            )
            print(f"   Reason: {reason}")
            print(f"{'='*60}\n")

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(float(anomaly_score), 4),
            anomaly_type=anomaly_type if is_anomaly else "normal",
            severity=severity if is_anomaly else "low",
            confidence=round(confidence, 3),
            failure_risk_pct=round(failure_prob * 100, 1),
            time_to_failure_hours=ttf,
            affected_components=affected_components,
            contributing_factors=contributing_factors,
        )

    def save(self, path: str) -> None:
        """Save detector state (baseline stats, digital twin)."""
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "vehicle_id": self.vehicle_id,
            "driver_profile": self.driver_profile,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "baseline_stats": self.baseline_stats,
            "personalized_thresholds": self.personalized_thresholds,
            "digital_twin": self.digital_twin,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """Load detector state."""
        model_dir = Path(path)
        metadata_path = model_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.vehicle_id = metadata.get("vehicle_id", self.vehicle_id)
                self.driver_profile = metadata.get("driver_profile", "normal")
                self.is_trained = metadata.get("is_trained", False)
                self.training_samples = metadata.get("training_samples", 0)
                self.baseline_stats = metadata.get("baseline_stats", {})
                self.personalized_thresholds = metadata.get(
                    "personalized_thresholds", self.ANOMALY_THRESHOLDS
                )
                self.digital_twin = metadata.get("digital_twin", self.digital_twin)


# =============================================================================
# ScoringEngine - KEPT AS-IS from original
# =============================================================================


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
        "low_regen": {"points": -10, "threshold": ("regen_pct", "less", 0.5)},
        "excessive_speed": {"points": -15, "threshold": ("speed_kph", "greater", 140)},
        "critical_battery_temp": {
            "points": -50,
            "threshold": ("battery_temp_c", "greater", 60),
        },
        # Positive events
        "excellent_regen": {"points": 10, "threshold": ("regen_pct", "greater", 0.9)},
        "good_regen": {"points": 5, "threshold": ("regen_pct", "between", [0.75, 0.9])},
        "smooth_driving": {
            "points": 5,
            "threshold": ("jerk_ms3", "between", [-2.0, 2.0]),
        },
        "efficient_speed": {
            "points": 5,
            "threshold": ("speed_kph", "between", [50, 100]),
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
            "Your brakes called - they want a gentler touch!",
        ],
        "harsh_acceleration": [
            "Easy on the accelerator! Smooth takeoffs improve range by up to 15%.",
            "Jackrabbit starts detected - your battery prefers a gradual approach.",
        ],
        "battery_overheat": [
            "⚠️ Battery running hot! Consider reducing power demand.",
            "Temperature alert: Your battery is climbing into the danger zone.",
        ],
        "excellent_regen": [
            "🌟 Excellent regeneration! You're putting energy back where it belongs.",
            "Regen master! That smooth braking just added miles to your range.",
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
        self.streaks = {
            "smooth_streak": 0,
            "regen_streak": 0,
            "no_overheat_streak": 0,
        }
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
        """Generate feedback text using templates."""
        if not events:
            return (
                "Keep up the good driving!"
                if score_delta >= 0
                else "Room for improvement."
            )

        primary_event = max(events, key=lambda x: abs(x["points"]))
        event_name = primary_event["event"]

        if event_name in self.FEEDBACK_TEMPLATES:
            import random

            return random.choice(self.FEEDBACK_TEMPLATES[event_name])

        if score_delta < 0:
            return f"Detected {event_name.replace('_', ' ')}. Try to improve."
        return f"Great job with {event_name.replace('_', ' ')}!"

    def _update_streaks(self, events: List[Dict]) -> None:
        """Update streak counters."""
        event_names = [e["event"] for e in events]

        if "smooth_driving" in event_names:
            self.streaks["smooth_streak"] += 1
        elif any(e in event_names for e in ["harsh_braking", "harsh_acceleration"]):
            self.streaks["smooth_streak"] = 0

        if "excellent_regen" in event_names:
            self.streaks["regen_streak"] += 1
        elif "low_regen" in event_names:
            self.streaks["regen_streak"] = 0

        if not any(e in event_names for e in ["battery_overheat", "motor_overheat"]):
            self.streaks["no_overheat_streak"] += 1
        else:
            self.streaks["no_overheat_streak"] = 0

    def _check_badges(self) -> List[str]:
        """Check for new badges."""
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
        """Calculate score for current telemetry."""
        events = []
        score_delta = 0

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

        self.total_score += score_delta
        self.session_events.extend(events)
        self._update_streaks(events)
        new_badges = self._check_badges()
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


# =============================================================================
# MLPipeline - Unified interface
# =============================================================================


class MLPipeline:
    """
    Unified ML Pipeline combining anomaly detection and scoring.
    Uses ml_pipeline models for production-grade predictions.
    """

    def __init__(self, vehicle_id: str, driver_name: str = "Driver"):
        """Initialize the complete ML pipeline."""
        self.vehicle_id = vehicle_id
        self.detector = AdvancedAnomalyDetector(vehicle_id)
        self.scorer = ScoringEngine(driver_name)

    def train(
        self, historical_data: pd.DataFrame, driver_profile: str = "normal"
    ) -> Dict:
        """Train baseline stats (models are pre-trained)."""
        return self.detector.train(historical_data, driver_profile)

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Process telemetry through the full pipeline."""
        anomaly_result = self.detector.predict(telemetry)
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
            # Performance
            "inference_latency_ms": self.detector.last_inference_latency_ms,
        }

    def save(self, path: str) -> None:
        """Save pipeline state."""
        self.detector.save(path)

    def load(self, path: str) -> None:
        """Load pipeline state."""
        self.detector.load(path)


if __name__ == "__main__":
    print("🔬 Testing Advanced ML Pipeline v2.0")
    print("=" * 50)

    # Test telemetry
    test_telemetry = {
        "speed_kph": 85.0,
        "motor_rpm": 5500,
        "motor_temp_c": 65.0,
        "inverter_temp_c": 55.0,
        "battery_soc_pct": 72.0,
        "battery_voltage_v": 400,
        "battery_current_a": 120,
        "battery_temp_c": 32.0,
        "battery_cell_delta_v": 0.03,
        "throttle_pct": 45,
        "brake_pct": 0,
        "regen_pct": 0.75,
        "accel_x": 0.1,
        "accel_y": 0.05,
        "accel_z": 1.0,
    }

    # Test pipeline
    pipeline = MLPipeline("TEST_VIN_001")

    # Process multiple frames to build up window
    print("\nProcessing 60 test frames...")
    for i in range(60):
        frame = test_telemetry.copy()
        frame["speed_kph"] += np.random.randn() * 5
        frame["battery_soc_pct"] -= i * 0.1
        result = pipeline.process(frame)

    print(f"\n✅ Final result:")
    print(f"   Is anomaly: {result['is_anomaly']}")
    print(f"   Severity: {result['severity']}")
    print(f"   Failure risk: {result['failure_risk_pct']:.1f}%")
    print(f"   Score: {result['total_score']} (delta: {result['score_delta']})")
    print(f"   Latency: {result['inference_latency_ms']:.2f}ms")

    print("\n✅ ML Pipeline v2.0 test complete")
