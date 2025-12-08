"""
SentinelEY - Personalized Digital Twin ML Model
Trains a per-vehicle ML model for anomaly detection and failure prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path

from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib


@dataclass
class PredictionResult:
    """Result from real-time prediction."""

    anomaly_score: float  # -1 to 1 (higher = more anomalous)
    is_anomaly: bool
    failure_risk_percent: float  # 0-100%
    time_to_failure_hours: Optional[float]
    anomaly_type: str
    severity: str  # low, medium, high, critical
    suggested_action: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class DigitalTwinModel:
    """
    Personalized ML model for each vehicle.

    Learns the vehicle's "normal" behavior pattern and detects anomalies
    based on driver behavior, vehicle manual constraints, and industry faults.

    Components:
    - Isolation Forest: Unsupervised anomaly detection
    - Random Forest Regressor: Failure prediction
    - Standard Scaler: Feature normalization
    """

    # Feature columns used for ML
    FEATURE_COLUMNS = [
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "power_draw_kw",
        "regen_efficiency",
        "net_power_kw",
        "battery_temp_c",
        "battery_soc_percent",
        "wear_index",
        "efficiency_wh_km",
        "ambient_temp_c",
    ]

    # Thresholds for anomaly classification
    DEFAULT_THRESHOLDS = {
        "battery_temp_warning": 50,
        "battery_temp_critical": 60,
        "power_draw_warning": 150,
        "efficiency_warning": 200,  # Wh/km
        "wear_index_warning": 0.1,
        "jerk_warning": 8.0,
        "soc_critical": 10,
    }

    def __init__(
        self, vehicle_id: str, contamination: float = 0.05, n_estimators: int = 100
    ):
        """
        Initialize the Digital Twin Model.

        Args:
            vehicle_id: Unique vehicle identifier
            contamination: Expected proportion of outliers (default: 5%)
            n_estimators: Number of trees for ensemble models
        """
        self.vehicle_id = vehicle_id
        self.contamination = contamination
        self.n_estimators = n_estimators

        # ML Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )

        self.failure_predictor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )

        self.anomaly_classifier = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )

        self.scaler = StandardScaler()

        # State
        self.is_trained = False
        self.training_timestamp: Optional[datetime] = None
        self.training_samples: int = 0

        # Learned thresholds (personalized)
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()

        # Driver profile learned statistics
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

        # Vehicle manual constraints
        self.manual_constraints: Dict[str, Any] = {}

        # Industry fault patterns
        self.fault_patterns: List[Dict[str, Any]] = []

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix from telemetry data.

        Args:
            data: DataFrame with telemetry columns

        Returns:
            Feature matrix as numpy array
        """
        # Use available columns, fill missing with 0
        features = []
        for col in self.FEATURE_COLUMNS:
            if col in data.columns:
                features.append(data[col].values)
            else:
                features.append(np.zeros(len(data)))

        return np.column_stack(features)

    def _calculate_baseline_stats(self, data: pd.DataFrame):
        """
        Calculate baseline statistics for each feature.

        Args:
            data: Training data DataFrame
        """
        for col in self.FEATURE_COLUMNS:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    self.baseline_stats[col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                        "q95": float(values.quantile(0.95)),
                    }

    def _personalize_thresholds(self, driver_profile: str):
        """
        Adjust thresholds based on driver profile.

        Args:
            driver_profile: aggressive, eco, or normal
        """
        if driver_profile == "aggressive":
            # Aggressive drivers have higher normal ranges
            self.thresholds["battery_temp_warning"] = 55
            self.thresholds["jerk_warning"] = 12.0
            self.thresholds["power_draw_warning"] = 180
        elif driver_profile == "eco":
            # Eco drivers have tighter thresholds
            self.thresholds["battery_temp_warning"] = 45
            self.thresholds["jerk_warning"] = 5.0
            self.thresholds["power_draw_warning"] = 100

    def _create_failure_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create synthetic failure risk labels for supervised learning.

        Uses rule-based labeling based on known fault conditions.

        Args:
            data: Training data

        Returns:
            Array of failure risk scores (0-100)
        """
        risk_scores = np.zeros(len(data))

        # Battery temperature risk
        if "battery_temp_c" in data.columns:
            temp = data["battery_temp_c"].values
            risk_scores += np.clip((temp - 40) * 5, 0, 50)

        # High power draw risk
        if "power_draw_kw" in data.columns:
            power = data["power_draw_kw"].values
            risk_scores += np.clip((power - 100) * 0.3, 0, 20)

        # Wear index risk
        if "wear_index" in data.columns:
            wear = data["wear_index"].values
            risk_scores += np.clip(wear * 200, 0, 20)

        # Low SoC risk
        if "battery_soc_percent" in data.columns:
            soc = data["battery_soc_percent"].values
            risk_scores += np.clip((20 - soc) * 2, 0, 10)

        return np.clip(risk_scores, 0, 100)

    def load_vehicle_manual(self, manual_path: str):
        """
        Load vehicle manual constraints for training.

        Args:
            manual_path: Path to vehicle_manual.json
        """
        try:
            with open(manual_path, "r") as f:
                manual = json.load(f)

            self.manual_constraints = manual.get("components", {})

            # Update thresholds from manual
            battery = self.manual_constraints.get("battery", {})
            if "max_temp_c" in battery:
                self.thresholds["battery_temp_critical"] = battery["max_temp_c"]
                self.thresholds["battery_temp_warning"] = battery["max_temp_c"] - 10

            # Load warning thresholds
            thresholds = manual.get("warning_thresholds", {})
            for key, value in thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key] = value

        except Exception as e:
            print(f"Warning: Could not load vehicle manual: {e}")

    def load_industry_faults(self, faults_path: str):
        """
        Load industry fault patterns for training.

        Args:
            faults_path: Path to industry_faults.csv
        """
        try:
            faults_df = pd.read_csv(faults_path)
            self.fault_patterns = faults_df.to_dict("records")
        except Exception as e:
            print(f"Warning: Could not load industry faults: {e}")

    def train(
        self,
        historical_data: pd.DataFrame,
        driver_profile: str = "normal",
        vehicle_manual_path: Optional[str] = None,
        industry_faults_path: Optional[str] = None,
    ):
        """
        Train the Digital Twin Model.

        Args:
            historical_data: Historical telemetry DataFrame
            driver_profile: Driver behavior profile
            vehicle_manual_path: Optional path to vehicle manual
            industry_faults_path: Optional path to industry faults CSV
        """
        if len(historical_data) < 100:
            raise ValueError("Need at least 100 samples for training")

        # Load external knowledge
        if vehicle_manual_path:
            self.load_vehicle_manual(vehicle_manual_path)

        if industry_faults_path:
            self.load_industry_faults(industry_faults_path)

        # Personalize thresholds
        self._personalize_thresholds(driver_profile)

        # Calculate baseline statistics
        self._calculate_baseline_stats(historical_data)

        # Extract and scale features
        X = self._extract_features(historical_data)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest for anomaly detection
        self.isolation_forest.fit(X_scaled)

        # Create synthetic labels and train failure predictor
        y_risk = self._create_failure_labels(historical_data)
        self.failure_predictor.fit(X_scaled, y_risk)

        # Create anomaly type labels for classifier
        y_anomaly_type = self._label_anomaly_types(historical_data)
        if len(np.unique(y_anomaly_type)) > 1:
            self.anomaly_classifier.fit(X_scaled, y_anomaly_type)

        # Update state
        self.is_trained = True
        self.training_timestamp = datetime.now()
        self.training_samples = len(historical_data)

        print(f"✅ Model trained for vehicle {self.vehicle_id}")
        print(f"   Samples: {self.training_samples}")
        print(f"   Features: {len(self.FEATURE_COLUMNS)}")
        print(f"   Driver Profile: {driver_profile}")

    def _label_anomaly_types(self, data: pd.DataFrame) -> np.ndarray:
        """
        Label anomaly types based on telemetry patterns.

        Returns:
            Array of anomaly type labels (0=normal, 1=thermal, 2=power, 3=wear, etc.)
        """
        labels = np.zeros(len(data), dtype=int)

        if "battery_temp_c" in data.columns:
            labels[data["battery_temp_c"] > self.thresholds["battery_temp_warning"]] = (
                1  # thermal
            )

        if "power_draw_kw" in data.columns:
            labels[data["power_draw_kw"] > self.thresholds["power_draw_warning"]] = (
                2  # power
            )

        if "wear_index" in data.columns:
            labels[data["wear_index"] > self.thresholds["wear_index_warning"]] = (
                3  # wear
            )

        if "efficiency_wh_km" in data.columns:
            labels[data["efficiency_wh_km"] > self.thresholds["efficiency_warning"]] = (
                4  # efficiency
            )

        return labels

    def predict_realtime(self, current_telemetry: Dict[str, float]) -> PredictionResult:
        """
        Make real-time prediction on current telemetry.

        Args:
            current_telemetry: Dictionary of current sensor values

        Returns:
            PredictionResult with anomaly info and recommendations
        """
        if not self.is_trained:
            return PredictionResult(
                anomaly_score=0.0,
                is_anomaly=False,
                failure_risk_percent=0.0,
                time_to_failure_hours=None,
                anomaly_type="unknown",
                severity="low",
                suggested_action="Model not trained yet. Please train first.",
                confidence=0.0,
            )

        # Create DataFrame from single reading
        df = pd.DataFrame([current_telemetry])

        # Extract and scale features
        X = self._extract_features(df)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        # Isolation Forest anomaly score
        # Returns -1 for anomalies, 1 for normal
        if_score = self.isolation_forest.decision_function(X_scaled)[0]
        if_prediction = self.isolation_forest.predict(X_scaled)[0]

        # Normalize score to 0-1 range (higher = more anomalous)
        anomaly_score = 1 - (if_score + 0.5) / 1.0
        anomaly_score = max(0, min(1, anomaly_score))

        is_anomaly = if_prediction == -1

        # Failure risk prediction
        failure_risk = self.failure_predictor.predict(X_scaled)[0]
        failure_risk = max(0, min(100, failure_risk))

        # Estimate time to failure (simple model)
        if failure_risk > 50:
            time_to_failure = max(1, (100 - failure_risk) * 2)  # hours
        else:
            time_to_failure = None

        # Classify anomaly type
        anomaly_type, severity = self._classify_anomaly(
            current_telemetry, anomaly_score
        )

        # Generate suggested action
        suggested_action = self._generate_action(
            anomaly_type, severity, current_telemetry
        )

        # Calculate confidence
        confidence = min(0.95, 0.5 + self.training_samples / 10000)

        # Z-score details
        z_scores = self._calculate_z_scores(current_telemetry)

        return PredictionResult(
            anomaly_score=round(anomaly_score, 3),
            is_anomaly=is_anomaly,
            failure_risk_percent=round(failure_risk, 1),
            time_to_failure_hours=(
                round(time_to_failure, 1) if time_to_failure else None
            ),
            anomaly_type=anomaly_type,
            severity=severity,
            suggested_action=suggested_action,
            confidence=round(confidence, 2),
            details={
                "z_scores": z_scores,
                "thresholds": self.thresholds,
                "isolation_forest_score": round(if_score, 3),
            },
        )

    def _calculate_z_scores(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """Calculate Z-scores for each feature."""
        z_scores = {}
        for col, value in telemetry.items():
            if col in self.baseline_stats:
                stats = self.baseline_stats[col]
                if stats["std"] > 0:
                    z = (value - stats["mean"]) / stats["std"]
                    z_scores[col] = round(z, 2)
        return z_scores

    def _classify_anomaly(
        self, telemetry: Dict[str, float], anomaly_score: float
    ) -> Tuple[str, str]:
        """
        Classify the type and severity of anomaly.

        Returns:
            Tuple of (anomaly_type, severity)
        """
        if anomaly_score < 0.3:
            return "normal", "low"

        # Check specific conditions
        battery_temp = telemetry.get("battery_temp_c", 25)
        power_draw = telemetry.get("power_draw_kw", 0)
        efficiency = telemetry.get("efficiency_wh_km", 150)
        wear_index = telemetry.get("wear_index", 0)
        jerk = telemetry.get("jerk_ms3", 0)
        soc = telemetry.get("battery_soc_percent", 100)

        # Determine anomaly type
        anomaly_type = "general"
        severity = "medium"

        if battery_temp > self.thresholds["battery_temp_critical"]:
            anomaly_type = "thermal_critical"
            severity = "critical"
        elif battery_temp > self.thresholds["battery_temp_warning"]:
            anomaly_type = "thermal_warning"
            severity = "high"
        elif power_draw > self.thresholds["power_draw_warning"]:
            anomaly_type = "power_anomaly"
            severity = "medium"
        elif efficiency > self.thresholds["efficiency_warning"]:
            anomaly_type = "efficiency_degradation"
            severity = "medium"
        elif wear_index > self.thresholds["wear_index_warning"]:
            anomaly_type = "mechanical_wear"
            severity = "medium"
        elif soc < self.thresholds["soc_critical"]:
            anomaly_type = "low_battery"
            severity = "high"
        elif jerk > self.thresholds["jerk_warning"]:
            anomaly_type = "harsh_driving"
            severity = "low"

        return anomaly_type, severity

    def _generate_action(
        self, anomaly_type: str, severity: str, telemetry: Dict[str, float]
    ) -> str:
        """Generate recommended action based on anomaly."""
        actions = {
            "normal": "Continue monitoring. All systems operating normally.",
            "thermal_critical": "STOP IMMEDIATELY! Battery overheating detected. Pull over safely and contact service.",
            "thermal_warning": "Reduce speed to cool battery. Avoid high acceleration.",
            "power_anomaly": "High power consumption detected. Check for dragging brakes or low tire pressure.",
            "efficiency_degradation": "Efficiency below normal. Schedule maintenance check for brakes and alignment.",
            "mechanical_wear": "Increased wear detected. Schedule preventive maintenance.",
            "low_battery": "Battery critically low! Find charging station immediately.",
            "harsh_driving": "Driving aggressively. Smooth acceleration saves energy and reduces wear.",
            "general": "Anomaly detected. Monitor closely and contact service if persists.",
        }

        return actions.get(anomaly_type, actions["general"])

    def save(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Directory path to save model files
        """
        os.makedirs(path, exist_ok=True)

        model_data = {
            "vehicle_id": self.vehicle_id,
            "is_trained": self.is_trained,
            "training_timestamp": (
                self.training_timestamp.isoformat() if self.training_timestamp else None
            ),
            "training_samples": self.training_samples,
            "thresholds": self.thresholds,
            "baseline_stats": self.baseline_stats,
            "manual_constraints": self.manual_constraints,
        }

        # Save metadata
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(model_data, f, indent=2)

        # Save sklearn models
        joblib.dump(
            self.isolation_forest, os.path.join(path, "isolation_forest.joblib")
        )
        joblib.dump(
            self.failure_predictor, os.path.join(path, "failure_predictor.joblib")
        )
        joblib.dump(
            self.anomaly_classifier, os.path.join(path, "anomaly_classifier.joblib")
        )
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))

        print(f"✅ Model saved to {path}")

    def load(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing model files
        """
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            model_data = json.load(f)

        self.vehicle_id = model_data["vehicle_id"]
        self.is_trained = model_data["is_trained"]
        self.training_timestamp = (
            datetime.fromisoformat(model_data["training_timestamp"])
            if model_data["training_timestamp"]
            else None
        )
        self.training_samples = model_data["training_samples"]
        self.thresholds = model_data["thresholds"]
        self.baseline_stats = model_data["baseline_stats"]
        self.manual_constraints = model_data.get("manual_constraints", {})

        # Load sklearn models
        self.isolation_forest = joblib.load(
            os.path.join(path, "isolation_forest.joblib")
        )
        self.failure_predictor = joblib.load(
            os.path.join(path, "failure_predictor.joblib")
        )
        self.anomaly_classifier = joblib.load(
            os.path.join(path, "anomaly_classifier.joblib")
        )
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))

        print(f"✅ Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and statistics."""
        return {
            "vehicle_id": self.vehicle_id,
            "is_trained": self.is_trained,
            "training_timestamp": (
                self.training_timestamp.isoformat() if self.training_timestamp else None
            ),
            "training_samples": self.training_samples,
            "feature_columns": self.FEATURE_COLUMNS,
            "thresholds": self.thresholds,
            "baseline_stats": self.baseline_stats,
        }


class FleetDigitalTwinManager:
    """Manages Digital Twin models for an entire fleet."""

    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize fleet manager.

        Args:
            models_dir: Directory to store trained models
        """
        self.models_dir = models_dir
        self.models: Dict[str, DigitalTwinModel] = {}
        os.makedirs(models_dir, exist_ok=True)

    def get_model(self, vehicle_id: str) -> DigitalTwinModel:
        """
        Get or create a model for a vehicle.

        Args:
            vehicle_id: Vehicle identifier

        Returns:
            DigitalTwinModel instance
        """
        if vehicle_id not in self.models:
            model = DigitalTwinModel(vehicle_id)

            # Try to load existing model
            model_path = os.path.join(self.models_dir, vehicle_id)
            if os.path.exists(model_path):
                try:
                    model.load(model_path)
                except Exception as e:
                    print(f"Warning: Could not load model for {vehicle_id}: {e}")

            self.models[vehicle_id] = model

        return self.models[vehicle_id]

    def train_model(
        self,
        vehicle_id: str,
        historical_data: pd.DataFrame,
        driver_profile: str = "normal",
        save: bool = True,
    ):
        """
        Train model for a specific vehicle.

        Args:
            vehicle_id: Vehicle identifier
            historical_data: Training data
            driver_profile: Driver behavior profile
            save: Whether to save the model after training
        """
        model = self.get_model(vehicle_id)

        # Get paths for knowledge files
        base_path = Path(__file__).parent.parent / "data" / "datasets"
        manual_path = base_path / "vehicle_manual.json"
        faults_path = base_path / "industry_faults.csv"

        model.train(
            historical_data,
            driver_profile=driver_profile,
            vehicle_manual_path=str(manual_path) if manual_path.exists() else None,
            industry_faults_path=str(faults_path) if faults_path.exists() else None,
        )

        if save:
            model_path = os.path.join(self.models_dir, vehicle_id)
            model.save(model_path)

    def predict(self, vehicle_id: str, telemetry: Dict[str, float]) -> PredictionResult:
        """
        Make prediction for a vehicle.

        Args:
            vehicle_id: Vehicle identifier
            telemetry: Current telemetry reading

        Returns:
            PredictionResult
        """
        model = self.get_model(vehicle_id)
        return model.predict_realtime(telemetry)

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models in the fleet."""
        summary = {"total_models": len(self.models), "trained_models": 0, "models": []}

        for vid, model in self.models.items():
            info = model.get_model_info()
            summary["models"].append(info)
            if model.is_trained:
                summary["trained_models"] += 1

        return summary


if __name__ == "__main__":
    # Demo: Train and test a model
    from physics import VehicleSimulator

    print("🤖 SentinelEY Digital Twin Model Demo")
    print("=" * 50)

    # Create simulator and generate training data
    simulator = VehicleSimulator(
        vehicle_id="DEMO-001", driver_profile="aggressive", weather_condition="hot"
    )

    print("\n📊 Generating training data (7 days)...")
    training_data = simulator.generate_history(days=7)
    print(f"   Generated {len(training_data)} samples")

    # Create and train model
    model = DigitalTwinModel(vehicle_id="DEMO-001")

    print("\n🎯 Training model...")
    model.train(training_data, driver_profile="aggressive")

    # Test prediction
    print("\n🔮 Testing real-time prediction...")
    simulator.reset()

    for i in range(3):
        reading = simulator.step(dt_seconds=60)
        result = model.predict_realtime(reading)

        print(f"\n   Reading {i+1}:")
        print(f"   - Speed: {reading['speed_kmh']:.1f} km/h")
        print(f"   - Battery Temp: {reading['battery_temp_c']:.1f}°C")
        print(f"   - Anomaly Score: {result.anomaly_score:.3f}")
        print(f"   - Is Anomaly: {result.is_anomaly}")
        print(f"   - Failure Risk: {result.failure_risk_percent:.1f}%")
        print(f"   - Type: {result.anomaly_type}")
        print(f"   - Action: {result.suggested_action}")

    # Test with fault injection
    print("\n⚠️ Testing fault injection...")
    fault_reading = simulator.inject_fault("overheat", severity=1.5)
    fault_result = model.predict_realtime(fault_reading)

    print(f"   Injected fault: overheat")
    print(f"   - Battery Temp: {fault_reading['battery_temp_c']:.1f}°C")
    print(f"   - Anomaly Score: {fault_result.anomaly_score:.3f}")
    print(f"   - Is Anomaly: {fault_result.is_anomaly}")
    print(f"   - Type: {fault_result.anomaly_type}")
    print(f"   - Severity: {fault_result.severity}")
    print(f"   - Action: {fault_result.suggested_action}")
