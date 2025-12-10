"""
Monitoring & Drift Detection Service
=====================================
Tracks inference metrics, detects data/model drift, and triggers retraining alerts.

Usage:
    from monitoring import MetricsLogger, DriftDetector
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np


BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class InferenceMetric:
    """Single inference metric record."""

    timestamp: float
    vehicle_id: str
    failure_probability: float
    severity: str
    latency_ms: float
    window_size: int


@dataclass
class DriftAlert:
    """Drift detection alert."""

    timestamp: float
    drift_type: str  # "data_drift", "prediction_drift", "performance_drift"
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_pct: float
    severity: str
    recommendation: str


class MetricsLogger:
    """Logs inference metrics for monitoring."""

    def __init__(self, log_dir: Optional[Path] = None, buffer_size: int = 1000):
        self.log_dir = log_dir or LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.session_start = time.time()
        self.total_inferences = 0
        self.total_alerts = 0

        # Hourly aggregates
        self.hourly_stats = {"predictions": [], "latencies": [], "alert_counts": 0}

    def log_inference(self, metric: InferenceMetric):
        """Log a single inference."""
        self.metrics_buffer.append(asdict(metric))
        self.total_inferences += 1
        self.hourly_stats["predictions"].append(metric.failure_probability)
        self.hourly_stats["latencies"].append(metric.latency_ms)

        if metric.failure_probability > 0.5:
            self.total_alerts += 1
            self.hourly_stats["alert_counts"] += 1

    def get_stats(self) -> Dict:
        """Get current session statistics."""
        uptime = time.time() - self.session_start
        predictions = self.hourly_stats["predictions"]
        latencies = self.hourly_stats["latencies"]

        return {
            "uptime_seconds": uptime,
            "total_inferences": self.total_inferences,
            "total_alerts": self.total_alerts,
            "alert_rate": self.total_alerts / max(self.total_inferences, 1),
            "avg_prediction": np.mean(predictions) if predictions else 0,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
        }

    def flush_to_disk(self):
        """Write buffered metrics to disk."""
        if not self.metrics_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"metrics_{timestamp}.jsonl"

        with open(log_file, "w") as f:
            for metric in self.metrics_buffer:
                f.write(json.dumps(metric) + "\n")

        print(f"Flushed {len(self.metrics_buffer)} metrics to {log_file}")

    def reset_hourly(self):
        """Reset hourly aggregates."""
        self.hourly_stats = {"predictions": [], "latencies": [], "alert_counts": 0}


class DriftDetector:
    """Detects data and model drift."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline_stats: Dict = {}
        self.current_window: deque = deque(maxlen=window_size)
        self.drift_threshold = 0.2  # 20% deviation triggers alert
        self.alerts: List[DriftAlert] = []

    def set_baseline(self, stats: Dict):
        """Set baseline statistics from training data."""
        self.baseline_stats = stats
        print(f"Baseline set: {stats}")

    def add_observation(self, features: Dict):
        """Add observation to current window."""
        self.current_window.append(features)

    def check_drift(self) -> List[DriftAlert]:
        """Check for drift against baseline."""
        if len(self.current_window) < self.window_size // 2:
            return []

        alerts = []

        # Compute current statistics
        current_df = list(self.current_window)

        for key, baseline_value in self.baseline_stats.items():
            if baseline_value == 0:
                continue

            current_values = [obs.get(key, 0) for obs in current_df if key in obs]
            if not current_values:
                continue

            current_value = np.mean(current_values)
            deviation = abs(current_value - baseline_value) / abs(baseline_value)

            if deviation > self.drift_threshold:
                severity = "high" if deviation > 0.5 else "medium"

                alert = DriftAlert(
                    timestamp=time.time(),
                    drift_type="data_drift",
                    metric_name=key,
                    current_value=float(current_value),
                    baseline_value=float(baseline_value),
                    deviation_pct=float(deviation * 100),
                    severity=severity,
                    recommendation=f"Feature '{key}' has drifted {deviation*100:.1f}% from baseline. Consider retraining.",
                )
                alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    def check_prediction_drift(
        self, recent_predictions: List[float], baseline_rate: float = 0.5
    ) -> Optional[DriftAlert]:
        """Check if prediction distribution has shifted."""
        if len(recent_predictions) < 50:
            return None

        current_rate = np.mean([1 if p > 0.5 else 0 for p in recent_predictions])
        deviation = abs(current_rate - baseline_rate) / baseline_rate

        if deviation > self.drift_threshold:
            return DriftAlert(
                timestamp=time.time(),
                drift_type="prediction_drift",
                metric_name="positive_rate",
                current_value=float(current_rate),
                baseline_value=float(baseline_rate),
                deviation_pct=float(deviation * 100),
                severity="high" if deviation > 0.3 else "medium",
                recommendation="Prediction distribution has shifted significantly. Evaluate model and consider retraining.",
            )

        return None


class RetrainTrigger:
    """Manages automated retrain triggers."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "drift_alert_threshold": 3,  # Alerts before triggering
            "min_samples_for_retrain": 10000,
            "max_days_since_train": 30,
            "performance_drop_threshold": 0.1,
        }
        self.accumulated_alerts = 0
        self.last_train_timestamp = time.time()
        self.retrain_triggered = False

    def evaluate_trigger(self, drift_alerts: List[DriftAlert], metrics: Dict) -> Dict:
        """Evaluate if retrain should be triggered."""
        self.accumulated_alerts += len(drift_alerts)

        days_since_train = (time.time() - self.last_train_timestamp) / 86400

        reasons = []
        should_retrain = False

        # Check drift alert accumulation
        if self.accumulated_alerts >= self.config["drift_alert_threshold"]:
            reasons.append(f"Accumulated {self.accumulated_alerts} drift alerts")
            should_retrain = True

        # Check time since last training
        if days_since_train > self.config["max_days_since_train"]:
            reasons.append(f"{days_since_train:.0f} days since last training")
            should_retrain = True

        # Check sample count
        total_samples = metrics.get("total_inferences", 0)
        if (
            total_samples >= self.config["min_samples_for_retrain"]
            and not self.retrain_triggered
        ):
            reasons.append(f"Collected {total_samples} new samples")
            should_retrain = True

        result = {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "priority": "high" if len(reasons) > 1 else "medium",
            "accumulated_alerts": self.accumulated_alerts,
            "days_since_train": days_since_train,
        }

        if should_retrain:
            self.retrain_triggered = True

        return result

    def reset_after_retrain(self):
        """Reset triggers after retraining."""
        self.accumulated_alerts = 0
        self.last_train_timestamp = time.time()
        self.retrain_triggered = False


def create_monitoring_dashboard() -> str:
    """Generate ASCII monitoring dashboard."""
    logger = MetricsLogger()
    stats = logger.get_stats()

    dashboard = f"""
╔═══════════════════════════════════════════════════════════╗
║           EV TELEMETRY ML PIPELINE - MONITORING           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  📊 Session Statistics                                    ║
║  ─────────────────────                                    ║
║  Uptime:           {stats.get('uptime_seconds', 0):>10.0f}s                        ║
║  Total Inferences: {stats.get('total_inferences', 0):>10,}                         ║
║  Total Alerts:     {stats.get('total_alerts', 0):>10,}                         ║
║  Alert Rate:       {stats.get('alert_rate', 0)*100:>10.1f}%                        ║
║                                                           ║
║  ⚡ Performance                                            ║
║  ─────────────────────                                    ║
║  Avg Latency:      {stats.get('avg_latency_ms', 0):>10.1f}ms                       ║
║  P95 Latency:      {stats.get('p95_latency_ms', 0):>10.1f}ms                       ║
║                                                           ║
║  🔍 Model Performance                                     ║
║  ─────────────────────                                    ║
║  Avg Prediction:   {stats.get('avg_prediction', 0):>10.3f}                         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    return dashboard


def main():
    """Demo monitoring components."""
    print("=" * 60)
    print("Monitoring & Drift Detection Demo")
    print("=" * 60)

    # Initialize components
    logger = MetricsLogger()
    drift_detector = DriftDetector()
    retrain_trigger = RetrainTrigger()

    # Set baseline
    drift_detector.set_baseline(
        {"battery_temp_mean": 30.0, "motor_temp_mean": 45.0, "failure_probability": 0.3}
    )

    # Simulate some inferences
    print("\nSimulating inferences...")
    for i in range(100):
        metric = InferenceMetric(
            timestamp=time.time(),
            vehicle_id=f"EV_{i % 10 + 1:03d}",
            failure_probability=np.random.random() * 0.4,
            severity="low",
            latency_ms=np.random.uniform(5, 20),
            window_size=300,
        )
        logger.log_inference(metric)

        # Add observation for drift detection
        drift_detector.add_observation(
            {
                "battery_temp_mean": 30 + np.random.normal(0, 5),
                "motor_temp_mean": 45 + np.random.normal(0, 8),
            }
        )

    # Check for drift
    print("\nChecking for drift...")
    alerts = drift_detector.check_drift()
    if alerts:
        for alert in alerts:
            print(
                f"  ⚠️ {alert.drift_type}: {alert.metric_name} - {alert.deviation_pct:.1f}% deviation"
            )
    else:
        print("  ✅ No drift detected")

    # Check retrain trigger
    print("\nEvaluating retrain triggers...")
    trigger_result = retrain_trigger.evaluate_trigger(alerts, logger.get_stats())
    print(f"  Should retrain: {trigger_result['should_retrain']}")
    if trigger_result["reasons"]:
        for reason in trigger_result["reasons"]:
            print(f"    - {reason}")

    # Print dashboard
    print(create_monitoring_dashboard())


if __name__ == "__main__":
    main()
