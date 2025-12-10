"""
SentinEV - Adaptive Threshold Manager
======================================
Auto-adjusts thresholds based on fleet-wide baseline drift.
"""

import time
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdConfig:
    """Threshold configuration with adaptive bounds."""

    name: str
    default_value: float
    current_value: float
    min_value: float
    max_value: float
    last_adjusted: float = 0


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for anomaly and failure detection.

    Adjusts thresholds based on:
    - Fleet-wide prediction distribution drift
    - Alert fatigue (too many alerts)
    - Missed failures (too few alerts)
    """

    def __init__(self, base_thresholds: Dict = None):
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.history: Dict[str, list] = {}
        self.adjustment_interval = 3600  # Check every hour
        self.last_adjustment = 0

        # Initialize from base thresholds or defaults
        base = base_thresholds or {
            "failure_prob": 0.5,
            "anomaly_score": 0.5,
            "severity_critical": 0.9,
            "severity_high": 0.7,
        }

        for name, value in base.items():
            self.thresholds[name] = ThresholdConfig(
                name=name,
                default_value=value,
                current_value=value,
                min_value=value * 0.5,  # Can go down to 50%
                max_value=min(value * 1.5, 0.99),  # Can go up to 150% or 0.99
            )
            self.history[name] = []

    def get_threshold(self, name: str) -> float:
        """Get current threshold value."""
        if name in self.thresholds:
            return self.thresholds[name].current_value
        return 0.5  # Default

    def record_prediction(self, name: str, value: float):
        """Record a prediction for drift tracking."""
        if name not in self.history:
            self.history[name] = []

        self.history[name].append({"value": value, "timestamp": time.time()})

        # Keep only last 1000 predictions
        if len(self.history[name]) > 1000:
            self.history[name] = self.history[name][-1000:]

    def compute_drift(self, name: str, baseline_mean: float = 0.3) -> float:
        """Compute drift from baseline for a metric."""
        if name not in self.history or len(self.history[name]) < 100:
            return 0.0

        recent = self.history[name][-100:]
        current_mean = np.mean([r["value"] for r in recent])

        if baseline_mean == 0:
            return 0.0

        drift = (current_mean - baseline_mean) / baseline_mean
        return drift

    def adjust_thresholds(
        self, alert_rate: float = 0.1, target_alert_rate: float = 0.05
    ):
        """
        Adjust thresholds based on alert patterns.

        Args:
            alert_rate: Current fraction of predictions that trigger alerts
            target_alert_rate: Desired alert rate (default 5%)
        """
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return  # Too soon for adjustment

        self.last_adjustment = now

        # If too many alerts, raise thresholds
        if alert_rate > target_alert_rate * 2:
            adjustment = 1.05  # Raise by 5%
            reason = "high_alert_rate"
        # If too few alerts (might be missing issues), lower thresholds
        elif alert_rate < target_alert_rate * 0.5:
            adjustment = 0.98  # Lower by 2%
            reason = "low_alert_rate"
        else:
            adjustment = 1.0
            reason = "stable"

        for name, config in self.thresholds.items():
            if adjustment != 1.0:
                new_value = config.current_value * adjustment
                new_value = max(config.min_value, min(config.max_value, new_value))
                config.current_value = new_value
                config.last_adjusted = now

        return {
            "adjustment": adjustment,
            "reason": reason,
            "alert_rate": alert_rate,
            "target_rate": target_alert_rate,
        }

    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current thresholds."""
        return {name: config.current_value for name, config in self.thresholds.items()}

    def reset_to_defaults(self):
        """Reset all thresholds to defaults."""
        for name, config in self.thresholds.items():
            config.current_value = config.default_value
            config.last_adjusted = time.time()


# Singleton
_threshold_manager: Optional[AdaptiveThresholdManager] = None


def get_threshold_manager(base_thresholds: Dict = None) -> AdaptiveThresholdManager:
    """Get or create threshold manager."""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = AdaptiveThresholdManager(base_thresholds)
    return _threshold_manager


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adaptive Threshold Manager")
    print("=" * 60)

    # Load thresholds from ml_pipeline if available
    thresholds_path = Path("ml_pipeline/models/thresholds.json")
    base = None
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            data = json.load(f)
            base = {
                "failure_prob": data.get("failure_predictor", {}).get(
                    "high_precision_threshold", 0.5
                )
            }

    manager = get_threshold_manager(base)

    print(f"\n📊 Current thresholds:")
    for name, value in manager.get_all_thresholds().items():
        print(f"   {name}: {value:.4f}")

    # Simulate predictions
    for i in range(200):
        manager.record_prediction("failure_prob", np.random.random() * 0.5)

    # Test drift computation
    drift = manager.compute_drift("failure_prob", baseline_mean=0.25)
    print(f"\n📈 Drift from baseline: {drift:.2%}")

    # Test adjustment (simulate high alert rate)
    manager.last_adjustment = 0  # Force adjustment
    result = manager.adjust_thresholds(alert_rate=0.15, target_alert_rate=0.05)
    print(f"\n🔧 Adjustment: {result}")
    print(f"   New thresholds: {manager.get_all_thresholds()}")

    print("\n✅ Adaptive Threshold Manager test complete")
