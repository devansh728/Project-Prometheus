"""
SentinEV - Streaming Infrastructure
====================================
Kafka-based telemetry streaming with window buffers for real-time inference.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd

# Kafka imports (optional - fallback to simulation if not available)
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("⚠️ aiokafka not installed. Using simulated Kafka.")


# =============================================================================
# Window Buffer for per-vehicle telemetry aggregation
# =============================================================================


@dataclass
class WindowConfig:
    """Configuration for sliding window."""

    short_window_seconds: int = 60  # 1 minute
    medium_window_seconds: int = 300  # 5 minutes
    long_window_seconds: int = 3600  # 1 hour
    inference_stride_seconds: int = 60  # Inference every 60 seconds


class VehicleWindowBuffer:
    """
    Per-vehicle buffer for computing windowed aggregate features.

    Maintains sliding windows of telemetry frames and computes
    aggregate features (mean, std, min, max) for inference.
    """

    NUMERIC_COLUMNS = [
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

    def __init__(self, vehicle_id: str, config: WindowConfig = None):
        self.vehicle_id = vehicle_id
        self.config = config or WindowConfig()
        self.frames: List[Dict] = []
        self.last_inference_ts: float = 0
        self.frame_count: int = 0

    def add_frame(self, frame: Dict) -> Optional[pd.DataFrame]:
        """
        Add telemetry frame to buffer.

        Returns aggregate features if inference is due, else None.
        """
        frame["_arrival_time"] = time.time()
        self.frames.append(frame)
        self.frame_count += 1

        # Prune old frames (keep 2x longest window)
        max_age = self.config.long_window_seconds * 2
        cutoff = time.time() - max_age
        self.frames = [f for f in self.frames if f.get("_arrival_time", 0) > cutoff]

        # Check if inference is due
        # Convert timestamp to float if string (ISO format from WebSocket)
        ts_raw = frame.get("timestamp", time.time())
        if isinstance(ts_raw, str):
            try:
                from datetime import datetime

                current_ts = datetime.fromisoformat(
                    ts_raw.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                current_ts = time.time()
        else:
            current_ts = float(ts_raw) if ts_raw else time.time()

        if current_ts - self.last_inference_ts >= self.config.inference_stride_seconds:
            if len(self.frames) >= 60:  # Minimum frames for inference
                self.last_inference_ts = current_ts
                return self._compute_aggregate_features()

        return None

    def _compute_aggregate_features(self) -> pd.DataFrame:
        """Compute aggregate features for all window sizes."""
        df = pd.DataFrame(self.frames)
        features = {}

        now = time.time()
        windows = [
            ("short", self.config.short_window_seconds),
            ("medium", self.config.medium_window_seconds),
            ("long", self.config.long_window_seconds),
        ]

        for window_name, window_seconds in windows:
            cutoff = now - window_seconds
            window_df = df[df["_arrival_time"] > cutoff]

            if len(window_df) < 5:
                continue

            for col in self.NUMERIC_COLUMNS:
                if col in window_df.columns:
                    series = pd.to_numeric(window_df[col], errors="coerce").dropna()
                    if len(series) > 0:
                        prefix = f"{col}"
                        features[f"{prefix}_mean"] = series.mean()
                        features[f"{prefix}_std"] = (
                            series.std() if len(series) > 1 else 0
                        )
                        features[f"{prefix}_min"] = series.min()
                        features[f"{prefix}_max"] = series.max()

        # Derived features
        if (
            "battery_voltage_v_mean" in features
            and "battery_current_a_mean" in features
        ):
            features["power_kw_mean"] = (
                features["battery_voltage_v_mean"]
                * features["battery_current_a_mean"]
                / 1000
            )

        if all(f"accel_{axis}_mean" in features for axis in ["x", "y", "z"]):
            features["accel_magnitude_mean"] = np.sqrt(
                features["accel_x_mean"] ** 2
                + features["accel_y_mean"] ** 2
                + features["accel_z_mean"] ** 2
            )

        features["vehicle_id"] = self.vehicle_id
        features["timestamp"] = now
        features["frame_count"] = self.frame_count

        return pd.DataFrame([features])

    def get_sequence_for_lstm(self, seq_len: int = 60) -> Optional[np.ndarray]:
        """Get sequence of recent frames for LSTM inference."""
        if len(self.frames) < seq_len:
            return None

        recent = self.frames[-seq_len:]
        df = pd.DataFrame(recent)

        # Extract numeric columns only
        features = []
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                features.append(
                    pd.to_numeric(df[col], errors="coerce").fillna(0).values
                )

        if not features:
            return None

        sequence = np.column_stack(features)  # Shape: (seq_len, n_features)
        return sequence


# =============================================================================
# Kafka Streaming Components
# =============================================================================


class TelemetryKafkaConsumer:
    """
    Kafka consumer for telemetry streaming.

    Routes messages to workers based on vehicle_id hash.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "sentinev.telemetry",
        group_id: str = "inference-workers",
        num_workers: int = 4,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.num_workers = num_workers
        self.consumer = None
        self.running = False
        self.message_handlers: Dict[int, Callable] = {}

    def register_worker(self, worker_id: int, handler: Callable):
        """Register a message handler for a worker."""
        self.message_handlers[worker_id] = handler

    def _route_to_worker(self, vehicle_id: str) -> int:
        """Consistent hash routing to worker."""
        hash_val = int(hashlib.md5(vehicle_id.encode()).hexdigest(), 16)
        return hash_val % self.num_workers

    async def start(self):
        """Start consuming messages."""
        if not KAFKA_AVAILABLE:
            print("⚠️ Kafka not available. Use simulated mode.")
            return

        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.running = True

        await self.consumer.start()
        print(f"✅ Kafka consumer started on {self.topic}")

        try:
            async for msg in self.consumer:
                if not self.running:
                    break

                data = msg.value
                vehicle_id = data.get("vehicle_id", "unknown")
                worker_id = self._route_to_worker(vehicle_id)

                if worker_id in self.message_handlers:
                    await self.message_handlers[worker_id](data)
        finally:
            await self.consumer.stop()

    async def stop(self):
        """Stop consuming."""
        self.running = False


class AlertKafkaProducer:
    """Kafka producer for alerts."""

    def __init__(
        self, bootstrap_servers: str = "localhost:9092", topic: str = "sentinev.alerts"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None

    async def start(self):
        """Start producer."""
        if not KAFKA_AVAILABLE:
            print("⚠️ Kafka not available. Alerts will be logged only.")
            return

        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self.producer.start()
        print(f"✅ Kafka producer started for {self.topic}")

    async def publish_alert(self, alert: Dict):
        """Publish alert to Kafka."""
        if self.producer:
            await self.producer.send_and_wait(self.topic, alert)
        else:
            print(
                f"🚨 Alert (local): {alert.get('severity', 'unknown')} - {alert.get('vehicle_id', 'unknown')}"
            )

    async def stop(self):
        """Stop producer."""
        if self.producer:
            await self.producer.stop()


# =============================================================================
# Simulated Kafka for development/testing
# =============================================================================


class SimulatedKafkaStream:
    """
    Simulated Kafka stream for development when Kafka is not available.
    Uses asyncio queues internally.
    """

    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    async def produce(self, topic: str, message: Dict):
        """Produce message to topic."""
        await self.queues[topic].put(message)

        # Notify subscribers
        for handler in self.subscribers[topic]:
            asyncio.create_task(handler(message))

    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic."""
        self.subscribers[topic].append(handler)

    async def consume(self, topic: str) -> Dict:
        """Consume message from topic."""
        return await self.queues[topic].get()


# Global simulated stream instance
_simulated_stream: Optional[SimulatedKafkaStream] = None


def get_simulated_stream() -> SimulatedKafkaStream:
    """Get or create simulated stream."""
    global _simulated_stream
    if _simulated_stream is None:
        _simulated_stream = SimulatedKafkaStream()
    return _simulated_stream


# =============================================================================
# Event-Time Synchronization Buffer
# =============================================================================


class EventTimeBuffer:
    """
    Buffer for handling out-of-order events.
    Holds events up to max_delay seconds before processing.
    """

    def __init__(self, max_delay_seconds: float = 30.0):
        self.max_delay = max_delay_seconds
        self.buffer: List[Dict] = []
        self.watermark: float = 0  # Lowest event time we've seen recently

    def add_event(self, event: Dict) -> List[Dict]:
        """
        Add event to buffer.

        Returns list of events ready to be processed (past watermark).
        """
        event_time = event.get("timestamp", time.time())
        self.buffer.append(event)

        # Update watermark
        current_time = time.time()
        new_watermark = current_time - self.max_delay

        if new_watermark > self.watermark:
            self.watermark = new_watermark

        # Extract events that are past the watermark
        ready_events = [
            e for e in self.buffer if e.get("timestamp", 0) <= self.watermark
        ]
        self.buffer = [e for e in self.buffer if e.get("timestamp", 0) > self.watermark]

        # Sort by event time
        ready_events.sort(key=lambda x: x.get("timestamp", 0))

        return ready_events


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Streaming Components")
    print("=" * 60)

    # Test window buffer
    buffer = VehicleWindowBuffer("TEST_001")

    # Simulate 100 frames
    for i in range(100):
        frame = {
            "timestamp": time.time(),
            "vehicle_id": "TEST_001",
            "speed_kph": 50 + np.random.randn() * 10,
            "motor_rpm": 3000 + np.random.randn() * 500,
            "motor_temp_c": 45 + np.random.randn() * 5,
            "battery_soc_pct": 80 - i * 0.1,
            "battery_voltage_v": 400,
            "battery_current_a": 50,
            "battery_temp_c": 30,
            "battery_cell_delta_v": 0.05,
            "accel_x": np.random.randn() * 0.1,
            "accel_y": np.random.randn() * 0.1,
            "accel_z": 1.0,
        }

        result = buffer.add_frame(frame)
        if result is not None:
            print(f"\n✅ Inference triggered at frame {i+1}")
            print(f"   Features computed: {len(result.columns)}")
            print(f"   Sample: speed_mean={result['speed_kph_mean'].values[0]:.1f}")

    print("\n✅ Streaming components test complete")
