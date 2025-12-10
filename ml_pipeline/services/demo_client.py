"""
Demo Client for Real-time Telemetry Inference
==============================================
Simulates vehicle telemetry streaming and displays predictions.

Usage:
    python demo_client.py --vehicle EV_001 --mode failure
"""

import json
import asyncio
import argparse
import random
import time
from pathlib import Path

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def generate_normal_frame(vehicle_id: str, timestamp: int) -> dict:
    """Generate a normal telemetry frame."""
    return {
        "vehicle_id": vehicle_id,
        "timestamp": timestamp,
        "speed_kph": random.uniform(30, 80),
        "motor_rpm": random.uniform(3000, 8000),
        "motor_temp_c": random.uniform(35, 55),
        "inverter_temp_c": random.uniform(30, 45),
        "battery_soc_pct": random.uniform(40, 90),
        "battery_voltage_v": random.uniform(360, 400),
        "battery_current_a": random.uniform(-50, 200),
        "battery_temp_c": random.uniform(20, 35),
        "battery_cell_delta_v": random.uniform(0.02, 0.08),
        "hvac_power_kw": random.uniform(0, 3),
        "throttle_pct": random.randint(10, 60),
        "brake_pct": 0,
        "regen_pct": 0,
        "accel_x": random.gauss(0, 0.1),
        "accel_y": random.gauss(0, 0.05),
        "accel_z": random.gauss(1.0, 0.05),
        "charging_state": "discharging",
        "dtc_codes": [],
    }


def generate_failure_frame(vehicle_id: str, timestamp: int, progress: float) -> dict:
    """Generate a frame with failure symptoms."""
    frame = generate_normal_frame(vehicle_id, timestamp)

    # Add failure symptoms that increase with progress
    frame["motor_temp_c"] += 30 * progress
    frame["inverter_temp_c"] += 20 * progress
    frame["battery_cell_delta_v"] += 0.15 * progress
    frame["accel_x"] *= 1 + 2 * progress  # Increased vibration
    frame["accel_y"] *= 1 + 2 * progress

    if progress > 0.7:
        frame["dtc_codes"] = ["P0A78"]  # Inverter overheating

    return frame


async def run_websocket_demo(vehicle_id: str, mode: str, host: str, port: int):
    """Run WebSocket-based demo."""
    if not HAS_WEBSOCKETS:
        print("ERROR: 'websockets' package not installed. Run: pip install websockets")
        return

    uri = f"ws://{host}:{port}/ws/telemetry"
    print(f"\nConnecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Streaming telemetry...\n")

            start_time = int(time.time())
            frame_count = 0

            # Simulate 5 minutes of telemetry (1 frame per second)
            for i in range(300):
                timestamp = start_time + i

                if mode == "failure":
                    progress = i / 300  # Gradually increase failure symptoms
                    frame = generate_failure_frame(vehicle_id, timestamp, progress)
                else:
                    frame = generate_normal_frame(vehicle_id, timestamp)

                await websocket.send(json.dumps(frame))
                frame_count += 1

                # Check for responses
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(response)

                    if data.get("type") == "prediction":
                        pred = data.get("prediction", {})
                        prob = pred.get("failure_probability", 0)
                        severity = pred.get("severity", "low")

                        print(
                            f"[{timestamp}] Prediction - Failure Prob: {prob:.3f}, Severity: {severity}"
                        )

                        if "alert" in data:
                            alert = data["alert"]
                            print(f"  🚨 ALERT: {alert['message']}")
                            print(f"     Action: {alert['recommended_action']}")

                except asyncio.TimeoutError:
                    pass

                if (i + 1) % 60 == 0:
                    print(f"  Sent {frame_count} frames...")

                await asyncio.sleep(0.01)  # Fast demo mode

            print(f"\nDemo complete. Sent {frame_count} frames.")

    except Exception as e:
        print(f"Connection error: {e}")


def run_http_demo(vehicle_id: str, mode: str, host: str, port: int):
    """Run HTTP-based demo (fallback if websockets unavailable)."""
    import requests

    url = f"http://{host}:{port}/telemetry"
    print(f"\nSending telemetry to {url}...")

    start_time = int(time.time())

    for i in range(100):
        timestamp = start_time + i

        if mode == "failure":
            progress = i / 100
            frame = generate_failure_frame(vehicle_id, timestamp, progress)
        else:
            frame = generate_normal_frame(vehicle_id, timestamp)

        try:
            response = requests.post(url, json=frame, timeout=5)
            data = response.json()

            if "prediction" in data:
                pred = data["prediction"]
                prob = pred.get("failure_probability", 0)
                print(f"[{i}] Failure Prob: {prob:.3f}")

            if "alert" in data:
                print(f"  🚨 ALERT: {data['alert']['message']}")

        except Exception as e:
            print(f"Error: {e}")
            break

        time.sleep(0.05)

    print("\nDemo complete.")


def main():
    parser = argparse.ArgumentParser(description="Demo client for telemetry inference")
    parser.add_argument("--vehicle", type=str, default="EV_001", help="Vehicle ID")
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal", "failure"],
        help="Simulation mode",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--http", action="store_true", help="Use HTTP instead of WebSocket"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("EV Telemetry Demo Client")
    print("=" * 50)
    print(f"Vehicle: {args.vehicle}")
    print(f"Mode: {args.mode}")
    print("=" * 50)

    if args.http:
        run_http_demo(args.vehicle, args.mode, args.host, args.port)
    else:
        asyncio.run(run_websocket_demo(args.vehicle, args.mode, args.host, args.port))


if __name__ == "__main__":
    main()
