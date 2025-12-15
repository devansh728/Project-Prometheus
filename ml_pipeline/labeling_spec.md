# Labeling Specification for EV Failure Prediction
# ================================================

## Overview
This document defines the labeling strategy for training ML models to detect anomalies 
and predict failures in electric vehicle telemetry data.

---

## 1. Anomaly Labels

### Definition
A telemetry window is labeled as anomaly=1 if ANY of the following conditions are met:

| Condition              | Threshold            | Description                                      |
|------------------------|----------------------|--------------------------------------------------|
| Reconstruction Error   | > 99th percentile    | LSTM-AE trained on healthy data flags anomalies  |
| DTC Codes Present      | count > 0            | Any diagnostic trouble code within the window    |
| Battery Temp           | > 55C or < -25C      | Extreme battery temperature                      |
| Inverter Temp          | > 100C               | Inverter approaching thermal limits              |
| Cell Voltage Delta     | > 0.3V               | Significant cell imbalance                       |
| Motor Temp Spike       | > 130C               | Motor overheating                                |

### Implementation (Python pseudocode):
    def label_anomaly(window):
        if reconstruction_error > threshold_99pct:
            return 1
        if len(window['dtc_codes'].dropna().sum()) > 0:
            return 1
        if window['battery_temp_c'].max() > 55:
            return 1
        if window['inverter_temp_c'].max() > 100:
            return 1
        if window['battery_cell_delta_v'].max() > 0.3:
            return 1
        return 0

---

## 2. Failure Labels

### Definition
A window is labeled failure=1 if a maintenance/failure event occurs within H days 
(horizon) after the window ends.

### Configurable Horizons
| Horizon    | Use Case                                    |
|------------|---------------------------------------------|
| H = 7 days | Short-term prediction for urgent maintenance|
| H = 14 days| Standard prediction window (default)        |
| H = 30 days| Long-term planning                          |

### Failure Events to Predict
| Event Type       | Example Events                                    |
|------------------|---------------------------------------------------|
| Battery Service  | Pack replacement, cell balancing, BMS reset       |
| Motor Service    | Bearing replacement, winding repair, coolant flush|
| Inverter Service | Module replacement, firmware update, thermal paste|
| Charging System  | Onboard charger repair, DC inlet replacement      |
| Thermal System   | Coolant pump, thermal management service          |

---

## 3. Severity Labels

### Multi-class Classification
| Severity   | Level | EV-Specific Failure Modes                                     |
|------------|-------|---------------------------------------------------------------|
| low        | 1     | Tire pressure adjustment, cabin air filter, software update  |
| medium     | 2     | 12V battery replacement, coolant top-up, brake pad replacement|
| high       | 3     | Motor bearing wear, inverter degradation, HV cable damage    |
| critical   | 4     | Battery pack failure, thermal runaway risk, drive unit failure|

### Severity Assignment Map
    SEVERITY_MAP = {
        # Critical (4)
        'battery_pack_replacement': 4,
        'thermal_runaway_incident': 4,
        'drive_unit_failure': 4,
        'tow_required': 4,
        
        # High (3)
        'motor_bearing_replacement': 3,
        'inverter_module_replacement': 3,
        'hv_battery_coolant_system': 3,
        'bms_replacement': 3,
        
        # Medium (2)
        '12v_battery_replacement': 2,
        'onboard_charger_repair': 2,
        'brake_system_service': 2,
        'suspension_repair': 2,
        
        # Low (1)
        'tire_service': 1,
        'wiper_replacement': 1,
        'software_update': 1,
        'cabin_filter': 1
    }

---

## 4. EV-Specific Failure Modes

| Failure Mode         | Lead Time | Telemetry Signature                                          |
|----------------------|-----------|--------------------------------------------------------------|
| Battery Degradation  | 72h       | SOC capacity fade, voltage sag under load, increased cell delta|
| Motor Bearing Wear   | 48h       | Vibration increase, efficiency drop, temperature spikes      |
| Inverter Overheating | 24h       | Inverter temp ramp, power derating, efficiency loss          |
| Thermal Runaway Risk | 6h        | Rapid battery temp increase, cell voltage divergence         |
| Charging Fault       | 48h       | Charging efficiency drop, current fluctuations               |
| Coolant System Fail  | 36h       | Rising temps despite low load, temperature oscillations      |

---

## 5. Train/Validation/Test Split

### Split Strategy
- Split by vehicle: Ensures model generalizes to unseen vehicles
- No data leakage: Test vehicles never seen during training

### Split Ratios
| Split      | Ratio | Vehicles (of 10) |
|------------|-------|------------------|
| Train      | 60%   | 6 vehicles       |
| Validation | 20%   | 2 vehicles       |
| Test       | 20%   | 2 vehicles       |

---

## 6. Window Labeling Structure

Each window gets multiple label columns:

    window_labels = {
        'window_id': 'w_00001',
        'vehicle_id': 'EV_001',
        'start_ts': 1702137600,
        'end_ts': 1702137660,
        
        # Anomaly labels
        'anomaly': 0,
        'anomaly_score': 0.12,
        
        # Failure labels (multi-horizon)
        'failure_7d': 0,
        'failure_14d': 1,
        'failure_30d': 1,
        
        # Severity (if failure predicted)
        'severity': 2,
        'failure_type': 'charging_fault'
    }

---

## 7. Edge Cases

| Case                        | Handling                                           |
|-----------------------------|----------------------------------------------------|
| Missing telemetry values    | Forward-fill up to 5 seconds, then mark incomplete |
| Multiple failures in horizon| Use highest severity failure                       |
| Overlapping failure windows | Label based on earliest failure                    |
| Vehicle in charging state   | Include in training, flag is_charging=True         |
| Data gaps > 60s             | Start new window sequence                          |
