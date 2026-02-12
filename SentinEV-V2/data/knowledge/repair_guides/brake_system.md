# EV Brake System: Regenerative and Friction Braking

## Overview

Modern EVs use a combination of regenerative and friction braking systems. Regenerative braking captures kinetic energy during deceleration and converts it back to electrical energy for battery charging.

## Common Issues

### Brake Fade

**Symptoms:** Reduced braking performance after repeated hard braking, spongy pedal feel.
**Causes:**

- Overheated brake fluid (boiling point exceeded)
- Glazed brake pads
- Aggressive driving in hot conditions

**Diagnosis Steps:**

1. Check brake fluid condition and level
2. Inspect brake pad thickness (minimum 3mm)
3. Measure rotor thickness and check for hot spots
4. Check for DTC codes C0035, C0040

**Recommended Action:**

- Replace brake fluid if contaminated
- Replace pads if < 4mm thickness
- Resurface or replace rotors if warped

**Estimated Time:** 1.5 - 2.5 hours
**Priority:** HIGH if detected anomaly score > 0.7

### Regenerative Braking Failure

**Symptoms:** Reduced range, normal friction brakes compensating more than usual.
**Causes:**

- Battery at 100% SOC (cannot accept more charge)
- Battery thermal protection active
- Inverter fault

**Diagnosis Steps:**

1. Check battery SOC at time of issue
2. Review thermal history
3. Scan for motor controller DTCs

**Recommended Action:** Software update or inverter inspection

---

## Brake Pad Wear Prediction Model

| Driving Profile | Average Wear Rate | Expected Life (km) |
| --------------- | ----------------- | ------------------ |
| Eco             | 0.015 mm/1000km   | 80,000             |
| Normal          | 0.025 mm/1000km   | 50,000             |
| Aggressive      | 0.045 mm/1000km   | 28,000             |

## DTC Reference

- **C0035**: Left Front Wheel Speed Sensor
- **C0040**: Right Front Wheel Speed Sensor
- **C0051**: ABS Pump Motor Circuit
- **C0110**: Pump Motor Stalled
