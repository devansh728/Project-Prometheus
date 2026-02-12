# CAPA Case: CAPA-2025-001

## Brake Pad Premature Wear in Aggressive Driving Profiles

### Issue Summary

**Severity:** HIGH  
**Status:** RESOLVED  
**Affected Vehicles:** XUV400 EV, ZS EV (2023 models with Supplier-A brake pads)  
**Affected Count:** 847 vehicles  
**Date Opened:** 2025-04-15  
**Date Closed:** 2025-07-20

### Problem Description

Field data analysis detected brake pad wear rates 2.3x higher than specification in vehicles with aggressive driving profiles. Issue correlated with Supplier-A brake pad compound (Part# BP-2023-A).

### Root Cause Analysis

1. **Immediate Cause:** Brake pad compound composition had lower thermal resistance than specified
2. **Underlying Cause:** Supplier-A changed raw material source without notification
3. **System Cause:** Incoming quality inspection did not include thermal degradation testing

### Corrective Actions

1. **Containment (Immediate):**
   - Proactive recall for all affected vehicles with > 15,000 km
   - Replaced brake pads with updated compound (Part# BP-2023-B)
2. **Permanent Fix:**
   - Updated brake pad specification to include thermal cycling requirements
   - Added thermal degradation test to incoming QC process
   - Supplier-A issued corrective action; verified with audit
3. **Preventive Measures:**
   - ML model updated to weight brake pressure variance higher for aggressive profiles
   - Alert threshold adjusted: vibration_amplitude > 0.2 triggers warning

### Verification

- Post-fix wear rate analysis confirms compliance with spec
- 500-vehicle sample tracked for 10,000 km with no recurrence
- Supplier-A audit passed (June 2025)

---

# CAPA Case: CAPA-2025-002

## Battery Thermal Runaway in Hot Climates

### Issue Summary

**Severity:** CRITICAL  
**Status:** OPEN - MONITORING  
**Affected Vehicles:** eVerito (2022-2023), Atto 3 (2023)  
**Affected Count:** 23 incidents reported  
**Date Opened:** 2025-10-01

### Problem Description

Thermal runaway incidents reported in vehicles operated in ambient temperatures > 42°C, particularly after DC fast charging.

### Preliminary Root Cause

1. Cooling pump motor efficiency degraded at high temperatures
2. Thermal interface material between cells showed delamination after 50k cycles

### Interim Actions

1. Software update: Limit DC fast charge power to 80% when ambient > 38°C
2. Proactive cooling system inspection campaign
3. Increased telemetry monitoring frequency for affected VINs

### Next Steps

- Complete tear-down analysis of affected packs
- Design change evaluation: Upgraded thermal interface material (TIM-v2)
- Target permanent fix release: Q1 2026
