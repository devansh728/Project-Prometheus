# 🚗 SentinEV - Agentic AI Predictive Maintenance System

> **"Your car talks to you before it breaks down."**

SentinEV is an intelligent **EV predictive maintenance platform** that combines multi-agent AI orchestration, real-time anomaly detection, voice-enabled customer engagement, and manufacturing quality feedback - all working together to keep you safe on the road.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple.svg)

---

## ✨ What Makes SentinEV Special?

Imagine you're driving down a mountain road. Your brakes are overheating, but you don't know it yet. SentinEV does:

1. **🔍 Detects** brake fade through real-time telemetry (temperature spike, efficiency drop)
2. **🧠 Predicts** potential failure using LSTM-AE + LightGBM models
3. **📞 Calls you** proactively: *"Hi Alex, your brakes are at 15% efficiency. Let's get you to safety."*
4. **📅 Books service** automatically at the nearest center with parts in stock
5. **📊 Learns** from the incident to prevent future failures across all vehicles

This isn't just monitoring - it's an AI co-pilot that acts on your behalf.

---

## 🎬 The Complete Journey: A Real Example

Let's follow **Alex** through a real scenario:

### Morning: Normal Drive
```
🚗 Alex starts driving to work
   └── SentinEV monitors 50+ sensors every second
   └── Battery: 78%, Motor: 42°C, Brakes: 35°C
   └── Status: All systems nominal ✅
```

### 10:30 AM: Mountain Pass
```
🏔️ Alex takes the scenic route through mountains
   └── Continuous downhill for 20 minutes
   └── Brake temperature rising: 35°C → 180°C → 320°C
   └── LSTM-AE detects anomaly pattern
   └── LightGBM predicts: 87% failure probability within 30 min
```

### 10:35 AM: AI Takes Action
```
📞 SentinEV initiates voice call:

AGENT: "Hi Alex, this is SentinEV. I need to let you know about 
        an urgent safety situation with your brakes."

ALEX:  "I'm kind of busy right now..."

AGENT: "I completely understand you're busy. However, this IS a 
        critical safety issue - your brake efficiency is at 15%. 
        Would you like me to text you the details and call back 
        in 30 minutes?"

ALEX:  "Wait, 15%? That sounds serious. What should I do?"

AGENT: "For immediate safety, please reduce speed and use engine 
        braking by downshifting. I've found Downtown EV Hub has 
        the specific ceramic brake pads in stock. They have a bay 
        at 2:30 PM. Shall I book it?"

ALEX:  "Yes please"

AGENT: "Done! You're confirmed for 2:30 PM. I've sent navigation 
        to your car. Drive safely, and we'll see you there."
```

### 2:30 PM: Service Center
```
🔧 Alex arrives at Downtown EV Hub
   └── Service Tracker: INTAKE → DIAGNOSIS → REPAIR
   └── Real-time progress visible in app
   └── 3:45 PM: "Your vehicle is ready for pickup!"
```

### 4:00 PM: Quality Loop
```
📊 CAPA Agent analyzes the incident:
   └── Pattern detected: 47 similar brake fades in mountain regions
   └── Root cause: Brake pad compound inadequate for sustained grades
   └── Recommendation: Use ceramic pads for mountain area customers
   └── Manufacturing notified for design review
```

---

## 📋 The 13 Phases of SentinEV

SentinEV was built in 13 progressive phases, each adding critical capabilities:

### Foundation (Phases 1-8)

| Phase | What We Built | Why It Matters |
|-------|--------------|----------------|
| 1 | **Synthetic Data Generator** | Realistic EV telemetry with physics-based simulation |
| 2 | **ML Pipeline** | LSTM Autoencoder + LightGBM for anomaly detection |
| 3 | **RAG Knowledge Base** | ChromaDB + vehicle manuals for expert diagnostics |
| 4 | **Multi-Agent Foundation** | Data Analysis, Safety, Diagnosis agents |
| 5 | **Scheduling System** | Service center matching, slot optimization |
| 6 | **Service Tracking** | Amazon-style lifecycle (INTAKE → PICKED_UP) |
| 7 | **Feedback Loop** | Customer ratings, sentiment analysis, NPS |
| 8 | **CAPA Analysis** | Root cause analysis, manufacturing insights |

### Advanced Intelligence (Phases 9-13)

| Phase | What We Built | Why It Matters |
|-------|--------------|----------------|
| 9 | **Real-Time Orchestrator** | Hash-based worker routing, severity decisions |
| 10 | **Voice-First Engagement** | Proactive calls, emotion detection, persona adaptation |
| 11 | **Autonomous Scheduling** | Priority queue, geo-optimization, emergency towing |
| 12 | **RCA/CAPA Insights** | Supplier risk scoring, model-year drift detection |
| 13 | **UEBA Governance** | Security monitoring, SLA enforcement, agent fingerprinting |

---

## 🤖 Phase 9: Real-Time Orchestrator

The brain of SentinEV - coordinates all agents and routes telemetry efficiently.

### How It Works

```
Vehicle Telemetry → Master Orchestrator → Hash-Based Worker Routing
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
        Worker Pool (4)           Severity Decision
       VIN hash % 4 = slot              ↓
              ↓                ┌────────┴────────┐
      Stateful Processing      Low    Medium    High/Critical
              ↓                 ↓        ↓           ↓
      Anomaly Detection    Monitor   Schedule   Voice Call
```

### Key Features

**Hash-Based Worker Routing**
```python
worker_id = hash(vehicle_id) % 4
# Same vehicle always goes to same worker → consistent state
```

**Severity-Based Decision Logic**
```python
if severity == "critical":
    trigger_voice_call(vehicle_id)
    queue_emergency_appointment()
elif severity == "high":
    create_prediction_alert()
    suggest_appointment()
else:
    monitor_and_log()
```

---

## 📞 Phase 10: Voice-First Customer Engagement

Not just alerts - actual phone conversations with emotion awareness.

### Emotion Detection

The Voice Agent detects emotional state from user speech:

| Emotion | Trigger Words | Agent Response |
|---------|--------------|----------------|
| **Panicked** | "oh my god", "can't stop", "crash" | Calm, step-by-step guidance |
| **Anxious** | "worried", "is it safe", "afraid" | Reassuring with timeline |
| **Frustrated** | "busy", "not now", "call later" | Acknowledge, offer callback |
| **Confused** | "don't understand", "what does" | Simplified explanations |

### Persona Adaptation

```python
# Agent adapts personality based on severity
if severity == "critical":
    persona = "Emergency"  # Calm but urgent
    speech_rate = "slow"
    empathy_level = "high"
else:
    persona = "Professional"  # Courteous, measured
```

### Example Flow: Frustrated User

```
USER:   "I'm busy right now"
DETECT: FRUSTRATED emotion + callback_later intent

AGENT:  "I completely understand you're busy, and I apologize 
         for the interruption. However, this IS a critical safety 
         issue with your brakes. For your safety, I strongly 
         recommend pulling over when possible. Would you like me 
         to send you a text and call back in 30 minutes?"
```

---

## 📅 Phase 11: Autonomous Scheduling & Demand Forecasting

Smart scheduling that accounts for urgency, location, and capacity.

### Priority Queue System

```python
# Appointments ordered by urgency score
urgency_score = calculate_urgency(
    severity=severity,           # critical=10, high=8, medium=5, low=2
    failure_probability=0.87,    # ML prediction
    customer_tier="premium"      # VIP gets +1
)

# Queue processes highest urgency first
priority_queue.push(appointment)
```

### Geo-Optimized Center Selection

```
Vehicle Location: 40.7128° N, 74.0060° W (Manhattan)

Candidate Centers:
├── Downtown EV Hub (2.3 mi) - Has brake pads ✅
├── Brooklyn Service (4.1 mi) - Has brake pads ✅  
└── Queens Auto (7.8 mi) - Parts on order ❌

Selected: Downtown EV Hub (closest with parts)
```

### Emergency Towing Dispatch

```python
# Critical breakdown? Tow truck dispatched automatically
tow_request = dispatch_towing(
    vehicle_location=(40.7128, -74.0060),
    destination_center="Downtown EV Hub",
    vehicle_id="VIN-001"
)
# ETA: 15 minutes, AAA dispatched
```

---

## 🔍 Phase 12: RCA/CAPA Manufacturing Feedback

Learn from every failure to prevent future ones.

### Supplier Risk Scoring

```
┌──────────────────────────────────────────────────────┐
│ SUPPLIER RISK DASHBOARD                              │
├──────────────────┬─────────┬──────────┬──────────────┤
│ Supplier         │ Score   │ Trend    │ Action       │
├──────────────────┼─────────┼──────────┼──────────────┤
│ BrakeTech Ind.   │ 4.08 ⚠️ │ Declining│ Schedule Audit│
│ PowerCell Energy │ 2.50 ✅ │ Stable   │ None         │
│ EMotion Drives   │ 6.20 🔴 │ Declining│ Review Contract│
└──────────────────┴─────────┴──────────┴──────────────┘
```

### Model Year Drift Detection

```python
# Detect quality degradation by model year
drift_analysis = analyze_model_year_drift("brakes")

# Result:
# 2022: 0.8% failure rate (baseline)
# 2023: 1.0% failure rate (+25%)
# 2024: 1.33% failure rate (+66%) ⚠️ ELEVATED
#
# Recommendation: Investigate 2024 brake pad supplier change
```

### AI Engineering Recommendations

```
┌─────────────────────────────────────────────────────────────┐
│ 🧠 AI ENGINEERING RECOMMENDATIONS                           │
├─────────────────────────────────────────────────────────────┤
│ 🔧 DESIGN (High Priority)                                   │
│    Revise brake rotor alloy composition for improved heat   │
│    dissipation in 2024+ models                              │
├─────────────────────────────────────────────────────────────┤
│ ⚙️ PROCESS (High Priority)                                  │
│    Implement automated brake pad thickness verification     │
│    at end-of-line testing                                   │
├─────────────────────────────────────────────────────────────┤
│ 🏭 SUPPLIER (Medium Priority)                               │
│    Schedule supplier audit for BrakeTech Industries -       │
│    elevated defect rate detected                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Phase 13: UEBA Security & Governance

Monitor agent behavior, detect anomalies, enforce compliance.

### Agent Baseline Monitoring

```python
# Each agent has allowed actions
AGENT_BASELINES = {
    "data_analysis": ["analyze_telemetry", "detect_anomaly", "score_behavior"],
    "scheduling":    ["check_availability", "book_appointment", "reschedule"],
    "voice":         ["initiate_call", "process_input", "confirm_booking"],
}

# Rogue action detection
if agent == "scheduling" and action == "access_telematics_database":
    # 🚨 BLOCKED - Scheduling shouldn't access telematics!
    ueba_alert(severity="critical", reason="Unauthorized action")
```

### Real-Time Security Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│ 🛡️ UEBA SECURITY DASHBOARD                                  │
├─────────────────────────────────────────────────────────────┤
│ Total Actions Monitored: 1,247                              │
│ Security Alerts: 2 (blocked)                                │
│ Active Agents: 4                                            │
│ System Status: ⚠️ ALERT                                     │
├─────────────────────────────────────────────────────────────┤
│ [INJECT ROGUE ACTION] ← Demo button                         │
│                                                             │
│ When clicked:                                               │
│   🚨 SECURITY BREACH DETECTED 🚨                            │
│   Scheduling Agent attempted unauthorized access to         │
│   telematics database                                        │
│   ACTION BLOCKED BY UEBA MONITOR                            │
└─────────────────────────────────────────────────────────────┘
```

### SLA Enforcement

```python
# Automatic SLA monitoring
sla_check = {
    "inference_latency": {"threshold_ms": 100, "actual_ms": 45, "status": "OK"},
    "voice_response":    {"threshold_ms": 500, "actual_ms": 380, "status": "OK"},
    "appointment_book":  {"threshold_ms": 2000, "actual_ms": 1200, "status": "OK"},
}
```

---

## 🗺️ Multi-Agent Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MASTER ORCHESTRATOR (LangGraph)                       │
│                    Coordinates all agents, manages state machine              │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                           UEBA SECURITY MONITOR                          │ │
│  │           Monitors all agent actions, blocks unauthorized ops            │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└──────────┬───────────────┬───────────────┬───────────────┬──────────────────┘
           │               │               │               │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │   DATA    │   │  SAFETY   │   │ DIAGNOSIS │   │SCHEDULING │
     │ ANALYSIS  │   │   AGENT   │   │   AGENT   │   │   AGENT   │
     │   AGENT   │   │           │   │           │   │           │
     └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
           │               │               │               │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │   VOICE   │   │ SERVICE   │   │ FEEDBACK  │   │   CAPA    │
     │   AGENT   │   │ TRACKER   │   │   AGENT   │   │   AGENT   │
     │           │   │   AGENT   │   │           │   │           │
     └───────────┘   └───────────┘   └───────────┘   └───────────┘
           │                                               │
           └───────────────────────────────────────────────┘
                         Manufacturing Feedback Loop
```

### Agent Responsibilities

| Agent | What It Does | Example Output |
|-------|-------------|----------------|
| **Master Orchestrator** | Routes telemetry, coordinates agents, manages workflow | "Routing VIN-001 to worker 2, severity=critical" |
| **Data Analysis** | ML anomaly detection, failure prediction | "Anomaly: brake_fade, probability=87%" |
| **Safety** | Immediate safety recommendations | "Reduce speed, use engine braking" |
| **Diagnosis** | RAG-powered root cause analysis | "Brake fluid boiled due to sustained downhill" |
| **Scheduling** | Appointment booking, center matching | "Booked at Downtown EV Hub, 2:30 PM" |
| **Voice** | Proactive calls, emotion detection | "I understand you're busy..." |
| **Service Tracker** | Lifecycle tracking | "Status: REPAIR → QUALITY_CHECK" |
| **Feedback** | Post-service sentiment analysis | "Customer satisfied, NPS=9" |
| **CAPA** | Manufacturing quality feedback | "47 similar cases, recommend design review" |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Google Gemini API key

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/SentinEV.git
cd SentinEV

# Backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Configure

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY
```

### 3. Run

```bash
# Terminal 1: Backend
python -m ml.enhanced_api

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 4. Experience

- **Dashboard**: http://localhost:3000
- **Vehicle Monitor**: http://localhost:3000/vehicles/VIN-001
- **API Docs**: http://localhost:8000/docs

---

## 📡 API Reference

### v2 API Endpoints (Phases 9-13)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v2/scheduling/queue/status` | Priority queue status |
| POST | `/v2/scheduling/towing` | Dispatch emergency towing |
| GET | `/v2/capa/supplier` | Supplier risk scores |
| GET | `/v2/capa/drift/{component}` | Model year drift analysis |
| GET | `/v2/capa/recommendations/{component}` | AI recommendations |
| WS | `/v2/kafka-bridge` | Real-time Kafka WebSocket |

### WebSocket Stream

```javascript
// Connect to real-time telemetry
const ws = new WebSocket('ws://localhost:8000/api/v1/vehicles/VIN-001/scenario-stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.v2);  // Phase 9-13 features
  // {
  //   worker_id: 2,
  //   voice_ready: true,
  //   agents_active: ["Master", "Data", "Diagnostics"],
  //   ueba_status: "normal"
  // }
};
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, Uvicorn, Python 3.11 |
| **AI/ML** | LangChain, LangGraph, Gemini Flash, LSTM-AE, LightGBM |
| **Vector DB** | ChromaDB + HuggingFace embeddings |
| **Database** | SQLite (WAL mode) |
| **Frontend** | Next.js 16, React 19, TailwindCSS, shadcn/ui |
| **Voice** | Web Speech API, gTTS |
| **Streaming** | WebSocket, Kafka-compatible |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🏆 Acknowledgments

Built for **EY Techathon 6.0** - Agentic AI Challenge

---

<p align="center">
  <b>SentinEV</b> - Your AI Co-Pilot for Safer Roads 🚗⚡
</p>
