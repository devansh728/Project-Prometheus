SentinEV: Complete Phase-Wise Implementation Blueprint
Master Execution Plan for Hackathon Victory
📋 EXECUTIVE OVERVIEW
This document provides a complete, phase-by-phase execution plan to build SentinEV from scratch. Each phase is designed to be:

Time-boxed for hackathon constraints
Dependency-aware so nothing blocks
Demo-safe so nothing breaks during presentation
Evaluation-aligned hitting every EY criterion
🗓️ PHASE TIMELINE OVERVIEW
text

┌─────────────────────────────────────────────────────────────────────────────┐
│                        HACKATHON EXECUTION TIMELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 0 ──► PHASE 1 ──► PHASE 2 ──► PHASE 3 ──► PHASE 4                   │
│  Foundation   Data       SentinEV    ServiceOps   RAG &                    │
│  & Setup      Layer      Core        AI Engine    Intelligence             │
│  (2 hrs)      (3 hrs)    (5 hrs)     (4 hrs)      (3 hrs)                  │
│                                                                             │
│  PHASE 5 ──► PHASE 6 ──► PHASE 7 ──► PHASE 8 ──► PHASE 9                   │
│  UserOps     ServiceOps  AdminOps    UEBA         Voice                    │
│  Mobile      Dashboard   Dashboard   Security     Agent                    │
│  (5 hrs)     (4 hrs)     (3 hrs)     (2 hrs)      (2 hrs)                  │
│                                                                             │
│  PHASE 10 ──────────────────────────────────────────────────────────────►  │
│  Integration, Polish & Demo Preparation (4 hrs)                            │
│                                                                             │
│  TOTAL ESTIMATED: 37 hours (2-3 day hackathon with 4-person team)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
PHASE 0: FOUNDATION & STRATEGIC SETUP
Duration: 2 Hours
0.1 Team Role Assignment
Role Distribution Matrix
Role	Responsibility	Focus Areas
Architect Lead	Backend + Orchestration	SentinEV Core, ServiceOpsAI, API design
ML Engineer	Models + RAG	Anomaly detection, Prognosis, ChromaDB
Frontend Lead	All three apps	UserOps, ServiceOps, AdminOps
Integration Lead	Glue + Demo	Voice, UEBA, Demo scripting
Communication Protocol
Establish a shared document for real-time status updates with three columns: In Progress, Blocked, and Completed. Every team member updates their status every 30 minutes during active development phases. Create dedicated voice channels for quick synchronization and use a shared timer visible to everyone to maintain phase discipline.

0.2 Technology Stack Decisions
Backend Stack
The backend foundation uses Python 3.11+ as the primary language because of its rich AI/ML ecosystem and rapid prototyping capability. FastAPI serves as the API framework due to its async support, automatic OpenAPI documentation, and type safety through Pydantic. For real-time communication, WebSockets handle telematics streaming while standard REST endpoints manage transactional operations.

Agent Orchestration
The agentic AI layer leverages LangGraph for workflow orchestration because it provides native support for stateful multi-agent systems with clear node-based visualization. This directly addresses the EY requirement for "Agentic AI framework." LangChain provides the foundational utilities for LLM interactions and tool definitions.

Database Layer
SQLite serves as the primary operational database for the hackathon context—zero configuration, file-based, sufficient for demo scale. ChromaDB provides the vector storage for RAG operations, chosen for its simplicity and local operation without external dependencies.

Frontend Stack
React Native with Expo powers the UserOps mobile application, enabling rapid development with hot reloading and easy deployment to physical devices or simulators. React with Vite drives both ServiceOps and AdminOps web dashboards, providing fast build times and modern development experience. Framer Motion handles all animations, offering declarative animation primitives that create professional motion design without complex configuration.

Voice and Communication
Voice interaction simulates through Web Speech API for browser-based demonstrations, providing text-to-speech and speech-to-text without external service dependencies. For a more polished demo, ElevenLabs API can generate natural-sounding voice with emotional tone control—a free tier provides sufficient credits for hackathon use.

0.3 Repository Structure Design
Monorepo Organization
Design a single repository containing all components with clear separation. The root level contains shared configuration files, documentation, and orchestration scripts. Three primary directories organize the codebase: backend containing all Python services, frontend containing all React/React Native applications, and shared containing common types, constants, and utilities.

Backend Directory Structure
The backend divides into sentinev-core housing the master agent and ML pipeline, serviceops-ai containing the local operations engine, and shared holding common database models and utility functions. Each service maintains its own api folder for endpoint definitions, agents folder for LangGraph node implementations, models folder for ML inference logic, and services folder for business logic.

Frontend Directory Structure
Three separate application directories exist: userops-mobile as the React Native application, serviceops-web as the service center dashboard, and adminops-web as the manufacturing insights dashboard. A shared ui-components library contains reusable animated components, theming configuration, and common hooks.

0.4 Design System Foundation
Visual Identity
Establish a cohesive visual language that communicates intelligence, trust, and automotive precision. The primary color palette centers on deep electric blue (#0066FF) representing technology and reliability, paired with dark graphite (#1A1A2E) for professional grounding. Accent colors include success green (#10B981) for healthy states, warning amber (#F59E0B) for attention states, and critical red (#EF4444) for urgent states.

Typography System
Select Inter as the primary typeface for its exceptional readability across all sizes and platforms. Establish a clear hierarchy: H1 at 32px bold for major screens, H2 at 24px semibold for sections, H3 at 18px medium for cards, body at 16px regular for content, and caption at 12px for metadata.

Animation Principles
All animations follow three core principles: purposeful (every motion communicates state or guides attention), swift (durations between 200-400ms to feel responsive), and natural (easing curves that mirror physical motion—ease-out for entering elements, ease-in for exiting elements).

Component Library Preview
Design tokens for core components: cards with 16px border radius and subtle shadow for depth, buttons with 8px radius and 48px minimum touch target, status indicators as 12px diameter circles with animated pulse for active states, and charts with smooth path animations and tooltips on hover/touch.

0.5 API Contract Definition
Before any implementation begins, define the critical API contracts between systems:

Telematics Ingestion Contract
The telemetry endpoint accepts vehicle identification, timestamp, and a structured sensor payload containing RPM, engine temperature, battery voltage, coolant level, vibration metrics, and GPS coordinates. The endpoint returns an acknowledgment with anomaly score if immediate analysis was performed.

Service Request Contract
Service requests flow from SentinEV Core to ServiceOpsAI containing vehicle identification, predicted failure type, severity level, failure probability, estimated remaining useful life, preferred scheduling window, and customer contact information. The response includes booking confirmation or alternative slot suggestions.

Agent Communication Contract
Inter-agent messages follow a standardized envelope containing source agent identifier, target agent identifier, message type, payload, timestamp, and correlation ID for tracing. This standardization enables UEBA monitoring across all agent interactions.

PHASE 1: DATA LAYER CONSTRUCTION
Duration: 3 Hours
1.1 Vehicle Fleet Design
Fleet Composition Strategy
Create a fleet of 10 distinct vehicles with intentionally varied characteristics to demonstrate system intelligence across different scenarios:

Vehicle 1-3: Normal Operation Vehicles
These vehicles exhibit consistently healthy behavior throughout the demo. Their telemetry stays within established baselines with minor natural variations. They exist to show the system does NOT generate false alarms—a critical trust signal.

Vehicle 4-6: Warning Trajectory Vehicles
These vehicles show gradual degradation patterns that trigger the predictive maintenance workflow. Each degrades differently: one shows battery degradation, another shows brake wear patterns, the third shows cooling system stress. This variety demonstrates the diagnostic breadth.

Vehicle 7-8: Critical Alert Vehicles
These vehicles are pre-configured to enter critical states during the demo, triggering emergency workflows. Their degradation is steep and unambiguous, ensuring the critical scenario demo never fails.

Vehicle 9-10: Edge Case Vehicles
One vehicle represents a fleet vehicle with different usage patterns. Another represents a vehicle with intermittent issues that require multiple diagnostic passes. These demonstrate system sophistication.

Vehicle Profile Schema
Each vehicle record contains:

Static Attributes: Vehicle ID, make, model, year, VIN, registration date, owner ID, current mileage, warranty status.

Dynamic Attributes: Current location, driving profile assignment (eco/normal/aggressive), maintenance history reference, current health score, last service date.

Behavioral Configuration: Baseline sensor ranges, degradation curve parameters, failure mode assignments, scenario trigger points.

1.2 Synthetic Telematics Design
Sensor Parameter Definitions
Temperature Sensors
Engine temperature baseline at 90°C with healthy range of 85-105°C. Values are influenced by ambient temperature, driving intensity, and cooling system health. Degradation manifests as higher baseline and slower cooling response.

Electrical Sensors
Battery voltage baseline at 12.6V for resting, 14.2V while running. Healthy discharge follows predictable curves based on accessory load. Degradation shows as reduced peak voltage and faster discharge rates.

Mechanical Sensors
Vibration sensors produce frequency spectrum data, simplified to dominant frequency and amplitude for the hackathon. Healthy baseline shows low amplitude across expected frequencies. Degradation introduces new frequency peaks or amplitude increases at specific frequencies.

Performance Sensors
RPM, throttle position, brake pressure, and acceleration values correlate with driving style. The system learns driver behavior patterns to distinguish driver-induced anomalies from vehicle-induced anomalies.

Degradation Modeling
Each failure mode follows a degradation trajectory function:

Linear Degradation: Used for wear items like brakes. Health decreases proportionally with usage. Simple to model: health = 100 - (usage * wear_rate).

Exponential Degradation: Used for cascading failures. Initial slow decline accelerates as system struggles. Formula: health = 100 * e^(-λ * time) where λ increases as other systems stress.

Step Degradation: Used for sudden issues like sensor failures. Health remains stable until a threshold event, then drops dramatically.

Probabilistic Injection: Random noise added to all signals to simulate real-world sensor variance. Healthy variance is small; degrading systems show increased variance before failure.

Telematics Streaming Design
The telematics simulator operates as a separate process that generates realistic data streams:

Streaming Architecture
The simulator maintains virtual time that can run faster than real-time for demonstration purposes. Each vehicle ticks independently, generating sensor readings based on its current state, driving profile, and degradation trajectory. Data streams through WebSocket connections to the SentinEV Core at configurable intervals (default: every 2 seconds for demo visibility).

Scenario Injection Points
The simulator accepts commands to trigger specific scenarios: "start degradation" for a vehicle, "inject anomaly" for immediate alert, "simulate trip" for realistic driving patterns. These commands enable scripted demo execution.

1.3 Historical Data Generation
Maintenance History Construction
For each vehicle, generate 12-24 months of synthetic maintenance history:

Routine Maintenance Records
Oil changes, tire rotations, inspections at appropriate mileage intervals. These establish baseline maintenance patterns and inform the AI about expected service frequencies.

Repair Records
A portion of vehicles have prior repairs matching their assigned failure modes. This creates the historical correlation data that enables RCA/CAPA analysis. Vehicle 7 with cooling issues has a prior coolant system repair; this becomes the pattern the CAPA agent detects.

Service Center Diversity
Historical records reference multiple service centers with varying quality ratings. This enables recommendations based on past service quality and specialization.

Diagnostic Trouble Code History
Generate realistic DTC patterns:

Isolated Codes: Single codes that resolved with simple fixes, showing the system can distinguish minor from major issues.

Code Clusters: Groups of related codes appearing together, demonstrating pattern recognition capability.

Recurring Codes: Same codes appearing multiple times over months, feeding into CAPA analysis for manufacturing feedback.

1.4 Knowledge Base Construction
Document Corpus for RAG
Prepare the knowledge base that powers explainable AI:

Vehicle Technical Documentation
Create condensed versions of owner's manual sections covering major systems: powertrain, electrical, cooling, braking, HVAC. Each document chunk contains system description, normal operating parameters, common issues, and recommended maintenance.

Repair Procedure Library
For each failure mode in the demo, prepare detailed repair procedure documents: symptom description, diagnostic steps, parts required, labor estimate, and preventive measures. These enable the RAG system to provide actionable service center guidance.

Historical Case Database
Synthesize 50-100 past cases matching the failure modes. Each case contains: vehicle profile, symptoms observed, diagnosis performed, root cause identified, repair performed, and outcome. These train the RAG system to recognize patterns and provide evidence-based recommendations.

CAPA Record Collection
Create manufacturing-focused documents linking field failures to design or supplier issues. Each CAPA record contains: failure mode, affected vehicles, root cause analysis, corrective action taken, preventive action implemented, and effectiveness verification. This corpus enables the manufacturing feedback loop.

Document Chunking Strategy
All documents are preprocessed with consistent chunking:

Chunk Size: 512 tokens maximum to fit within embedding model context and maintain semantic coherence.

Overlap: 50 tokens overlap between chunks to preserve context at boundaries.

Metadata Enrichment: Each chunk carries metadata including source document, section, component system, severity relevance, and vehicle model applicability.

1.5 Service Center Data
Service Center Registry
Create 5 service centers with distinct characteristics:

Center 1: Premium Flagship
Full capabilities, all specialists available, premium parts inventory, longest wait times due to demand.

Center 2: Quick Service Express
Limited to routine maintenance and minor repairs, fast turnaround, limited specialist availability.

Center 3: Regional Full-Service
Balanced capabilities, moderate inventory, average wait times.

Center 4: EV Specialist Center
Focused on electric/hybrid systems, specialized equipment, limited general repair capability.

Center 5: Budget Service Option
Lower-cost positioning, longer turnaround, adequate for non-urgent repairs.

Mechanic Roster
Each center has 3-8 mechanics with:

Skill Certifications: General, electrical, powertrain, brake specialist, EV certified.

Availability Schedule: Work hours, current assignments, efficiency ratings.

Capacity Metrics: Average jobs per day, specialization efficiency multipliers.

Inventory Status
Maintain live inventory for demo-critical parts:

Brake Components: Pads, rotors, calipers for common vehicle models.

Cooling Components: Thermostats, water pumps, coolant, radiators.

Electrical Components: Batteries, alternators, starters, sensors.

Stock Levels: Quantities that enable realistic "out of stock" scenarios for edge case demonstration.

PHASE 2: SENTINEV CORE (GLOBAL MASTER BRAIN)
Duration: 5 Hours
2.1 API Gateway Architecture
Endpoint Design Philosophy
The API layer follows a purpose-driven organization rather than resource-based REST:

Ingestion Endpoints: Accept data from vehicles and external systems. Designed for high throughput and minimal latency.

Query Endpoints: Serve data to frontend applications. Designed for rich responses with computed fields.

Command Endpoints: Trigger agent actions. Designed for reliability with acknowledgment patterns.

Streaming Endpoints: Maintain real-time connections. Designed for efficient state synchronization.

Core Endpoint Catalog
Telemetry Ingestion
WebSocket endpoint accepting continuous sensor streams. Each message contains vehicle ID, timestamp, and sensor payload. The endpoint immediately queues data for processing and returns acknowledgment. Failed messages are logged but don't block the stream.

Vehicle Registration
REST endpoint for new vehicle onboarding. Accepts vehicle specifications, owner information, and initial configuration preferences. Returns vehicle ID and initial health assessment based on manufacturer baselines.

Health Status Query
REST endpoint returning comprehensive vehicle health for a given vehicle ID. Response includes current sensor readings, anomaly score, failure predictions, recommended actions, and historical trend summary.

Agent Status Monitoring
REST endpoint returning current state of all agents for observability and UEBA integration. Includes agent identifiers, current activity, recent actions, and resource utilization.

Middleware Stack
Request Validation: Pydantic models enforce schema compliance on all inputs.

Authentication Simulation: Token-based auth headers validated (simplified for hackathon—tokens are accepted without cryptographic verification).

Rate Limiting: Simple in-memory counters prevent abuse in demo scenarios.

Correlation Tracing: Every request receives a unique correlation ID propagated through all downstream operations.

Audit Logging: All requests logged with timestamp, endpoint, actor, and outcome for UEBA analysis.

2.2 Agent Architecture Design
Master Agent Role Definition
The Master Agent serves as the orchestration controller with these specific responsibilities:

Decision Authority: Only the Master Agent can initiate customer contact, approve service requests, and authorize autonomous actions.

State Ownership: The Master Agent maintains global awareness of all vehicle states, active workflows, and pending decisions.

Escalation Handling: When worker agents encounter uncertainty or exceptions, they escalate to the Master Agent rather than failing silently.

Coordination: The Master Agent sequences worker agent activities to prevent conflicts and ensure coherent outcomes.

Worker Agent Definitions
Data Analysis Agent
Continuously processes incoming telemetry streams. Maintains sliding windows of sensor data per vehicle. Computes statistical features: means, variances, trends, anomaly scores. Outputs structured health assessments consumed by other agents.

Diagnosis Agent
Receives health assessments and applies failure prediction models. Maintains component-specific degradation models. Outputs failure probabilities, remaining useful life estimates, and component-specific diagnoses. Queries RAG system for explanation context.

Customer Engagement Agent
Manages all customer communications. Maintains customer preference profiles. Determines communication timing and channel. Generates personalized messages using RAG-enhanced context. Tracks communication outcomes for learning.

Scheduling Agent
Interfaces with ServiceOpsAI for appointment management. Translates diagnosis outputs into service requests. Handles booking confirmations and reschedules. Maintains awareness of customer schedule preferences.

RCA/CAPA Agent
Analyzes patterns across vehicles and historical data. Identifies recurring failures and correlates with manufacturing data. Generates insights for quality improvement. Maintains feedback loop to manufacturing systems.

Security Monitor Agent (UEBA)
Observes all agent activities and API calls. Maintains behavioral baselines per agent. Detects deviations from expected behavior patterns. Raises alerts and can block suspicious actions.

LangGraph Workflow Design
The agent system implements as a LangGraph StateGraph:

State Schema
The shared state contains: current vehicle context, pending analyses, active diagnoses, customer interaction status, scheduling requests, UEBA observations, and decision history.

Node Definitions
Each agent becomes a node in the graph. Nodes read relevant state portions, perform their function, and write results back to state.

Edge Definitions
Conditional edges route flow based on state values:

If anomaly score exceeds threshold → route to Diagnosis Agent
If diagnosis severity is critical → route to immediate Customer Engagement
If customer accepts service → route to Scheduling Agent
All transitions → route through UEBA monitoring edge
Checkpointing
State checkpoints enable workflow resumption if any step fails, crucial for demo reliability.

2.3 ML Pipeline Integration
Anomaly Detection Module
Model Architecture Concept
The LSTM Autoencoder learns to reconstruct normal sensor patterns. During inference, reconstruction error indicates deviation from learned normalcy. This architecture is effective because it doesn't require labeled failure data—it only needs examples of healthy operation.

Input Processing
Raw telemetry undergoes preprocessing: normalization to standard ranges, windowing into sequences of 20 timesteps, feature alignment across sensor types.

Inference Logic
For hackathon implementation, the model can be simulated with rule-based logic that mimics autoencoder behavior:

Compute distance from established baselines
Weight by sensor importance
Aggregate into single anomaly score
Apply learned threshold for classification
Output Format
Returns structured assessment: anomaly score (0-1), contributing sensors ranked by deviation magnitude, timestamp of assessment, confidence interval.

Prognosis Module
Trend Analysis Logic
The prognostic model examines degradation trends over longer windows (7-14 days of aggregated data):

Compute rolling statistics: slopes, acceleration of decline
Identify degradation patterns matching known failure trajectories
Estimate remaining useful life using extrapolation
Failure Probability Calculation
Combine multiple signals into probability estimate:

Current anomaly score contribution
Trend trajectory contribution
Historical failure rate for similar patterns
Vehicle-specific factors (age, mileage, maintenance history)
Output Format
Returns structured prediction: failure probability, estimated RUL in days, primary failure mode predicted, confidence score.

Diagnosis Classification
Component Mapping Logic
Map sensor patterns to specific components:

Temperature + coolant patterns → Cooling System
Voltage + charging patterns → Electrical System
Vibration + braking patterns → Brake System
Multiple weak signals → General Wear
Severity Assignment
Classify based on safety and urgency:

Critical: Immediate safety risk, requires urgent action
High: Significant degradation, service within days
Medium: Moderate wear, schedule convenience appointment
Low: Minor observation, mention at next regular service
2.4 Decision Engine
Action Determination Logic
The Master Agent applies a decision matrix to convert diagnostic outputs into actions:

No Action Conditions

Anomaly score below 0.3 AND no failure predictions AND healthy trends
Continue monitoring, update dashboards, no customer contact
Notification Conditions

Anomaly score 0.3-0.5 OR failure probability emerging OR minor trends
Send app notification with health summary
No voice contact unless customer has opted in to proactive calls
Proactive Contact Conditions

Anomaly score 0.5-0.7 OR failure probability 0.4-0.7 OR concerning trends
Initiate voice contact with calm, informational tone
Explain situation and recommend scheduling
Respect customer response (accept or defer)
Urgent Intervention Conditions

Anomaly score above 0.7 OR failure probability above 0.7 OR critical severity
Initiate urgent voice contact with serious tone
Strongly recommend immediate action
If customer declines, log refusal and schedule follow-up
Confidence Score Calculation
Every decision carries a confidence score (USP implementation):

Input Signals

Model prediction confidences
Data quality metrics (completeness, consistency)
Historical accuracy for similar predictions
Agreement between multiple models/signals
Aggregation Formula
Weighted combination of input signals with learned weights based on historical outcomes.

Presentation
Confidence displayed alongside every recommendation in all interfaces, building trust through transparency.

2.5 State Management
Global State Store Design
Maintain centralized state for all entities:

Vehicle State Table

Vehicle ID as primary key
Current sensor snapshot
Latest health assessment
Active predictions
Pending workflows
Last customer interaction
Workflow State Table

Workflow ID as primary key
Initiating event
Current stage
Assigned agents
Intermediate results
Timeout thresholds
Customer State Table

Customer ID as primary key
Linked vehicles
Preference profile
Communication history
Satisfaction scores
Event Log Design
Every state change produces an immutable event:

Event Schema

Timestamp (microsecond precision)
Event type (enumerated)
Actor (agent ID or "system")
Subject (entity affected)
Previous state (snapshot)
New state (snapshot)
Correlation ID (trace linkage)
Event Usage

Replay for debugging
UEBA analysis for behavioral patterns
Audit trail for compliance
Metrics computation for dashboards
PHASE 3: SERVICEOPS AI (LOCAL OPERATIONS ENGINE)
Duration: 4 Hours
3.1 Service Center Management
Center Registration System
Onboarding Data Model
Each service center registers with: center identification, physical address with geocoordinates, contact information, operating hours by day of week, service capabilities (maintenance, repair, specialty), equipment inventory, and certification levels.

Capacity Calculation
System computes effective daily capacity based on:

Number of service bays
Technician count and skill distribution
Average job duration by type
Historical efficiency factors
Dynamic Status Tracking
Real-time tracking of center status: current utilization percentage, queue depth, wait time estimate, and alert flags for overload conditions.

Geographic Intelligence
Location Services
Implement Haversine distance calculation for nearest center determination. Consider not just distance but also traffic patterns (simplified: urban centers have higher delay factors) and center specialization matching the required service type.

Multi-Center Coordination
When multiple centers could serve a request, rank by:

Required capability match
Distance with traffic adjustment
Wait time comparison
Historical quality rating
Customer preference if expressed
3.2 Workforce Management
Mechanic Skill Matrix
Certification Tracking
Each mechanic profile contains certifications with expiration tracking:

General automotive service
Brand-specific training
Specialty certifications (EV, hybrid, diagnostic)
Safety certifications
Skill-to-Job Matching
Jobs are tagged with required skills. Scheduling only considers mechanics with all required certifications. Premium jobs may prefer mechanics with higher experience levels.

Schedule Optimization
Slot Generation Algorithm
Generate available slots by:

Start with center operating hours
Overlay mechanic schedules (shifts, breaks, PTO)
Block slots already committed
Apply buffer time between appointments
Weight remaining slots by efficiency factors
Priority Queue Management
Pending jobs ranked by priority score:

text

Priority = (Severity × 3) + (Failure_Probability × 2) + 
           (Wait_Time_Days × 0.5) + (Customer_Value × 1)
Higher priority jobs can trigger rescheduling suggestions for lower priority existing appointments.

Labor Forecasting
Demand Prediction
Based on current queue, predicted failures across monitored fleet, and historical seasonal patterns, forecast:

Required mechanic-hours by specialty for next 7 days
Potential bottlenecks where demand exceeds capacity
Overtime requirements
Capacity Alerts
When forecasted demand exceeds available capacity by threshold (e.g., 80%), generate alerts to:

AdminOps dashboard for management visibility
Suggest customer communication adjustments (earlier outreach, alternative center suggestions)
3.3 Inventory Management
Parts Catalog Structure
Inventory Schema
Each part tracked with: part number, description, compatible vehicles, quantity on hand, reorder point, lead time, primary supplier, and unit cost.

Location Awareness
Parts inventory tracked per service center, enabling intelligent routing: if Center A lacks a required part but Center B has stock, the system can suggest Center B or arrange inter-center transfer.

Parts Allocation Logic
Reservation System
When appointment is booked, required parts are soft-reserved:

Reduces available quantity for other bookings
Reservation expires if appointment not confirmed within timeframe
Hard allocation occurs at check-in
Shortage Handling
If parts unavailable:

Check other centers for transfer possibility
Check if substitute part approved
If no alternatives, communicate to customer before booking
Trigger reorder alert to supply chain
Supply Chain Integration
Reorder Automation
When inventory falls below reorder point:

Generate purchase suggestion
Route to AdminOps for approval (simulated: auto-approve in demo)
Track expected delivery
Update availability forecast
3.4 Service Lifecycle Management
Status State Machine
Implement Amazon-style tracking with defined states:

text

REQUESTED → BOOKED → CONFIRMED → CHECK_IN → 
DIAGNOSIS → PARTS_ALLOCATED → REPAIR_IN_PROGRESS → 
QUALITY_CHECK → READY → COMPLETED → FEEDBACK_COLLECTED
State Transition Rules
Each transition has:

Preconditions (what must be true to enter)
Actions (what happens on entry)
Timeout (maximum time in state before escalation)
Allowed next states
Progress Tracking
Customer Visibility
Customer app shows:

Current state with friendly description
Estimated time to completion
Service advisor contact
Real-time updates on significant changes
Internal Tracking
ServiceOps dashboard shows:

All jobs by state
Jobs approaching timeout
Jobs with blockers (parts, labor)
Efficiency metrics
Feedback Collection
Post-Service Survey
Triggered when state reaches COMPLETED:

Rating (1-5 stars)
Comment (free text)
Net Promoter Score question
Issue report option
Feedback Integration
Results feed into:

Service center quality scores
Mechanic performance metrics
CAPA analysis (recurring complaints indicate systemic issues)
3.5 API Surface for SentinEV
Single Integration Point Philosophy
ServiceOpsAI exposes a clean, minimal API to SentinEV Core:

Service Request Endpoint
Input: Vehicle ID, failure type, severity, urgency, customer preferences
Output: Available slots at qualified centers, ranked by recommendation score

Booking Confirmation Endpoint
Input: Selected slot ID, customer confirmation
Output: Booking reference, service details, center information

Status Query Endpoint
Input: Booking reference
Output: Current state, ETA, any blockers

Feedback Submission Endpoint
Input: Booking reference, rating, comments
Output: Acknowledgment

This minimal surface area reduces integration complexity and points of failure.

PHASE 4: RAG & INTELLIGENCE LAYER
Duration: 3 Hours
4.1 ChromaDB Setup
Collection Architecture
Organize vector storage into purpose-specific collections:

Technical Documentation Collection
Contains: vehicle manuals, system descriptions, operating parameters
Metadata: vehicle_model, component_system, document_type
Use case: Explaining what the vehicle system does and normal behavior

Repair Procedures Collection
Contains: service procedures, diagnostic guides, repair instructions
Metadata: component_system, skill_level, estimated_duration, parts_required
Use case: Guiding service center actions

Historical Cases Collection
Contains: past diagnostic cases, repair outcomes, lessons learned
Metadata: failure_type, vehicle_model, severity, resolution_success
Use case: Pattern matching for diagnosis confidence

CAPA Records Collection
Contains: root cause analyses, corrective actions, preventive measures
Metadata: failure_mode, affected_model_years, supplier_involved, resolution_status
Use case: Manufacturing feedback and quality improvement

Embedding Strategy
Model Selection
Use sentence-transformers (all-MiniLM-L6-v2) for embedding generation—fast, effective, runs locally without API dependencies.

Preprocessing Pipeline
Documents undergo:

Text extraction and cleaning
Chunking with overlap
Metadata attachment
Embedding generation
Storage with metadata indexing
4.2 Retrieval Pipeline
Query Construction
Transform diagnostic outputs into effective search queries:

Symptom-Based Query
Combine: observed anomalies, sensor deviations, DTC codes
Example: "high engine temperature low coolant efficiency aggressive driving EV-X"

Context Enhancement
Add: vehicle model, mileage range, geographic region, driving profile
This improves relevance by matching to similar historical scenarios.

Search Execution
Multi-Collection Search
Query relevant collections in parallel:

Technical docs for background context
Historical cases for pattern matching
Repair procedures for action guidance
Relevance Scoring
Combine semantic similarity with metadata matching:

Base score from cosine similarity
Boost for exact vehicle model match
Boost for matching severity level
Penalty for outdated documents
Result Aggregation
Merge results across collections, deduplicate, rank by combined relevance, select top-k (typically 5-10 chunks) for context window.

4.3 Response Generation
Context Assembly
Construct LLM prompt with retrieved context:

System Instruction
Define the AI's role: automotive diagnostic assistant with access to technical documentation and historical cases. Emphasize evidence-based responses, safety prioritization, and clear communication.

Retrieved Context Block
Insert relevant chunks with source attribution. Each chunk prefixed with its source document and relevance to the query.

Specific Query
The actual question being answered: explaining diagnosis to customer, recommending service actions, generating CAPA insights.

Output Formatting
Customer-Facing Explanations

Clear, non-technical language
Empathetic tone acknowledging inconvenience
Concrete next steps with options
Safety emphasis where relevant
Service Center Guidance

Technical precision
Specific parts and procedures
Time and skill requirements
Alternative approaches if primary fails
Manufacturing Insights

Data-driven observations
Statistical patterns with sample sizes
Specific supplier or component callouts
Recommended corrective actions
4.4 Explanation Engine
Decision Transparency
For every significant AI decision, generate explanations:

Anomaly Explanation
"Your vehicle's engine temperature has been running 15% higher than its normal baseline over the past 3 days. This pattern, combined with slightly reduced coolant efficiency, suggests early stress on the cooling system."

Prediction Explanation
"Based on similar patterns in vehicles with your driving profile and mileage, we estimate a 74% likelihood that cooling system service will be needed within 7 days. This prediction draws from 47 similar historical cases."

Recommendation Explanation
"We recommend preventive service now because: (1) early cooling system service typically costs 40% less than emergency repair, (2) prevents potential overheating during your commute, and (3) our nearest qualified center has availability this week."

Confidence Communication
Translate confidence scores to human-understandable language:

90%+: "We're highly confident..."
70-90%: "Our analysis strongly suggests..."
50-70%: "Indicators point toward..."
Below 50%: "We're monitoring possible..."
PHASE 5: USEROPS MOBILE APPLICATION
Duration: 5 Hours
5.1 Application Structure
Screen Architecture
Authentication Flow

Splash screen with animated logo
Login screen with email/password
Registration flow for new users
Profile setup with vehicle linking
Main Navigation
Bottom tab navigation with four primary sections:

Home (dashboard overview)
Vehicles (vehicle management and details)
Services (booking and tracking)
Profile (settings and history)
Modal Flows
Overlay screens for focused interactions:

Voice call simulation
Chat conversation
Booking flow
Feedback submission
Navigation Design
Stack-Based Flow
Each tab maintains its own navigation stack, enabling drill-down into detail screens while preserving tab position.

Deep Linking Support
Enable notifications to open specific screens: voice call incoming, service update, health alert.

5.2 Home Dashboard Design
Hero Health Card
Visual Design
Prominent card occupying top third of screen. Vehicle silhouette with animated health indicator—pulsing green ring for healthy, amber for warning, red for critical. Current health score displayed prominently with trend indicator (up/down/stable).

Animation Specification

Health ring pulses with 2-second cycle, slowing when healthy, accelerating when critical
Score counter animates on load with counting effect
Trend arrow has subtle bounce on appearance
Interaction
Tap expands to full vehicle health detail screen.

Quick Action Tiles
Design Grid
2x2 grid of action tiles below hero card:

"Chat with AI" — opens chatbot
"Schedule Service" — opens booking flow
"Call Advisor" — initiates voice interaction
"View History" — opens maintenance timeline
Visual Treatment
Each tile has icon with subtle shadow, label text, and optional badge for pending items.

Alert Banner
Purpose
When system has proactive recommendations, banner appears between hero and tiles.

Animation
Slides in from top with spring physics. Gentle pulse on badge indicator. Swipe to dismiss with rubber-band effect.

Content
Brief headline ("Brake Service Recommended") with tap to expand for full details.

5.3 Vehicle Management Screens
Vehicle List
Card Design
Each registered vehicle as horizontal card showing:

Vehicle image (placeholder or user photo)
Make, model, year
Health score badge
Last active indicator
Empty State
When no vehicles registered, animated illustration encouraging registration with prominent "Add Vehicle" button.

Vehicle Detail Screen
Tabbed Organization
Three tabs within detail screen:

Overview (current status, specs)
Health (live sensors, predictions)
History (maintenance timeline)
Overview Tab
Vehicle image hero, key specifications, current mileage input, driving profile selector.

Health Tab
Real-time sensor displays with live-updating values:

Temperature gauge with animated needle
Battery voltage bar with charge indicator
Vibration level oscilloscope-style animation
Overall health donut chart with component breakdown
History Tab
Timeline of past services with expandable cards showing details.

5.4 Chat Interface
Conversation Design
Visual Layout
Standard chat bubble interface. User messages right-aligned in brand color. AI messages left-aligned in neutral color with subtle avatar.

Typing Indicator
Three-dot animation when AI is "thinking"—dots bounce in sequence.

Rich Messages
AI responses can include:

Formatted text with headers and bullets
Inline vehicle health cards
Action buttons (book service, call advisor)
Image attachments (vehicle diagrams)
Suggested Prompts
Initial State
When conversation empty, show suggested questions:

"How is my vehicle's health?"
"When is my next service due?"
"Explain my recent alert"
"Help me understand my battery health"
Contextual Suggestions
After AI response, relevant follow-up suggestions appear above input field.

Input Experience
Text Input
Expandable text field with send button. Character limit with counter for very long messages.

Voice Input
Microphone button for speech-to-text. Animated waveform during recording. Transcription appears in input field for review before sending.

5.5 Voice Call Simulation
Incoming Call Screen
Visual Design
Full-screen overlay mimicking native phone call interface:

Pulsing circle with AI avatar
Caller ID showing "SentinEV Service"
Accept (green) and decline (red) buttons
Animation

Ripple effect emanating from avatar
Buttons have press animations
Decline triggers slide-out; accept transitions to active call
Active Call Screen
Layout

AI avatar centered
Waveform visualization responding to audio
Mute and speaker toggle buttons
End call button
Conversation Flow
Pre-scripted conversation with timing:

AI greeting with concern explanation
Pause for user acknowledgment
Recommendation with explanation
Offer to schedule
Confirmation and close
Visual Feedback
Real-time transcription optionally displayed below avatar, showing both AI speech and detected user responses.

5.6 Service Booking Flow
Slot Selection
Calendar Interface
Horizontal date selector for next 14 days. Available slots shown as time blocks. Color coding: green for available, amber for limited, gray for unavailable.

Center Cards
Each slot shows associated service center with:

Name and distance
Wait time estimate
Quality rating
Available slot times
Booking Confirmation
Summary Screen
Before confirmation, display:

Selected date and time
Service center details with map
Expected service type
Estimated duration
Estimated cost range
Confirmation Animation
On booking success, celebratory checkmark animation with confetti subtle effect.

Tracking Screen
Progress Visualization
Vertical timeline showing service states:

Past states with checkmarks
Current state highlighted with pulse
Future states dimmed
Updates
Real-time updates pushed through notification system, screen updates without refresh.

5.7 Gamification Elements
Safe Driving Score
Scoring Algorithm
Based on driving behavior data:

Smooth acceleration: +points
Gentle braking: +points
Steady speeds: +points
Hard stops: -points
Aggressive acceleration: -points
Visual Display
Circular score display with animation on update. Score history graph showing trend. Comparison to fleet average for social motivation.

Achievement System
Badge Categories

Maintenance badges: "Regular Service Champion", "Preventive Pro"
Driving badges: "Eco Driver", "Smooth Operator"
Engagement badges: "AI Communicator", "Feedback Provider"
Visual Design
Badge icons with unlock animation. Collection screen showing earned and locked badges with progress indicators.

Points Economy
Earning Points

Completing scheduled maintenance
Accepting AI recommendations
Maintaining good driving scores
Providing feedback
Potential Redemption
Display accumulated points with teaser for future redemption options (service discounts, priority booking)—doesn't need to be functional for demo.

PHASE 6: SERVICEOPS DASHBOARD
Duration: 4 Hours
6.1 Dashboard Layout Architecture
Navigation Structure
Sidebar Navigation
Left-aligned vertical menu:

Dashboard (overview)
Service Queue
Schedule
Mechanics
Inventory
Analytics
Header Bar
Top bar with:

Center selector (multi-center operators)
Search functionality
Notification bell
User profile
Responsive Design
Desktop Primary
Optimized for 1280px+ screens with full sidebar and multi-column layouts.

Tablet Adaptation
Collapsible sidebar, adjusted grid layouts for 768px-1279px.

Mobile Minimal
For demo purposes, basic responsive handling; not primary focus.

6.2 Command Center Dashboard
Key Metrics Row
Metric Card Design
Four primary metrics displayed as cards:

Active Jobs (count with breakdown by status)
Today's Appointments (count with timeline)
Queue Wait Time (average with trend)
Capacity Utilization (percentage with bar)
Animation
Metrics animate from previous value on refresh. Trend arrows animate with subtle bounce.

Active Jobs Grid
Card Grid Layout
Each active job as card showing:

Customer name and vehicle
Current status badge
Assigned mechanic
Elapsed time
Priority indicator
Color Coding
Border color indicates priority: critical (red), high (amber), normal (blue), low (gray).

Interaction
Click opens job detail panel on right side of screen.

Live Alerts Feed
Alert Stream
Scrolling list of recent alerts:

New high-priority job received
Parts shortage warning
Mechanic reassignment needed
Job status transitions
Priority Visual
Critical alerts flash briefly on appearance. Color-coded icons for alert types.

6.3 Schedule Calendar
Calendar View Options
Day View
Vertical timeline with all bays shown as columns. Appointments as blocks positioned by time and duration.

Week View
Grid with days as columns, time slots as rows. Summary blocks showing appointment counts.

List View
Table format with sortable columns for date, time, vehicle, status, mechanic.

Drag-and-Drop Scheduling
Interaction Design
Appointments draggable between time slots and mechanics/bays. Visual preview during drag. Conflict detection with warning overlay. Confirmation dialog for schedule changes affecting customers.

Animation
Smooth transitions during drag. Snap-to-slot behavior. Undo option after change.

Optimization Indicators
Suggestion Badges
AI-generated optimization suggestions appear as badges:

"Earlier slot available" for jobs that could move up
"Mechanic swap recommended" for skill optimization
"Parts arriving tomorrow" for blocked jobs
6.4 Job Detail Panel
Job Header
Information Display

Customer name with contact options
Vehicle with specs
Job type and description
Priority badge
Status timeline progress
Status Control
Status Transition
Buttons or dropdown to advance job status. Validation before transition (e.g., parts allocated before repair can start). Confirmation for status changes.

Timeline Visualization
Horizontal timeline showing completed and remaining states with timestamps.

Parts and Labor
Parts List
Table of required parts with:

Part number and description
Quantity needed
Allocation status
Stock location
Labor Assignment
Assigned mechanic profile with:

Photo and name
Skills badge
Current load indicator
Reassignment option
Communication Log
Interaction History
Timeline of all customer interactions:

AI voice calls (with transcripts)
App notifications sent
Customer responses
Status updates shared
6.5 Mechanic Management
Roster View
Card Grid
Each mechanic as card showing:

Photo and name
Skill certification badges
Current status (available, busy, break)
Today's job count
Skill Filter
Filter mechanics by certification for quick assignment matching.

Individual Profile
Detail Panel
Expanded view showing:

Full skill list with expiration dates
Schedule overview
Performance metrics (jobs completed, efficiency, ratings)
Current assignments
Workload Visualization
Capacity Bar
Visual bar showing allocated hours vs. available hours. Color shifts to amber when nearing capacity, red when overloaded.

Forecast View
Weekly view showing predicted workload from scheduled jobs and anticipated demand.

6.6 Inventory Dashboard
Stock Overview
Category Summary
Visual summary by category:

Brake components: X items, Y units total
Electrical components: X items, Y units total
Cooling components: X items, Y units total
Status Indicators
Each category shows:

Green: Well stocked
Amber: Below optimal
Red: Critical shortage
Part Detail Table
Sortable Grid
All parts in searchable, sortable table:

Part number
Description
Quantity on hand
Reorder point
Status flag
Quick Actions
Inline buttons for reorder, transfer request, location check.

Alerts Panel
Shortage Warnings
List of parts needing attention:

Below reorder point
Reserved but needed for scheduled jobs
Long lead time items with upcoming demand
PHASE 7: ADMINOPS DASHBOARD
Duration: 3 Hours
7.1 Executive Dashboard
High-Level Metrics
Metric Cards Row
Four executive metrics:

Fleet Health Index (average across all monitored vehicles)
Predicted Failures (count in next 30 days)
CAPA Items Open (pending manufacturing issues)
Service Satisfaction (aggregate rating)
Trend Indicators
Each metric shows:

Current value
Change from previous period
Trend sparkline
Issue Heatmap
Visual Design
Grid heatmap showing:

X-axis: Vehicle models
Y-axis: Component systems
Cell color: Issue frequency (green → amber → red)
Interaction
Click cell to drill down to specific model+component issue list.

Alert Summary
Priority Distribution
Donut chart showing alerts by severity: critical, high, medium, low.

Trend Graph
Line chart showing alert volume over time with trend line.

7.2 RCA/CAPA Management
CAPA Case List
Table View
All CAPA cases in sortable table:

Case ID
Failure mode
Affected vehicles count
Status (open, investigating, resolved, closed)
Owner
Age in days
Status Filters
Quick filters for open, overdue, recently closed.

CAPA Detail Screen
Case Header

Case ID and title
Severity classification
Date opened and target date
Assigned team
Root Cause Section

Problem description
Investigation findings
Root cause determination
Supporting data (charts, vehicle lists)
Corrective Actions

Immediate containment actions
Long-term corrective actions
Preventive measures
Verification requirements
Evidence Attachments
Links to related documents, analysis reports, and historical data.

Pattern Detection Panel
AI-Generated Insights
System automatically identifies patterns:

"15 battery failures in past 30 days, all in vehicles with XYZ supplier batteries"
"Cooling issues 3x more frequent in hot climate regions"
"Brake wear accelerated in aggressive driving profiles"
Visualization
Each insight with supporting chart and drill-down capability.

7.3 Manufacturing Feedback Loop
Quality Insights Report
Report Sections
Generated report containing:

Executive summary
Top 5 issues by frequency
Top 5 issues by severity
Supplier quality trends
Recommended actions
Export Options
PDF generation for sharing with manufacturing teams.

Supplier Scorecard
Supplier Table
Track supplier quality metrics:

Supplier name
Component categories
Failure rate per 1000 units
Quality trend (improving/stable/declining)
Action status
Comparison View
Chart comparing suppliers on key metrics.

Design Feedback Queue
Suggestion Pipeline
List of design improvement suggestions from field data:

Component/system affected
Issue description
Supporting data
Suggested modification
Estimated impact
Workflow Status
Track suggestions through: submitted → reviewed → accepted/rejected → implemented.

7.4 Analytics Center
Failure Trend Analysis
Time Series Charts
Interactive charts showing:

Failure rates over time
By component category
By vehicle model
By geographic region
Comparison Tools
Select multiple series for overlay comparison.

Predictive Accuracy Metrics
Model Performance
Track prediction quality:

True positive rate (predicted failures that occurred)
False positive rate (predictions not followed by failure)
Lead time accuracy (predicted vs. actual time to failure)
Trend Monitoring
Flag if accuracy degrades, suggesting model recalibration.

Fleet Health Distribution
Histogram View
Distribution of vehicles by health score:

How many vehicles in healthy range
Warning range distribution
Critical vehicles count
Risk Stratification
Segment fleet into risk categories for targeted action.

PHASE 8: UEBA SECURITY LAYER
Duration: 2 Hours
8.1 Behavioral Baseline Definition
Agent Behavior Profiles
Define expected behavior for each agent type:

Data Analysis Agent

Accesses: Telemetry data, vehicle profiles
Does not access: Customer contact info, scheduling APIs
Call patterns: Continuous telemetry reads, batch analysis writes
Resource usage: Moderate CPU for analysis
Diagnosis Agent

Accesses: Health assessments, historical data, RAG system
Does not access: Direct customer communication
Call patterns: Event-triggered analysis
Resource usage: Spiky during diagnosis
Customer Engagement Agent

Accesses: Customer profiles, communication channels
Does not access: Raw telemetry, inventory systems
Call patterns: Scheduled and event-triggered calls
Resource usage: Low background, burst for calls
Scheduling Agent

Accesses: ServiceOpsAI APIs, customer preferences
Does not access: Telemetry data, manufacturing data
Call patterns: Request-response with ServiceOps
Resource usage: Low, network-bound
Baseline Metrics
For each agent, track:

API endpoints accessed (should match profile)
Data volumes read/written
Timing patterns (expected frequency)
Error rates (normal bounds)
8.2 Anomaly Detection Logic
Rule-Based Detection
Access Violation Rules
If agent accesses unauthorized endpoint → ALERT
Example: Scheduling Agent queries telemetry endpoint

Volume Anomaly Rules
If data volume exceeds 3× normal baseline → ALERT
Example: Data Agent suddenly reads 10× normal vehicle count

Timing Anomaly Rules
If agent active during off-hours → ALERT
Example: Diagnosis Agent running at 3 AM with no triggering event

Sequence Violation Rules
If action order violates expected workflow → ALERT
Example: Customer call before diagnosis completed

Scoring System
Each anomaly contributes to risk score:

Access violation: +40 points
Volume anomaly: +20 points
Timing anomaly: +10 points
Sequence violation: +30 points
Thresholds:

Below 30: Log only
30-60: Warning alert
Above 60: Block action, escalate
8.3 Alert and Response
Alert Generation
Alert Schema
Each UEBA alert contains:

Timestamp
Agent ID
Anomaly type
Severity score
Evidence (what triggered detection)
Context (surrounding activity)
Alert Routing

Low severity: Dashboard display only
Medium severity: Dashboard + notification
High severity: Dashboard + notification + action blocking
Demo-Ready Incident
Pre-Configured Scenario
For demonstration, pre-configure a scenario:

Scheduling Agent attempts to access telemetry data
UEBA detects access violation
Action is blocked
Alert appears on AdminOps dashboard
Audit log shows blocked request
Narration Script
"Here we see UEBA in action. The Scheduling Agent just attempted to access vehicle telematics data, which is outside its normal behavior profile. UEBA immediately detected this anomaly, blocked the unauthorized access, and raised this alert for security review."

8.4 Audit Trail
Comprehensive Logging
Every agent action logged with:

Timestamp (millisecond precision)
Agent ID
Action type
Target resource
Parameters
Outcome (success/failure)
Duration
UEBA risk score
Log Visualization
Timeline View
Scrollable timeline of all agent activity with:

Color coding by agent type
Highlight for anomalous events
Drill-down to full event details
Filter Capabilities
Filter by:

Time range
Agent ID
Action type
Risk score threshold
PHASE 9: VOICE AGENT INTEGRATION
Duration: 2 Hours
9.1 Voice Generation System
Text-to-Speech Setup
Technology Selection
Use Web Speech API for zero-dependency demo capability. Optional upgrade to ElevenLabs for more natural voice quality if API access available.

Voice Configuration

Voice selection: Professional, gender-neutral or per user preference
Speed: Slightly slower than default for clarity
Pitch: Standard, adjusted for emotional content
Emotional Tone Mapping
Tone Categories

Calm/Informational
Used for: Normal advisories, health updates, routine recommendations
Voice parameters: Standard speed, neutral pitch, friendly phrasing
Example phrases: "I wanted to let you know about...", "Your vehicle is running well, and..."

Concerned/Advisory
Used for: Warning level predictions, recommended actions
Voice parameters: Slightly slower, slightly lower pitch, empathetic phrasing
Example phrases: "We've noticed something that needs attention...", "I recommend scheduling service soon because..."

Urgent/Serious
Used for: Critical alerts, safety-related issues
Voice parameters: Clear and direct pace, firm tone, safety-first phrasing
Example phrases: "This is important for your safety...", "Please take immediate action to..."

Reassuring/Supportive
Used for: Following up on concerns, confirming resolutions
Voice parameters: Warm tone, measured pace, positive phrasing
Example phrases: "Great news about your service...", "Everything has been taken care of..."

9.2 Conversation Script Design
Standard Flow Template
Opening

Greeting with user name
Identification as SentinEV service
Brief purpose statement
Main Content

Condition explanation (from RAG)
Severity communication
Recommendation with reasoning
Engagement

Check for understanding
Answer questions
Address concerns
Closing

Action confirmation or next steps
Appreciation for user's time
Clear conclusion
Scenario-Specific Scripts
Warning Scenario Script

Opening: "Hello [Name], this is your SentinEV service assistant calling about your [Vehicle Model]."

Explanation: "I've been monitoring your vehicle's health, and I wanted to let you know about something we've detected. Your braking system is showing early signs of wear, likely due to your driving patterns over the past few weeks. While your vehicle is currently safe to drive, our analysis suggests that without attention, you may experience reduced braking performance in about 7 to 10 days."

Recommendation: "I recommend scheduling a preventive brake service, which typically takes about an hour. This is usually much more cost-effective than waiting for a more serious issue to develop. Would you like me to find an available appointment at a service center near you?"

If Yes: "Great, I found availability at [Center Name], which is about [X] miles from your location. They have openings on [Date] at [Time]. Would that work for you?"

If No: "I understand. I'll send you a summary through the app so you have all the details. Is there anything else you'd like to know about your vehicle's condition?"

Closing: "Thank you for your time, [Name]. Drive safely, and don't hesitate to reach out if you have any questions."

9.3 Speech Recognition Handling
Input Processing
Recognition Triggers
System listens for:

Explicit responses: "yes", "no", "okay", "not now"
Questions: "why?", "how much?", "when?"
Concerns: "is it dangerous?", "what happens if..."
Response Mapping
Map recognized input to conversation branches:

Affirmative → proceed with booking
Negative → acknowledge and offer alternatives
Question → retrieve and provide additional context
Unclear → request clarification politely
Fallback Handling
If recognition fails:
"I'm sorry, I didn't quite catch that. Could you repeat your response?"

After second failure:
"I want to make sure I understand you correctly. I'll send you a summary through the app, and you can respond there or call us back. Does that work for you?"

9.4 Demo Reliability
Pre-Recorded Backup
Audio Files
Record complete voice-over for demo scenarios:

Full warning scenario call
Full critical scenario call
Key response variations
Trigger Mechanism
If live TTS fails, seamlessly fall back to pre-recorded audio. UI should not reveal whether audio is live or pre-recorded.

Timing Synchronization
Script Timing
Each script segment timed precisely. Demo operator knows exactly when to trigger user responses. Conversation flows predictably without awkward pauses.

Visual Synchronization
Voice audio synchronized with:

Mobile app "incoming call" screen
Transcript display updating in real-time
Status changes in backend dashboards
PHASE 10: INTEGRATION, POLISH & DEMO PREPARATION
Duration: 4 Hours
