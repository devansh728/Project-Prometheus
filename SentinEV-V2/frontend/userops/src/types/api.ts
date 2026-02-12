/**
 * TypeScript DTOs matching backend API contracts
 */

// Enums
export enum Severity {
  INFO = 'INFO',
  WARNING = 'WARNING',
  CRITICAL = 'CRITICAL',
  EMERGENCY = 'EMERGENCY',
}

export enum ServiceState {
  REQUESTED = 'REQUESTED',
  BOOKED = 'BOOKED',
  CONFIRMED = 'CONFIRMED',
  CHECK_IN = 'CHECK_IN',
  DIAGNOSIS = 'DIAGNOSIS',
  PARTS_ALLOCATED = 'PARTS_ALLOCATED',
  REPAIR_IN_PROGRESS = 'REPAIR_IN_PROGRESS',
  QUALITY_CHECK = 'QUALITY_CHECK',
  READY = 'READY',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED',
}

// Telemetry
export interface TelemetryData {
  vehicle_id: string;
  timestamp: string;
  sensors: {
    engine_temp: number;
    battery_voltage: number;
    coolant_level: number;
    brake_pressure: number;
    vibration_amplitude: number;
  };
  anomaly_score: number;
  gps: {
    lat: number;
    lon: number;
  };
}

// Vehicle Health
export interface VehicleHealth {
  vehicle_id: string;
  state: string;
  severity: Severity | null;
  primary_concern: string | null;
  recommended_action: string | null;
  engagement_action: string | null;
  actions_log: string[];
  decision: {
    action: string;
    priority: number;
    notify_customer: boolean;
    delay_seconds: number;
    rationale: string;
  } | null;
  error: string | null;
}

// Agent Workflow Events (WebSocket)
export interface WorkflowEvent {
  event: 'state_change' | 'agent_complete' | 'workflow_complete';
  state?: string;
  agent?: string;
  actions?: string;
  timestamp: string;
  result?: {
    severity: Severity | null;
    engagement_action: string | null;
    primary_concern: string | null;
  };
}

// Service Job
export interface ServiceJob {
  job_id: string;
  vehicle_id: string;
  customer_id: string;
  service_center_id: string;
  failure_type: string;
  severity: string;
  state: ServiceState;
  mechanic_id: string | null;
  scheduled_at: string | null;
  created_at: string;
  completed_at: string | null;
  estimated_completion: string | null;
  state_history: Array<{
    from: string | null;
    to: string;
    timestamp: string;
  }>;
}

// Scheduling
export interface SchedulingSlot {
  service_center: {
    id: string;
    name: string;
    distance_km: number;
    quality_rating: number;
  };
  mechanic: {
    id: string;
    name: string;
    skill_match_score: number;
  };
  available_at: string;
  estimated_duration_minutes: number;
  parts_available: boolean;
  estimated_cost: number;
  recommendation_score: number;
}

export interface ScheduleRequest {
  vehicle_id: string;
  customer_id: string;
  customer_lat: number;
  customer_lon: number;
  failure_type: string;
  severity: string;
  preferred_datetime?: string;
}

export interface ScheduleResponse {
  job_id: string;
  state: string;
  service_center: string;
  mechanic: string;
  scheduled_at: string | null;
  estimated_duration_minutes: number;
  estimated_cost: number;
  alternatives: SchedulingSlot[];
}

// Chat Message
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    severity?: Severity;
    action_type?: string;
  };
}

// Vehicle Info
export interface Vehicle {
  id: string;
  vin: string;
  make: string;
  model: string;
  year: number;
  mileage: number;
  health_score: number;
  category: 'normal' | 'warning' | 'critical';
  driving_profile: string;
}

// Fleet Telemetry Response
export interface FleetTelemetry {
  [vehicle_id: string]: TelemetryData;
}
