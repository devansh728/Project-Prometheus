/**
 * TypeScript DTOs matching backend ServiceOps API contracts
 */

// Enums
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

export enum Severity {
  INFO = 'INFO',
  WARNING = 'WARNING',
  CRITICAL = 'CRITICAL',
  EMERGENCY = 'EMERGENCY',
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
  state_history: StateTransition[];
}

export interface StateTransition {
  from: string | null;
  to: string;
  timestamp: string;
}

// Service Center
export interface ServiceCenter {
  id: string;
  name: string;
  lat: number;
  lon: number;
  capacity: number;
  quality_rating: number;
  specializations: string[];
}

// Mechanic
export interface Mechanic {
  id: string;
  name: string;
  skills: string[];
  certification_level: string;
  current_load: number;
  max_daily_capacity: number;
  efficiency_rating: number;
  available: boolean;
}

// Inventory Part
export interface InventoryPart {
  part_id: string;
  name: string;
  quantity: number;
  reorder_point: number;
  cost: number;
  compatible_components: string[];
}

// Dashboard Metrics
export interface DashboardMetrics {
  total_active_jobs: number;
  jobs_by_state: Record<string, number>;
  avg_completion_time_hours: number;
  mechanic_utilization: number;
  parts_low_stock: number;
  customer_satisfaction: number;
}

// Workload
export interface MechanicWorkload {
  mechanic_id: string;
  mechanic_name: string;
  current_jobs: number;
  max_capacity: number;
  utilization_percent: number;
  skills: string[];
}

// Schedule Slot
export interface ScheduleSlot {
  time: string;
  mechanic_id: string;
  mechanic_name: string;
  job_id: string | null;
  available: boolean;
}

// API Response Types
export interface JobsResponse {
  jobs: ServiceJob[];
  total: number;
}

export interface InventoryResponse {
  parts: InventoryPart[];
  low_stock_alerts: InventoryPart[];
}
