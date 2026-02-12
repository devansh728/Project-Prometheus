/**
 * AdminOps TypeScript Types
 */

export interface FleetHealth {
  total_vehicles: number;
  healthy: number;
  warning: number;
  critical: number;
  avg_health_score: number;
}

export interface CAPAPattern {
  id: string;
  title: string;
  component: string;
  occurrences: number;
  affected_vehicles: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'investigating' | 'resolved';
  root_cause?: string;
  corrective_action?: string;
  created_at: string;
}

export interface SupplierScorecard {
  supplier_id: string;
  name: string;
  component_category: string;
  quality_score: number;
  defect_rate: number;
  avg_resolution_time_hours: number;
  total_parts_supplied: number;
  issues_reported: number;
}

export interface FailureTrend {
  date: string;
  brake: number;
  battery: number;
  coolant: number;
  suspension: number;
  other: number;
}

export interface ModelAccuracy {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  last_updated: string;
}

export interface UEBAEvent {
  id: string;
  agent_id: string;
  event_type: 'access_anomaly' | 'volume_spike' | 'timing_anomaly' | 'pattern_deviation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: string;
  resolved: boolean;
}

export interface AgentProfile {
  agent_id: string;
  agent_name: string;
  normal_access_hours: string;
  avg_daily_requests: number;
  typical_endpoints: string[];
  last_activity: string;
}
