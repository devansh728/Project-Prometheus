/**
 * AdminOps Zustand Store with Mock Data
 */
import { create } from 'zustand';
import { FleetHealth, CAPAPattern, SupplierScorecard, FailureTrend, UEBAEvent, AgentProfile } from '../types/api';

const MOCK_CAPA: CAPAPattern[] = [
  { id: 'CAPA-2025-001', title: 'Brake Pad Premature Wear', component: 'brake', occurrences: 47, affected_vehicles: 23, severity: 'high', status: 'investigating', root_cause: 'Supplier batch defect', created_at: '2025-01-15' },
  { id: 'CAPA-2025-002', title: 'Battery Thermal Runaway Risk', component: 'battery', occurrences: 12, affected_vehicles: 8, severity: 'critical', status: 'open', created_at: '2025-01-20' },
  { id: 'CAPA-2024-015', title: 'Coolant Hose Degradation', component: 'coolant', occurrences: 89, affected_vehicles: 45, severity: 'medium', status: 'resolved', root_cause: 'Temperature cycling stress', corrective_action: 'Upgraded hose material', created_at: '2024-11-05' },
];

const MOCK_SUPPLIERS: SupplierScorecard[] = [
  { supplier_id: 'SUP-001', name: 'BrakeTech India', component_category: 'brake', quality_score: 78, defect_rate: 2.3, avg_resolution_time_hours: 48, total_parts_supplied: 15000, issues_reported: 47 },
  { supplier_id: 'SUP-002', name: 'PowerCell Systems', component_category: 'battery', quality_score: 92, defect_rate: 0.8, avg_resolution_time_hours: 24, total_parts_supplied: 2500, issues_reported: 12 },
  { supplier_id: 'SUP-003', name: 'CoolFlow Industries', component_category: 'coolant', quality_score: 85, defect_rate: 1.5, avg_resolution_time_hours: 36, total_parts_supplied: 8000, issues_reported: 32 },
];

const MOCK_TRENDS: FailureTrend[] = [
  { date: '2025-01-01', brake: 8, battery: 3, coolant: 5, suspension: 2, other: 4 },
  { date: '2025-01-08', brake: 12, battery: 2, coolant: 7, suspension: 3, other: 6 },
  { date: '2025-01-15', brake: 15, battery: 5, coolant: 4, suspension: 4, other: 3 },
  { date: '2025-01-22', brake: 10, battery: 8, coolant: 6, suspension: 2, other: 5 },
  { date: '2025-01-29', brake: 7, battery: 4, coolant: 8, suspension: 5, other: 4 },
  { date: '2025-02-05', brake: 5, battery: 3, coolant: 5, suspension: 3, other: 2 },
];

const MOCK_UEBA: UEBAEvent[] = [
  { id: 'EVT-001', agent_id: 'master_agent', event_type: 'volume_spike', severity: 'medium', description: 'Unusual request volume: 342 requests in 5 minutes (baseline: 50)', timestamp: '2025-02-07T02:15:00Z', resolved: false },
  { id: 'EVT-002', agent_id: 'diagnosis_agent', event_type: 'timing_anomaly', severity: 'low', description: 'Activity outside normal hours (3:45 AM)', timestamp: '2025-02-06T22:15:00Z', resolved: true },
  { id: 'EVT-003', agent_id: 'scheduling_agent', event_type: 'access_anomaly', severity: 'high', description: 'Attempted access to restricted CAPA endpoint', timestamp: '2025-02-06T14:30:00Z', resolved: false },
];

const MOCK_AGENTS: AgentProfile[] = [
  { agent_id: 'master_agent', agent_name: 'Master Orchestrator', normal_access_hours: '00:00-23:59', avg_daily_requests: 850, typical_endpoints: ['/analyze', '/workflow', '/status'], last_activity: '2025-02-07T03:10:00Z' },
  { agent_id: 'diagnosis_agent', agent_name: 'Diagnosis Agent', normal_access_hours: '06:00-22:00', avg_daily_requests: 420, typical_endpoints: ['/telemetry', '/rag/query', '/anomaly'], last_activity: '2025-02-07T03:05:00Z' },
  { agent_id: 'scheduling_agent', agent_name: 'Scheduling Agent', normal_access_hours: '08:00-20:00', avg_daily_requests: 180, typical_endpoints: ['/serviceops/schedule', '/find-slots'], last_activity: '2025-02-07T02:55:00Z' },
];

interface StoreState {
  fleetHealth: FleetHealth;
  capaPatterns: CAPAPattern[];
  suppliers: SupplierScorecard[];
  failureTrends: FailureTrend[];
  uebaEvents: UEBAEvent[];
  agentProfiles: AgentProfile[];
  activeView: string;
  setActiveView: (view: string) => void;
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useStore = create<StoreState>((set) => ({
  fleetHealth: { total_vehicles: 1250, healthy: 1089, warning: 138, critical: 23, avg_health_score: 87 },
  capaPatterns: MOCK_CAPA,
  suppliers: MOCK_SUPPLIERS,
  failureTrends: MOCK_TRENDS,
  uebaEvents: MOCK_UEBA,
  agentProfiles: MOCK_AGENTS,
  activeView: 'fleet',
  setActiveView: (view) => set({ activeView: view }),
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
