/**
 * Zustand Store - Global State for ServiceOps Dashboard
 */
import { create } from 'zustand';
import { ServiceJob, Mechanic, InventoryPart, ServiceState, DashboardMetrics } from '../types/api';

// Mock data for demo
const MOCK_JOBS: ServiceJob[] = [
  {
    job_id: 'JOB-001',
    vehicle_id: 'VH001',
    customer_id: 'CUST-001',
    service_center_id: 'SC001',
    failure_type: 'brake_degradation',
    severity: 'WARNING',
    state: ServiceState.DIAGNOSIS,
    mechanic_id: 'M001',
    scheduled_at: new Date().toISOString(),
    created_at: new Date(Date.now() - 3600000).toISOString(),
    completed_at: null,
    estimated_completion: new Date(Date.now() + 7200000).toISOString(),
    state_history: [
      { from: null, to: 'REQUESTED', timestamp: new Date(Date.now() - 3600000).toISOString() },
      { from: 'REQUESTED', to: 'BOOKED', timestamp: new Date(Date.now() - 3000000).toISOString() },
      { from: 'BOOKED', to: 'CHECK_IN', timestamp: new Date(Date.now() - 2400000).toISOString() },
      { from: 'CHECK_IN', to: 'DIAGNOSIS', timestamp: new Date(Date.now() - 1800000).toISOString() },
    ],
  },
  {
    job_id: 'JOB-002',
    vehicle_id: 'VH002',
    customer_id: 'CUST-002',
    service_center_id: 'SC001',
    failure_type: 'battery_thermal',
    severity: 'CRITICAL',
    state: ServiceState.REPAIR_IN_PROGRESS,
    mechanic_id: 'M002',
    scheduled_at: new Date().toISOString(),
    created_at: new Date(Date.now() - 7200000).toISOString(),
    completed_at: null,
    estimated_completion: new Date(Date.now() + 3600000).toISOString(),
    state_history: [],
  },
  {
    job_id: 'JOB-003',
    vehicle_id: 'VH003',
    customer_id: 'CUST-003',
    service_center_id: 'SC001',
    failure_type: 'coolant_leak',
    severity: 'WARNING',
    state: ServiceState.REQUESTED,
    mechanic_id: null,
    scheduled_at: null,
    created_at: new Date(Date.now() - 1800000).toISOString(),
    completed_at: null,
    estimated_completion: null,
    state_history: [],
  },
  {
    job_id: 'JOB-004',
    vehicle_id: 'VH004',
    customer_id: 'CUST-004',
    service_center_id: 'SC001',
    failure_type: 'suspension_wear',
    severity: 'INFO',
    state: ServiceState.QUALITY_CHECK,
    mechanic_id: 'M001',
    scheduled_at: new Date().toISOString(),
    created_at: new Date(Date.now() - 14400000).toISOString(),
    completed_at: null,
    estimated_completion: new Date(Date.now() + 1800000).toISOString(),
    state_history: [],
  },
  {
    job_id: 'JOB-005',
    vehicle_id: 'VH005',
    customer_id: 'CUST-005',
    service_center_id: 'SC001',
    failure_type: 'tire_pressure',
    severity: 'WARNING',
    state: ServiceState.PARTS_ALLOCATED,
    mechanic_id: 'M003',
    scheduled_at: new Date().toISOString(),
    created_at: new Date(Date.now() - 5400000).toISOString(),
    completed_at: null,
    estimated_completion: new Date(Date.now() + 5400000).toISOString(),
    state_history: [],
  },
];

const MOCK_MECHANICS: Mechanic[] = [
  { id: 'M001', name: 'Rajesh Kumar', skills: ['brake', 'suspension', 'general'], certification_level: 'Senior', current_load: 2, max_daily_capacity: 4, efficiency_rating: 4.8, available: true },
  { id: 'M002', name: 'Amit Sharma', skills: ['battery', 'electrical', 'thermal'], certification_level: 'Expert', current_load: 1, max_daily_capacity: 3, efficiency_rating: 4.9, available: true },
  { id: 'M003', name: 'Priya Patel', skills: ['coolant', 'engine', 'general'], certification_level: 'Senior', current_load: 1, max_daily_capacity: 4, efficiency_rating: 4.7, available: true },
  { id: 'M004', name: 'Vikram Singh', skills: ['tire', 'brake', 'alignment'], certification_level: 'Standard', current_load: 0, max_daily_capacity: 5, efficiency_rating: 4.5, available: true },
];

const MOCK_PARTS: InventoryPart[] = [
  { part_id: 'P001', name: 'Brake Pad Set', quantity: 8, reorder_point: 5, cost: 2500, compatible_components: ['brake'] },
  { part_id: 'P002', name: 'Coolant 1L', quantity: 3, reorder_point: 10, cost: 450, compatible_components: ['coolant'] },
  { part_id: 'P003', name: 'Battery Cell Module', quantity: 2, reorder_point: 3, cost: 15000, compatible_components: ['battery'] },
  { part_id: 'P004', name: 'Suspension Spring', quantity: 12, reorder_point: 4, cost: 3200, compatible_components: ['suspension'] },
  { part_id: 'P005', name: 'Tire 205/55R16', quantity: 6, reorder_point: 8, cost: 5500, compatible_components: ['tire'] },
];

interface StoreState {
  // Jobs
  jobs: ServiceJob[];
  selectedJob: ServiceJob | null;
  setJobs: (jobs: ServiceJob[]) => void;
  setSelectedJob: (job: ServiceJob | null) => void;
  updateJobState: (jobId: string, newState: ServiceState) => void;

  // Mechanics
  mechanics: Mechanic[];
  setMechanics: (mechanics: Mechanic[]) => void;

  // Inventory
  parts: InventoryPart[];
  setParts: (parts: InventoryPart[]) => void;

  // UI
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  activeView: string;
  setActiveView: (view: string) => void;
}

export const useStore = create<StoreState>((set) => ({
  // Jobs
  jobs: MOCK_JOBS,
  selectedJob: null,
  setJobs: (jobs) => set({ jobs }),
  setSelectedJob: (job) => set({ selectedJob: job }),
  updateJobState: (jobId, newState) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.job_id === jobId ? { ...job, state: newState } : job
      ),
    })),

  // Mechanics
  mechanics: MOCK_MECHANICS,
  setMechanics: (mechanics) => set({ mechanics }),

  // Inventory
  parts: MOCK_PARTS,
  setParts: (parts) => set({ parts }),

  // UI
  sidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  activeView: 'dashboard',
  setActiveView: (view) => set({ activeView: view }),
}));
