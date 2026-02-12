/**
 * ServiceOps API Service
 * HTTP client for backend integration
 */
import axios, { AxiosInstance } from 'axios';
import {
  ServiceJob,
  Mechanic,
  InventoryPart,
  ServiceCenter,
  ServiceState,
  DashboardMetrics,
  MechanicWorkload,
} from '../types/api';

const API_BASE = '/api/v1/serviceops';

class ServiceOpsApi {
  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({
      baseURL: API_BASE,
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // Jobs
  async getJobs(state?: ServiceState): Promise<ServiceJob[]> {
    const params = state ? { state } : {};
    const response = await this.http.get('/jobs', { params });
    return response.data.jobs || [];
  }

  async getJob(jobId: string): Promise<ServiceJob> {
    const response = await this.http.get(`/jobs/${jobId}`);
    return response.data;
  }

  async updateJobState(jobId: string, newState: ServiceState): Promise<ServiceJob> {
    const response = await this.http.post(`/jobs/${jobId}/transition`, {
      new_state: newState,
    });
    return response.data;
  }

  async assignMechanic(jobId: string, mechanicId: string): Promise<ServiceJob> {
    const response = await this.http.post(`/jobs/${jobId}/assign`, {
      mechanic_id: mechanicId,
    });
    return response.data;
  }

  // Service Centers
  async getServiceCenters(): Promise<ServiceCenter[]> {
    const response = await this.http.get('/service-centers');
    return response.data.centers || [];
  }

  // Mechanics
  async getMechanics(centerId?: string): Promise<Mechanic[]> {
    const params = centerId ? { center_id: centerId } : {};
    const response = await this.http.get('/mechanics', { params });
    return response.data.mechanics || [];
  }

  async getWorkload(centerId: string): Promise<MechanicWorkload[]> {
    const response = await this.http.get(`/workload/${centerId}`);
    return response.data.workload || [];
  }

  // Inventory
  async getInventory(centerId: string): Promise<InventoryPart[]> {
    const response = await this.http.get(`/inventory/${centerId}`);
    return response.data.parts || [];
  }

  async getLowStockAlerts(centerId: string): Promise<InventoryPart[]> {
    const response = await this.http.get(`/inventory/${centerId}/alerts`);
    return response.data.low_stock_alerts || [];
  }

  // Dashboard Metrics (mock aggregation)
  async getMetrics(): Promise<DashboardMetrics> {
    // In production, this would be a dedicated endpoint
    return {
      total_active_jobs: 24,
      jobs_by_state: {
        REQUESTED: 3,
        BOOKED: 5,
        CONFIRMED: 2,
        CHECK_IN: 4,
        DIAGNOSIS: 3,
        REPAIR_IN_PROGRESS: 5,
        QUALITY_CHECK: 2,
      },
      avg_completion_time_hours: 4.2,
      mechanic_utilization: 78,
      parts_low_stock: 3,
      customer_satisfaction: 4.7,
    };
  }
}

export const api = new ServiceOpsApi();
