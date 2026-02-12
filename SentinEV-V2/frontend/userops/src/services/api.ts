/**
 * API Service - HTTP & WebSocket connections to backend
 */
import axios, { AxiosInstance } from 'axios';
import {
  TelemetryData,
  VehicleHealth,
  ScheduleRequest,
  ScheduleResponse,
  SchedulingSlot,
  ServiceJob,
  FleetTelemetry,
  WorkflowEvent,
} from '../types/api';

// Environment configuration
const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://10.79.149.8000/api/v1';
const WS_BASE_URL = process.env.EXPO_PUBLIC_WS_URL || 'ws://10.79.239.149:8000/api/v1';

class ApiService {
  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.http.interceptors.request.use(
      (config) => {
        // Add auth token if available
        // config.headers.Authorization = `Bearer ${token}`;
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.http.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // ---- Telemetry ----
  async getTelemetry(vehicleId: string): Promise<TelemetryData> {
    const response = await this.http.get(`/telemetry/${vehicleId}`);
    return response.data;
  }

  async getFleetTelemetry(): Promise<FleetTelemetry> {
    const response = await this.http.get('/telemetry/fleet');
    return response.data;
  }

  // ---- Agent Analysis ----
  async analyzeVehicle(vehicleId: string): Promise<VehicleHealth> {
    const response = await this.http.get(`/agent/analyze/quick/${vehicleId}`);
    return response.data;
  }

  // ---- Scheduling ----
  async findSlots(
    customerLat: number,
    customerLon: number,
    failureType: string,
    severity: string
  ): Promise<{ slots: SchedulingSlot[] }> {
    const response = await this.http.post('/serviceops/find-slots', {
      customer_lat: customerLat,
      customer_lon: customerLon,
      failure_type: failureType,
      severity,
      max_distance_km: 50,
    });
    return response.data;
  }

  async scheduleService(request: ScheduleRequest): Promise<ScheduleResponse> {
    const response = await this.http.post('/serviceops/schedule', request);
    return response.data;
  }

  async getJob(jobId: string): Promise<ServiceJob> {
    const response = await this.http.get(`/serviceops/jobs/${jobId}`);
    return response.data;
  }

  // ---- RAG ----
  async queryKnowledge(query: string, queryType: string = 'general') {
    const response = await this.http.post('/rag/query', {
      query,
      query_type: queryType,
      n_results: 5,
    });
    return response.data;
  }

  // ---- System Status ----
  async getStatus() {
    const response = await this.http.get('/status');
    return response.data;
  }
}

// WebSocket Service
class WebSocketService {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  connectTelemetry(vehicleId: string): void {
    if (this.ws) {
      this.ws.close();
    }

    this.ws = new WebSocket(`${WS_BASE_URL}/telemetry/stream/${vehicleId}`);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.emit('telemetry', data);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };

    this.ws.onclose = () => {
      this.emit('close', null);
    };
  }

  connectAgentStream(vehicleId: string): void {
    if (this.ws) {
      this.ws.close();
    }

    this.ws = new WebSocket(`${WS_BASE_URL}/agent/ws/${vehicleId}`);

    this.ws.onmessage = (event) => {
      const data: WorkflowEvent = JSON.parse(event.data);
      this.emit('workflow', data);
    };

    this.ws.onerror = (error) => {
      console.error('Agent WebSocket error:', error);
      this.emit('error', error);
    };

    this.ws.onclose = () => {
      this.emit('close', null);
    };
  }

  on(event: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.listeners.get(event)?.delete(callback);
    };
  }

  private emit(event: string, data: any): void {
    this.listeners.get(event)?.forEach((callback) => callback(data));
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Singletons
export const api = new ApiService();
export const wsService = new WebSocketService();
