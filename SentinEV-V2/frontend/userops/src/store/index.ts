/**
 * Zustand Store - Global State Management
 */
import { create } from 'zustand';
import { TelemetryData, Vehicle, VehicleHealth, ServiceJob, ChatMessage, Severity } from '../types/api';

// User profile type
interface User {
  name: string;
  email: string;
  phone: string;
}

// Gamification state
interface GamificationState {
  drivingScore: number;
  totalPoints: number;
  sessionPoints: number;
  streak: number;
  badges: string[];
  nextServiceDays: number;
}

// Mock vehicle data (would come from API in production)
const MOCK_VEHICLES: Vehicle[] = [
  {
    id: 'VH001',
    vin: '1HGBH41JXMN109186',
    make: 'Tata',
    model: 'Nexon EV Max',
    year: 2024,
    mileage: 15420,
    health_score: 92.5,
    category: 'normal',
    driving_profile: 'eco',
  },
];

interface AppState {
  // User
  user: User | null;
  setUser: (user: User | null) => void;

  // Vehicles
  vehicles: Vehicle[];
  selectedVehicle: Vehicle | null;
  setSelectedVehicle: (vehicle: Vehicle | null) => void;

  // Telemetry
  telemetry: TelemetryData | null;
  setTelemetry: (data: TelemetryData | null) => void;
  isTelemetryActive: boolean;
  setTelemetryActive: (active: boolean) => void;

  // Health Analysis
  vehicleHealth: VehicleHealth | null;
  isAnalyzing: boolean;
  setVehicleHealth: (health: VehicleHealth | null) => void;
  setIsAnalyzing: (analyzing: boolean) => void;

  // Service Jobs
  activeJob: ServiceJob | null;
  setActiveJob: (job: ServiceJob | null) => void;

  // Chat
  chatMessages: ChatMessage[];
  addChatMessage: (message: ChatMessage) => void;
  clearChat: () => void;

  // Gamification
  gamification: GamificationState;
  addPoints: (points: number) => void;
  updateGamification: (updates: Partial<GamificationState>) => void;

  // UI State
  isLoading: boolean;
  setLoading: (loading: boolean) => void;
  
  // Incoming Call
  incomingCall: boolean;
  setIncomingCall: (incoming: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  // User
  user: null,
  setUser: (user) => set({ user }),

  // Vehicles
  vehicles: MOCK_VEHICLES,
  selectedVehicle: null,
  setSelectedVehicle: (vehicle) => set({ selectedVehicle: vehicle }),

  // Telemetry
  telemetry: null,
  setTelemetry: (data) => set({ telemetry: data }),
  isTelemetryActive: false,
  setTelemetryActive: (active) => set({ isTelemetryActive: active }),

  // Health
  vehicleHealth: null,
  isAnalyzing: false,
  setVehicleHealth: (health) => set({ vehicleHealth: health }),
  setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),

  // Jobs
  activeJob: null,
  setActiveJob: (job) => set({ activeJob: job }),

  // Chat
  chatMessages: [
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm your SentinEV AI assistant. I'm monitoring your vehicle's health in real-time. How can I help you today?",
      timestamp: new Date().toISOString(),
    },
  ],
  addChatMessage: (message) =>
    set((state) => ({ chatMessages: [...state.chatMessages, message] })),
  clearChat: () =>
    set({
      chatMessages: [
        {
          id: '1',
          role: 'assistant',
          content: "Hello! I'm your SentinEV AI assistant. How can I help you today?",
          timestamp: new Date().toISOString(),
        },
      ],
    }),

  // Gamification
  gamification: {
    drivingScore: 94,
    totalPoints: 2450,
    sessionPoints: 0,
    streak: 7,
    badges: ['🌿 Eco Champion', '🛡️ Safe Driver', '⚡ EV Pioneer'],
    nextServiceDays: 45,
  },
  addPoints: (points) =>
    set((state) => ({
      gamification: {
        ...state.gamification,
        totalPoints: state.gamification.totalPoints + points,
        sessionPoints: state.gamification.sessionPoints + points,
      },
    })),
  updateGamification: (updates) =>
    set((state) => ({
      gamification: { ...state.gamification, ...updates },
    })),

  // UI
  isLoading: false,
  setLoading: (loading) => set({ isLoading: loading }),
  
  incomingCall: false,
  setIncomingCall: (incoming) => set({ incomingCall: incoming }),
}));
