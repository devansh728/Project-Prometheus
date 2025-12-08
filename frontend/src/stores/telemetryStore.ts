// SentinEV Telemetry Store - Real-time vehicle data

import { create } from 'zustand';
import { TelemetryData, AnomalyData, ScoringData, ScenarioState, Prediction, Notification } from '@/types';

interface TelemetryStore {
    // Connection state
    isConnected: boolean;
    vehicleId: string | null;

    // Real-time data
    telemetry: TelemetryData | null;
    anomaly: AnomalyData | null;
    scoring: ScoringData | null;
    scenario: ScenarioState | null;

    // Prediction tracking
    activePrediction: Prediction | null;
    predictionTimeout: number | null;

    // Notifications
    notifications: Notification[];

    // Actions
    setConnected: (connected: boolean, vehicleId?: string) => void;
    setTelemetry: (data: TelemetryData) => void;
    setAnomaly: (data: AnomalyData) => void;
    setScoring: (data: ScoringData) => void;
    setScenario: (scenario: ScenarioState | null) => void;
    setPrediction: (prediction: Prediction | null) => void;
    addNotification: (notification: Notification) => void;
    markNotificationRead: (id: string) => void;
    clearNotifications: () => void;
    reset: () => void;
}

export const useTelemetryStore = create<TelemetryStore>((set) => ({
    // Initial state
    isConnected: false,
    vehicleId: null,
    telemetry: null,
    anomaly: null,
    scoring: null,
    scenario: null,
    activePrediction: null,
    predictionTimeout: null,
    notifications: [],

    // Actions
    setConnected: (connected, vehicleId) =>
        set({ isConnected: connected, vehicleId: vehicleId || null }),

    setTelemetry: (data) => set({ telemetry: data }),

    setAnomaly: (data) => set({ anomaly: data }),

    setScoring: (data) => set({ scoring: data }),

    setScenario: (scenario) => set({ scenario }),

    setPrediction: (prediction) => set({
        activePrediction: prediction,
        predictionTimeout: prediction?.timeout_seconds || null
    }),

    addNotification: (notification) =>
        set((state) => ({
            notifications: [notification, ...state.notifications].slice(0, 50) // Keep last 50
        })),

    markNotificationRead: (id) =>
        set((state) => ({
            notifications: state.notifications.map(n =>
                n.id === id ? { ...n, read: true } : n
            )
        })),

    clearNotifications: () => set({ notifications: [] }),

    reset: () => set({
        isConnected: false,
        vehicleId: null,
        telemetry: null,
        anomaly: null,
        scoring: null,
        scenario: null,
        activePrediction: null,
        predictionTimeout: null,
        notifications: [],
    }),
}));
