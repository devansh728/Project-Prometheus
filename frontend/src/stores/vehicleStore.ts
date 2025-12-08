// SentinEV Vehicle Store - Vehicle management state

import { create } from 'zustand';
import { Vehicle, Scenario } from '@/types';

interface VehicleStore {
    // Vehicles
    vehicles: Vehicle[];
    selectedVehicle: string | null;

    // Scenarios
    scenarios: Scenario[];
    activeScenario: string | null;

    // Loading states
    isLoading: boolean;
    error: string | null;

    // Actions
    setVehicles: (vehicles: Vehicle[]) => void;
    addVehicle: (vehicle: Vehicle) => void;
    selectVehicle: (vin: string | null) => void;
    setScenarios: (scenarios: Scenario[]) => void;
    setActiveScenario: (scenarioId: string | null) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
}

export const useVehicleStore = create<VehicleStore>((set) => ({
    // Initial state
    vehicles: [],
    selectedVehicle: null,
    scenarios: [],
    activeScenario: null,
    isLoading: false,
    error: null,

    // Actions
    setVehicles: (vehicles) => set({ vehicles }),

    addVehicle: (vehicle) =>
        set((state) => ({
            vehicles: [...state.vehicles.filter(v => v.vehicle_id !== vehicle.vehicle_id), vehicle]
        })),

    selectVehicle: (vin) => set({ selectedVehicle: vin }),

    setScenarios: (scenarios) => set({ scenarios }),

    setActiveScenario: (scenarioId) => set({ activeScenario: scenarioId }),

    setLoading: (loading) => set({ isLoading: loading }),

    setError: (error) => set({ error }),
}));
