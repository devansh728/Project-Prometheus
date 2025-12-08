// SentinEV Chat Store - AI chatbot state

import { create } from 'zustand';
import { ChatMessage } from '@/types';

interface ChatStore {
    // Messages per vehicle
    messagesByVehicle: Record<string, ChatMessage[]>;

    // Session tracking
    sessionId: string | null;

    // Loading state
    isTyping: boolean;

    // Actions
    addMessage: (vehicleId: string, message: ChatMessage) => void;
    setTyping: (isTyping: boolean) => void;
    setSessionId: (sessionId: string) => void;
    clearMessages: (vehicleId: string) => void;
    getMessages: (vehicleId: string) => ChatMessage[];
}

export const useChatStore = create<ChatStore>((set, get) => ({
    // Initial state
    messagesByVehicle: {},
    sessionId: null,
    isTyping: false,

    // Actions
    addMessage: (vehicleId, message) =>
        set((state) => ({
            messagesByVehicle: {
                ...state.messagesByVehicle,
                [vehicleId]: [...(state.messagesByVehicle[vehicleId] || []), message],
            },
        })),

    setTyping: (isTyping) => set({ isTyping }),

    setSessionId: (sessionId) => set({ sessionId }),

    clearMessages: (vehicleId) =>
        set((state) => ({
            messagesByVehicle: {
                ...state.messagesByVehicle,
                [vehicleId]: [],
            },
        })),

    getMessages: (vehicleId) => get().messagesByVehicle[vehicleId] || [],
}));
