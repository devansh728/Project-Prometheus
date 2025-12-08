// SentinEV WebSocket Hook - Fixed version

'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { useTelemetryStore } from '@/stores/telemetryStore';
import { useVehicleStore } from '@/stores/vehicleStore';
import type { WebSocketMessage, Prediction, SafetyAdvice, DiagnosisResult } from '@/types';

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/v1';

interface UseWebSocketOptions {
    onPrediction?: (prediction: Prediction) => void;
    onSafetyAdvice?: (advice: SafetyAdvice) => void;
    onDiagnosis?: (diagnosis: DiagnosisResult) => void;
    onTimeout?: () => void;
    onRedirectToChat?: (vehicleId: string, context: Record<string, unknown>) => void;
    onVoiceCallTrigger?: (vehicleId: string, alertType: string, context: Record<string, unknown>) => void;
}

export function useWebSocket(vin: string | null, options: UseWebSocketOptions = {}) {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectRef = useRef<NodeJS.Timeout | null>(null);
    const reconnectCountRef = useRef(0);
    const isCleaningUpRef = useRef(false);
    const vinRef = useRef(vin);

    const [isReconnecting, setIsReconnecting] = useState(false);
    const [isConnected, setIsConnectedState] = useState(false);

    // Store refs for options to avoid dependency issues
    const optionsRef = useRef(options);
    optionsRef.current = options;

    // Update vin ref
    vinRef.current = vin;

    // Get store actions via selectors (stable references)
    const storeSetConnected = useTelemetryStore((state) => state.setConnected);
    const storeSetTelemetry = useTelemetryStore((state) => state.setTelemetry);
    const storeSetAnomaly = useTelemetryStore((state) => state.setAnomaly);
    const storeSetScoring = useTelemetryStore((state) => state.setScoring);
    const storeSetScenario = useTelemetryStore((state) => state.setScenario);
    const storeSetPrediction = useTelemetryStore((state) => state.setPrediction);
    const storeAddNotification = useTelemetryStore((state) => state.addNotification);
    const storeReset = useTelemetryStore((state) => state.reset);
    const storeSetActiveScenario = useVehicleStore((state) => state.setActiveScenario);

    // Send command through WebSocket
    const sendCommand = useCallback((command: object) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(command));
            return true;
        }
        console.warn('WebSocket not connected');
        return false;
    }, []);

    // Command helpers
    const startScenario = useCallback((scenarioId: string) => {
        return sendCommand({ command: 'start_scenario', scenario_id: scenarioId });
    }, [sendCommand]);

    const stopScenario = useCallback(() => {
        return sendCommand({ command: 'stop_scenario' });
    }, [sendCommand]);

    const injectFault = useCallback((faultType: string, severity: number = 1.0) => {
        return sendCommand({ command: 'inject_fault', fault_type: faultType, severity });
    }, [sendCommand]);

    const acceptPrediction = useCallback((predictionId: string) => {
        return sendCommand({ command: 'accept_prediction', prediction_id: predictionId });
    }, [sendCommand]);

    const rejectPrediction = useCallback((predictionId: string) => {
        return sendCommand({ command: 'reject_prediction', prediction_id: predictionId });
    }, [sendCommand]);

    const confirmService = useCallback((diagnosisId: string) => {
        return sendCommand({ command: 'confirm_service', diagnosis_id: diagnosisId });
    }, [sendCommand]);

    const declineService = useCallback((diagnosisId: string) => {
        return sendCommand({ command: 'decline_service', diagnosis_id: diagnosisId });
    }, [sendCommand]);

    // Connect on mount with vin, disconnect on unmount
    useEffect(() => {
        if (!vin) return;

        isCleaningUpRef.current = false;
        reconnectCountRef.current = 0;

        const wsUrl = `${WS_BASE}/vehicles/${vin}/scenario-stream`;
        console.log('Connecting to WebSocket:', wsUrl);

        const createConnection = () => {
            if (isCleaningUpRef.current) return;

            try {
                const ws = new WebSocket(wsUrl);
                wsRef.current = ws;

                ws.onopen = () => {
                    console.log('WebSocket connected');
                    setIsConnectedState(true);
                    storeSetConnected(true, vin);
                    setIsReconnecting(false);
                    reconnectCountRef.current = 0;
                };

                ws.onmessage = (event) => {
                    try {
                        const message: WebSocketMessage = JSON.parse(event.data);

                        switch (message.type) {
                            case 'connection':
                                console.log('Connection confirmed:', message.available_commands);
                                break;

                            case 'telemetry':
                                if (message.data) {
                                    storeSetTelemetry(message.data.telemetry);
                                    storeSetAnomaly(message.data.anomaly);
                                    storeSetScoring(message.data.scoring);
                                }
                                if (message.scenario) {
                                    storeSetScenario(message.scenario);
                                }
                                if (message.prediction) {
                                    storeSetPrediction(message.prediction);
                                    optionsRef.current.onPrediction?.(message.prediction);
                                }
                                if (message.notifications) {
                                    message.notifications.forEach(n => storeAddNotification(n));
                                }
                                break;

                            case 'scenario_started':
                                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                storeSetActiveScenario((message as any).scenario_id || null);
                                break;

                            case 'scenario_stopped':
                                storeSetActiveScenario(null);
                                storeSetScenario(null);
                                break;

                            case 'prediction_accepted':
                                storeSetPrediction(null);
                                if (message.advice) {
                                    optionsRef.current.onSafetyAdvice?.(message.advice);
                                }
                                break;

                            case 'prediction_rejected':
                                storeSetPrediction(null);
                                break;

                            case 'prediction_timeout':
                                optionsRef.current.onTimeout?.();
                                break;

                            case 'auto_diagnosis':
                                storeSetPrediction(null);
                                if (message.diagnosis) {
                                    optionsRef.current.onDiagnosis?.(message.diagnosis);
                                }
                                break;

                            case 'service_confirmed':
                            case 'service_declined':
                                break;

                            case 'error':
                                console.error('WebSocket error message:', message);
                                break;

                            case 'redirect_to_chat':
                                // Store context and trigger redirect
                                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                const redirectMsg = message as any;
                                if (redirectMsg.context) {
                                    localStorage.setItem('chatContext', JSON.stringify(redirectMsg.context));
                                }
                                optionsRef.current.onRedirectToChat?.(redirectMsg.vehicle_id, redirectMsg.context);
                                break;

                            case 'voice_call_trigger':
                                // Trigger voice call for critical brake scenarios
                                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                const voiceMsg = message as any;
                                console.log('🔊 Voice call trigger received:', voiceMsg);
                                optionsRef.current.onVoiceCallTrigger?.(
                                    voiceMsg.vehicle_id,
                                    voiceMsg.alert_type,
                                    voiceMsg.context
                                );
                                break;
                        }
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };

                ws.onerror = () => {
                    // Don't log the error object as it's empty in browsers
                    console.log('WebSocket connection error');
                };

                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    setIsConnectedState(false);
                    storeSetConnected(false, undefined);

                    // Auto-reconnect with backoff, max 5 attempts
                    if (!isCleaningUpRef.current && vinRef.current && reconnectCountRef.current < 5) {
                        setIsReconnecting(true);
                        reconnectCountRef.current++;
                        const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current), 30000);
                        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectCountRef.current})`);

                        reconnectRef.current = setTimeout(() => {
                            reconnectRef.current = null;
                            createConnection();
                        }, delay);
                    } else if (reconnectCountRef.current >= 5) {
                        console.log('Max reconnection attempts reached');
                        setIsReconnecting(false);
                    }
                };
            } catch (error) {
                console.error('Failed to create WebSocket:', error);
            }
        };

        createConnection();

        // Cleanup function
        return () => {
            console.log('Cleaning up WebSocket connection');
            isCleaningUpRef.current = true;

            if (reconnectRef.current) {
                clearTimeout(reconnectRef.current);
                reconnectRef.current = null;
            }
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            setIsConnectedState(false);
            setIsReconnecting(false);
            reconnectCountRef.current = 0;
            storeReset();
        };
    }, [vin]); // ONLY depend on vin - all store functions are stable

    // Manual disconnect
    const disconnect = useCallback(() => {
        isCleaningUpRef.current = true;
        if (reconnectRef.current) {
            clearTimeout(reconnectRef.current);
            reconnectRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnectedState(false);
        setIsReconnecting(false);
        storeReset();
    }, [storeReset]);

    // Manual reconnect
    const connect = useCallback(() => {
        // This will trigger useEffect by changing the vin ref... but actually we need to 
        // just recreate. For now, just reload the page or navigate away and back.
        console.log('Manual reconnect not implemented - refresh the page');
    }, []);

    return {
        isConnected,
        isReconnecting,
        sendCommand,
        startScenario,
        stopScenario,
        injectFault,
        acceptPrediction,
        rejectPrediction,
        confirmService,
        declineService,
        connect,
        disconnect,
    };
}
