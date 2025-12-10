// Vehicle Monitor Page - Real-time telemetry with predictions

'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
    Gauge, Thermometer, Battery, Zap, Activity,
    AlertTriangle, Play, Square, MessageSquare,
    CheckCircle, XCircle, Clock, Wrench, Phone
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Separator } from '@/components/ui/separator';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { CriticalAlertModal } from '@/components/ui/CriticalAlertModal';
import { VoiceCallModal } from '@/components/ui/VoiceCallModal';
import { AgentConsoleLog } from '@/components/ui/AgentConsoleLog';
import { FailurePredictionCard } from '@/components/ui/FailurePredictionCard';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useTelemetryStore } from '@/stores/telemetryStore';
import { useVehicleStore } from '@/stores/vehicleStore';
import { listScenarios, initializeVehicle } from '@/lib/api';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import type { Prediction, SafetyAdvice, DiagnosisResult } from '@/types';

export default function VehicleMonitorPage() {
    const params = useParams();
    const router = useRouter();
    const vin = params.vin as string;

    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [selectedScenario, setSelectedScenario] = useState<string>('');
    const [predictionModalOpen, setPredictionModalOpen] = useState(false);
    const [safetyAdvice, setSafetyAdvice] = useState<SafetyAdvice | null>(null);
    const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
    const [countdown, setCountdown] = useState<number | null>(null);

    // Critical Alert & Voice Call state (for brake_fade scenario)
    const [criticalAlertOpen, setCriticalAlertOpen] = useState(false);
    const [voiceCallOpen, setVoiceCallOpen] = useState(false);
    const [criticalAlertData, setCriticalAlertData] = useState<{
        type: string;
        component: string;
        severity: string;
        message: string;
        efficiency?: number;
        temperature?: number;
        timestamp: string;
        vehicle_id: string;
    } | null>(null);

    const { scenarios, setScenarios, activeScenario } = useVehicleStore();
    const {
        isConnected, telemetry, anomaly, scoring, scenario,
        activePrediction, notifications
    } = useTelemetryStore();

    // Watch for critical anomalies (brake_fade scenario)
    // Track if we've already shown the alert to prevent repeated triggers
    const [hasShownCriticalAlert, setHasShownCriticalAlert] = useState(false);

    useEffect(() => {
        // Multiple ways to detect brake-related critical issues:
        // 1. Anomaly type contains brake/thermal_brake/fade
        const typeContainsBrake = anomaly?.type?.toLowerCase().includes('brake') ||
            anomaly?.type?.toLowerCase().includes('fade') ||
            anomaly?.type?.toLowerCase().includes('thermal');

        // 2. Scenario event is brake-related
        const scenarioIsBrake = scenario?.event?.toLowerCase().includes('brake') ||
            scenario?.description?.toLowerCase().includes('brake');

        // 3. Brake temperature is critically high (direct telemetry check)
        const brakeOverheating = (telemetry?.brake_temp_c || 0) > 300;

        const isBrakeAnomaly = typeContainsBrake || scenarioIsBrake || brakeOverheating;

        // Check if it's critical severity
        const isCriticalLevel = anomaly?.severity === 'critical' ||
            anomaly?.severity === 'high' ||
            (anomaly?.failure_risk_pct && anomaly.failure_risk_pct > 70) ||
            (brakeOverheating && anomaly?.is_anomaly);

        if (anomaly?.is_anomaly && isCriticalLevel && isBrakeAnomaly && !hasShownCriticalAlert) {
            console.log('🚨 Critical brake anomaly detected!', {
                type: anomaly.type,
                severity: anomaly.severity,
                failure_risk: anomaly.failure_risk_pct,
                brake_temp: telemetry?.brake_temp_c,
                scenario_event: scenario?.event
            });
            setCriticalAlertData({
                type: anomaly.type || 'brake_fade',
                component: 'Brakes',
                severity: 'critical',
                message: `Brake fade detected! Brake efficiency at ${(100 - (anomaly.failure_risk_pct || 85)).toFixed(0)}%. Immediate service required.`,
                efficiency: Math.max(15, 100 - (anomaly.failure_risk_pct || 85)),
                temperature: telemetry?.brake_temp_c || 350,
                timestamp: new Date().toISOString(),
                vehicle_id: vin
            });
            setCriticalAlertOpen(true);
            setHasShownCriticalAlert(true);
        }
    }, [anomaly, telemetry, vin, scenario, hasShownCriticalAlert]);

    // Initialize vehicle first
    useEffect(() => {
        async function init() {
            try {
                await initializeVehicle(vin, 'normal');
                const scenariosRes = await listScenarios();
                setScenarios(scenariosRes.scenarios || []);
            } catch (error) {
                console.log('Vehicle may already be initialized');
            }
        }
        init();
    }, [vin, setScenarios]);

    // Prediction handlers
    const handlePrediction = useCallback((prediction: Prediction) => {
        setPredictionModalOpen(true);
        setCountdown(prediction.timeout_seconds || 30);
    }, []);

    const handleSafetyAdvice = useCallback((advice: SafetyAdvice) => {
        setPredictionModalOpen(false);
        setSafetyAdvice(advice);
        toast.success(`+${advice.points_awarded} points!`, { description: 'Warning accepted' });
    }, []);

    const handleDiagnosis = useCallback((diag: DiagnosisResult) => {
        setPredictionModalOpen(false);
        setDiagnosis(diag);
        toast.warning('Diagnosis required', { description: diag.title });
    }, []);

    const handleTimeout = useCallback(() => {
        toast.warning('Prediction timeout', { description: 'Routing to diagnosis...' });
    }, []);

    // Handle redirect to chat for scheduling
    const handleRedirectToChat = useCallback((vehicleId: string, context: Record<string, unknown>) => {
        toast.success('Opening Scheduler...', { description: 'Redirecting to chat for appointment booking' });
        // Navigate to chat page with auto flag
        router.push(`/chat/${vehicleId}?auto=true`);
    }, [router]);

    // Handle voice call trigger for critical brake scenarios
    const handleVoiceCallTrigger = useCallback((vehicleId: string, alertType: string, context: Record<string, unknown>) => {
        console.log('🔊 Voice call trigger received in Vehicle Monitor:', { vehicleId, alertType, context });

        // Set the critical alert data and open the voice call modal directly
        setCriticalAlertData({
            type: alertType,
            component: String(context.component || 'Brakes'),
            severity: 'critical',
            message: String(context.summary || 'Critical brake failure detected. Immediate service required.'),
            efficiency: Number(context.brake_efficiency) || 15,
            temperature: Number(context.temperature) || 350,
            timestamp: new Date().toISOString(),
            vehicle_id: vehicleId
        });

        // Open the critical alert modal first, which will then trigger voice call
        setCriticalAlertOpen(true);
        setHasShownCriticalAlert(true);

        toast.warning('🚨 CRITICAL ALERT', { description: 'Incoming call from SentinEV AI' });
    }, []);

    // WebSocket connection
    const {
        startScenario, stopScenario, acceptPrediction, rejectPrediction,
        confirmService, declineService, isReconnecting
    } = useWebSocket(vin, {
        onPrediction: handlePrediction,
        onSafetyAdvice: handleSafetyAdvice,
        onDiagnosis: handleDiagnosis,
        onTimeout: handleTimeout,
        onRedirectToChat: handleRedirectToChat,
        onVoiceCallTrigger: handleVoiceCallTrigger,
    });

    // Countdown timer for prediction
    useEffect(() => {
        if (countdown === null || countdown <= 0) return;

        const timer = setInterval(() => {
            setCountdown(prev => {
                if (prev === null || prev <= 1) {
                    clearInterval(timer);
                    return null;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(timer);
    }, [countdown]);

    // Handle scenario control
    const handleStartScenario = () => {
        if (selectedScenario) {
            startScenario(selectedScenario);
            toast.info('Scenario started', { description: `Running ${selectedScenario}` });
        }
    };

    const handleStopScenario = () => {
        stopScenario();
        toast.info('Scenario stopped');
    };

    // Handle prediction response
    const handleAccept = () => {
        if (activePrediction) {
            acceptPrediction(activePrediction.prediction_id);
            setPredictionModalOpen(false);
        }
    };

    const handleReject = () => {
        if (activePrediction) {
            rejectPrediction(activePrediction.prediction_id);
            setPredictionModalOpen(false);
        }
    };

    // Helper for severity colors
    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical': return 'text-red-500 bg-red-500/10 border-red-500';
            case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500';
            case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500';
            default: return 'text-blue-500 bg-blue-500/10 border-blue-500';
        }
    };

    // Helper for temp color
    const getTempColor = (temp: number, thresholds: [number, number]) => {
        if (temp >= thresholds[1]) return 'text-red-500';
        if (temp >= thresholds[0]) return 'text-yellow-500';
        return 'text-green-500';
    };

    return (
        <div className="min-h-screen flex flex-col">
            <Navbar onMenuClick={() => setSidebarOpen(true)} />

            <div className="flex flex-1">
                <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                <main className="flex-1 p-6 overflow-auto">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h1 className="text-2xl font-bold flex items-center gap-2">
                                <Gauge className="h-6 w-6" />
                                Vehicle: {vin}
                            </h1>
                            <div className="flex items-center gap-2 mt-1">
                                <div className={cn(
                                    'h-2 w-2 rounded-full',
                                    isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                                )} />
                                <span className="text-sm text-muted-foreground">
                                    {isConnected ? 'Connected' : isReconnecting ? 'Reconnecting...' : 'Disconnected'}
                                </span>
                            </div>
                        </div>

                        <Link href={`/chat/${vin}`}>
                            <Button variant="outline">
                                <MessageSquare className="mr-2 h-4 w-4" />
                                AI Chat
                            </Button>
                        </Link>
                    </div>

                    {/* Scenario Control */}
                    <Card className="mb-6">
                        <CardHeader>
                            <CardTitle className="text-lg">Scenario Control</CardTitle>
                            <CardDescription>Start a test scenario to simulate driving conditions</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex items-center gap-4">
                                <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                                    <SelectTrigger className="w-64">
                                        <SelectValue placeholder="Select scenario..." />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {scenarios.map((s) => (
                                            <SelectItem key={s.id} value={s.id}>
                                                {s.name}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>

                                {activeScenario ? (
                                    <Button variant="destructive" onClick={handleStopScenario}>
                                        <Square className="mr-2 h-4 w-4" />
                                        Stop
                                    </Button>
                                ) : (
                                    <Button onClick={handleStartScenario} disabled={!selectedScenario}>
                                        <Play className="mr-2 h-4 w-4" />
                                        Start
                                    </Button>
                                )}

                                {scenario && (
                                    <Badge variant="outline" className="capitalize ml-4">
                                        Phase: {scenario.phase}
                                    </Badge>
                                )}
                            </div>

                            {scenario?.description && (
                                <p className="text-sm text-muted-foreground mt-2">
                                    {scenario.description}
                                </p>
                            )}
                        </CardContent>
                    </Card>

                    {/* Main Grid */}
                    <div className="grid gap-6 lg:grid-cols-3">
                        {/* Telemetry Column */}
                        <div className="lg:col-span-2 space-y-6">
                            {/* Speed & Battery */}
                            <div className="grid gap-4 sm:grid-cols-2">
                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                                            <Gauge className="h-4 w-4" />
                                            Speed
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-4xl font-bold">
                                            {telemetry?.speed_kmh?.toFixed(0) || 0}
                                            <span className="text-lg font-normal text-muted-foreground ml-1">km/h</span>
                                        </div>
                                        <Progress
                                            value={(telemetry?.speed_kmh || 0) / 2}
                                            className="mt-2 h-2"
                                        />
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                                            <Battery className="h-4 w-4" />
                                            Battery
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-4xl font-bold">
                                            {telemetry?.battery_soc_pct?.toFixed(0) || 0}
                                            <span className="text-lg font-normal text-muted-foreground ml-1">%</span>
                                        </div>
                                        <Progress
                                            value={telemetry?.battery_soc_pct || 0}
                                            className="mt-2 h-2"
                                        />
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Temperatures */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                                        <Thermometer className="h-4 w-4" />
                                        Temperatures
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                                        <div>
                                            <p className="text-sm text-muted-foreground">Battery</p>
                                            <p className={cn('text-2xl font-bold', getTempColor(telemetry?.battery_temp_c || 0, [35, 50]))}>
                                                {telemetry?.battery_temp_c?.toFixed(1) || 0}°C
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Motor</p>
                                            <p className={cn('text-2xl font-bold', getTempColor(telemetry?.motor_temp_c || 0, [85, 100]))}>
                                                {telemetry?.motor_temp_c?.toFixed(1) || 0}°C
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Inverter</p>
                                            <p className={cn('text-2xl font-bold', getTempColor(telemetry?.inverter_temp_c || 0, [75, 90]))}>
                                                {telemetry?.inverter_temp_c?.toFixed(1) || 0}°C
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Brakes</p>
                                            <p className={cn('text-2xl font-bold', getTempColor(telemetry?.brake_temp_c || 0, [200, 350]))}>
                                                {telemetry?.brake_temp_c?.toFixed(1) || 0}°C
                                            </p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Power & Regen */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                                        <Zap className="h-4 w-4" />
                                        Power
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid gap-4 sm:grid-cols-3">
                                        <div>
                                            <p className="text-sm text-muted-foreground">Power Draw</p>
                                            <p className="text-2xl font-bold">
                                                {telemetry?.power_draw_kw?.toFixed(1) || 0} kW
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Regen</p>
                                            <p className="text-2xl font-bold text-green-500">
                                                {telemetry?.regen_power_kw?.toFixed(1) || 0} kW
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Regen Efficiency</p>
                                            <p className="text-2xl font-bold">
                                                {((telemetry?.regen_efficiency || 0) * 100).toFixed(0)}%
                                            </p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Agent Console Log - Demo */}
                            <AgentConsoleLog
                                anomalyScore={(anomaly?.failure_risk_pct || 0) / 100}
                                failureProbability={anomaly?.failure_risk_pct || 0}
                                isAnomaly={anomaly?.is_anomaly || false}
                                severity={anomaly?.severity || 'low'}
                                anomalyType={anomaly?.type || 'normal'}
                                brakeTemp={telemetry?.brake_temp_c || 0}
                            />
                        </div>

                        {/* Right Column - Scores & Anomalies */}
                        <div className="space-y-6">
                            {/* Score Card */}
                            <Card className={cn(
                                'border-2 transition-colors',
                                scoring && scoring.delta > 0 ? 'border-green-500/50' :
                                    scoring && scoring.delta < 0 ? 'border-red-500/50' : 'border-transparent'
                            )}>
                                <CardHeader>
                                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                                        <Activity className="h-4 w-4" />
                                        Driver Score
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="text-center">
                                        <div className={cn(
                                            'text-5xl font-bold',
                                            scoring && scoring.total >= 0 ? 'text-green-500' : 'text-red-500'
                                        )}>
                                            {scoring?.total || 0}
                                        </div>
                                        {scoring && scoring.delta !== 0 && (
                                            <div className={cn(
                                                'text-lg font-medium mt-1',
                                                scoring.delta > 0 ? 'text-green-500' : 'text-red-500'
                                            )}>
                                                {scoring.delta > 0 ? '+' : ''}{scoring.delta}
                                            </div>
                                        )}
                                        {scoring?.feedback && (
                                            <p className="text-sm text-muted-foreground mt-2">
                                                {scoring.feedback}
                                            </p>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Failure Prediction Card - Enhanced for Demo */}
                            <FailurePredictionCard
                                failureProbability={anomaly?.failure_risk_pct || 0}
                                anomalyScore={(anomaly?.failure_risk_pct || 0) / 100}
                                severity={anomaly?.severity || 'low'}
                                component={anomaly?.type?.includes('brake') ? 'Brakes' :
                                    anomaly?.type?.includes('battery') ? 'Battery' :
                                        anomaly?.type?.includes('motor') ? 'Motor' : 'System'}
                                daysToFailure={3}
                            />

                            {/* Notifications */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm font-medium">Recent Notifications</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="h-48 overflow-y-auto">
                                        {notifications.length === 0 ? (
                                            <p className="text-sm text-muted-foreground text-center py-4">
                                                No notifications
                                            </p>
                                        ) : (
                                            <div className="space-y-2">
                                                {notifications.slice(0, 10).map((n, index) => (
                                                    <div key={`${n.id}-${index}`} className="text-sm p-2 rounded bg-muted/50">
                                                        {n.message}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>

                    {/* Prediction Modal */}
                    <Dialog open={predictionModalOpen} onOpenChange={setPredictionModalOpen}>
                        <DialogContent className="sm:max-w-lg">
                            <DialogHeader>
                                <DialogTitle className="flex items-center gap-2">
                                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                                    Prediction Warning
                                </DialogTitle>
                                <DialogDescription>
                                    {activePrediction?.message}
                                </DialogDescription>
                            </DialogHeader>

                            {activePrediction && (
                                <div className="py-4 space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-muted-foreground">Component</span>
                                        <Badge variant="outline">{activePrediction.component}</Badge>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-muted-foreground">Severity</span>
                                        <Badge className={getSeverityColor(activePrediction.severity)}>
                                            {activePrediction.severity}
                                        </Badge>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-muted-foreground">Time to Failure</span>
                                        <span className="font-medium">{activePrediction.days_to_failure} days</span>
                                    </div>

                                    {countdown !== null && (
                                        <div className="text-center py-4">
                                            <div className="flex items-center justify-center gap-2 text-muted-foreground">
                                                <Clock className="h-4 w-4" />
                                                <span>Auto-routing to diagnosis in</span>
                                            </div>
                                            <div className="text-3xl font-bold mt-2">{countdown}s</div>
                                        </div>
                                    )}
                                </div>
                            )}

                            <DialogFooter className="gap-2">
                                <Button variant="outline" onClick={handleReject}>
                                    <XCircle className="mr-2 h-4 w-4" />
                                    Ignore
                                </Button>
                                <Button onClick={handleAccept} className="bg-green-600 hover:bg-green-700">
                                    <CheckCircle className="mr-2 h-4 w-4" />
                                    Accept Warning
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>

                    {/* Safety Advice Modal */}
                    <Dialog open={!!safetyAdvice} onOpenChange={() => setSafetyAdvice(null)}>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle className="flex items-center gap-2 text-green-500">
                                    <CheckCircle className="h-5 w-5" />
                                    {safetyAdvice?.title}
                                </DialogTitle>
                            </DialogHeader>

                            {safetyAdvice && (
                                <div className="py-4 space-y-4">
                                    <p className="text-sm">{safetyAdvice.message}</p>

                                    <div>
                                        <h4 className="font-medium mb-2">Tips:</h4>
                                        <ul className="list-disc list-inside space-y-1">
                                            {safetyAdvice.tips.map((tip, i) => (
                                                <li key={i} className="text-sm text-muted-foreground">{tip}</li>
                                            ))}
                                        </ul>
                                    </div>

                                    <div className="text-center py-4">
                                        <Badge className="text-lg bg-green-600">
                                            +{safetyAdvice.points_awarded} points
                                        </Badge>
                                    </div>
                                </div>
                            )}

                            <DialogFooter>
                                <Button onClick={() => setSafetyAdvice(null)}>Got it!</Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>

                    {/* Diagnosis Modal */}
                    <Dialog open={!!diagnosis} onOpenChange={() => setDiagnosis(null)}>
                        <DialogContent className="sm:max-w-lg">
                            <DialogHeader>
                                <DialogTitle className="flex items-center gap-2 text-orange-500">
                                    <Wrench className="h-5 w-5" />
                                    {diagnosis?.title}
                                </DialogTitle>
                            </DialogHeader>

                            {diagnosis && (
                                <div className="py-4 space-y-4">
                                    <p className="text-sm">{diagnosis.summary}</p>

                                    <div>
                                        <h4 className="font-medium mb-2">Diagnostic Steps:</h4>
                                        <div className="space-y-2">
                                            {diagnosis.steps.map((step) => (
                                                <div key={step.step} className="flex items-start gap-2 text-sm">
                                                    <span>{step.icon}</span>
                                                    <div>
                                                        <span className="font-medium">{step.title}:</span>
                                                        <span className="text-muted-foreground ml-1">{step.finding}</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <Separator />

                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <span className="text-muted-foreground">Estimated Cost</span>
                                            <p className="font-medium">{diagnosis.estimated_cost}</p>
                                        </div>
                                        <div>
                                            <span className="text-muted-foreground">Urgency</span>
                                            <p className="font-medium capitalize">{diagnosis.urgency}</p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <DialogFooter className="gap-2">
                                <Button variant="outline" onClick={() => {
                                    if (diagnosis) declineService(diagnosis.diagnosis_id);
                                    setDiagnosis(null);
                                }}>
                                    Decline
                                </Button>
                                <Button onClick={() => {
                                    if (diagnosis) confirmService(diagnosis.diagnosis_id);
                                    setDiagnosis(null);
                                    toast.success('Service scheduled!');
                                }}>
                                    Schedule Service
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>

                    {/* Critical Alert Modal - Brake Fade Scenario */}
                    <CriticalAlertModal
                        open={criticalAlertOpen}
                        onOpenChange={setCriticalAlertOpen}
                        alertData={criticalAlertData}
                        onAnswerCall={() => {
                            setCriticalAlertOpen(false);
                            setVoiceCallOpen(true);
                        }}
                        onDismiss={() => setCriticalAlertOpen(false)}
                    />

                    {/* Voice Call Modal */}
                    <VoiceCallModal
                        open={voiceCallOpen}
                        onOpenChange={setVoiceCallOpen}
                        vehicleId={vin}
                        alertType="brake_fade"
                        alertData={{
                            brake_efficiency: criticalAlertData?.efficiency,
                            component: criticalAlertData?.component,
                            severity: criticalAlertData?.severity
                        }}
                        ownerName="Alex"
                        onBookingConfirmed={(booking) => {
                            toast.success('Appointment booked!', {
                                description: `${booking.center_name} at ${booking.time}`,
                            });
                        }}
                    />
                </main>
            </div>
        </div>
    );
}
