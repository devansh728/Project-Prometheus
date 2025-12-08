'use client';

import { useState, useEffect, useCallback } from 'react';
import {
    Shield, ShieldAlert, ShieldCheck, AlertOctagon, Activity,
    Lock, Unlock, Eye, RefreshCw, Zap, AlertTriangle, Clock
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface AgentAction {
    timestamp: string;
    agent: string;
    action: string;
    details: Record<string, any>;
    is_anomaly?: boolean;
    blocked?: boolean;
}

interface UEBAAlert {
    timestamp: string;
    agent: string;
    action: string;
    reason: string;
    severity: string;
}

export default function SecurityDashboard() {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [loading, setLoading] = useState(true);
    const [showRedOverlay, setShowRedOverlay] = useState(false);
    const [lastAlert, setLastAlert] = useState<UEBAAlert | null>(null);

    // Data states
    const [alerts, setAlerts] = useState<UEBAAlert[]>([]);
    const [actionLog, setActionLog] = useState<AgentAction[]>([]);
    const [agentSummaries, setAgentSummaries] = useState<Record<string, any>>({});

    // Agent baseline behavior
    const agentBaselines = {
        data_analysis: {
            name: 'Data Analysis Agent',
            allowed: ['analyze_telemetry', 'detect_anomaly', 'score_behavior', 'predict_failure'],
            description: 'Processes vehicle telemetry and detects anomalies',
            icon: Activity
        },
        safety: {
            name: 'Safety Agent',
            allowed: ['generate_tips', 'send_warning', 'add_points'],
            description: 'Provides safety recommendations and warnings',
            icon: ShieldCheck
        },
        diagnosis: {
            name: 'Diagnosis Agent',
            allowed: ['run_diagnosis', 'estimate_cost', 'query_rag'],
            description: 'Diagnoses vehicle issues and estimates repairs',
            icon: Zap
        },
        scheduling: {
            name: 'Scheduling Agent',
            allowed: ['check_availability', 'book_appointment', 'reschedule', 'cancel'],
            description: 'Manages service appointments',
            icon: Clock
        }
    };

    // Fetch data
    useEffect(() => {
        fetchAllData();
    }, []);

    async function fetchAllData() {
        setLoading(true);
        try {
            const [alertsRes, logsRes] = await Promise.all([
                fetch(`${API_BASE}/ueba/alerts`).then(r => r.json()),
                fetch(`${API_BASE}/ueba/agent-logs`).then(r => r.json())
            ]);

            setAlerts(alertsRes.alerts || []);
            setActionLog(logsRes.action_log || []);
        } catch (error) {
            console.error('Failed to fetch UEBA data:', error);
            toast.error('Failed to load security data');
        } finally {
            setLoading(false);
        }
    }

    // Inject rogue action for demo
    const injectRogueAction = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/ueba/inject-rogue-action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    agent: 'scheduling',
                    action: 'access_telematics_database',
                    target: 'vehicle_telematics'
                })
            });

            const data = await res.json();

            if (data.blocked) {
                // Show red overlay
                setShowRedOverlay(true);
                setLastAlert(data.alert);

                // Play alert sound (optional)
                const audio = new Audio('/alert.mp3');
                audio.volume = 0.5;
                audio.play().catch(() => { });

                toast.error(data.message, {
                    duration: 5000,
                    style: { background: '#dc2626', color: 'white' }
                });

                // Refresh data
                fetchAllData();

                // Hide overlay after 5 seconds
                setTimeout(() => {
                    setShowRedOverlay(false);
                }, 5000);
            }
        } catch (error) {
            toast.error('Failed to inject rogue action');
        }
    }, []);

    const getAgentColor = (agent: string) => {
        const colors: Record<string, string> = {
            data_analysis: 'text-blue-500',
            safety: 'text-green-500',
            diagnosis: 'text-purple-500',
            scheduling: 'text-orange-500'
        };
        return colors[agent] || 'text-gray-500';
    };

    const getActionIcon = (action: AgentAction) => {
        if (action.is_anomaly || action.blocked) {
            return <ShieldAlert className="h-4 w-4 text-red-500" />;
        }
        return <ShieldCheck className="h-4 w-4 text-green-500" />;
    };

    const formatTimestamp = (ts: string) => {
        try {
            return new Date(ts).toLocaleTimeString();
        } catch {
            return ts;
        }
    };

    return (
        <>
            {/* RED OVERLAY - Security Breach Screen Lock */}
            {showRedOverlay && (
                <div className="fixed inset-0 z-50 bg-red-600/90 flex items-center justify-center animate-pulse">
                    <div className="text-center text-white max-w-2xl p-8">
                        <AlertOctagon className="h-32 w-32 mx-auto mb-6 animate-bounce" />
                        <h1 className="text-4xl font-bold mb-4">🚨 SECURITY BREACH DETECTED 🚨</h1>
                        <p className="text-xl mb-6">
                            {lastAlert?.agent} agent attempted unauthorized access to {lastAlert?.reason || 'protected resource'}
                        </p>
                        <div className="p-4 bg-black/30 rounded-lg mb-6">
                            <p className="text-lg">
                                <Lock className="inline h-5 w-5 mr-2" />
                                ACTION BLOCKED BY UEBA MONITOR
                            </p>
                        </div>
                        <Button
                            variant="outline"
                            className="border-white text-white hover:bg-white/20"
                            onClick={() => setShowRedOverlay(false)}
                        >
                            <Unlock className="h-4 w-4 mr-2" />
                            Dismiss Alert
                        </Button>
                    </div>
                </div>
            )}

            <div className="min-h-screen flex flex-col bg-background">
                <Navbar onMenuClick={() => setSidebarOpen(true)} />

                <div className="flex flex-1">
                    <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                    <main className="flex-1 p-6 overflow-auto">
                        {/* Header */}
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h1 className="text-2xl font-bold flex items-center gap-2">
                                    <Shield className="h-6 w-6" />
                                    UEBA Security Dashboard
                                </h1>
                                <p className="text-muted-foreground">User & Entity Behavior Analytics for Agent Monitoring</p>
                            </div>
                            <div className="flex gap-2">
                                <Button
                                    variant="destructive"
                                    onClick={injectRogueAction}
                                    className="gap-2"
                                >
                                    <ShieldAlert className="h-4 w-4" />
                                    Inject Rogue Action
                                </Button>
                                <Button variant="outline" onClick={fetchAllData}>
                                    <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                                    Refresh
                                </Button>
                            </div>
                        </div>

                        {/* Summary Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">
                                        Total Actions
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="text-2xl font-bold">{actionLog.length}</div>
                                    <p className="text-xs text-muted-foreground">Monitored</p>
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">
                                        Security Alerts
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="text-2xl font-bold text-red-500">{alerts.length}</div>
                                    <p className="text-xs text-muted-foreground">Blocked actions</p>
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">
                                        Active Agents
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="text-2xl font-bold text-green-500">4</div>
                                    <p className="text-xs text-muted-foreground">Monitored</p>
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">
                                        Security Status
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className={cn(
                                        "text-2xl font-bold",
                                        alerts.length > 0 ? "text-yellow-500" : "text-green-500"
                                    )}>
                                        {alerts.length > 0 ? 'ALERT' : 'SECURE'}
                                    </div>
                                    <p className="text-xs text-muted-foreground">System status</p>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Main Content Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Agent Baselines */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Eye className="h-5 w-5" />
                                        Agent Baselines
                                    </CardTitle>
                                    <CardDescription>Allowed actions per agent</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4">
                                        {Object.entries(agentBaselines).map(([key, agent]) => {
                                            const Icon = agent.icon;
                                            return (
                                                <div key={key} className="p-3 border rounded-lg">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <Icon className={cn("h-4 w-4", getAgentColor(key))} />
                                                        <span className="font-medium text-sm">{agent.name}</span>
                                                    </div>
                                                    <p className="text-xs text-muted-foreground mb-2">{agent.description}</p>
                                                    <div className="flex flex-wrap gap-1">
                                                        {agent.allowed.map(action => (
                                                            <Badge key={action} variant="outline" className="text-xs">
                                                                {action}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Action Log */}
                            <Card className="lg:col-span-2">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Activity className="h-5 w-5" />
                                        Agent Action Log
                                    </CardTitle>
                                    <CardDescription>Real-time monitoring of all agent actions</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2 max-h-[400px] overflow-y-auto">
                                        {actionLog.length === 0 ? (
                                            <div className="text-center py-8 text-muted-foreground">
                                                <Shield className="h-12 w-12 mx-auto mb-4 opacity-50" />
                                                <p>No actions logged yet</p>
                                                <p className="text-xs mt-2">Actions will appear here as agents operate</p>
                                            </div>
                                        ) : (
                                            actionLog.slice().reverse().map((action, i) => (
                                                <div
                                                    key={i}
                                                    className={cn(
                                                        "flex items-start gap-3 p-3 rounded-lg",
                                                        action.is_anomaly || action.blocked
                                                            ? "bg-red-500/10 border border-red-500/50"
                                                            : "bg-muted/50"
                                                    )}
                                                >
                                                    {getActionIcon(action)}
                                                    <div className="flex-1 min-w-0">
                                                        <div className="flex items-center justify-between">
                                                            <span className={cn("font-medium text-sm", getAgentColor(action.agent))}>
                                                                {action.agent}
                                                            </span>
                                                            <span className="text-xs text-muted-foreground">
                                                                {formatTimestamp(action.timestamp)}
                                                            </span>
                                                        </div>
                                                        <p className="text-sm mt-1">{action.action}</p>
                                                        {action.is_anomaly && (
                                                            <Badge variant="destructive" className="mt-2 text-xs">
                                                                BLOCKED - Unauthorized Action
                                                            </Badge>
                                                        )}
                                                    </div>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Security Alerts */}
                        {alerts.length > 0 && (
                            <Card className="mt-6 border-red-500/50">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-red-500">
                                        <AlertTriangle className="h-5 w-5" />
                                        Security Alerts
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {alerts.map((alert, i) => (
                                            <div
                                                key={i}
                                                className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/50 rounded-lg"
                                            >
                                                <ShieldAlert className="h-5 w-5 text-red-500 shrink-0" />
                                                <div className="flex-1">
                                                    <div className="flex items-center justify-between">
                                                        <span className="font-semibold text-red-400">{alert.agent} Agent</span>
                                                        <span className="text-xs text-muted-foreground">
                                                            {formatTimestamp(alert.timestamp)}
                                                        </span>
                                                    </div>
                                                    <p className="text-sm mt-1">Attempted: {alert.action}</p>
                                                    <p className="text-xs text-muted-foreground mt-1">{alert.reason}</p>
                                                    <Badge variant="destructive" className="mt-2">
                                                        {alert.severity || 'high'} severity
                                                    </Badge>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* How UEBA Works */}
                        <Card className="mt-6">
                            <CardHeader>
                                <CardTitle className="text-sm">How UEBA Monitoring Works</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                    <div className="p-4 bg-muted/50 rounded-lg">
                                        <h4 className="font-semibold mb-2 flex items-center gap-2">
                                            <Eye className="h-4 w-4 text-blue-500" />
                                            1. Baseline Establishment
                                        </h4>
                                        <p className="text-xs text-muted-foreground">
                                            Each agent has a defined set of allowed actions based on its responsibilities.
                                        </p>
                                    </div>
                                    <div className="p-4 bg-muted/50 rounded-lg">
                                        <h4 className="font-semibold mb-2 flex items-center gap-2">
                                            <Activity className="h-4 w-4 text-purple-500" />
                                            2. Continuous Monitoring
                                        </h4>
                                        <p className="text-xs text-muted-foreground">
                                            Every agent action is logged and analyzed against the baseline in real-time.
                                        </p>
                                    </div>
                                    <div className="p-4 bg-muted/50 rounded-lg">
                                        <h4 className="font-semibold mb-2 flex items-center gap-2">
                                            <ShieldAlert className="h-4 w-4 text-red-500" />
                                            3. Anomaly Detection
                                        </h4>
                                        <p className="text-xs text-muted-foreground">
                                            Unauthorized actions are immediately blocked and generate security alerts.
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </main>
                </div>
            </div>
        </>
    );
}
