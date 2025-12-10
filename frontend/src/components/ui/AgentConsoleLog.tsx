// Agent Console Log Component - Shows real-time agent activity for demo
'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Terminal, Bot, Brain, Wrench, Calendar, Shield, Phone } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LogEntry {
    id: string;
    timestamp: Date;
    agent: 'Master' | 'Data' | 'Diagnostics' | 'Scheduling' | 'Voice' | 'UEBA';
    message: string;
    type: 'info' | 'warning' | 'error' | 'success';
}

const AGENT_ICONS = {
    'Master': Bot,
    'Data': Brain,
    'Diagnostics': Wrench,
    'Scheduling': Calendar,
    'Voice': Phone,
    'UEBA': Shield,
};

const AGENT_COLORS = {
    'Master': 'text-purple-400',
    'Data': 'text-blue-400',
    'Diagnostics': 'text-orange-400',
    'Scheduling': 'text-green-400',
    'Voice': 'text-pink-400',
    'UEBA': 'text-red-400',
};

interface AgentConsoleLogProps {
    anomalyScore?: number;
    failureProbability?: number;
    isAnomaly?: boolean;
    severity?: string;
    anomalyType?: string;
    brakeTemp?: number;
    agentsActive?: string[];  // v2: Active agents from WebSocket
    workerId?: number;  // v2: Worker routing ID
}

export function AgentConsoleLog({
    anomalyScore = 0,
    failureProbability = 0,
    isAnomaly = false,
    severity = 'low',
    anomalyType = 'normal',
    brakeTemp = 0,
    agentsActive = [],
    workerId = 0,
}: AgentConsoleLogProps) {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Generate logs based on telemetry state
    useEffect(() => {
        const newLogs: LogEntry[] = [];
        const timestamp = new Date();

        // Always show frame received
        newLogs.push({
            id: `${Date.now()}-1`,
            timestamp,
            agent: 'Master',
            message: 'Frame received from vehicle telematics',
            type: 'info',
        });

        // Show anomaly score
        if (anomalyScore > 0.1) {
            newLogs.push({
                id: `${Date.now()}-2`,
                timestamp,
                agent: 'Data',
                message: `anomaly_score=${anomalyScore.toFixed(2)} (LSTM-AE reconstruction)`,
                type: anomalyScore > 0.5 ? 'warning' : 'info',
            });
        }

        // Show brake temperature warning
        if (brakeTemp > 200) {
            newLogs.push({
                id: `${Date.now()}-3`,
                timestamp,
                agent: 'Diagnostics',
                message: `brake_temp=${brakeTemp.toFixed(0)}°C - ${brakeTemp > 300 ? 'CRITICAL OVERHEAT' : 'elevated temperature'}`,
                type: brakeTemp > 300 ? 'error' : 'warning',
            });
        }

        // Show failure prediction
        if (failureProbability > 50) {
            newLogs.push({
                id: `${Date.now()}-4`,
                timestamp,
                agent: 'Data',
                message: `failure_probability=${failureProbability.toFixed(0)}% (LightGBM prediction)`,
                type: 'warning',
            });
        }

        // Show severity routing
        if (isAnomaly && (severity === 'high' || severity === 'critical')) {
            newLogs.push({
                id: `${Date.now()}-5`,
                timestamp,
                agent: 'Master',
                message: `Severity=${severity.toUpperCase()} → Initiating customer outreach`,
                type: 'error',
            });

            newLogs.push({
                id: `${Date.now()}-6`,
                timestamp,
                agent: 'Voice',
                message: 'Preparing proactive call to vehicle owner...',
                type: 'info',
            });

            newLogs.push({
                id: `${Date.now()}-7`,
                timestamp,
                agent: 'UEBA',
                message: 'Monitoring agent behavior - all actions authorized',
                type: 'success',
            });
        }

        // Show anomaly type detection
        if (isAnomaly && anomalyType !== 'normal') {
            newLogs.push({
                id: `${Date.now()}-8`,
                timestamp,
                agent: 'Diagnostics',
                message: `detecting ${anomalyType.replace('_', ' ')} signature...`,
                type: 'warning',
            });
        }

        setLogs(prev => {
            const combined = [...prev, ...newLogs];
            return combined.slice(-20); // Keep last 20 logs
        });
    }, [anomalyScore, failureProbability, isAnomaly, severity, anomalyType, brakeTemp]);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <Card className="bg-gray-950 border-gray-800">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2 text-gray-300">
                    <Terminal className="h-4 w-4 text-green-400" />
                    Agent Console
                    <Badge variant="outline" className="ml-auto text-green-400 border-green-400/50">
                        LIVE
                    </Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div
                    ref={scrollRef}
                    className="h-48 overflow-y-auto font-mono text-xs space-y-1 scrollbar-thin"
                >
                    {logs.length === 0 ? (
                        <div className="text-gray-500 text-center py-8">
                            Waiting for telemetry...
                        </div>
                    ) : (
                        logs.map((log) => {
                            const Icon = AGENT_ICONS[log.agent];
                            return (
                                <div
                                    key={log.id}
                                    className={cn(
                                        "flex items-start gap-2 py-0.5",
                                        log.type === 'error' && 'animate-pulse'
                                    )}
                                >
                                    <span className="text-gray-500 shrink-0">
                                        [{formatTime(log.timestamp)}]
                                    </span>
                                    <Icon className={cn("h-3 w-3 shrink-0 mt-0.5", AGENT_COLORS[log.agent])} />
                                    <span className={cn("shrink-0", AGENT_COLORS[log.agent])}>
                                        [{log.agent}]
                                    </span>
                                    <span className={cn(
                                        log.type === 'error' ? 'text-red-400' :
                                            log.type === 'warning' ? 'text-yellow-400' :
                                                log.type === 'success' ? 'text-green-400' :
                                                    'text-gray-300'
                                    )}>
                                        {log.message}
                                    </span>
                                </div>
                            );
                        })
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
