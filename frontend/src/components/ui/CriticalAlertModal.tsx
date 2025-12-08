'use client';

import { useState, useEffect, useRef } from 'react';
import { AlertTriangle, Phone, X, Activity, Thermometer, Gauge, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

interface CriticalAlertData {
    type: string;
    component: string;
    severity: string;
    message: string;
    efficiency?: number;
    temperature?: number;
    timestamp: string;
    vehicle_id: string;
}

interface CriticalAlertModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    alertData: CriticalAlertData | null;
    onAnswerCall: () => void;
    onDismiss: () => void;
}

export function CriticalAlertModal({
    open,
    onOpenChange,
    alertData,
    onAnswerCall,
    onDismiss
}: CriticalAlertModalProps) {
    const [isRinging, setIsRinging] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // Start ringing animation when modal opens
    useEffect(() => {
        if (open && alertData) {
            // Start ringing after a brief delay
            const timer = setTimeout(() => {
                setIsRinging(true);
                // Play ring sound if available
                if (audioRef.current) {
                    audioRef.current.play().catch(() => { });
                }
            }, 2000);

            return () => clearTimeout(timer);
        } else {
            setIsRinging(false);
        }
    }, [open, alertData]);

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical':
            case 'high':
                return 'text-red-500 bg-red-500/20';
            case 'medium':
                return 'text-orange-500 bg-orange-500/20';
            default:
                return 'text-yellow-500 bg-yellow-500/20';
        }
    };

    if (!alertData) return null;

    return (
        <>
            {/* Hidden audio for ring tone */}
            <audio ref={audioRef} src="/ring.mp3" loop className="hidden" />

            <Dialog open={open} onOpenChange={onOpenChange}>
                <DialogContent className={cn(
                    "sm:max-w-[500px] border-2",
                    alertData.severity === 'critical' ? "border-red-500 bg-gradient-to-b from-red-950/90 to-slate-950" : "border-orange-500"
                )}>
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-red-500">
                            <ShieldAlert className="h-6 w-6 animate-pulse" />
                            CRITICAL FAILURE PREDICTED
                        </DialogTitle>
                        <DialogDescription className="text-slate-300">
                            Immediate attention required for vehicle safety
                        </DialogDescription>
                    </DialogHeader>

                    {/* Alert Content */}
                    <div className="space-y-4">
                        {/* Component & Severity */}
                        <div className="flex items-center justify-between">
                            <Badge className={cn("text-sm", getSeverityColor(alertData.severity))}>
                                {alertData.severity.toUpperCase()} SEVERITY
                            </Badge>
                            <span className="text-sm text-muted-foreground">
                                {new Date(alertData.timestamp).toLocaleTimeString()}
                            </span>
                        </div>

                        {/* Alert Message */}
                        <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
                            <h3 className="font-semibold text-red-400 mb-2 flex items-center gap-2">
                                <AlertTriangle className="h-5 w-5" />
                                {alertData.component} Failure Detected
                            </h3>
                            <p className="text-sm">{alertData.message}</p>
                        </div>

                        {/* Metrics */}
                        <div className="grid grid-cols-2 gap-4">
                            {alertData.efficiency !== undefined && (
                                <div className="p-3 bg-slate-800 rounded-lg">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Gauge className="h-4 w-4 text-red-400" />
                                        <span className="text-xs text-muted-foreground">Efficiency</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-2xl font-bold text-red-400">
                                            {alertData.efficiency}%
                                        </span>
                                        <Progress value={alertData.efficiency} className="h-2 flex-1" />
                                    </div>
                                </div>
                            )}
                            {alertData.temperature !== undefined && (
                                <div className="p-3 bg-slate-800 rounded-lg">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Thermometer className="h-4 w-4 text-orange-400" />
                                        <span className="text-xs text-muted-foreground">Temperature</span>
                                    </div>
                                    <span className="text-2xl font-bold text-orange-400">
                                        {alertData.temperature}°C
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Agent Activity */}
                        <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                            <div className="flex items-center gap-2 text-sm text-blue-400">
                                <Activity className="h-4 w-4" />
                                <span>Data Analysis Agent processing...</span>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                Cross-referencing with historical maintenance records
                            </p>
                        </div>

                        {/* Incoming Call Section */}
                        {isRinging && (
                            <div className="p-4 bg-green-500/10 border border-green-500/50 rounded-lg animate-pulse">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="p-3 bg-green-500 rounded-full animate-bounce">
                                            <Phone className="h-5 w-5 text-white" />
                                        </div>
                                        <div>
                                            <p className="font-semibold text-green-400">Incoming Call</p>
                                            <p className="text-sm text-muted-foreground">SentinEV AI Assistant</p>
                                        </div>
                                    </div>
                                    <Button
                                        className="bg-green-500 hover:bg-green-600"
                                        onClick={() => {
                                            setIsRinging(false);
                                            if (audioRef.current) {
                                                audioRef.current.pause();
                                                audioRef.current.currentTime = 0;
                                            }
                                            onAnswerCall();
                                        }}
                                    >
                                        <Phone className="h-4 w-4 mr-2" />
                                        Answer
                                    </Button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="flex justify-end gap-2 pt-4 border-t border-slate-700">
                        <Button variant="ghost" onClick={onDismiss}>
                            <X className="h-4 w-4 mr-2" />
                            Dismiss
                        </Button>
                        {!isRinging && (
                            <Button
                                className="bg-green-500 hover:bg-green-600"
                                onClick={onAnswerCall}
                            >
                                <Phone className="h-4 w-4 mr-2" />
                                Request Call
                            </Button>
                        )}
                    </div>
                </DialogContent>
            </Dialog>
        </>
    );
}
