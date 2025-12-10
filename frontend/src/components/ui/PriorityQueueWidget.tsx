// Priority Queue Widget - Shows vehicles ordered by urgency for demo
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { ListOrdered, Clock, AlertTriangle, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

const API_V2 = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';

interface QueueItem {
    vehicle_id: string;
    component: string;
    urgency_score: number;
    severity: string;
    failure_probability?: number;
}

export function PriorityQueueWidget() {
    const [queue, setQueue] = useState<QueueItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [processing, setProcessing] = useState(false);

    useEffect(() => {
        fetchQueue();
    }, []);

    async function fetchQueue() {
        setLoading(true);
        try {
            const res = await fetch(`${API_V2}/v2/scheduling/queue/status`);
            const data = await res.json();
            setQueue(data.items || []);
        } catch (error) {
            console.error('Failed to fetch queue:', error);
            // Demo fallback data
            setQueue([
                { vehicle_id: 'EV-07', component: 'Brakes', urgency_score: 9.2, severity: 'critical', failure_probability: 87 },
                { vehicle_id: 'EV-12', component: 'Battery', urgency_score: 7.5, severity: 'high', failure_probability: 65 },
                { vehicle_id: 'EV-03', component: 'Motor', urgency_score: 5.1, severity: 'medium', failure_probability: 42 },
            ]);
        } finally {
            setLoading(false);
        }
    }

    async function processQueue() {
        setProcessing(true);
        try {
            const res = await fetch(`${API_V2}/v2/scheduling/queue/process`, {
                method: 'POST'
            });
            const data = await res.json();
            console.log('Queue processed:', data);
            fetchQueue();
        } catch (error) {
            console.error('Failed to process queue:', error);
        } finally {
            setProcessing(false);
        }
    }

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical': return 'bg-red-500 text-white';
            case 'high': return 'bg-orange-500 text-white';
            case 'medium': return 'bg-yellow-500 text-black';
            default: return 'bg-blue-500 text-white';
        }
    };

    return (
        <Card className="border-2 border-purple-500/30">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <ListOrdered className="h-4 w-4 text-purple-500" />
                    Priority Queue
                    <Badge variant="outline" className="ml-auto">
                        {queue.length} vehicles
                    </Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                {queue.length === 0 ? (
                    <div className="text-center py-4 text-muted-foreground">
                        Queue empty
                    </div>
                ) : (
                    <div className="space-y-2">
                        {queue.map((item, index) => (
                            <div
                                key={item.vehicle_id}
                                className={cn(
                                    "flex items-center justify-between p-2 rounded-lg border",
                                    index === 0 && "bg-red-500/10 border-red-500/50 animate-pulse"
                                )}
                            >
                                <div className="flex items-center gap-2">
                                    <span className="text-lg font-bold text-muted-foreground">
                                        #{index + 1}
                                    </span>
                                    <div>
                                        <code className="text-xs bg-muted px-1 rounded">
                                            {item.vehicle_id}
                                        </code>
                                        <p className="text-xs text-muted-foreground">
                                            {item.component}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="text-right">
                                        <p className="text-sm font-bold">{item.urgency_score.toFixed(1)}</p>
                                        <p className="text-xs text-muted-foreground">score</p>
                                    </div>
                                    <Badge className={getSeverityColor(item.severity)}>
                                        {item.severity}
                                    </Badge>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                <Button
                    className="w-full mt-4"
                    onClick={processQueue}
                    disabled={processing || queue.length === 0}
                >
                    <Zap className={cn("h-4 w-4 mr-2", processing && "animate-spin")} />
                    Process Queue
                </Button>
            </CardContent>
        </Card>
    );
}
