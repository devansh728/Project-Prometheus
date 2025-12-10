// Failure Prediction Card - Enhanced display for demo
'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, Clock, TrendingUp, Cpu } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FailurePredictionCardProps {
    failureProbability: number;
    anomalyScore: number;
    severity: string;
    component?: string;
    daysToFailure?: number;
}

export function FailurePredictionCard({
    failureProbability,
    anomalyScore,
    severity,
    component = 'Unknown',
    daysToFailure = 3,
}: FailurePredictionCardProps) {
    const [animatedProb, setAnimatedProb] = useState(0);
    const [animatedScore, setAnimatedScore] = useState(0);

    // Animate values
    useEffect(() => {
        const probInterval = setInterval(() => {
            setAnimatedProb(prev => {
                const diff = failureProbability - prev;
                if (Math.abs(diff) < 1) return failureProbability;
                return prev + diff * 0.1;
            });
        }, 50);

        return () => clearInterval(probInterval);
    }, [failureProbability]);

    useEffect(() => {
        const scoreInterval = setInterval(() => {
            setAnimatedScore(prev => {
                const diff = anomalyScore - prev;
                if (Math.abs(diff) < 0.01) return anomalyScore;
                return prev + diff * 0.1;
            });
        }, 50);

        return () => clearInterval(scoreInterval);
    }, [anomalyScore]);

    const getSeverityColor = (sev: string) => {
        switch (sev) {
            case 'critical': return 'bg-red-500 text-white';
            case 'high': return 'bg-orange-500 text-white';
            case 'medium': return 'bg-yellow-500 text-black';
            default: return 'bg-blue-500 text-white';
        }
    };

    const getProbabilityColor = (prob: number) => {
        if (prob >= 80) return 'text-red-500';
        if (prob >= 60) return 'text-orange-500';
        if (prob >= 40) return 'text-yellow-500';
        return 'text-green-500';
    };

    return (
        <Card className={cn(
            "border-2 transition-all duration-300",
            failureProbability >= 70 ? "border-red-500/50 bg-red-500/5 animate-pulse" :
                failureProbability >= 50 ? "border-orange-500/50" :
                    "border-transparent"
        )}>
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <TrendingUp className="h-4 w-4" />
                    Failure Prediction
                    {severity !== 'low' && (
                        <Badge className={cn("ml-auto", getSeverityColor(severity))}>
                            {severity.toUpperCase()}
                        </Badge>
                    )}
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Main probability display */}
                <div className="text-center">
                    <div className={cn(
                        "text-5xl font-bold transition-colors",
                        getProbabilityColor(animatedProb)
                    )}>
                        {animatedProb.toFixed(0)}%
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                        Failure Probability (LightGBM)
                    </p>
                </div>

                {/* Anomaly Score */}
                <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                        <span className="flex items-center gap-1 text-muted-foreground">
                            <Cpu className="h-3 w-3" />
                            Anomaly Score
                        </span>
                        <span className="font-mono">{animatedScore.toFixed(2)}</span>
                    </div>
                    <Progress
                        value={animatedScore * 100}
                        className={cn(
                            "h-2",
                            animatedScore > 0.5 ? "[&>div]:bg-red-500" :
                                animatedScore > 0.3 ? "[&>div]:bg-orange-500" :
                                    "[&>div]:bg-green-500"
                        )}
                    />
                </div>

                {/* Time to failure */}
                {failureProbability > 50 && (
                    <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                        <div className="flex items-center gap-2 text-sm">
                            <Clock className="h-4 w-4 text-muted-foreground" />
                            Time to Failure
                        </div>
                        <span className="font-bold text-orange-500">
                            ~{daysToFailure} days
                        </span>
                    </div>
                )}

                {/* Component affected */}
                {failureProbability > 30 && (
                    <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                        <div className="flex items-center gap-2 text-sm">
                            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                            Affected Component
                        </div>
                        <Badge variant="outline">{component}</Badge>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
