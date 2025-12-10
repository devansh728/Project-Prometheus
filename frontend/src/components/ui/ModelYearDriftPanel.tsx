// Model Year Drift Panel - Shows quality drift by model year
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Calendar, TrendingUp, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

const API_V2 = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';

interface YearData {
    year: number;
    failure_rate: number;
    production_volume: number;
    delta_vs_avg: number;
    is_elevated: boolean;
}

interface ModelYearDriftProps {
    component?: string;
}

export function ModelYearDriftPanel({ component = 'brakes' }: ModelYearDriftProps) {
    const [data, setData] = useState<YearData[]>([]);
    const [loading, setLoading] = useState(false);
    const [summary, setSummary] = useState<string>('');

    useEffect(() => {
        fetchDrift();
    }, [component]);

    async function fetchDrift() {
        setLoading(true);
        try {
            const res = await fetch(`${API_V2}/v2/capa/drift/${component}`);
            const result = await res.json();
            if (result.drift_data) {
                setData(result.drift_data);
                setSummary(result.summary || '');
            }
        } catch (error) {
            console.error('Failed to fetch drift:', error);
            // Demo fallback
            setData([
                { year: 2022, failure_rate: 0.8, production_volume: 15000, delta_vs_avg: -20, is_elevated: false },
                { year: 2023, failure_rate: 1.0, production_volume: 22000, delta_vs_avg: 0, is_elevated: false },
                { year: 2024, failure_rate: 1.33, production_volume: 28000, delta_vs_avg: 33, is_elevated: true },
            ]);
            setSummary('Elevated failure rates detected in model years: 2024');
        } finally {
            setLoading(false);
        }
    }

    const maxRate = Math.max(...data.map(d => d.failure_rate), 1);

    return (
        <Card className={cn(
            "border-2",
            data.some(d => d.is_elevated) && "border-red-500/50 bg-red-500/5"
        )}>
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    Model Year Quality Drift
                    <Badge variant="outline" className="ml-auto capitalize">
                        {component}
                    </Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                {summary && (
                    <div className="mb-4 p-2 rounded bg-red-500/10 border border-red-500/30 flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
                        <span className="text-sm text-red-400">{summary}</span>
                    </div>
                )}

                <div className="space-y-3">
                    {data.map((year) => (
                        <div key={year.year} className="space-y-1">
                            <div className="flex items-center justify-between text-sm">
                                <span className="font-medium">{year.year}</span>
                                <div className="flex items-center gap-2">
                                    <span className={cn(
                                        "font-mono",
                                        year.is_elevated && "text-red-500 font-bold"
                                    )}>
                                        {year.failure_rate.toFixed(2)}%
                                    </span>
                                    {year.delta_vs_avg !== 0 && (
                                        <Badge
                                            variant={year.delta_vs_avg > 0 ? "destructive" : "default"}
                                            className="text-xs"
                                        >
                                            {year.delta_vs_avg > 0 ? '+' : ''}{year.delta_vs_avg}%
                                        </Badge>
                                    )}
                                </div>
                            </div>
                            <Progress
                                value={(year.failure_rate / maxRate) * 100}
                                className={cn(
                                    "h-3",
                                    year.is_elevated ? "[&>div]:bg-red-500" : "[&>div]:bg-green-500"
                                )}
                            />
                            <p className="text-xs text-muted-foreground">
                                {year.production_volume.toLocaleString()} units
                            </p>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
