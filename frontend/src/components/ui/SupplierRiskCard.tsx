// Supplier Risk Card - Shows supplier risk scores for demo
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Factory, AlertTriangle, TrendingDown, Package } from 'lucide-react';
import { cn } from '@/lib/utils';

const API_V2 = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';

interface SupplierRisk {
    supplier_id: string;
    supplier_name: string;
    risk_score: number;
    risk_level: string;
    defect_rate: number;
    response_time_days: number;
    quality_trend: string;
}

export function SupplierRiskCard() {
    const [suppliers, setSuppliers] = useState<SupplierRisk[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchSupplierRisks();
    }, []);

    async function fetchSupplierRisks() {
        setLoading(true);
        try {
            const res = await fetch(`${API_V2}/v2/capa/supplier`);
            const data = await res.json();
            setSuppliers(data.suppliers || []);
        } catch (error) {
            console.error('Failed to fetch supplier risks:', error);
            // Demo fallback
            setSuppliers([
                { supplier_id: 'SUP-001', supplier_name: 'BrakeTech Industries', risk_score: 4.08, risk_level: 'medium', defect_rate: 0.12, response_time_days: 5, quality_trend: 'declining' },
                { supplier_id: 'SUP-002', supplier_name: 'PowerCell Energy', risk_score: 2.5, risk_level: 'low', defect_rate: 0.05, response_time_days: 3, quality_trend: 'stable' },
                { supplier_id: 'SUP-003', supplier_name: 'EMotion Drives', risk_score: 6.2, risk_level: 'high', defect_rate: 0.18, response_time_days: 8, quality_trend: 'declining' },
            ]);
        } finally {
            setLoading(false);
        }
    }

    const getRiskColor = (level: string) => {
        switch (level) {
            case 'critical': return 'bg-red-500 text-white';
            case 'high': return 'bg-orange-500 text-white';
            case 'medium': return 'bg-yellow-500 text-black';
            default: return 'bg-green-500 text-white';
        }
    };

    const getTrendIcon = (trend: string) => {
        if (trend === 'declining') return <TrendingDown className="h-3 w-3 text-red-500" />;
        if (trend === 'improving') return <TrendingDown className="h-3 w-3 text-green-500 rotate-180" />;
        return <span className="text-xs text-muted-foreground">→</span>;
    };

    return (
        <Card className="border-orange-500/30">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Factory className="h-4 w-4 text-orange-500" />
                    Supplier Risk Scores
                    {suppliers.filter(s => s.risk_level === 'high' || s.risk_level === 'critical').length > 0 && (
                        <Badge variant="destructive" className="ml-auto">
                            {suppliers.filter(s => s.risk_level === 'high' || s.risk_level === 'critical').length} Alerts
                        </Badge>
                    )}
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-3">
                    {suppliers.map((supplier) => (
                        <div
                            key={supplier.supplier_id}
                            className={cn(
                                "p-3 rounded-lg border",
                                supplier.risk_level === 'high' && "bg-orange-500/10 border-orange-500/30",
                                supplier.risk_level === 'critical' && "bg-red-500/10 border-red-500/30"
                            )}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <Package className="h-4 w-4 text-muted-foreground" />
                                    <span className="font-medium text-sm">{supplier.supplier_name}</span>
                                </div>
                                <Badge className={getRiskColor(supplier.risk_level)}>
                                    {supplier.risk_score.toFixed(2)}
                                </Badge>
                            </div>
                            <div className="flex items-center gap-4 text-xs">
                                <span className="flex items-center gap-1">
                                    <AlertTriangle className="h-3 w-3" />
                                    {(supplier.defect_rate * 100).toFixed(1)}% defect
                                </span>
                                <span>
                                    {supplier.response_time_days}d response
                                </span>
                                <span className="flex items-center gap-1">
                                    {getTrendIcon(supplier.quality_trend)}
                                    {supplier.quality_trend}
                                </span>
                            </div>
                            <Progress
                                value={supplier.risk_score * 10}
                                className={cn(
                                    "h-1 mt-2",
                                    supplier.risk_score > 6 && "[&>div]:bg-red-500",
                                    supplier.risk_score > 4 && supplier.risk_score <= 6 && "[&>div]:bg-orange-500",
                                    supplier.risk_score <= 4 && "[&>div]:bg-green-500"
                                )}
                            />
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
