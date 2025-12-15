'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { TelemetryData } from '@/types';

interface TelemetryDetailPanelProps {
    telemetry: TelemetryData | null;
    className?: string;
}

// Define feature categories matching the 66 ML features
const FEATURE_CATEGORIES = {
    'Vehicle Dynamics': [
        { key: 'speed_kmh', label: 'Speed', unit: 'km/h', decimals: 1 },
        { key: 'motor_rpm', label: 'Motor RPM', unit: 'rpm', decimals: 0 },
        { key: 'throttle_pct', label: 'Throttle', unit: '%', decimals: 1 },
        { key: 'brake_pct', label: 'Brake', unit: '%', decimals: 1 },
        { key: 'regen_pct', label: 'Regen', unit: '%', decimals: 1 },
    ],
    'Temperatures': [
        { key: 'battery_temp_c', label: 'Battery', unit: '°C', decimals: 1, warnThreshold: 45, critThreshold: 55 },
        { key: 'motor_temp_c', label: 'Motor', unit: '°C', decimals: 1, warnThreshold: 90, critThreshold: 110 },
        { key: 'inverter_temp_c', label: 'Inverter', unit: '°C', decimals: 1, warnThreshold: 75, critThreshold: 90 },
        { key: 'brake_temp_c', label: 'Brakes', unit: '°C', decimals: 1, warnThreshold: 200, critThreshold: 300 },
        { key: 'coolant_temp_c', label: 'Coolant', unit: '°C', decimals: 1, warnThreshold: 60, critThreshold: 80 },
    ],
    'Battery': [
        { key: 'battery_soc_pct', label: 'SOC', unit: '%', decimals: 1 },
        { key: 'battery_voltage_v', label: 'Voltage', unit: 'V', decimals: 1 },
        { key: 'battery_current_a', label: 'Current', unit: 'A', decimals: 1 },
        { key: 'battery_cell_delta_v', label: 'Cell Delta', unit: 'V', decimals: 3, warnThreshold: 0.1, critThreshold: 0.2 },
    ],
    'Power': [
        { key: 'power_draw_kw', label: 'Power Draw', unit: 'kW', decimals: 1 },
        { key: 'hvac_power_kw', label: 'HVAC Power', unit: 'kW', decimals: 1 },
        { key: 'regen_efficiency', label: 'Regen Efficiency', unit: '', decimals: 2 },
    ],
    'Accelerometer': [
        { key: 'accel_x', label: 'Accel X', unit: 'g', decimals: 2 },
        { key: 'accel_y', label: 'Accel Y', unit: 'g', decimals: 2 },
        { key: 'accel_z', label: 'Accel Z', unit: 'g', decimals: 2 },
        { key: 'jerk_ms3', label: 'Jerk', unit: 'm/s³', decimals: 2 },
    ],
    'ML Status': [
        { key: 'frame_count', label: 'Frame Count', unit: '', decimals: 0 },
        { key: 'wear_index', label: 'Wear Index', unit: '', decimals: 2 },
    ],
};

// Get value color based on thresholds
function getValueColor(value: number, warn?: number, crit?: number): string {
    if (crit !== undefined && value >= crit) return 'text-red-500 font-bold';
    if (warn !== undefined && value >= warn) return 'text-yellow-500';
    return 'text-foreground';
}

export function TelemetryDetailPanel({ telemetry, className }: TelemetryDetailPanelProps) {
    if (!telemetry) {
        return (
            <Card className={className}>
                <CardHeader>
                    <CardTitle className="text-sm font-medium">Telemetry Details</CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-muted-foreground text-center py-8">
                        Waiting for telemetry data...
                    </p>
                </CardContent>
            </Card>
        );
    }

    // Get active faults
    const activeFaults = telemetry.active_faults || [];

    return (
        <Card className={className}>
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">
                        📊 All Telemetry Features
                    </CardTitle>
                    {activeFaults.length > 0 && (
                        <Badge variant="destructive" className="text-xs">
                            {activeFaults.length} Fault{activeFaults.length > 1 ? 's' : ''} Active
                        </Badge>
                    )}
                </div>
            </CardHeader>
            <CardContent>
                <ScrollArea className="h-[400px] pr-4">
                    <div className="space-y-4">
                        {/* Active Faults Warning */}
                        {activeFaults.length > 0 && (
                            <div className="p-2 rounded bg-red-500/10 border border-red-500/30">
                                <p className="text-xs font-medium text-red-500">
                                    🚨 Active Faults: {activeFaults.join(', ')}
                                </p>
                            </div>
                        )}

                        {/* Feature Categories */}
                        {Object.entries(FEATURE_CATEGORIES).map(([category, features]) => (
                            <div key={category}>
                                <h4 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                                    {category}
                                </h4>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                                    {features.map((feature) => {
                                        const rawValue = (telemetry as unknown as Record<string, unknown>)[feature.key];
                                        const value = typeof rawValue === 'number' ? rawValue : 0;
                                        const colorClass = getValueColor(
                                            value,
                                            'warnThreshold' in feature ? feature.warnThreshold : undefined,
                                            'critThreshold' in feature ? feature.critThreshold : undefined
                                        );

                                        return (
                                            <div key={feature.key} className="flex justify-between text-sm py-0.5">
                                                <span className="text-muted-foreground">{feature.label}</span>
                                                <span className={colorClass}>
                                                    {value.toFixed(feature.decimals)}{feature.unit}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        ))}

                        {/* Raw JSON for debugging */}
                        <div>
                            <h4 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                                Raw Data (All Keys)
                            </h4>
                            <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded max-h-32 overflow-auto font-mono">
                                {Object.entries(telemetry)
                                    .filter(([, v]) => typeof v === 'number')
                                    .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(2) : v}`)
                                    .join(', ')}
                            </div>
                        </div>
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
