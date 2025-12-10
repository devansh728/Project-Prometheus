'use client';

import { useState, useEffect } from 'react';
import {
    Calendar, Users, Clock, BarChart3, Wrench, TrendingUp,
    AlertTriangle, CheckCircle, RefreshCw, Zap, Plus, Trash2, Sparkles
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { TetrisCalendar, TetrisDay, TetrisAppointment, generateDemoTetrisData } from '@/components/ui/TetrisCalendar';
import { PriorityQueueWidget } from '@/components/ui/PriorityQueueWidget';
import { TowingDispatchPanel } from '@/components/ui/TowingDispatchPanel';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface LaborPrediction {
    date: string;
    day: string;
    scheduled_appointments: number;
    scheduled_hours: number;
    total_labor_hours: number;
    capacity_hours: number;
    utilization_pct: number;
    status: string;
}

interface StaffingRecommendation {
    date: string;
    day: string;
    action: string;
    message: string;
    urgency: string;
}

interface HeatmapDay {
    date: string;
    day: string;
    hours: { hour: string; appointments: number; capacity: number; intensity: number }[];
}

interface Technician {
    id: string;
    name: string;
    specialties: string[];
    capacity_hours: number;
}

interface Appointment {
    id: string;
    vehicle_id: string;
    component: string;
    scheduled_date: string;
    scheduled_time: string;
    status: string;
    urgency: string;
    diagnosis_summary: string;
    estimated_cost: string;
    center_name: string;
    assigned_technician: string;
}

export default function SchedulerDashboard() {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [loading, setLoading] = useState(true);
    const [forecastDays, setForecastDays] = useState('7');
    const [appointmentFilter, setAppointmentFilter] = useState('all');

    // Data states
    const [laborForecast, setLaborForecast] = useState<LaborPrediction[]>([]);
    const [staffingRecs, setStaffingRecs] = useState<StaffingRecommendation[]>([]);
    const [heatmap, setHeatmap] = useState<HeatmapDay[]>([]);
    const [technicians, setTechnicians] = useState<Technician[]>([]);
    const [overview, setOverview] = useState<any>(null);
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [optimizing, setOptimizing] = useState(false);
    const [tetrisData, setTetrisData] = useState<TetrisDay[]>([]);

    // Generate/Clear/Optimize state
    const [generating, setGenerating] = useState(false);
    const [generateCount, setGenerateCount] = useState(5);
    const [showGenerateModal, setShowGenerateModal] = useState(false);
    const [clearing, setClearing] = useState(false);
    const [optimizationResult, setOptimizationResult] = useState<any>(null);

    // Only fetch appointments on load - all other stats come from optimization
    useEffect(() => {
        fetchAppointmentsOnly();
    }, []);

    async function fetchAppointmentsOnly() {
        setLoading(true);
        try {
            // Only fetch appointments - stats come from optimize endpoint
            const appointmentsRes = await fetch(`${API_BASE}/admin/appointments-list?days=${forecastDays}`).then(r => r.json());
            const apts = appointmentsRes.appointments || [];
            setAppointments(apts);

            // Generate Tetris calendar from appointments
            const tetris = convertAppointmentsToTetris(apts);
            setTetrisData(tetris);

            // Clear optimization result when appointments change
            // Stats will be populated after clicking Optimize
        } catch (error) {
            console.error('Failed to fetch appointments:', error);
        } finally {
            setLoading(false);
        }
    }

    // Alias for refresh button
    function fetchAllData() {
        fetchAppointmentsOnly();
    }

    // Convert appointments to Tetris calendar format
    function convertAppointmentsToTetris(apts: Appointment[]): TetrisDay[] {
        const dayMap = new Map<string, TetrisAppointment[]>();

        apts.forEach(apt => {
            const date = apt.scheduled_date;
            if (!dayMap.has(date)) {
                dayMap.set(date, []);
            }

            // Determine appointment type based on urgency
            let type: 'emergency' | 'routine' | 'moved' | 'parts_blocked' = 'routine';
            if (apt.urgency === 'critical' || apt.urgency === 'high') type = 'emergency';
            else if (apt.urgency === 'medium') type = 'routine';
            else if (apt.urgency === 'low') type = 'routine';

            dayMap.get(date)!.push({
                id: apt.id,
                vehicle_id: apt.vehicle_id,
                component: apt.component,
                time: apt.scheduled_time || '09:00',
                duration: apt.component.toLowerCase().includes('brake') ? 2 : 1,
                type,
                technician: apt.assigned_technician,
                diagnosis: apt.diagnosis_summary
            });
        });

        // Generate 14 days of data
        const days: TetrisDay[] = [];
        const baseDate = new Date();
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

        for (let i = 0; i < 14; i++) {
            const date = new Date(baseDate);
            date.setDate(baseDate.getDate() + i);
            const dateStr = date.toISOString().split('T')[0];

            days.push({
                date: dateStr,
                day: dayNames[date.getDay()],
                appointments: dayMap.get(dateStr) || [],
                parts_blocked_slots: []  // Will be set by optimization
            });
        }

        // Always return DB data - no demo fallback
        return days;
    }

    async function handleOptimize(date: string) {
        setOptimizing(true);
        setOptimizationResult(null);
        try {
            const res = await fetch(`${API_BASE}/admin/optimize-schedule?date=${date}`, {
                method: 'POST'
            });
            const result = await res.json();

            // Debug logging
            console.log('📊 Optimization result:', result);
            console.log('📊 labor_forecast:', result.labor_forecast?.length);
            console.log('📊 technicians:', result.technicians?.length);
            console.log('📊 total_scheduled:', result.total_scheduled);

            setOptimizationResult(result);

            // Populate all stats from optimization result
            if (result.labor_forecast) {
                setLaborForecast(result.labor_forecast);
            }
            if (result.staffing_recommendations) {
                setStaffingRecs(result.staffing_recommendations);
            }
            if (result.heatmap) {
                setHeatmap(result.heatmap);
            }
            if (result.technicians) {
                setTechnicians(result.technicians);
            }
            if (result.total_scheduled !== undefined) {
                setOverview({
                    total_scheduled: result.total_scheduled,
                    total_labor_hours: result.total_labor_hours,
                    avg_utilization: result.avg_utilization,
                    capacity_hours: result.capacity_hours
                });
            }

            toast.success(result.message || 'Schedule optimized!');

            // Show detailed toast for rescheduled appointments
            if (result.rescheduled?.length > 0) {
                toast.info(`⚡ ${result.rescheduled.length} low-priority appointments moved for emergencies`);
            }

            // Update calendar with sorted appointments from optimize response
            if (result.appointments && result.appointments.length > 0) {
                setAppointments(result.appointments);
                const tetris = convertAppointmentsToTetris(result.appointments);
                setTetrisData(tetris);
                console.log('📊 Updated calendar with', result.appointments.length, 'sorted appointments');
            }

            fetchAllData();
        } catch (error) {
            toast.error('Optimization failed');
        } finally {
            setOptimizing(false);
        }
    }

    async function handleGenerate() {
        setGenerating(true);
        try {
            const res = await fetch(`${API_BASE}/admin/generate-appointments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count: generateCount })
            });
            const result = await res.json();
            toast.success(result.message || `Generated ${generateCount} appointments`);
            setShowGenerateModal(false);
            fetchAllData();
        } catch (error) {
            toast.error('Failed to generate appointments');
        } finally {
            setGenerating(false);
        }
    }

    async function handleClearAll() {
        if (!confirm('Are you sure you want to clear ALL appointments? This cannot be undone.')) {
            return;
        }

        setClearing(true);
        try {
            const res = await fetch(`${API_BASE}/admin/appointments/clear`, {
                method: 'DELETE'
            });
            const result = await res.json();
            toast.success(result.message || 'All appointments cleared');
            setOptimizationResult(null);
            fetchAllData();
        } catch (error) {
            toast.error('Failed to clear appointments');
        } finally {
            setClearing(false);
        }
    }


    const getStatusColor = (status: string) => {
        switch (status) {
            case 'low': return 'bg-blue-500';
            case 'optimal': return 'bg-green-500';
            case 'busy': return 'bg-yellow-500';
            case 'over_capacity': return 'bg-red-500';
            default: return 'bg-gray-500';
        }
    };

    const getUrgencyVariant = (urgency: string): 'default' | 'secondary' | 'destructive' | 'outline' => {
        switch (urgency) {
            case 'high': return 'destructive';
            case 'medium': return 'secondary';
            default: return 'outline';
        }
    };

    const getHeatColor = (intensity: number) => {
        if (intensity === 0) return 'bg-slate-100 dark:bg-slate-800';
        if (intensity < 0.3) return 'bg-green-200 dark:bg-green-900';
        if (intensity < 0.6) return 'bg-yellow-200 dark:bg-yellow-900';
        if (intensity < 0.9) return 'bg-orange-300 dark:bg-orange-800';
        return 'bg-red-400 dark:bg-red-700';
    };

    return (
        <div className="min-h-screen flex flex-col bg-background">
            <Navbar onMenuClick={() => setSidebarOpen(true)} />

            <div className="flex flex-1">
                <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                <main className="flex-1 p-6 overflow-auto">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h1 className="text-2xl font-bold flex items-center gap-2">
                                <Calendar className="h-6 w-6" />
                                Scheduler Dashboard
                            </h1>
                            <p className="text-muted-foreground">Labor forecasting & appointment optimization</p>
                        </div>
                        <div className="flex gap-2 flex-wrap">
                            {/* Generate Appointments */}
                            <Button
                                variant="outline"
                                onClick={() => setShowGenerateModal(true)}
                            >
                                <Plus className="h-4 w-4 mr-2" />
                                Generate
                            </Button>

                            {/* Clear All */}
                            <Button
                                variant="outline"
                                onClick={handleClearAll}
                                disabled={clearing}
                                className="text-red-500 hover:text-red-600"
                            >
                                <Trash2 className={cn("h-4 w-4 mr-2", clearing && "animate-spin")} />
                                Clear All
                            </Button>

                            {/* Optimize Workload */}
                            <Button
                                onClick={() => handleOptimize(new Date().toISOString().split('T')[0])}
                                disabled={optimizing}
                                className="bg-gradient-to-r from-purple-500 to-blue-500 text-white"
                            >
                                <Sparkles className={cn("h-4 w-4 mr-2", optimizing && "animate-spin")} />
                                Optimize Workload
                            </Button>

                            <Select value={forecastDays} onValueChange={setForecastDays}>
                                <SelectTrigger className="w-32">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="7">7 Days</SelectItem>
                                    <SelectItem value="14">14 Days</SelectItem>
                                    <SelectItem value="30">30 Days</SelectItem>
                                </SelectContent>
                            </Select>
                            <Button variant="outline" onClick={fetchAllData}>
                                <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                                Refresh
                            </Button>
                        </div>
                    </div>

                    {/* Generate Modal */}
                    {showGenerateModal && (
                        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                            <Card className="w-96">
                                <CardHeader>
                                    <CardTitle>Generate Appointments</CardTitle>
                                    <CardDescription>Create sample appointments for demo</CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div>
                                        <label className="text-sm font-medium">Number of Appointments</label>
                                        <input
                                            type="number"
                                            min={1}
                                            max={20}
                                            value={generateCount}
                                            onChange={(e) => setGenerateCount(Number(e.target.value))}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    </div>
                                    <div className="flex gap-2 justify-end">
                                        <Button variant="outline" onClick={() => setShowGenerateModal(false)}>
                                            Cancel
                                        </Button>
                                        <Button onClick={handleGenerate} disabled={generating}>
                                            {generating ? 'Generating...' : `Generate ${generateCount} Appointments`}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    )}

                    {/* Optimization Result Panel */}
                    {optimizationResult && (
                        <Card className="mb-6 border-purple-500/50 bg-gradient-to-r from-purple-500/10 to-blue-500/10">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <Sparkles className="h-5 w-5 text-purple-500" />
                                    Workload Optimization Results
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => setOptimizationResult(null)}
                                        className="ml-auto"
                                    >
                                        ✕
                                    </Button>
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    {/* Appointments Summary */}
                                    <div className="space-y-2">
                                        <h4 className="font-medium text-sm">📅 Appointments</h4>
                                        <div className="text-2xl font-bold">{optimizationResult.total_appointments}</div>
                                        <div className="flex gap-2 text-xs">
                                            <Badge variant="destructive">{optimizationResult.emergency_count} Emergency</Badge>
                                            <Badge variant="secondary">{optimizationResult.routine_count} Routine</Badge>
                                        </div>
                                        {optimizationResult.rescheduled?.length > 0 && (
                                            <div className="mt-2 p-2 bg-blue-500/20 rounded text-xs">
                                                ⚡ <strong>{optimizationResult.rescheduled.length}</strong> low-priority moved to 4 PM
                                            </div>
                                        )}
                                    </div>

                                    {/* Technicians Required */}
                                    <div className="space-y-2">
                                        <h4 className="font-medium text-sm">👷 Technicians Required</h4>
                                        <div className="text-2xl font-bold">{optimizationResult.total_technicians_needed}</div>
                                        <div className="space-y-1 text-xs">
                                            {Object.entries(optimizationResult.technicians_required || {}).map(([specialty, count]) => (
                                                <div key={specialty} className="flex justify-between">
                                                    <span className="capitalize">{specialty.replace(/_/g, ' ')}</span>
                                                    <Badge variant="outline">{count as number}</Badge>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Supply Chain Alerts */}
                                    <div className="space-y-2">
                                        <h4 className="font-medium text-sm">📦 Supply Chain</h4>
                                        {optimizationResult.supply_chain_alerts?.length > 0 ? (
                                            <div className="space-y-2">
                                                {optimizationResult.supply_chain_alerts.map((alert: any, i: number) => (
                                                    <div key={i} className="p-2 bg-yellow-500/20 rounded text-xs">
                                                        <strong>{alert.part}</strong>
                                                        <div>{alert.status} - ETA: {alert.eta || `Qty: ${alert.quantity}`}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="text-sm text-green-500">✅ All parts in stock</div>
                                        )}
                                        {optimizationResult.blocked_slots?.length > 0 && (
                                            <div className="mt-2 p-2 bg-gray-500/20 rounded text-xs">
                                                🚫 Morning slots blocked - parts arriving Tuesday
                                            </div>
                                        )}
                                    </div>

                                    {/* CAPA Manufacturing Insights */}
                                    {optimizationResult.capa_insights && (
                                        <div className="md:col-span-3 mt-4 p-4 bg-gradient-to-r from-red-500/10 to-orange-500/10 rounded-lg border border-red-500/30">
                                            <h4 className="font-medium text-sm flex items-center gap-2 text-red-600 dark:text-red-400">
                                                🏭 Manufacturing Quality Insights
                                            </h4>
                                            <div className="mt-2 space-y-2">
                                                <p className="text-sm font-semibold">
                                                    {optimizationResult.capa_insights.headline}
                                                </p>
                                                <p className="text-sm text-muted-foreground">
                                                    <strong>Top Issue:</strong> {optimizationResult.capa_insights.top_component}
                                                    ({optimizationResult.capa_insights.affected_count} vehicles affected)
                                                </p>
                                                <p className="text-sm text-orange-600 dark:text-orange-400">
                                                    ⚠️ {optimizationResult.capa_insights.recommendation}
                                                </p>
                                                <a
                                                    href={optimizationResult.capa_insights.action_link}
                                                    className="inline-block mt-2 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                                                >
                                                    View CAPA Dashboard →
                                                </a>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Summary Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Total Scheduled
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{overview?.total_scheduled || 0}</div>
                                <p className="text-xs text-muted-foreground">Next {forecastDays} days</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Total Labor Hours
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">
                                    {laborForecast.reduce((sum, p) => sum + p.total_labor_hours, 0).toFixed(0)}h
                                </div>
                                <p className="text-xs text-muted-foreground">Forecasted</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Avg Utilization
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">
                                    {laborForecast.length > 0
                                        ? (laborForecast.reduce((sum, p) => sum + p.utilization_pct, 0) / laborForecast.length).toFixed(0)
                                        : 0}%
                                </div>
                                <p className="text-xs text-muted-foreground">Capacity used</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Staffing Alerts
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-orange-500">
                                    {staffingRecs.filter(r => r.urgency === 'high').length}
                                </div>
                                <p className="text-xs text-muted-foreground">High priority</p>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Main Content Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Labor Forecast Chart */}
                        <Card className="lg:col-span-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <BarChart3 className="h-5 w-5" />
                                    Labor Hours Forecast
                                </CardTitle>
                                <CardDescription>Predicted labor hours vs capacity</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {laborForecast.map((pred) => (
                                        <div key={pred.date} className="space-y-1">
                                            <div className="flex justify-between text-sm">
                                                <span className="font-medium">{pred.day} ({pred.date})</span>
                                                <span className="flex items-center gap-2">
                                                    <Badge variant="outline" className={cn("text-xs", getStatusColor(pred.status), "text-white")}>
                                                        {pred.status}
                                                    </Badge>
                                                    {pred.total_labor_hours}h / {pred.capacity_hours}h
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Progress
                                                    value={Math.min(pred.utilization_pct, 100)}
                                                    className="h-2"
                                                />
                                                <span className="text-xs w-12 text-right">{pred.utilization_pct}%</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Staffing Recommendations */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Users className="h-5 w-5" />
                                    Staffing Alerts
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {staffingRecs.length === 0 ? (
                                    <div className="text-center text-muted-foreground py-4">
                                        <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-500" />
                                        <p>No staffing alerts</p>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        {staffingRecs.map((rec, i) => (
                                            <div key={i} className="p-3 rounded-lg border">
                                                <div className="flex items-start justify-between">
                                                    <Badge variant={getUrgencyVariant(rec.urgency)}>{rec.urgency}</Badge>
                                                    <span className="text-xs text-muted-foreground">{rec.date}</span>
                                                </div>
                                                <p className="text-sm mt-2">{rec.message}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Capacity Heatmap */}
                        <Card className="lg:col-span-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Zap className="h-5 w-5" />
                                    Capacity Heatmap
                                </CardTitle>
                                <CardDescription>Hourly appointment density by day</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs">
                                        <thead>
                                            <tr>
                                                <th className="text-left p-1">Day</th>
                                                {['8AM', '9AM', '10AM', '11AM', '12PM', '1PM', '2PM', '3PM', '4PM', '5PM'].map(h => (
                                                    <th key={h} className="p-1 text-center">{h}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {heatmap.map((day) => (
                                                <tr key={day.date}>
                                                    <td className="p-1 font-medium whitespace-nowrap">
                                                        {day.day.slice(0, 3)}
                                                    </td>
                                                    {day.hours.map((h, i) => (
                                                        <td key={i} className="p-1">
                                                            <div
                                                                className={cn(
                                                                    "w-6 h-6 rounded flex items-center justify-center text-xs font-medium",
                                                                    getHeatColor(h.intensity)
                                                                )}
                                                                title={`${h.appointments}/${h.capacity} appointments`}
                                                            >
                                                                {h.appointments}
                                                            </div>
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                <div className="flex gap-2 mt-4 text-xs">
                                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-slate-100 rounded"></div> Empty</span>
                                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-green-200 rounded"></div> Low</span>
                                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-yellow-200 rounded"></div> Medium</span>
                                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-orange-300 rounded"></div> High</span>
                                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-red-400 rounded"></div> Full</span>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Technicians */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Wrench className="h-5 w-5" />
                                    Technicians
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {technicians.map((tech) => (
                                        <div key={tech.id} className="flex items-center justify-between p-2 rounded border">
                                            <div>
                                                <p className="font-medium text-sm">{tech.name}</p>
                                                <div className="flex gap-1 mt-1">
                                                    {tech.specialties.map(s => (
                                                        <Badge key={s} variant="outline" className="text-xs">{s}</Badge>
                                                    ))}
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <p className="text-sm font-medium">{tech.capacity_hours}h</p>
                                                <p className="text-xs text-muted-foreground">capacity</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Priority Queue & Towing - Demo */}
                        <PriorityQueueWidget />
                        <TowingDispatchPanel />

                        {/* Optimize Schedule */}
                        <Card className="lg:col-span-3">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5" />
                                    Schedule Optimization
                                </CardTitle>
                                <CardDescription>Run optimization to assign technicians optimally</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="flex flex-wrap gap-2">
                                    {laborForecast.slice(0, 7).map((pred) => (
                                        <Button
                                            key={pred.date}
                                            variant="outline"
                                            disabled={optimizing}
                                            onClick={() => handleOptimize(pred.date)}
                                        >
                                            {optimizing ? (
                                                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                                            ) : (
                                                <Zap className="h-4 w-4 mr-2" />
                                            )}
                                            Optimize {pred.day}
                                        </Button>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Appointments List */}
                        <Card className="lg:col-span-3">
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <CardTitle className="flex items-center gap-2">
                                            <Clock className="h-5 w-5" />
                                            Appointments
                                        </CardTitle>
                                        <CardDescription>All scheduled and completed appointments</CardDescription>
                                    </div>
                                    <div className="flex gap-1">
                                        <Button
                                            size="sm"
                                            variant={appointmentFilter === 'all' ? 'default' : 'outline'}
                                            onClick={() => setAppointmentFilter('all')}
                                        >
                                            All ({appointments.length})
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant={appointmentFilter === 'scheduled' ? 'default' : 'outline'}
                                            onClick={() => setAppointmentFilter('scheduled')}
                                        >
                                            Scheduled
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant={appointmentFilter === 'completed' ? 'default' : 'outline'}
                                            onClick={() => setAppointmentFilter('completed')}
                                        >
                                            Completed
                                        </Button>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b">
                                                <th className="text-left p-2 font-medium">Date/Time</th>
                                                <th className="text-left p-2 font-medium">Vehicle</th>
                                                <th className="text-left p-2 font-medium">Component</th>
                                                <th className="text-left p-2 font-medium">Urgency</th>
                                                <th className="text-left p-2 font-medium">Status</th>
                                                <th className="text-left p-2 font-medium">Technician</th>
                                                <th className="text-left p-2 font-medium">Center</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {appointments
                                                .filter(a => appointmentFilter === 'all' || a.status === appointmentFilter)
                                                .slice(0, 20)
                                                .map((appt) => (
                                                    <tr key={appt.id} className="border-b hover:bg-muted/50">
                                                        <td className="p-2">
                                                            <div className="font-medium">{appt.scheduled_date}</div>
                                                            <div className="text-xs text-muted-foreground">{appt.scheduled_time}</div>
                                                        </td>
                                                        <td className="p-2">
                                                            <code className="text-xs bg-muted px-1 rounded">{appt.vehicle_id}</code>
                                                        </td>
                                                        <td className="p-2">
                                                            <Badge variant="outline" className="text-xs">{appt.component}</Badge>
                                                        </td>
                                                        <td className="p-2">
                                                            <Badge
                                                                variant={appt.urgency === 'critical' ? 'destructive' : appt.urgency === 'high' ? 'secondary' : 'outline'}
                                                                className="text-xs"
                                                            >
                                                                {appt.urgency}
                                                            </Badge>
                                                        </td>
                                                        <td className="p-2">
                                                            <Badge
                                                                className={cn(
                                                                    "text-xs",
                                                                    appt.status === 'completed' && "bg-green-500",
                                                                    appt.status === 'scheduled' && "bg-blue-500",
                                                                    appt.status === 'in_progress' && "bg-yellow-500"
                                                                )}
                                                            >
                                                                {appt.status}
                                                            </Badge>
                                                        </td>
                                                        <td className="p-2">
                                                            <span className={appt.assigned_technician === 'Unassigned' ? 'text-muted-foreground italic' : 'font-medium'}>
                                                                {appt.assigned_technician}
                                                            </span>
                                                        </td>
                                                        <td className="p-2 text-xs text-muted-foreground max-w-[150px] truncate">
                                                            {appt.center_name}
                                                        </td>
                                                    </tr>
                                                ))}
                                        </tbody>
                                    </table>
                                    {appointments.length === 0 && (
                                        <div className="text-center py-8 text-muted-foreground">
                                            No appointments found
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Tetris Calendar View */}
                    <div className="mt-6">
                        <TetrisCalendar
                            days={tetrisData.length > 0 ? tetrisData : generateDemoTetrisData()}
                            onSlotClick={(day, time) => toast.info(`Slot clicked: ${day} at ${time}`)}
                            onAppointmentClick={(apt) => toast.info(`Appointment: ${apt.vehicle_id} - ${apt.component}`)}
                            highlightEmergency={true}
                        />
                    </div>
                </main>
            </div>
        </div>
    );
}
