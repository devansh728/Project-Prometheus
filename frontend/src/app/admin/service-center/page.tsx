'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import {
    Phone, Trash2, RefreshCw, ChevronRight,
    Car, Wrench, AlertTriangle, CheckCircle, Clock
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api/v1';

interface Appointment {
    id: string;
    vehicle_id: string;
    center_id: string;
    component: string;
    diagnosis_summary: string;
    urgency: string;
    status: string;
    scheduled_date: string;
    scheduled_time: string;
    stage: string;
    is_voice_booked: boolean;
    progress_pct: number;
    center_name?: string;
}

const STAGES = [
    { value: 'INTAKE', label: 'Intake', icon: Car },
    { value: 'DIAGNOSIS', label: 'Diagnosis', icon: Wrench },
    { value: 'WAITING_PARTS', label: 'Waiting Parts', icon: Clock },
    { value: 'REPAIR', label: 'Repair', icon: Wrench },
    { value: 'QUALITY_CHECK', label: 'QC', icon: CheckCircle },
    { value: 'READY', label: 'Ready', icon: CheckCircle },
    { value: 'PICKED_UP', label: 'Picked Up', icon: Car },
];

export default function ServiceCenterPage() {
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [loading, setLoading] = useState(true);
    const [updatingId, setUpdatingId] = useState<string | null>(null);

    useEffect(() => {
        fetchAppointments();
    }, []);

    async function fetchAppointments() {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/service-center/appointments`);
            const data = await res.json();
            setAppointments(data.appointments || []);
        } catch (error) {
            console.error('Failed to fetch appointments:', error);
            toast.error('Failed to load appointments');
        } finally {
            setLoading(false);
        }
    }

    async function updateStage(appointmentId: string, newStage: string) {
        setUpdatingId(appointmentId);
        try {
            const res = await fetch(`${API_BASE}/service-center/appointments/${appointmentId}/stage`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stage: newStage })
            });
            const data = await res.json();

            if (data.success) {
                toast.success(`Stage updated to ${newStage}`);
                if (data.notification_triggered) {
                    toast.info('📞 Customer notification triggered!');
                }
                fetchAppointments();
            }
        } catch (error) {
            toast.error('Failed to update stage');
        } finally {
            setUpdatingId(null);
        }
    }

    async function deleteAppointment(appointmentId: string) {
        if (!confirm('Delete this appointment?')) return;

        try {
            const res = await fetch(`${API_BASE}/admin/appointments/${appointmentId}`, {
                method: 'DELETE'
            });
            const data = await res.json();

            if (data.success) {
                toast.success('Appointment deleted');
                fetchAppointments();
            }
        } catch (error) {
            toast.error('Failed to delete appointment');
        }
    }

    async function notifyReady(appointmentId: string) {
        try {
            const res = await fetch(`${API_BASE}/service-center/appointments/${appointmentId}/notify-ready`, {
                method: 'POST'
            });
            const data = await res.json();

            if (data.success) {
                toast.success(`📞 Voice call initiated for ${data.notification.vehicle_id}`);
            }
        } catch (error) {
            toast.error('Failed to send notification');
        }
    }

    function getUrgencyColor(urgency: string) {
        switch (urgency) {
            case 'critical': return 'bg-red-500';
            case 'high': return 'bg-orange-500';
            case 'medium': return 'bg-yellow-500';
            default: return 'bg-green-500';
        }
    }

    function getStageIndex(stage: string) {
        return STAGES.findIndex(s => s.value === stage);
    }

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold">Service Center</h1>
                    <p className="text-muted-foreground">Track and manage service appointments</p>
                </div>
                <Button onClick={fetchAppointments} variant="outline">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                </Button>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                    <CardContent className="pt-6">
                        <div className="text-2xl font-bold">{appointments.length}</div>
                        <p className="text-sm text-muted-foreground">Total Appointments</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="pt-6">
                        <div className="text-2xl font-bold text-purple-500">
                            {appointments.filter(a => a.is_voice_booked).length}
                        </div>
                        <p className="text-sm text-muted-foreground">Voice Booked</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="pt-6">
                        <div className="text-2xl font-bold text-orange-500">
                            {appointments.filter(a => a.stage === 'REPAIR').length}
                        </div>
                        <p className="text-sm text-muted-foreground">In Repair</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="pt-6">
                        <div className="text-2xl font-bold text-green-500">
                            {appointments.filter(a => a.stage === 'READY').length}
                        </div>
                        <p className="text-sm text-muted-foreground">Ready for Pickup</p>
                    </CardContent>
                </Card>
            </div>

            {/* Appointments List */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Wrench className="h-5 w-5" />
                        Active Appointments
                    </CardTitle>
                    <CardDescription>Click stage buttons to advance workflow</CardDescription>
                </CardHeader>
                <CardContent>
                    {loading ? (
                        <div className="text-center py-8 text-muted-foreground">Loading...</div>
                    ) : appointments.length === 0 ? (
                        <div className="text-center py-8 text-muted-foreground">
                            No appointments. Generate from Scheduler or book via Voice Agent.
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {appointments.map(apt => (
                                <div
                                    key={apt.id}
                                    className={`border rounded-lg p-4 ${apt.is_voice_booked ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20' : ''}`}
                                >
                                    <div className="flex items-start justify-between mb-3">
                                        <div className="flex items-center gap-3">
                                            <div className={`w-3 h-3 rounded-full ${getUrgencyColor(apt.urgency)}`} />
                                            <div>
                                                <div className="font-semibold flex items-center gap-2">
                                                    {apt.vehicle_id}
                                                    {apt.is_voice_booked && (
                                                        <Badge variant="secondary" className="bg-purple-100 text-purple-700">
                                                            <Phone className="h-3 w-3 mr-1" />
                                                            Voice Booked
                                                        </Badge>
                                                    )}
                                                    <Badge variant="outline">{apt.urgency}</Badge>
                                                </div>
                                                <div className="text-sm text-muted-foreground">
                                                    {apt.component} • {apt.scheduled_date} {apt.scheduled_time}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {apt.stage === 'READY' && (
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={() => notifyReady(apt.id)}
                                                    className="text-green-600"
                                                >
                                                    <Phone className="h-4 w-4 mr-1" />
                                                    Call Customer
                                                </Button>
                                            )}
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => deleteAppointment(apt.id)}
                                                className="text-red-500 hover:text-red-700"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>

                                    {/* Stage Progress Bar */}
                                    <div className="flex items-center gap-1 mt-3">
                                        {STAGES.map((stage, idx) => {
                                            const currentIdx = getStageIndex(apt.stage);
                                            const isComplete = idx < currentIdx;
                                            const isCurrent = idx === currentIdx;
                                            const StageIcon = stage.icon;

                                            return (
                                                <div key={stage.value} className="flex items-center flex-1">
                                                    <button
                                                        onClick={() => updateStage(apt.id, stage.value)}
                                                        disabled={updatingId === apt.id}
                                                        className={`
                                                            flex-1 py-2 px-1 text-xs rounded transition-all
                                                            ${isComplete ? 'bg-green-500 text-white' : ''}
                                                            ${isCurrent ? 'bg-blue-500 text-white ring-2 ring-blue-300' : ''}
                                                            ${!isComplete && !isCurrent ? 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300' : ''}
                                                        `}
                                                    >
                                                        <div className="flex flex-col items-center">
                                                            <StageIcon className="h-3 w-3 mb-1" />
                                                            <span className="truncate">{stage.label}</span>
                                                        </div>
                                                    </button>
                                                    {idx < STAGES.length - 1 && (
                                                        <ChevronRight className={`h-4 w-4 mx-0.5 ${isComplete ? 'text-green-500' : 'text-gray-300'}`} />
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>

                                    {/* Diagnosis Summary */}
                                    {apt.diagnosis_summary && (
                                        <div className="mt-3 text-sm text-muted-foreground bg-muted/50 p-2 rounded">
                                            {apt.diagnosis_summary}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
