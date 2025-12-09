'use client';

import { useState, useEffect } from 'react';
import {
    Wrench, Calendar, Clock, CheckCircle, PlayCircle, XCircle,
    User, Car, FileText, RefreshCw, Package, AlertTriangle, MapPin
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

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

interface Technician {
    id: string;
    name: string;
    specialties: string[];
    capacity_hours: number;
}

export default function ServiceCenterDashboard() {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [loading, setLoading] = useState(true);
    const [statusFilter, setStatusFilter] = useState('all');

    // Data states
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [technicians, setTechnicians] = useState<Technician[]>([]);
    const [partsInventory, setPartsInventory] = useState<any>(null);

    // Fetch data
    useEffect(() => {
        fetchAllData();
    }, [statusFilter]);

    async function fetchAllData() {
        setLoading(true);
        try {
            const [appointmentsRes, techRes, partsRes] = await Promise.all([
                fetch(`${API_BASE}/admin/appointments-list?days=7`).then(r => r.json()),
                fetch(`${API_BASE}/admin/technicians`).then(r => r.json()),
                fetch(`${API_BASE}/inventory/parts`).then(r => r.json())
            ]);

            let filtered = appointmentsRes.appointments || [];
            if (statusFilter !== 'all') {
                filtered = filtered.filter((a: Appointment) => a.status === statusFilter);
            }

            setAppointments(filtered);
            setTechnicians(techRes.technicians || []);
            setPartsInventory(partsRes);
        } catch (error) {
            console.error('Failed to fetch data:', error);
            toast.error('Failed to load service center data');
        } finally {
            setLoading(false);
        }
    }

    async function updateAppointmentStatus(appointmentId: string, newStatus: string) {
        try {
            const res = await fetch(`${API_BASE}/scheduling/appointments/${appointmentId}/status?status=${newStatus}`, {
                method: 'PUT'
            });

            const data = await res.json();

            if (data.success) {
                toast.success(`Appointment status updated to ${newStatus}`);

                // If service completed, trigger CAPA generation
                if (newStatus === 'completed') {
                    const apt = appointments.find(a => a.id === appointmentId);
                    if (apt && (apt.urgency === 'high' || apt.urgency === 'critical')) {
                        await generateCAPAReport(apt);
                    }
                }

                // Auto-switch filter to show the updated appointment if needed
                if (statusFilter !== 'all' && statusFilter !== newStatus) {
                    setStatusFilter('all');
                }

                // Refresh data
                await fetchAllData();
            } else {
                toast.error(data.message || 'Failed to update status');
            }
        } catch (error) {
            console.error('Failed to update status:', error);
            toast.error('Failed to update appointment status. Please try again.');
        }
    }

    async function generateCAPAReport(appointment: Appointment) {
        try {
            const res = await fetch(`${API_BASE}/capa/generate/${appointment.vehicle_id}?component=${appointment.component}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    diagnosis_summary: appointment.diagnosis_summary,
                    failure_mode: `${appointment.component} failure requiring emergency service`,
                    region: 'Mountainous',
                    vehicle_data: { urgency: appointment.urgency }
                })
            });

            const data = await res.json();
            if (data.success) {
                toast.success(`CAPA Report ${data.capa_id} auto-generated for manufacturing review`, {
                    duration: 5000
                });
            }
        } catch (error) {
            console.error('CAPA generation failed:', error);
        }
    }

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'completed':
                return <Badge className="bg-green-500"><CheckCircle className="h-3 w-3 mr-1" /> Completed</Badge>;
            case 'in_progress':
                return <Badge className="bg-blue-500"><PlayCircle className="h-3 w-3 mr-1" /> In Progress</Badge>;
            case 'scheduled':
                return <Badge className="bg-yellow-500"><Clock className="h-3 w-3 mr-1" /> Scheduled</Badge>;
            case 'cancelled':
                return <Badge className="bg-red-500"><XCircle className="h-3 w-3 mr-1" /> Cancelled</Badge>;
            default:
                return <Badge>{status}</Badge>;
        }
    };

    const getUrgencyBadge = (urgency: string) => {
        switch (urgency) {
            case 'critical':
                return <Badge variant="destructive" className="animate-pulse"><AlertTriangle className="h-3 w-3 mr-1" /> Critical</Badge>;
            case 'high':
                return <Badge variant="destructive"><AlertTriangle className="h-3 w-3 mr-1" /> Emergency</Badge>;
            case 'medium':
                return <Badge variant="secondary">Medium</Badge>;
            default:
                return <Badge variant="outline">Low</Badge>;
        }
    };

    const todayAppointments = appointments.filter(a => {
        const today = new Date().toISOString().split('T')[0];
        return a.scheduled_date === today;
    });

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
                                <Wrench className="h-6 w-6" />
                                Service Center Dashboard
                            </h1>
                            <p className="text-muted-foreground">Technician view for appointment management</p>
                        </div>
                        <div className="flex gap-2">
                            <Select value={statusFilter} onValueChange={setStatusFilter}>
                                <SelectTrigger className="w-40">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All Status</SelectItem>
                                    <SelectItem value="scheduled">Scheduled</SelectItem>
                                    <SelectItem value="in_progress">In Progress</SelectItem>
                                    <SelectItem value="completed">Completed</SelectItem>
                                </SelectContent>
                            </Select>
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
                                    Today's Appointments
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{todayAppointments.length}</div>
                                <p className="text-xs text-muted-foreground">
                                    {todayAppointments.filter(a => a.urgency === 'high').length} emergency
                                </p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    In Progress
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-blue-500">
                                    {appointments.filter(a => a.status === 'in_progress').length}
                                </div>
                                <p className="text-xs text-muted-foreground">Currently servicing</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Technicians Active
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-green-500">
                                    {technicians.length}
                                </div>
                                <p className="text-xs text-muted-foreground">Available today</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Parts Status
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-green-500">OK</div>
                                <p className="text-xs text-muted-foreground">All parts in stock</p>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Main Content Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Appointments List */}
                        <Card className="lg:col-span-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Calendar className="h-5 w-5" />
                                    Appointments Queue
                                </CardTitle>
                                <CardDescription>Manage service appointments</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    {appointments.map((apt) => (
                                        <div
                                            key={apt.id}
                                            className={cn(
                                                "p-4 border rounded-lg",
                                                (apt.urgency === 'high' || apt.urgency === 'critical') && apt.status === 'scheduled' && "border-red-500/50 bg-red-500/5"
                                            )}
                                        >
                                            <div className="flex items-start justify-between mb-3">
                                                <div>
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <code className="text-sm font-mono">{apt.id}</code>
                                                        {getUrgencyBadge(apt.urgency)}
                                                        {getStatusBadge(apt.status)}
                                                    </div>
                                                    <h4 className="font-semibold flex items-center gap-2">
                                                        <Car className="h-4 w-4 text-muted-foreground" />
                                                        {apt.vehicle_id}
                                                    </h4>
                                                </div>
                                                <div className="text-right text-sm text-muted-foreground">
                                                    <p>{apt.scheduled_date}</p>
                                                    <p className="font-medium">{apt.scheduled_time}</p>
                                                </div>
                                            </div>

                                            <div className="grid grid-cols-2 gap-4 mb-3 text-sm">
                                                <div>
                                                    <span className="text-muted-foreground">Component:</span>
                                                    <span className="ml-2 font-medium">{apt.component}</span>
                                                </div>
                                                <div>
                                                    <span className="text-muted-foreground">Technician:</span>
                                                    <span className="ml-2 font-medium">{apt.assigned_technician || 'Unassigned'}</span>
                                                </div>
                                            </div>

                                            <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                                                {apt.diagnosis_summary}
                                            </p>

                                            {/* Action Buttons */}
                                            <div className="flex gap-2 pt-3 border-t">
                                                {apt.status === 'scheduled' && (
                                                    <Button
                                                        size="sm"
                                                        className="bg-blue-500 hover:bg-blue-600"
                                                        onClick={() => updateAppointmentStatus(apt.id, 'in_progress')}
                                                    >
                                                        <PlayCircle className="h-4 w-4 mr-1" />
                                                        Start Service
                                                    </Button>
                                                )}
                                                {apt.status === 'in_progress' && (
                                                    <Button
                                                        size="sm"
                                                        className="bg-green-500 hover:bg-green-600"
                                                        onClick={() => updateAppointmentStatus(apt.id, 'completed')}
                                                    >
                                                        <CheckCircle className="h-4 w-4 mr-1" />
                                                        Complete
                                                    </Button>
                                                )}
                                                {apt.urgency === 'high' && apt.status === 'completed' && (
                                                    <Badge variant="outline" className="text-purple-400 border-purple-400">
                                                        <FileText className="h-3 w-3 mr-1" />
                                                        CAPA Generated
                                                    </Badge>
                                                )}
                                            </div>
                                        </div>
                                    ))}

                                    {appointments.length === 0 && (
                                        <div className="text-center py-8 text-muted-foreground">
                                            <Wrench className="h-12 w-12 mx-auto mb-4 opacity-50" />
                                            <p>No appointments found</p>
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Right Sidebar */}
                        <div className="space-y-6">
                            {/* Technicians */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <User className="h-4 w-4" />
                                        Technicians
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {technicians.map((tech) => (
                                            <div key={tech.id} className="flex items-center justify-between p-2 bg-muted rounded">
                                                <div>
                                                    <p className="font-medium text-sm">{tech.name}</p>
                                                    <div className="flex gap-1 mt-1">
                                                        {tech.specialties.slice(0, 2).map(s => (
                                                            <Badge key={s} variant="outline" className="text-xs">{s}</Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                                <Badge className="bg-green-500 text-xs">Active</Badge>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Parts Inventory */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <Package className="h-4 w-4" />
                                        Parts Inventory
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2 text-sm">
                                        <div className="flex justify-between p-2 bg-muted rounded">
                                            <span>Ceramic Brake Pads</span>
                                            <Badge className="bg-green-500">12</Badge>
                                        </div>
                                        <div className="flex justify-between p-2 bg-muted rounded">
                                            <span>Battery Cells</span>
                                            <Badge className="bg-green-500">24</Badge>
                                        </div>
                                        <div className="flex justify-between p-2 bg-muted rounded">
                                            <span>Inverter Chips</span>
                                            <Badge className="bg-yellow-500">4</Badge>
                                        </div>
                                        <div className="flex justify-between p-2 bg-muted rounded">
                                            <span>Motor Resolvers</span>
                                            <Badge className="bg-green-500">8</Badge>
                                        </div>
                                    </div>

                                    {/* Supply Chain Alert */}
                                    <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                                        <p className="text-xs text-yellow-400 flex items-center gap-1">
                                            <AlertTriangle className="h-3 w-3" />
                                            Inverter chips shipment arriving Tuesday
                                        </p>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Service Center Info */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <MapPin className="h-4 w-4" />
                                        Service Center
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="font-medium">Downtown EV Hub</p>
                                    <p className="text-xs text-muted-foreground">SC-001</p>
                                    <div className="mt-2 text-sm">
                                        <p>🕐 8:00 AM - 6:00 PM</p>
                                        <p>📞 +1-555-EV-REPAIR</p>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
