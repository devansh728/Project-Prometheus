"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import {
    Package,
    Wrench,
    ClipboardCheck,
    Car,
    CheckCircle2,
    Clock,
    AlertCircle,
    ChevronLeft,
    RefreshCw,
    MessageCircle,
    Phone
} from "lucide-react";
import { cn } from "@/lib/utils";

// Service lifecycle stages
const STAGES = [
    { id: "INTAKE", label: "Vehicle Received", icon: Car, description: "Your vehicle has been checked in" },
    { id: "DIAGNOSIS", label: "Diagnosing", icon: ClipboardCheck, description: "Our technician is inspecting your vehicle" },
    { id: "WAITING_PARTS", label: "Parts Ordered", icon: Package, description: "Waiting for parts delivery" },
    { id: "REPAIR", label: "Repair in Progress", icon: Wrench, description: "Active repair work underway" },
    { id: "QUALITY_CHECK", label: "Quality Check", icon: ClipboardCheck, description: "Final verification before handover" },
    { id: "READY", label: "Ready for Pickup", icon: CheckCircle2, description: "Your vehicle is ready!" },
    { id: "PICKED_UP", label: "Picked Up", icon: Car, description: "Thank you for choosing us" },
];

interface TimelineStage {
    stage: string;
    label: string;
    status: "completed" | "current" | "upcoming";
    timestamp: string | null;
    note: string;
}

interface ServiceTicket {
    id: string;
    appointment_id: string;
    vehicle_id: string;
    status: string;
    stage_log: Array<{ timestamp: string; stage: string; note: string }>;
    estimated_completion: string;
    technician_id: string;
    technician_notes: string;
    progress_percent: number;
    message: string;
    is_complete: boolean;
}

export default function ServiceTrackingPage() {
    const params = useParams();
    const router = useRouter();
    const vin = params.vin as string;

    const [ticket, setTicket] = useState<ServiceTicket | null>(null);
    const [timeline, setTimeline] = useState<TimelineStage[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);

    const fetchTicket = async () => {
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

            // Fetch ticket
            const ticketRes = await fetch(`${apiUrl}/api/v1/tickets/${vin}`);
            const ticketData = await ticketRes.json();

            if (ticketData.error) {
                setError(ticketData.error);
                setLoading(false);
                return;
            }

            setTicket(ticketData);

            // Fetch timeline
            const timelineRes = await fetch(`${apiUrl}/api/v1/tickets/${ticketData.id}/timeline`);
            const timelineData = await timelineRes.json();
            setTimeline(timelineData.timeline || []);

            setError(null);
        } catch (err) {
            console.error("Error fetching ticket:", err);
            setError("Failed to load service status");
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        fetchTicket();

        // Set up polling for real-time updates
        const interval = setInterval(fetchTicket, 30000); // Every 30 seconds
        return () => clearInterval(interval);
    }, [vin]);

    // WebSocket for real-time updates
    useEffect(() => {
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
        const ws = new WebSocket(`${wsUrl}/ws/vehicles/${vin}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "ticket_update") {
                toast.info(data.message || "Service status updated");
                fetchTicket();
            }
        };

        return () => ws.close();
    }, [vin]);

    const handleRefresh = () => {
        setRefreshing(true);
        fetchTicket();
    };

    const getCurrentStageIndex = () => {
        if (!ticket) return -1;
        return STAGES.findIndex(s => s.id === ticket.status);
    };

    const formatDate = (isoString: string | null) => {
        if (!isoString) return "";
        const date = new Date(isoString);
        return date.toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
            hour12: true,
        });
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
                <div className="animate-pulse text-white text-xl">Loading service status...</div>
            </div>
        );
    }

    if (error || !ticket) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
                <div className="max-w-4xl mx-auto">
                    <Button
                        variant="ghost"
                        onClick={() => router.back()}
                        className="text-white/70 hover:text-white mb-6"
                    >
                        <ChevronLeft className="w-4 h-4 mr-2" /> Back
                    </Button>

                    <Card className="bg-white/5 backdrop-blur-xl border-white/10">
                        <CardContent className="p-12 text-center">
                            <AlertCircle className="w-16 h-16 text-yellow-400 mx-auto mb-4" />
                            <h2 className="text-2xl font-bold text-white mb-2">No Active Service</h2>
                            <p className="text-white/60 mb-6">
                                {error || "There's no active service ticket for this vehicle."}
                            </p>
                            <Button onClick={() => router.push(`/vehicles/${vin}`)}>
                                Go to Vehicle Dashboard
                            </Button>
                        </CardContent>
                    </Card>
                </div>
            </div>
        );
    }

    const currentStageIndex = getCurrentStageIndex();
    const currentStage = STAGES[currentStageIndex];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4 md:p-8">
            <div className="max-w-4xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <Button
                        variant="ghost"
                        onClick={() => router.back()}
                        className="text-white/70 hover:text-white"
                    >
                        <ChevronLeft className="w-4 h-4 mr-2" /> Back
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={handleRefresh}
                        disabled={refreshing}
                        className="border-white/20 text-white hover:bg-white/10"
                    >
                        <RefreshCw className={cn("w-4 h-4 mr-2", refreshing && "animate-spin")} />
                        Refresh
                    </Button>
                </div>

                {/* Main Status Card */}
                <Card className="bg-white/5 backdrop-blur-xl border-white/10 overflow-hidden">
                    {/* Progress Header */}
                    <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <p className="text-white/80 text-sm">Service Ticket</p>
                                <h1 className="text-2xl font-bold text-white">{ticket.id}</h1>
                            </div>
                            {ticket.is_complete && (
                                <div className="bg-green-500/20 px-4 py-2 rounded-full">
                                    <span className="text-green-400 font-semibold">Complete</span>
                                </div>
                            )}
                        </div>

                        {/* Progress Bar */}
                        <div className="relative">
                            <div className="h-2 bg-white/20 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-white rounded-full transition-all duration-500"
                                    style={{ width: `${ticket.progress_percent}%` }}
                                />
                            </div>
                            <p className="text-white/80 text-sm mt-2">
                                {ticket.progress_percent}% Complete
                            </p>
                        </div>
                    </div>

                    {/* Current Status */}
                    <CardContent className="p-6">
                        <div className="flex items-start gap-4 mb-6">
                            {currentStage && (
                                <>
                                    <div className={cn(
                                        "p-4 rounded-xl",
                                        ticket.status === "READY" ? "bg-green-500/20" : "bg-purple-500/20"
                                    )}>
                                        <currentStage.icon className={cn(
                                            "w-8 h-8",
                                            ticket.status === "READY" ? "text-green-400" : "text-purple-400"
                                        )} />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-semibold text-white">
                                            {currentStage.label}
                                        </h2>
                                        <p className="text-white/60">{ticket.message}</p>
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Estimated Completion */}
                        {ticket.estimated_completion && !ticket.is_complete && (
                            <div className="flex items-center gap-2 text-white/70 mb-6">
                                <Clock className="w-4 h-4" />
                                <span>Estimated ready: {formatDate(ticket.estimated_completion)}</span>
                            </div>
                        )}

                        {/* Technician Notes */}
                        {ticket.technician_notes && (
                            <div className="bg-white/5 rounded-lg p-4 mb-6">
                                <h3 className="text-sm font-semibold text-white/80 mb-1">Technician Notes</h3>
                                <p className="text-white/60">{ticket.technician_notes}</p>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Timeline */}
                <Card className="bg-white/5 backdrop-blur-xl border-white/10">
                    <CardHeader>
                        <CardTitle className="text-white">Service Timeline</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="relative">
                            {/* Timeline line */}
                            <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-white/10" />

                            {/* Timeline items */}
                            <div className="space-y-6">
                                {STAGES.map((stage, index) => {
                                    const timelineItem = timeline.find(t => t.stage === stage.id);
                                    const isComplete = index < currentStageIndex;
                                    const isCurrent = index === currentStageIndex;
                                    const isUpcoming = index > currentStageIndex;

                                    // Skip WAITING_PARTS if not in timeline
                                    if (stage.id === "WAITING_PARTS" && !timelineItem) {
                                        return null;
                                    }

                                    return (
                                        <div key={stage.id} className="relative pl-10">
                                            {/* Timeline dot */}
                                            <div className={cn(
                                                "absolute left-2 w-5 h-5 rounded-full border-2 flex items-center justify-center",
                                                isComplete && "bg-green-500 border-green-500",
                                                isCurrent && "bg-purple-500 border-purple-500 ring-4 ring-purple-500/30",
                                                isUpcoming && "bg-slate-700 border-slate-600"
                                            )}>
                                                {isComplete && <CheckCircle2 className="w-3 h-3 text-white" />}
                                            </div>

                                            {/* Content */}
                                            <div className={cn(
                                                "p-4 rounded-lg transition-all",
                                                isCurrent && "bg-purple-500/10 border border-purple-500/30",
                                                isComplete && "bg-green-500/5",
                                                isUpcoming && "opacity-50"
                                            )}>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <stage.icon className={cn(
                                                        "w-4 h-4",
                                                        isComplete && "text-green-400",
                                                        isCurrent && "text-purple-400",
                                                        isUpcoming && "text-slate-500"
                                                    )} />
                                                    <h3 className={cn(
                                                        "font-semibold",
                                                        isComplete && "text-green-400",
                                                        isCurrent && "text-purple-400",
                                                        isUpcoming && "text-slate-500"
                                                    )}>
                                                        {stage.label}
                                                    </h3>
                                                </div>

                                                <p className="text-white/50 text-sm">{stage.description}</p>

                                                {timelineItem?.timestamp && (
                                                    <p className="text-white/40 text-xs mt-1">
                                                        {formatDate(timelineItem.timestamp)}
                                                    </p>
                                                )}

                                                {timelineItem?.note && (
                                                    <p className="text-white/60 text-sm mt-2 italic">
                                                        "{timelineItem.note}"
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Action Buttons */}
                <div className="grid grid-cols-2 gap-4">
                    <Button
                        variant="outline"
                        className="border-white/20 text-white hover:bg-white/10 h-auto py-4"
                        onClick={() => router.push(`/chat/${vin}`)}
                    >
                        <MessageCircle className="w-5 h-5 mr-2" />
                        <div className="text-left">
                            <div className="font-semibold">Chat with Us</div>
                            <div className="text-xs text-white/60">Ask questions about your service</div>
                        </div>
                    </Button>

                    <Button
                        variant="outline"
                        className="border-white/20 text-white hover:bg-white/10 h-auto py-4"
                    >
                        <Phone className="w-5 h-5 mr-2" />
                        <div className="text-left">
                            <div className="font-semibold">Call Service Center</div>
                            <div className="text-xs text-white/60">Speak with a technician</div>
                        </div>
                    </Button>
                </div>

                {/* Ready for Pickup CTA */}
                {ticket.status === "READY" && (
                    <Card className="bg-gradient-to-r from-green-600 to-emerald-600 border-0">
                        <CardContent className="p-6 text-center">
                            <CheckCircle2 className="w-12 h-12 text-white mx-auto mb-3" />
                            <h2 className="text-2xl font-bold text-white mb-2">
                                Your Vehicle is Ready! 🎉
                            </h2>
                            <p className="text-white/90 mb-4">
                                Pick up your vehicle at your convenience during business hours.
                            </p>
                            <Button
                                variant="secondary"
                                size="lg"
                                onClick={() => router.push(`/chat/${vin}`)}
                            >
                                Schedule Pickup Time
                            </Button>
                        </CardContent>
                    </Card>
                )}
            </div>
        </div>
    );
}
