'use client';

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Clock, Package, Wrench, ChevronLeft, ChevronRight, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

// Appointment types with colors
export type AppointmentType = 'emergency' | 'routine' | 'moved' | 'parts_blocked';

export interface TetrisAppointment {
    id: string;
    vehicle_id: string;
    component: string;
    time: string;  // e.g., "09:00"
    duration: number; // hours
    type: AppointmentType;
    technician?: string;
    status?: string;
    diagnosis?: string;
    bumped_from?: string; // For moved appointments
    parts_arrival?: string; // For parts-blocked
}

export interface TetrisDay {
    date: string;
    day: string;
    appointments: TetrisAppointment[];
    parts_blocked_slots?: { start: string; end: string; reason: string }[];
}

interface TetrisCalendarProps {
    days: TetrisDay[];
    onSlotClick?: (day: string, time: string) => void;
    onAppointmentClick?: (appointment: TetrisAppointment) => void;
    highlightEmergency?: boolean;
}

// Time slots for the day (8 AM to 6 PM)
const TIME_SLOTS = [
    '08:00', '09:00', '10:00', '11:00', '12:00',
    '13:00', '14:00', '15:00', '16:00', '17:00'
];

export function TetrisCalendar({
    days,
    onSlotClick,
    onAppointmentClick,
    highlightEmergency = false
}: TetrisCalendarProps) {
    const [currentWeekStart, setCurrentWeekStart] = useState(0);
    const [animatingBlock, setAnimatingBlock] = useState<string | null>(null);

    const visibleDays = days.slice(currentWeekStart, currentWeekStart + 7);

    const getBlockColor = (type: AppointmentType) => {
        switch (type) {
            case 'emergency':
                return 'bg-red-500 hover:bg-red-600 border-red-600';
            case 'routine':
                return 'bg-blue-500 hover:bg-blue-600 border-blue-600';
            case 'moved':
                return 'bg-blue-400 hover:bg-blue-500 border-blue-500 opacity-80 border-dashed';
            case 'parts_blocked':
                return 'bg-gray-500 hover:bg-gray-600 border-gray-600';
            default:
                return 'bg-slate-500';
        }
    };

    const getTypeIcon = (type: AppointmentType) => {
        switch (type) {
            case 'emergency':
                return <AlertTriangle className="h-3 w-3" />;
            case 'routine':
                return <Clock className="h-3 w-3" />;
            case 'moved':
                return <Zap className="h-3 w-3" />;
            case 'parts_blocked':
                return <Package className="h-3 w-3" />;
            default:
                return <Wrench className="h-3 w-3" />;
        }
    };

    const getTypeLabel = (type: AppointmentType) => {
        switch (type) {
            case 'emergency':
                return 'Emergency Repair';
            case 'routine':
                return 'Routine Service';
            case 'moved':
                return 'Rescheduled';
            case 'parts_blocked':
                return 'Parts Unavailable';
            default:
                return type;
        }
    };

    // Check if a slot is blocked due to parts
    const isSlotPartsBlocked = (day: TetrisDay, time: string) => {
        if (!day.parts_blocked_slots) return false;
        return day.parts_blocked_slots.some(slot =>
            time >= slot.start && time < slot.end
        );
    };

    // Get appointment at a specific time slot
    const getAppointmentAtSlot = (day: TetrisDay, time: string) => {
        return day.appointments.find(apt => {
            const aptHour = parseInt(apt.time.split(':')[0]);
            const slotHour = parseInt(time.split(':')[0]);
            return slotHour >= aptHour && slotHour < aptHour + apt.duration;
        });
    };

    // Check if this is the start of an appointment
    const isAppointmentStart = (appointment: TetrisAppointment, time: string) => {
        return appointment.time === time;
    };

    return (
        <Card className="w-full">
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                        <Wrench className="h-5 w-5" />
                        Service Bay Schedule
                    </CardTitle>
                    <div className="flex items-center gap-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setCurrentWeekStart(Math.max(0, currentWeekStart - 7))}
                            disabled={currentWeekStart === 0}
                        >
                            <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setCurrentWeekStart(Math.min(days.length - 7, currentWeekStart + 7))}
                            disabled={currentWeekStart >= days.length - 7}
                        >
                            <ChevronRight className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
                {/* Legend */}
                <div className="flex flex-wrap gap-3 mt-2 text-xs">
                    <div className="flex items-center gap-1">
                        <div className="w-4 h-4 rounded bg-red-500" />
                        <span>Emergency</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-4 h-4 rounded bg-blue-500" />
                        <span>Routine</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-4 h-4 rounded bg-blue-400 border border-dashed border-blue-600" />
                        <span>Moved</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-4 h-4 rounded bg-gray-500" />
                        <span>Parts Blocked</span>
                    </div>
                </div>
            </CardHeader>
            <CardContent className="pt-0">
                <TooltipProvider>
                    <div className="relative overflow-x-auto">
                        {/* Grid */}
                        <div className="grid grid-cols-[auto,repeat(7,1fr)] gap-px bg-muted rounded-lg overflow-hidden">
                            {/* Time column header */}
                            <div className="bg-card p-2 font-medium text-xs text-muted-foreground"></div>

                            {/* Day headers */}
                            {visibleDays.map((day) => (
                                <div key={day.date} className="bg-card p-2 text-center">
                                    <div className="font-medium text-xs">{day.day}</div>
                                    <div className="text-xs text-muted-foreground">
                                        {new Date(day.date).getDate()}
                                    </div>
                                </div>
                            ))}

                            {/* Time rows */}
                            {TIME_SLOTS.map((time) => (
                                <React.Fragment key={`row-${time}`}>
                                    {/* Time label */}
                                    <div
                                        className="bg-card p-2 text-xs text-muted-foreground font-mono"
                                    >
                                        {time}
                                    </div>

                                    {/* Day cells */}
                                    {visibleDays.map((day) => {
                                        const appointment = getAppointmentAtSlot(day, time);
                                        const isStart = appointment && isAppointmentStart(appointment, time);
                                        const isPartsBlocked = isSlotPartsBlocked(day, time);

                                        // Skip if this slot is part of an appointment but not the start
                                        if (appointment && !isStart) {
                                            return (
                                                <div
                                                    key={`${day.date}-${time}`}
                                                    className="bg-card"
                                                />
                                            );
                                        }

                                        return (
                                            <div
                                                key={`${day.date}-${time}`}
                                                className={cn(
                                                    "bg-card min-h-[3rem] relative",
                                                    !appointment && !isPartsBlocked && "hover:bg-muted/50 cursor-pointer"
                                                )}
                                                onClick={() => !appointment && onSlotClick?.(day.date, time)}
                                            >
                                                {/* Parts blocked overlay */}
                                                {isPartsBlocked && !appointment && (
                                                    <div className="absolute inset-0 bg-gray-300/50 dark:bg-gray-700/50 flex items-center justify-center">
                                                        <Package className="h-4 w-4 text-gray-500" />
                                                    </div>
                                                )}

                                                {/* Appointment block */}
                                                {appointment && isStart && (
                                                    <Tooltip>
                                                        <TooltipTrigger asChild>
                                                            <div
                                                                className={cn(
                                                                    "absolute left-0 right-0 mx-1 rounded-md border-2 text-white text-xs p-1.5 cursor-pointer transition-all",
                                                                    getBlockColor(appointment.type),
                                                                    highlightEmergency && appointment.type === 'emergency' && "animate-pulse ring-2 ring-red-400",
                                                                    animatingBlock === appointment.id && "scale-105 shadow-lg"
                                                                )}
                                                                style={{
                                                                    height: `calc(${appointment.duration * 3}rem - 4px)`,
                                                                    zIndex: 10
                                                                }}
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    onAppointmentClick?.(appointment);
                                                                }}
                                                            >
                                                                <div className="flex items-start gap-1">
                                                                    {getTypeIcon(appointment.type)}
                                                                    <div className="flex-1 min-w-0">
                                                                        <div className="font-medium truncate">
                                                                            {appointment.component}
                                                                        </div>
                                                                        <div className="opacity-80 truncate text-[10px]">
                                                                            {appointment.vehicle_id}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                                {appointment.type === 'moved' && appointment.bumped_from && (
                                                                    <div className="text-[10px] mt-0.5 opacity-70">
                                                                        ← From {appointment.bumped_from}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </TooltipTrigger>
                                                        <TooltipContent side="right" className="max-w-xs">
                                                            <div className="space-y-1">
                                                                <div className="font-semibold flex items-center gap-2">
                                                                    {getTypeIcon(appointment.type)}
                                                                    {getTypeLabel(appointment.type)}
                                                                </div>
                                                                <div className="text-sm">
                                                                    <p><strong>Vehicle:</strong> {appointment.vehicle_id}</p>
                                                                    <p><strong>Component:</strong> {appointment.component}</p>
                                                                    <p><strong>Time:</strong> {appointment.time} ({appointment.duration}h)</p>
                                                                    {appointment.technician && (
                                                                        <p><strong>Tech:</strong> {appointment.technician}</p>
                                                                    )}
                                                                    {appointment.diagnosis && (
                                                                        <p><strong>Diagnosis:</strong> {appointment.diagnosis}</p>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </TooltipContent>
                                                    </Tooltip>
                                                )}
                                            </div>
                                        );
                                    })}
                                </React.Fragment>
                            ))}
                        </div>
                    </div>
                </TooltipProvider>
            </CardContent>
        </Card>
    );
}

// Demo data generator for testing
export function generateDemoTetrisData(): TetrisDay[] {
    const baseDate = new Date();
    const days: TetrisDay[] = [];

    for (let i = 0; i < 14; i++) {
        const date = new Date(baseDate);
        date.setDate(baseDate.getDate() + i);
        const dateStr = date.toISOString().split('T')[0];
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

        const day: TetrisDay = {
            date: dateStr,
            day: dayNames[date.getDay()],
            appointments: [],
            parts_blocked_slots: []
        };

        // Add some demo appointments
        if (i === 0) {
            // Today - show the emergency brake scenario
            day.appointments = [
                {
                    id: 'apt-1',
                    vehicle_id: 'VIN-BRAKE-001',
                    component: 'Brakes',
                    time: '14:00',
                    duration: 2,
                    type: 'emergency',
                    technician: 'Mike Chen',
                    diagnosis: 'Brake fade - ceramic pad replacement',
                    status: 'scheduled'
                },
                {
                    id: 'apt-2',
                    vehicle_id: 'VIN-ROUTINE-002',
                    component: 'Oil Change',
                    time: '09:00',
                    duration: 1,
                    type: 'routine',
                    technician: 'Sarah Lee'
                },
                {
                    id: 'apt-3',
                    vehicle_id: 'VIN-MOVED-003',
                    component: 'Tire Rotation',
                    time: '16:00',
                    duration: 1,
                    type: 'moved',
                    technician: 'Sarah Lee',
                    bumped_from: '14:00'  // Was bumped by emergency
                }
            ];
        } else if (i === 1) {
            day.appointments = [
                {
                    id: 'apt-4',
                    vehicle_id: 'VIN-BAT-004',
                    component: 'Battery Check',
                    time: '10:00',
                    duration: 2,
                    type: 'routine',
                    technician: 'Mike Chen'
                }
            ];
            // Parts blocked for inverter work
            day.parts_blocked_slots = [
                { start: '13:00', end: '17:00', reason: 'Inverter chips arriving Tuesday' }
            ];
        } else if (i === 2) {
            day.appointments = [
                {
                    id: 'apt-5',
                    vehicle_id: 'VIN-INV-005',
                    component: 'Inverter',
                    time: '10:00',
                    duration: 3,
                    type: 'routine',
                    technician: 'Alex Kim',
                    diagnosis: 'Inverter cooling system service'
                },
                {
                    id: 'apt-6',
                    vehicle_id: 'VIN-MOTOR-006',
                    component: 'Motor',
                    time: '14:00',
                    duration: 2,
                    type: 'routine',
                    technician: 'Mike Chen'
                }
            ];
        } else if (i % 3 === 0) {
            day.appointments = [
                {
                    id: `apt-${i}-1`,
                    vehicle_id: `VIN-${i}-001`,
                    component: 'Suspension',
                    time: '09:00',
                    duration: 2,
                    type: 'routine',
                    technician: 'Sarah Lee'
                }
            ];
        }

        days.push(day);
    }

    return days;
}
