// Sidebar Component

'use client';

import Link from 'next/link';
import { usePathname, useParams } from 'next/navigation';
import {
    LayoutDashboard,
    Car,
    Gauge,
    MessageSquare,
    Settings,
    X,
    Calendar,
    Shield,
    Factory,
    Wrench
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { useTelemetryStore } from '@/stores/telemetryStore';
import { useVehicleStore } from '@/stores/vehicleStore';
import { cn } from '@/lib/utils';

interface SidebarProps {
    open?: boolean;
    onClose?: () => void;
}

export function Sidebar({ open = true, onClose }: SidebarProps) {
    const pathname = usePathname();
    const params = useParams();
    const currentVin = params?.vin as string | undefined;

    const vehicles = useVehicleStore((state) => state.vehicles);
    const anomaly = useTelemetryStore((state) => state.anomaly);
    const scoring = useTelemetryStore((state) => state.scoring);
    const scenario = useTelemetryStore((state) => state.scenario);

    const mainNav = [
        { href: '/', label: 'Dashboard', icon: LayoutDashboard },
        { href: '/admin', label: 'Admin', icon: Settings },
    ];

    const adminNav = [
        { href: '/admin/scheduler', label: 'Scheduler', icon: Calendar },
        { href: '/admin/capa', label: 'RCA/CAPA', icon: Factory },
        { href: '/admin/security', label: 'UEBA Security', icon: Shield },
        { href: '/service-center', label: 'Service Center', icon: Wrench },
    ];

    const vehicleNav = currentVin ? [
        { href: `/vehicles/${currentVin}`, label: 'Monitor', icon: Gauge },
        { href: `/chat/${currentVin}`, label: 'AI Chat', icon: MessageSquare },
    ] : [];

    return (
        <aside
            className={cn(
                'fixed inset-y-0 left-0 z-40 w-64 bg-background border-r transform transition-transform duration-200 ease-in-out md:relative md:translate-x-0',
                open ? 'translate-x-0' : '-translate-x-full'
            )}
        >
            {/* Mobile close button */}
            <div className="flex items-center justify-between p-4 md:hidden">
                <span className="font-semibold">Menu</span>
                <Button variant="ghost" size="icon" onClick={onClose}>
                    <X className="h-4 w-4" />
                </Button>
            </div>

            <div className="h-full overflow-y-auto py-4">
                {/* Main Navigation */}
                <div className="px-3 py-2">
                    <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">
                        Navigation
                    </h2>
                    <div className="space-y-1">
                        {mainNav.map((item) => {
                            const Icon = item.icon;
                            const isActive = pathname === item.href;

                            return (
                                <Link key={item.href} href={item.href}>
                                    <Button
                                        variant={isActive ? 'secondary' : 'ghost'}
                                        className="w-full justify-start"
                                        onClick={onClose}
                                    >
                                        <Icon className="mr-2 h-4 w-4" />
                                        {item.label}
                                    </Button>
                                </Link>
                            );
                        })}
                    </div>
                </div>

                <Separator className="my-4" />

                {/* Admin Dashboards */}
                <div className="px-3 py-2">
                    <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">
                        Admin Views
                    </h2>
                    <div className="space-y-1">
                        {adminNav.map((item) => {
                            const Icon = item.icon;
                            const isActive = pathname === item.href;

                            return (
                                <Link key={item.href} href={item.href}>
                                    <Button
                                        variant={isActive ? 'secondary' : 'ghost'}
                                        className="w-full justify-start"
                                        onClick={onClose}
                                    >
                                        <Icon className="mr-2 h-4 w-4" />
                                        {item.label}
                                    </Button>
                                </Link>
                            );
                        })}
                    </div>
                </div>

                <Separator className="my-4" />

                {/* Current Vehicle Section */}
                {currentVin && (
                    <div className="px-3 py-2">
                        <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">
                            Vehicle: {currentVin}
                        </h2>
                        <div className="space-y-1">
                            {vehicleNav.map((item) => {
                                const Icon = item.icon;
                                const isActive = pathname === item.href;

                                return (
                                    <Link key={item.href} href={item.href}>
                                        <Button
                                            variant={isActive ? 'secondary' : 'ghost'}
                                            className="w-full justify-start"
                                            onClick={onClose}
                                        >
                                            <Icon className="mr-2 h-4 w-4" />
                                            {item.label}
                                        </Button>
                                    </Link>
                                );
                            })}
                        </div>

                        {/* Live Status */}
                        <div className="mt-4 px-4 space-y-2">
                            {/* Anomaly Status */}
                            {anomaly && (
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Anomaly</span>
                                    <Badge
                                        variant={anomaly.is_anomaly ? 'destructive' : 'secondary'}
                                    >
                                        {anomaly.is_anomaly ? anomaly.severity : 'Normal'}
                                    </Badge>
                                </div>
                            )}

                            {/* Score */}
                            {scoring && (
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Score</span>
                                    <span className={cn(
                                        'font-mono font-bold',
                                        scoring.total >= 0 ? 'text-green-500' : 'text-red-500'
                                    )}>
                                        {scoring.total >= 0 ? '+' : ''}{scoring.total}
                                    </span>
                                </div>
                            )}

                            {/* Scenario Phase */}
                            {scenario && (
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Phase</span>
                                    <Badge variant="outline" className="capitalize">
                                        {scenario.phase}
                                    </Badge>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                <Separator className="my-4" />

                {/* Recent Vehicles */}
                <div className="px-3 py-2">
                    <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">
                        Fleet
                    </h2>
                    <div className="space-y-1">
                        {vehicles.length === 0 ? (
                            <p className="px-4 text-sm text-muted-foreground">
                                No vehicles initialized
                            </p>
                        ) : (
                            vehicles.map((vehicle) => (
                                <Link
                                    key={vehicle.vehicle_id}
                                    href={`/vehicles/${vehicle.vehicle_id}`}
                                >
                                    <Button
                                        variant={currentVin === vehicle.vehicle_id ? 'secondary' : 'ghost'}
                                        className="w-full justify-start"
                                        onClick={onClose}
                                    >
                                        <Car className="mr-2 h-4 w-4" />
                                        {vehicle.vehicle_id}
                                        <Badge variant="outline" className="ml-auto text-xs">
                                            {vehicle.driver_profile}
                                        </Badge>
                                    </Button>
                                </Link>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </aside>
    );
}
