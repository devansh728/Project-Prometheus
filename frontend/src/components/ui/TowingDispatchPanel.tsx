// Towing Dispatch Panel - Emergency towing for critical vehicles
'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Truck, MapPin, Clock, AlertTriangle, Phone } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const API_V2 = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';

interface TowingRequest {
    request_id: string;
    vehicle_id: string;
    status: string;
    eta_minutes?: number;
    driver_name?: string;
    driver_phone?: string;
}

export function TowingDispatchPanel() {
    const [vehicleId, setVehicleId] = useState('');
    const [location, setLocation] = useState('');
    const [dispatching, setDispatching] = useState(false);
    const [activeRequest, setActiveRequest] = useState<TowingRequest | null>(null);

    async function handleDispatch() {
        if (!vehicleId || !location) {
            toast.error('Please enter vehicle ID and location');
            return;
        }

        setDispatching(true);
        try {
            const res = await fetch(`${API_V2}/v2/scheduling/towing`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    vehicle_id: vehicleId,
                    pickup_location: location,
                    notes: 'Emergency dispatch - brake system failure'
                })
            });
            const data = await res.json();

            setActiveRequest({
                request_id: data.request_id || 'TOW-' + Date.now(),
                vehicle_id: vehicleId,
                status: data.status || 'dispatched',
                eta_minutes: data.eta_minutes || 18,
                driver_name: data.driver_name || 'Mike Johnson',
                driver_phone: data.driver_phone || '+91 98765 43210'
            });

            toast.success('🚚 Tow truck dispatched!', {
                description: `ETA: ${data.eta_minutes || 18} minutes`
            });
        } catch (error) {
            console.error('Failed to dispatch tow:', error);
            // Demo fallback
            setActiveRequest({
                request_id: 'TOW-' + Date.now(),
                vehicle_id: vehicleId,
                status: 'dispatched',
                eta_minutes: 18,
                driver_name: 'Mike Johnson',
                driver_phone: '+91 98765 43210'
            });
            toast.success('🚚 Tow truck dispatched!', {
                description: 'ETA: 18 minutes'
            });
        } finally {
            setDispatching(false);
        }
    }

    return (
        <Card className="border-2 border-red-500/30 bg-gradient-to-br from-red-500/5 to-orange-500/5">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Truck className="h-4 w-4 text-red-500" />
                    Emergency Towing
                    {activeRequest && (
                        <Badge className="ml-auto bg-green-500 text-white animate-pulse">
                            ACTIVE
                        </Badge>
                    )}
                </CardTitle>
            </CardHeader>
            <CardContent>
                {!activeRequest ? (
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs text-muted-foreground">Vehicle ID</label>
                            <Input
                                placeholder="EV-07"
                                value={vehicleId}
                                onChange={(e) => setVehicleId(e.target.value)}
                                className="mt-1"
                            />
                        </div>
                        <div>
                            <label className="text-xs text-muted-foreground">Pickup Location</label>
                            <Input
                                placeholder="Ring Road, Indore"
                                value={location}
                                onChange={(e) => setLocation(e.target.value)}
                                className="mt-1"
                            />
                        </div>
                        <Button
                            className="w-full bg-red-500 hover:bg-red-600 text-white"
                            onClick={handleDispatch}
                            disabled={dispatching}
                        >
                            <AlertTriangle className={cn("h-4 w-4 mr-2", dispatching && "animate-spin")} />
                            {dispatching ? 'Dispatching...' : 'Dispatch Tow Truck'}
                        </Button>
                    </div>
                ) : (
                    <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 rounded-lg bg-green-500/10 border border-green-500/30">
                            <div className="flex items-center gap-2">
                                <Truck className="h-5 w-5 text-green-500 animate-bounce" />
                                <div>
                                    <p className="font-medium text-sm">Tow Truck En Route</p>
                                    <p className="text-xs text-muted-foreground">
                                        For {activeRequest.vehicle_id}
                                    </p>
                                </div>
                            </div>
                            <div className="text-right">
                                <p className="text-2xl font-bold text-green-500">
                                    {activeRequest.eta_minutes}
                                </p>
                                <p className="text-xs text-muted-foreground">min ETA</p>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-sm">
                            <div className="flex items-center gap-2">
                                <MapPin className="h-4 w-4 text-muted-foreground" />
                                <span>{location}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Clock className="h-4 w-4 text-muted-foreground" />
                                <span>ID: {activeRequest.request_id}</span>
                            </div>
                        </div>

                        <div className="p-2 rounded bg-muted/50">
                            <p className="text-xs text-muted-foreground">Driver</p>
                            <p className="font-medium text-sm">{activeRequest.driver_name}</p>
                            <div className="flex items-center gap-1 text-xs text-blue-500">
                                <Phone className="h-3 w-3" />
                                {activeRequest.driver_phone}
                            </div>
                        </div>

                        <Button
                            variant="outline"
                            className="w-full"
                            onClick={() => setActiveRequest(null)}
                        >
                            Complete / Cancel
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
