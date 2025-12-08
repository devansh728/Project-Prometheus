// Dashboard (Home) Page

'use client';

import { useState, useEffect } from 'react';
import { Plus, Car, Activity, AlertTriangle, TrendingUp, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { useVehicleStore } from '@/stores/vehicleStore';
import { listVehicles, listScenarios, initializeVehicle } from '@/lib/api';
import { toast } from 'sonner';
import Link from 'next/link';

export default function DashboardPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newVin, setNewVin] = useState('');
  const [driverProfile, setDriverProfile] = useState<'normal' | 'aggressive' | 'eco'>('normal');
  const [isLoading, setIsLoading] = useState(false);

  const { vehicles, setVehicles, scenarios, setScenarios, addVehicle } = useVehicleStore();

  // Load vehicles and scenarios on mount
  useEffect(() => {
    async function loadData() {
      try {
        const [vehiclesRes, scenariosRes] = await Promise.all([
          listVehicles().catch(() => ({ vehicles: [] })),
          listScenarios().catch(() => ({ scenarios: [] })),
        ]);
        setVehicles(vehiclesRes.vehicles || []);
        setScenarios(scenariosRes.scenarios || []);
      } catch (error) {
        console.error('Failed to load data:', error);
      }
    }
    loadData();
  }, [setVehicles, setScenarios]);

  // Initialize new vehicle
  async function handleInitVehicle() {
    if (!newVin.trim()) {
      toast.error('Please enter a VIN');
      return;
    }

    setIsLoading(true);
    try {
      const vehicle = await initializeVehicle(newVin.trim(), driverProfile);
      addVehicle(vehicle);
      toast.success(`Vehicle ${newVin} initialized!`);
      setDialogOpen(false);
      setNewVin('');
    } catch (error) {
      toast.error(`Failed to initialize: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />

      <div className="flex flex-1">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        <main className="flex-1 p-6 overflow-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold">Dashboard</h1>
              <p className="text-muted-foreground">Monitor your EV fleet in real-time</p>
            </div>

            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Vehicle
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Initialize New Vehicle</DialogTitle>
                  <DialogDescription>
                    Enter the VIN and select a driver profile. This will generate training data and train ML models.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Vehicle ID (VIN)</label>
                    <Input
                      placeholder="e.g., VIN-001"
                      value={newVin}
                      onChange={(e) => setNewVin(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Driver Profile</label>
                    <Select value={driverProfile} onValueChange={(v) => setDriverProfile(v as typeof driverProfile)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="normal">Normal</SelectItem>
                        <SelectItem value="aggressive">Aggressive</SelectItem>
                        <SelectItem value="eco">Eco</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleInitVehicle} disabled={isLoading}>
                    {isLoading ? (
                      <>
                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                        Training...
                      </>
                    ) : (
                      'Initialize'
                    )}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          {/* Stats Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Vehicles</CardTitle>
                <Car className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{vehicles.length}</div>
                <p className="text-xs text-muted-foreground">Initialized in fleet</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Scenarios</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{scenarios.length}</div>
                <p className="text-xs text-muted-foreground">Available to run</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Alerts</CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">0</div>
                <p className="text-xs text-muted-foreground">Active warnings</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Score</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-500">+0</div>
                <p className="text-xs text-muted-foreground">Fleet average</p>
              </CardContent>
            </Card>
          </div>

          {/* Vehicles Grid */}
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4">Fleet Vehicles</h2>
            {vehicles.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Car className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-center">
                    No vehicles initialized yet.<br />
                    Click "Add Vehicle" to get started.
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {vehicles.map((vehicle) => (
                  <Link key={vehicle.vehicle_id} href={`/vehicles/${vehicle.vehicle_id}`}>
                    <Card className="cursor-pointer hover:border-primary transition-colors">
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="flex items-center gap-2">
                            <Car className="h-5 w-5" />
                            {vehicle.vehicle_id}
                          </CardTitle>
                          <Badge variant={vehicle.model_trained ? 'default' : 'secondary'}>
                            {vehicle.model_trained ? 'Ready' : 'Training'}
                          </Badge>
                        </div>
                        <CardDescription>
                          Profile: {vehicle.driver_profile}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="text-sm text-muted-foreground">
                          Training samples: {vehicle.training_samples}
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                ))}
              </div>
            )}
          </div>

          {/* Scenarios */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Available Scenarios</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {scenarios.map((scenario) => (
                <Card key={scenario.id}>
                  <CardHeader>
                    <CardTitle className="text-base">{scenario.name}</CardTitle>
                    <CardDescription>{scenario.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{scenario.component}</Badge>
                      <Badge
                        variant={
                          scenario.severity === 'critical' ? 'destructive' :
                            scenario.severity === 'high' ? 'default' : 'secondary'
                        }
                      >
                        {scenario.severity}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
