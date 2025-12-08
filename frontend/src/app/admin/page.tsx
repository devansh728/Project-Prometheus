// Admin Panel Page

'use client';

import { useState, useEffect } from 'react';
import {
    Settings, Database, RefreshCw, Car, Shield,
    CheckCircle, AlertTriangle, Trash2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import {
    rebuildKnowledgeBase,
    listSavedModels,
    getUebaReport,
    deleteVehicleModel
} from '@/lib/api';
import { toast } from 'sonner';

interface SavedModel {
    vehicle_id: string;
    driver_profile: string;
    training_samples: number;
    is_trained: boolean;
}

interface UebaReport {
    total_actions: number;
    blocked_actions: number;
    suspicious_patterns: number;
    last_report_time: string;
}

export default function AdminPage() {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isRebuildingRAG, setIsRebuildingRAG] = useState(false);
    const [models, setModels] = useState<SavedModel[]>([]);
    const [uebaReport, setUebaReport] = useState<UebaReport | null>(null);
    const [isLoadingModels, setIsLoadingModels] = useState(true);

    // Load data on mount
    useEffect(() => {
        async function loadData() {
            try {
                const [modelsRes, uebaRes] = await Promise.all([
                    listSavedModels().catch(() => ({ models: [], count: 0 })),
                    getUebaReport().catch(() => null),
                ]);
                setModels((modelsRes as { models: SavedModel[] }).models || []);
                setUebaReport(uebaRes as UebaReport);
            } catch (error) {
                console.error('Failed to load admin data:', error);
            } finally {
                setIsLoadingModels(false);
            }
        }
        loadData();
    }, []);

    // Rebuild knowledge base
    async function handleRebuildRAG() {
        setIsRebuildingRAG(true);
        try {
            await rebuildKnowledgeBase();
            toast.success('Knowledge base rebuilt successfully');
        } catch (error) {
            toast.error(`Failed to rebuild: ${error}`);
        } finally {
            setIsRebuildingRAG(false);
        }
    }

    // Delete model
    async function handleDeleteModel(vehicleId: string) {
        try {
            await deleteVehicleModel(vehicleId);
            setModels(models.filter(m => m.vehicle_id !== vehicleId));
            toast.success(`Model for ${vehicleId} deleted`);
        } catch (error) {
            toast.error(`Failed to delete: ${error}`);
        }
    }

    return (
        <div className="min-h-screen flex flex-col">
            <Navbar onMenuClick={() => setSidebarOpen(true)} />

            <div className="flex flex-1">
                <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                <main className="flex-1 p-6 overflow-auto">
                    {/* Header */}
                    <div className="mb-8">
                        <h1 className="text-3xl font-bold flex items-center gap-2">
                            <Settings className="h-8 w-8" />
                            Admin Panel
                        </h1>
                        <p className="text-muted-foreground">Manage RAG, models, and security</p>
                    </div>

                    <div className="grid gap-6 lg:grid-cols-2">
                        {/* RAG Knowledge Base */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Database className="h-5 w-5" />
                                    RAG Knowledge Base
                                </CardTitle>
                                <CardDescription>
                                    Rebuild the vector store from fault patterns and vehicle manual
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="text-muted-foreground">Data Sources</span>
                                        <Badge variant="secondary">3 files</Badge>
                                    </div>
                                    <ul className="text-sm text-muted-foreground list-disc list-inside">
                                        <li>industry_faults.json (50 fault patterns)</li>
                                        <li>vehicle_manual.json (component specs)</li>
                                        <li>capa_records.json (RCA records)</li>
                                    </ul>

                                    <Separator />

                                    <Button
                                        onClick={handleRebuildRAG}
                                        disabled={isRebuildingRAG}
                                        className="w-full"
                                    >
                                        {isRebuildingRAG ? (
                                            <>
                                                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                Rebuilding...
                                            </>
                                        ) : (
                                            <>
                                                <RefreshCw className="mr-2 h-4 w-4" />
                                                Rebuild Knowledge Base
                                            </>
                                        )}
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>

                        {/* UEBA Security */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Shield className="h-5 w-5" />
                                    UEBA Security Monitor
                                </CardTitle>
                                <CardDescription>
                                    User and Entity Behavior Analytics
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                {uebaReport ? (
                                    <div className="space-y-4">
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <p className="text-sm text-muted-foreground">Total Actions</p>
                                                <p className="text-2xl font-bold">{uebaReport.total_actions}</p>
                                            </div>
                                            <div>
                                                <p className="text-sm text-muted-foreground">Blocked</p>
                                                <p className="text-2xl font-bold text-red-500">
                                                    {uebaReport.blocked_actions}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2">
                                            {uebaReport.suspicious_patterns > 0 ? (
                                                <>
                                                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                                                    <span className="text-sm">
                                                        {uebaReport.suspicious_patterns} suspicious patterns
                                                    </span>
                                                </>
                                            ) : (
                                                <>
                                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                                    <span className="text-sm">No suspicious activity</span>
                                                </>
                                            )}
                                        </div>

                                        <p className="text-xs text-muted-foreground">
                                            Last updated: {new Date(uebaReport.last_report_time).toLocaleString()}
                                        </p>
                                    </div>
                                ) : (
                                    <p className="text-muted-foreground">Loading security data...</p>
                                )}
                            </CardContent>
                        </Card>

                        {/* Saved Models */}
                        <Card className="lg:col-span-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Car className="h-5 w-5" />
                                    Saved ML Models
                                </CardTitle>
                                <CardDescription>
                                    Trained vehicle models stored on disk
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                {isLoadingModels ? (
                                    <p className="text-muted-foreground">Loading models...</p>
                                ) : models.length === 0 ? (
                                    <p className="text-muted-foreground">No saved models found</p>
                                ) : (
                                    <div className="space-y-4">
                                        {models.map((model) => (
                                            <div
                                                key={model.vehicle_id}
                                                className="flex items-center justify-between p-4 border rounded-lg"
                                            >
                                                <div className="flex items-center gap-4">
                                                    <Car className="h-8 w-8 text-muted-foreground" />
                                                    <div>
                                                        <p className="font-medium">{model.vehicle_id}</p>
                                                        <p className="text-sm text-muted-foreground">
                                                            Profile: {model.driver_profile} •
                                                            Samples: {model.training_samples}
                                                        </p>
                                                    </div>
                                                </div>

                                                <div className="flex items-center gap-2">
                                                    <Badge variant={model.is_trained ? 'default' : 'secondary'}>
                                                        {model.is_trained ? 'Trained' : 'Pending'}
                                                    </Badge>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => handleDeleteModel(model.vehicle_id)}
                                                    >
                                                        <Trash2 className="h-4 w-4 text-destructive" />
                                                    </Button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </main>
            </div>
        </div>
    );
}
