'use client';

import { useState, useEffect } from 'react';
import {
    Factory, AlertTriangle, CheckCircle, Clock, TrendingUp,
    FileText, Search, Filter, ChevronDown, ChevronRight,
    MapPin, Wrench, ExternalLink, RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface CAPAReport {
    capa_id: string;
    created_date: string;
    status: string;
    failure_mode: string;
    root_cause: {
        primary: string;
        contributing_factors: string[];
        root_cause_category: string;
    };
    affected_components: string[];
    affected_models: string[];
    detection_method: string;
    occurrence: {
        production_volume?: number;
        failures?: number;
        rate_ppm?: number;
    };
    corrective_actions: Array<{
        action: string;
        responsible: string;
        due_date?: string;
        status: string;
    }>;
    preventive_actions: Array<{
        action: string;
        responsible: string;
        status: string;
    }>;
    verification: {
        method: string;
        result: string;
        closure_date: string | null;
    };
    lessons_learned: string;
    design_feedback: string;
}

interface PatternAnalysis {
    component: string;
    matching_records: number;
    total_production_volume: number;
    total_failures: number;
    failure_rate_ppm: number;
    root_cause_distribution: Record<string, number>;
    failure_mode_distribution: Record<string, number>;
    pattern_summary: string;
    recommendation: string;
    region_analysis: Record<string, number>;
}

interface ManufacturingSummary {
    total_capas: number;
    open_capas: number;
    closed_capas: number;
    component_distribution: Record<string, number>;
    root_cause_distribution: Record<string, number>;
    high_priority_actions: Array<{
        capa_id: string;
        action: string;
        responsible: string;
        due_date: string;
        component: string;
    }>;
}

export default function CAPADashboard() {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState<string>('all');
    const [componentFilter, setComponentFilter] = useState<string>('all');

    // Data states
    const [reports, setReports] = useState<CAPAReport[]>([]);
    const [summary, setSummary] = useState<ManufacturingSummary | null>(null);
    const [patternAnalysis, setPatternAnalysis] = useState<PatternAnalysis | null>(null);
    const [expandedReport, setExpandedReport] = useState<string | null>(null);
    const [selectedComponent, setSelectedComponent] = useState<string>('brakes');

    // Fetch data
    useEffect(() => {
        fetchAllData();
    }, [statusFilter, componentFilter]);

    async function fetchAllData() {
        setLoading(true);
        try {
            const [reportsRes, summaryRes] = await Promise.all([
                fetch(`${API_BASE}/capa/reports?${statusFilter !== 'all' ? `status=${statusFilter}` : ''}${componentFilter !== 'all' ? `&component=${componentFilter}` : ''}`).then(r => r.json()),
                fetch(`${API_BASE}/capa/manufacturing-summary`).then(r => r.json())
            ]);

            setReports(reportsRes.reports || []);
            setSummary(summaryRes);
        } catch (error) {
            console.error('Failed to fetch CAPA data:', error);
            toast.error('Failed to load CAPA data');
        } finally {
            setLoading(false);
        }
    }

    async function fetchPatternAnalysis(component: string) {
        try {
            const res = await fetch(`${API_BASE}/capa/pattern-analysis/${component}`);
            const data = await res.json();
            setPatternAnalysis(data);
            setSelectedComponent(component);
        } catch (error) {
            console.error('Failed to fetch pattern analysis:', error);
        }
    }

    async function generateCAPAReport() {
        // Demo: Generate a CAPA report for brakes
        try {
            const res = await fetch(`${API_BASE}/capa/generate/VIN-DEMO-001?component=brakes`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    diagnosis_summary: 'Brake pad replacement due to thermal degradation',
                    failure_mode: 'Brake fade during mountain descent',
                    region: 'Mountainous',
                    vehicle_data: { brake_temp: 350, usage_hours: 2500 }
                })
            });

            const data = await res.json();
            if (data.success) {
                toast.success(`CAPA Report ${data.capa_id} generated!`);
                fetchAllData();
                setPatternAnalysis(data.pattern_analysis);
            }
        } catch (error) {
            toast.error('Failed to generate CAPA report');
        }
    }

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'open': return 'bg-yellow-500';
            case 'closed': return 'bg-green-500';
            case 'in_progress': return 'bg-blue-500';
            default: return 'bg-gray-500';
        }
    };

    const getCategoryColor = (category: string) => {
        const colors: Record<string, string> = {
            'Manufacturing Defect': 'bg-red-500',
            'Software/Algorithm': 'bg-purple-500',
            'Material Specification': 'bg-orange-500',
            'Supplier Change': 'bg-yellow-500',
            'Design Margin': 'bg-blue-500',
            'Production Process': 'bg-cyan-500',
            'Control Algorithm': 'bg-indigo-500'
        };
        return colors[category] || 'bg-gray-500';
    };

    const filteredReports = reports.filter(r =>
        r.capa_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.failure_mode.toLowerCase().includes(searchQuery.toLowerCase())
    );

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
                                <Factory className="h-6 w-6" />
                                RCA/CAPA Dashboard
                            </h1>
                            <p className="text-muted-foreground">Manufacturing Quality Insights & Corrective Actions</p>
                        </div>
                        <div className="flex gap-2">
                            <Button variant="outline" onClick={() => fetchPatternAnalysis('brakes')}>
                                <TrendingUp className="h-4 w-4 mr-2" />
                                Analyze Brakes
                            </Button>
                            <Button onClick={generateCAPAReport}>
                                <FileText className="h-4 w-4 mr-2" />
                                Generate CAPA
                            </Button>
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
                                    Total CAPAs
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{summary?.total_capas || 0}</div>
                                <p className="text-xs text-muted-foreground">All time</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Open CAPAs
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-yellow-500">{summary?.open_capas || 0}</div>
                                <p className="text-xs text-muted-foreground">Requires action</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Closed CAPAs
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-green-500">{summary?.closed_capas || 0}</div>
                                <p className="text-xs text-muted-foreground">Resolved</p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground">
                                    Pending Actions
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-orange-500">
                                    {summary?.high_priority_actions?.length || 0}
                                </div>
                                <p className="text-xs text-muted-foreground">High priority</p>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Pattern Analysis Alert */}
                    {patternAnalysis && (
                        <Card className="mb-6 border-orange-500/50 bg-orange-500/10">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-orange-500">
                                    <AlertTriangle className="h-5 w-5" />
                                    Pattern Analysis: {patternAnalysis.component.toUpperCase()}
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <p className="text-sm mb-4">{patternAnalysis.pattern_summary}</p>
                                        <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                                            <h4 className="font-semibold text-red-400 mb-2">🚨 Recommendation</h4>
                                            <p className="text-sm">{patternAnalysis.recommendation}</p>
                                        </div>
                                    </div>
                                    <div>
                                        <h4 className="font-semibold mb-3 flex items-center gap-2">
                                            <MapPin className="h-4 w-4" />
                                            Regional Distribution
                                        </h4>
                                        <div className="space-y-2">
                                            {Object.entries(patternAnalysis.region_analysis).map(([region, count]) => (
                                                <div key={region} className="flex items-center justify-between">
                                                    <span className="text-sm capitalize">{region}</span>
                                                    <div className="flex items-center gap-2">
                                                        <Progress value={(count / 75) * 100} className="w-24 h-2" />
                                                        <span className="text-sm font-medium w-8">{count}</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="mt-4 p-3 bg-slate-800 rounded-lg">
                                            <p className="text-xs text-muted-foreground">
                                                Failure Rate: <span className="font-bold text-red-400">{patternAnalysis.failure_rate_ppm.toLocaleString()} PPM</span>
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Main Content Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* CAPA Reports List */}
                        <Card className="lg:col-span-2">
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <CardTitle className="flex items-center gap-2">
                                        <FileText className="h-5 w-5" />
                                        CAPA Reports
                                    </CardTitle>
                                    <div className="flex gap-2">
                                        <div className="relative">
                                            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                                            <Input
                                                placeholder="Search..."
                                                className="pl-8 w-40"
                                                value={searchQuery}
                                                onChange={(e) => setSearchQuery(e.target.value)}
                                            />
                                        </div>
                                        <Select value={statusFilter} onValueChange={setStatusFilter}>
                                            <SelectTrigger className="w-28">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="all">All</SelectItem>
                                                <SelectItem value="open">Open</SelectItem>
                                                <SelectItem value="closed">Closed</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {filteredReports.map((report) => (
                                        <div
                                            key={report.capa_id}
                                            className="border rounded-lg overflow-hidden"
                                        >
                                            {/* Report Header */}
                                            <div
                                                className="p-4 cursor-pointer hover:bg-muted/50 flex items-center justify-between"
                                                onClick={() => setExpandedReport(
                                                    expandedReport === report.capa_id ? null : report.capa_id
                                                )}
                                            >
                                                <div className="flex items-center gap-3">
                                                    {expandedReport === report.capa_id ? (
                                                        <ChevronDown className="h-4 w-4" />
                                                    ) : (
                                                        <ChevronRight className="h-4 w-4" />
                                                    )}
                                                    <div>
                                                        <div className="flex items-center gap-2">
                                                            <code className="text-sm font-mono">{report.capa_id}</code>
                                                            <Badge className={cn("text-xs", getStatusColor(report.status))}>
                                                                {report.status}
                                                            </Badge>
                                                            <Badge variant="outline" className={cn("text-xs", getCategoryColor(report.root_cause.root_cause_category))}>
                                                                {report.root_cause.root_cause_category}
                                                            </Badge>
                                                        </div>
                                                        <p className="text-sm text-muted-foreground mt-1 line-clamp-1">
                                                            {report.failure_mode}
                                                        </p>
                                                    </div>
                                                </div>
                                                <div className="text-right text-xs text-muted-foreground">
                                                    <p>{report.created_date}</p>
                                                    <div className="flex gap-1 mt-1">
                                                        {report.affected_components.map(comp => (
                                                            <Badge key={comp} variant="outline" className="text-xs">
                                                                {comp}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Expanded Content */}
                                            {expandedReport === report.capa_id && (
                                                <div className="px-4 pb-4 border-t bg-muted/30">
                                                    <div className="grid grid-cols-2 gap-4 mt-4">
                                                        <div>
                                                            <h4 className="font-semibold text-sm mb-2">Root Cause</h4>
                                                            <p className="text-sm">{report.root_cause.primary}</p>
                                                            <ul className="text-xs text-muted-foreground mt-2 space-y-1">
                                                                {report.root_cause.contributing_factors.map((f, i) => (
                                                                    <li key={i}>• {f}</li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                        <div>
                                                            <h4 className="font-semibold text-sm mb-2">Corrective Actions</h4>
                                                            <ul className="text-xs space-y-2">
                                                                {report.corrective_actions.map((action, i) => (
                                                                    <li key={i} className="flex items-start gap-2">
                                                                        {action.status === 'completed' ? (
                                                                            <CheckCircle className="h-4 w-4 text-green-500 shrink-0" />
                                                                        ) : (
                                                                            <Clock className="h-4 w-4 text-yellow-500 shrink-0" />
                                                                        )}
                                                                        <span>{action.action}</span>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    </div>
                                                    <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                                                        <h4 className="font-semibold text-sm text-blue-400 mb-1">💡 Lessons Learned</h4>
                                                        <p className="text-xs">{report.lessons_learned}</p>
                                                    </div>
                                                    <div className="mt-3 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                                                        <h4 className="font-semibold text-sm text-purple-400 mb-1">🏭 Design Feedback</h4>
                                                        <p className="text-xs">{report.design_feedback}</p>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}

                                    {filteredReports.length === 0 && (
                                        <div className="text-center py-8 text-muted-foreground">
                                            No CAPA reports found
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Right Sidebar */}
                        <div className="space-y-6">
                            {/* Component Distribution */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <Wrench className="h-4 w-4" />
                                        Component Distribution
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2">
                                        {Object.entries(summary?.component_distribution || {}).map(([comp, count]) => (
                                            <div
                                                key={comp}
                                                className="flex items-center justify-between p-2 rounded hover:bg-muted cursor-pointer"
                                                onClick={() => fetchPatternAnalysis(comp)}
                                            >
                                                <span className="text-sm capitalize">{comp}</span>
                                                <Badge variant="outline">{count}</Badge>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Root Cause Categories */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm">Root Cause Categories</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2">
                                        {Object.entries(summary?.root_cause_distribution || {}).map(([cat, count]) => (
                                            <div key={cat} className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <div className={cn("w-2 h-2 rounded-full", getCategoryColor(cat))} />
                                                    <span className="text-xs">{cat}</span>
                                                </div>
                                                <span className="text-xs font-medium">{count}</span>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* High Priority Actions */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <AlertTriangle className="h-4 w-4 text-orange-500" />
                                        Pending Actions
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2">
                                        {summary?.high_priority_actions?.slice(0, 5).map((action, i) => (
                                            <div key={i} className="p-2 bg-muted rounded text-xs">
                                                <div className="flex items-center justify-between mb-1">
                                                    <code className="text-orange-400">{action.capa_id}</code>
                                                    <Badge variant="outline" className="text-xs">{action.component}</Badge>
                                                </div>
                                                <p className="line-clamp-2">{action.action}</p>
                                                <p className="text-muted-foreground mt-1">→ {action.responsible}</p>
                                            </div>
                                        ))}
                                        {(!summary?.high_priority_actions || summary.high_priority_actions.length === 0) && (
                                            <div className="text-center py-4 text-muted-foreground text-xs">
                                                No pending actions
                                            </div>
                                        )}
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
