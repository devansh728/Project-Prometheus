// LLM Recommendation Panel - AI-generated engineering recommendations
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Brain, Sparkles, Lightbulb, RefreshCw, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

const API_V2 = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';

interface Recommendation {
    category: string;
    recommendation: string;
    priority: 'high' | 'medium' | 'low';
}

interface LLMRecommendationPanelProps {
    component?: string;
}

export function LLMRecommendationPanel({ component = 'brakes' }: LLMRecommendationPanelProps) {
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [loading, setLoading] = useState(false);
    const [generated, setGenerated] = useState(false);

    async function generateRecommendations() {
        setLoading(true);
        try {
            const res = await fetch(`${API_V2}/v2/capa/recommendations/${component}?include_patterns=true`);
            const data = await res.json();
            if (data.recommendations) {
                setRecommendations(data.recommendations);
            }
            setGenerated(true);
        } catch (error) {
            console.error('Failed to generate recommendations:', error);
            // Demo fallback
            setRecommendations([
                { category: 'Design', recommendation: 'Consider revising brake rotor alloy composition for improved heat dissipation in 2024+ models', priority: 'high' },
                { category: 'Process', recommendation: 'Implement automated brake pad thickness verification at end-of-line testing', priority: 'high' },
                { category: 'Supplier', recommendation: 'Schedule supplier audit for BrakeTech Industries - elevated defect rate detected', priority: 'medium' },
                { category: 'Testing', recommendation: 'Add thermal cycling test for brake components in mountainous driving scenarios', priority: 'medium' },
            ]);
            setGenerated(true);
        } finally {
            setLoading(false);
        }
    }

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'high': return 'bg-red-500 text-white';
            case 'medium': return 'bg-yellow-500 text-black';
            default: return 'bg-blue-500 text-white';
        }
    };

    const getCategoryIcon = (category: string) => {
        switch (category.toLowerCase()) {
            case 'design': return '🔧';
            case 'process': return '⚙️';
            case 'supplier': return '🏭';
            case 'testing': return '🧪';
            default: return '💡';
        }
    };

    return (
        <Card className="bg-gradient-to-br from-purple-500/5 to-blue-500/5 border-purple-500/30">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Brain className="h-4 w-4 text-purple-500" />
                    AI Engineering Recommendations
                    <Sparkles className="h-3 w-3 text-yellow-500" />
                </CardTitle>
            </CardHeader>
            <CardContent>
                {!generated ? (
                    <div className="text-center py-6">
                        <Lightbulb className="h-8 w-8 mx-auto text-muted-foreground mb-3" />
                        <p className="text-sm text-muted-foreground mb-4">
                            Generate AI-powered recommendations based on RCA/CAPA patterns
                        </p>
                        <Button
                            onClick={generateRecommendations}
                            disabled={loading}
                            className="bg-gradient-to-r from-purple-500 to-blue-500"
                        >
                            <Sparkles className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                            {loading ? 'Analyzing...' : 'Generate Recommendations'}
                        </Button>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {recommendations.map((rec, index) => (
                            <div
                                key={index}
                                className="p-3 rounded-lg border bg-background/50"
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <span>{getCategoryIcon(rec.category)}</span>
                                        <Badge variant="outline">{rec.category}</Badge>
                                    </div>
                                    <Badge className={getPriorityColor(rec.priority)}>
                                        {rec.priority}
                                    </Badge>
                                </div>
                                <p className="text-sm">{rec.recommendation}</p>
                            </div>
                        ))}

                        <Button
                            variant="outline"
                            className="w-full mt-2"
                            onClick={generateRecommendations}
                            disabled={loading}
                        >
                            <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                            Regenerate
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
