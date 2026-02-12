/**
 * Agent Insights Panel - Real-time log of agent observations
 * Shows: Data Analysis, Behavior, Trend, Diagnosis, Engagement agent updates
 */
import React, { useEffect, useState, useRef } from 'react';
import { View, Text, StyleSheet, ScrollView, Animated } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { theme } from '../../theme';

interface AgentInsight {
  id: string;
  agentType: 'data_analysis' | 'behavior' | 'trend' | 'diagnosis' | 'engagement';
  message: string;
  timestamp: Date;
  severity: 'info' | 'warning' | 'critical';
}

interface AgentInsightsPanelProps {
  isActive: boolean;
  isFaultActive?: boolean;
  degradationLevel?: number;
}

// Predefined insights for brake fade scenario
const NORMAL_INSIGHTS: Omit<AgentInsight, 'id' | 'timestamp'>[] = [
  { agentType: 'data_analysis', message: 'Telemetry patterns within normal range', severity: 'info' },
  { agentType: 'behavior', message: 'Driving behavior: Normal acceleration patterns', severity: 'info' },
  { agentType: 'trend', message: 'No significant trends detected in brake metrics', severity: 'info' },
];

const BRAKE_FADE_INSIGHTS: Omit<AgentInsight, 'id' | 'timestamp'>[] = [
  { agentType: 'data_analysis', message: 'Brake temperature trending +15% above baseline', severity: 'warning' },
  { agentType: 'behavior', message: 'Aggressive braking frequency detected ↑ 23%', severity: 'warning' },
  { agentType: 'trend', message: 'Brake efficiency declining: -8% over last 48 hours', severity: 'warning' },
  { agentType: 'diagnosis', message: 'Cross-referencing with manufacturer degradation curves...', severity: 'info' },
  { agentType: 'data_analysis', message: 'Vibration amplitude increased in braking zones', severity: 'warning' },
  { agentType: 'diagnosis', message: 'Brake pad wear pattern: Front-left accelerated wear', severity: 'warning' },
  { agentType: 'trend', message: 'Projected failure window: 6-7 days at current rate', severity: 'critical' },
  { agentType: 'diagnosis', message: 'Failure probability: 65-75% within prediction window', severity: 'critical' },
  { agentType: 'engagement', message: 'Initiating proactive customer contact protocol...', severity: 'info' },
  { agentType: 'engagement', message: 'Voice agent scheduled for customer notification', severity: 'info' },
];

const AGENT_COLORS: Record<string, string> = {
  data_analysis: '#3B82F6',
  behavior: '#8B5CF6',
  trend: '#F59E0B',
  diagnosis: '#EF4444',
  engagement: '#10B981',
};

const AGENT_LABELS: Record<string, string> = {
  data_analysis: 'Data Analysis',
  behavior: 'Behavior Agent',
  trend: 'Trend Analyzer',
  diagnosis: 'Diagnosis Agent',
  engagement: 'Engagement',
};

export const AgentInsightsPanel: React.FC<AgentInsightsPanelProps> = ({
  isActive,
  isFaultActive = false,
  degradationLevel = 0,
}) => {
  const [insights, setInsights] = useState<AgentInsight[]>([]);
  const scrollViewRef = useRef<ScrollView>(null);
  const insightQueue = isFaultActive ? BRAKE_FADE_INSIGHTS : NORMAL_INSIGHTS;
  const insightIndexRef = useRef(0);

  useEffect(() => {
    if (!isActive) {
      setInsights([]);
      insightIndexRef.current = 0;
      return;
    }

    const interval = setInterval(() => {
      if (insightIndexRef.current < insightQueue.length) {
        const newInsight: AgentInsight = {
          ...insightQueue[insightIndexRef.current],
          id: `insight-${Date.now()}`,
          timestamp: new Date(),
        };
        
        setInsights(prev => [...prev.slice(-8), newInsight]); // Keep last 8
        insightIndexRef.current++;
        
        // Auto-scroll to bottom
        setTimeout(() => {
          scrollViewRef.current?.scrollToEnd({ animated: true });
        }, 100);
      } else if (isFaultActive) {
        // Loop back for fault scenario
        insightIndexRef.current = 3; // Skip initial ones
      }
    }, isFaultActive ? 2500 : 4000);

    return () => clearInterval(interval);
  }, [isActive, isFaultActive]);

  // Reset insights when fault status changes
  useEffect(() => {
    setInsights([]);
    insightIndexRef.current = 0;
  }, [isFaultActive]);

  if (!isActive) return null;

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
        style={styles.gradient}
      >
        <View style={styles.header}>
          <Text style={styles.title}>🧠 Agent Insights</Text>
          <View style={[
            styles.statusDot,
            { backgroundColor: isFaultActive ? theme.colors.warning : theme.colors.success }
          ]} />
        </View>

        <ScrollView 
          ref={scrollViewRef}
          style={styles.insightsList}
          showsVerticalScrollIndicator={false}
        >
          {insights.map((insight, idx) => (
            <Animated.View
              key={insight.id}
              style={[
                styles.insightItem,
                { opacity: 1 - (insights.length - idx - 1) * 0.1 }
              ]}
            >
              <View style={[
                styles.agentBadge,
                { backgroundColor: AGENT_COLORS[insight.agentType] + '25' }
              ]}>
                <View style={[
                  styles.agentDot,
                  { backgroundColor: AGENT_COLORS[insight.agentType] }
                ]} />
                <Text style={[
                  styles.agentName,
                  { color: AGENT_COLORS[insight.agentType] }
                ]}>
                  {AGENT_LABELS[insight.agentType]}
                </Text>
              </View>
              <Text style={[
                styles.insightMessage,
                insight.severity === 'critical' && styles.criticalMessage,
                insight.severity === 'warning' && styles.warningMessage,
              ]}>
                {insight.message}
              </Text>
              <Text style={styles.timestamp}>
                {insight.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </Text>
            </Animated.View>
          ))}
          
          {insights.length === 0 && (
            <View style={styles.emptyState}>
              <Text style={styles.emptyText}>Analyzing telemetry data...</Text>
            </View>
          )}
        </ScrollView>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: theme.spacing.lg,
    marginTop: theme.spacing.md,
    borderRadius: theme.borderRadius.lg,
    overflow: 'hidden',
    ...theme.shadows.md,
  },
  gradient: {
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    borderRadius: theme.borderRadius.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  title: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  insightsList: {
    maxHeight: 200,
  },
  insightItem: {
    marginBottom: theme.spacing.sm,
    paddingBottom: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.05)',
  },
  agentBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
    marginBottom: 4,
  },
  agentDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  agentName: {
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  insightMessage: {
    fontSize: 13,
    color: theme.colors.textSecondary,
    lineHeight: 18,
  },
  criticalMessage: {
    color: theme.colors.danger,
  },
  warningMessage: {
    color: theme.colors.warning,
  },
  timestamp: {
    fontSize: 10,
    color: theme.colors.textMuted,
    marginTop: 4,
  },
  emptyState: {
    padding: theme.spacing.lg,
    alignItems: 'center',
  },
  emptyText: {
    color: theme.colors.textMuted,
    fontSize: 13,
  },
});

export default AgentInsightsPanel;
