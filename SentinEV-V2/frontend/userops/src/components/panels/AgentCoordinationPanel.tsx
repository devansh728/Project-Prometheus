/**
 * AgentCoordinationPanel - Live multi-agent visualization
 * Shows coordination between Master Agent and worker agents
 */
import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
  withSpring,
  FadeIn,
  FadeInDown,
} from 'react-native-reanimated';
import { theme } from '../../theme';

interface AgentStatus {
  name: string;
  displayName: string;
  status: 'active' | 'processing' | 'standby' | 'complete';
  icon: string;
  lastAction?: string;
}

interface AgentCoordinationPanelProps {
  isActive: boolean;
}

const AGENTS: AgentStatus[] = [
  { name: 'data_analysis', displayName: 'Data Analysis', status: 'active', icon: '📊', lastAction: 'Analyzing' },
  { name: 'diagnosis', displayName: 'Diagnosis', status: 'active', icon: '🔍', lastAction: 'Healthy' },
  { name: 'engagement', displayName: 'Engagement', status: 'active', icon: '💬', lastAction: 'Monitoring' },
  { name: 'scheduling', displayName: 'Scheduling', status: 'standby', icon: '📅', lastAction: 'Standby' },
  { name: 'ueba_security', displayName: 'UEBA Security', status: 'active', icon: '🛡️', lastAction: 'Secured' },
];

const AgentRow: React.FC<{ agent: AgentStatus; index: number }> = ({ agent, index }) => {
  const pulseOpacity = useSharedValue(1);
  const connectionWidth = useSharedValue(0);
  
  useEffect(() => {
    if (agent.status === 'active' || agent.status === 'processing') {
      pulseOpacity.value = withRepeat(
        withSequence(
          withTiming(0.5, { duration: 1000 }),
          withTiming(1, { duration: 1000 })
        ),
        -1,
        true
      );
    }
    connectionWidth.value = withTiming(1, { duration: 500 + index * 200 });
  }, [agent.status]);

  const dotStyle = useAnimatedStyle(() => ({
    opacity: pulseOpacity.value,
  }));

  const connectionStyle = useAnimatedStyle(() => ({
    transform: [{ scaleX: connectionWidth.value }],
  }));

  const getStatusColor = () => {
    switch (agent.status) {
      case 'active': return theme.colors.success;
      case 'processing': return theme.colors.warning;
      case 'standby': return theme.colors.textMuted;
      case 'complete': return theme.colors.info;
      default: return theme.colors.textMuted;
    }
  };

  return (
    <Animated.View 
      entering={FadeInDown.delay(index * 100).springify()}
      style={styles.agentRow}
    >
      {/* Connection Line */}
      <View style={styles.connectionContainer}>
        <View style={styles.connectionDot} />
        <Animated.View style={[styles.connectionLine, connectionStyle]} />
      </View>
      
      {/* Agent Info */}
      <View style={styles.agentInfo}>
        <Text style={styles.agentIcon}>{agent.icon}</Text>
        <View style={styles.agentDetails}>
          <Text style={styles.agentName}>{agent.displayName}</Text>
          <Text style={styles.agentAction}>→ {agent.lastAction}</Text>
        </View>
      </View>
      
      {/* Status Indicator */}
      <Animated.View style={[styles.statusDot, { backgroundColor: getStatusColor() }, dotStyle]} />
    </Animated.View>
  );
};

export const AgentCoordinationPanel: React.FC<AgentCoordinationPanelProps> = ({ isActive }) => {
  const [lastAnalysis, setLastAnalysis] = useState(2);
  const masterPulse = useSharedValue(1);
  
  useEffect(() => {
    if (isActive) {
      masterPulse.value = withRepeat(
        withSequence(
          withTiming(1.05, { duration: 1500 }),
          withTiming(1, { duration: 1500 })
        ),
        -1,
        true
      );
      
      // Simulate analysis timing
      const interval = setInterval(() => {
        setLastAnalysis(prev => (prev >= 5 ? 1 : prev + 1));
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [isActive]);

  const masterStyle = useAnimatedStyle(() => ({
    transform: [{ scale: masterPulse.value }],
  }));

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Agent Coordination</Text>
        <View style={[styles.liveIndicator, isActive && styles.liveIndicatorActive]}>
          <View style={[styles.liveDot, isActive && styles.liveDotActive]} />
          <Text style={[styles.liveText, isActive && styles.liveTextActive]}>
            {isActive ? 'LIVE' : 'OFFLINE'}
          </Text>
        </View>
      </View>
      
      <LinearGradient
        colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
        style={styles.content}
      >
        {/* Master Agent Header */}
        <Animated.View style={[styles.masterAgent, masterStyle]}>
          <LinearGradient
            colors={['#3B82F6', '#1D4ED8']}
            style={styles.masterGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.masterIcon}>🧠</Text>
            <View style={styles.masterInfo}>
              <Text style={styles.masterTitle}>Master Agent</Text>
              <Text style={styles.masterStatus}>● ACTIVE</Text>
            </View>
          </LinearGradient>
        </Animated.View>

        {/* Worker Agents */}
        <View style={styles.agentsList}>
          {AGENTS.map((agent, index) => (
            <AgentRow key={agent.name} agent={agent} index={index} />
          ))}
        </View>

        {/* ML Models Status */}
        <View style={styles.mlSection}>
          <Text style={styles.mlTitle}>ML Models</Text>
          <View style={styles.mlModels}>
            <View style={styles.mlModel}>
              <Text style={styles.mlCheck}>✓</Text>
              <Text style={styles.mlName}>LSTM-AE</Text>
            </View>
            <View style={styles.mlModel}>
              <Text style={styles.mlCheck}>✓</Text>
              <Text style={styles.mlName}>LightGBM</Text>
            </View>
            <View style={styles.mlModel}>
              <Text style={styles.mlCheck}>✓</Text>
              <Text style={styles.mlName}>RAG</Text>
            </View>
          </View>
        </View>

        {/* Footer Stats */}
        <View style={styles.footer}>
          <View style={styles.footerItem}>
            <Text style={styles.footerLabel}>Last Analysis</Text>
            <Text style={styles.footerValue}>{lastAnalysis}s ago</Text>
          </View>
          <View style={styles.footerDivider} />
          <View style={styles.footerItem}>
            <Text style={styles.footerLabel}>Confidence</Text>
            <Text style={styles.footerValue}>98.2%</Text>
          </View>
        </View>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: theme.spacing.lg,
    marginBottom: theme.spacing.lg,
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
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(100, 116, 139, 0.2)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  liveIndicatorActive: {
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: theme.colors.textMuted,
    marginRight: 4,
  },
  liveDotActive: {
    backgroundColor: '#EF4444',
  },
  liveText: {
    fontSize: 10,
    fontWeight: '700',
    color: theme.colors.textMuted,
    letterSpacing: 1,
  },
  liveTextActive: {
    color: '#EF4444',
  },
  content: {
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  masterAgent: {
    marginBottom: theme.spacing.md,
  },
  masterGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
  },
  masterIcon: {
    fontSize: 28,
    marginRight: theme.spacing.md,
  },
  masterInfo: {
    flex: 1,
  },
  masterTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#fff',
  },
  masterStatus: {
    fontSize: 11,
    color: '#10B981',
    fontWeight: '600',
  },
  agentsList: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.sm,
  },
  agentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: theme.spacing.sm,
  },
  connectionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    width: 24,
    marginRight: theme.spacing.sm,
  },
  connectionDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: theme.colors.primary,
  },
  connectionLine: {
    flex: 1,
    height: 2,
    backgroundColor: theme.colors.primary,
    marginLeft: 2,
    transformOrigin: 'left',
  },
  agentInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  agentIcon: {
    fontSize: 18,
    marginRight: 10,
  },
  agentDetails: {
    flex: 1,
  },
  agentName: {
    fontSize: 13,
    fontWeight: '600',
    color: theme.colors.textPrimary,
  },
  agentAction: {
    fontSize: 11,
    color: theme.colors.textMuted,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  mlSection: {
    marginTop: theme.spacing.md,
    padding: theme.spacing.sm,
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: theme.borderRadius.md,
  },
  mlTitle: {
    fontSize: 11,
    fontWeight: '600',
    color: theme.colors.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
  },
  mlModels: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  mlModel: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  mlCheck: {
    fontSize: 12,
    color: theme.colors.success,
    marginRight: 4,
  },
  mlName: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    fontWeight: '500',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: theme.spacing.md,
    paddingTop: theme.spacing.sm,
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
  },
  footerItem: {
    alignItems: 'center',
  },
  footerLabel: {
    fontSize: 10,
    color: theme.colors.textMuted,
  },
  footerValue: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.primary,
  },
  footerDivider: {
    width: 1,
    height: 24,
    backgroundColor: theme.colors.glassBorder,
    marginHorizontal: 24,
  },
});

export default AgentCoordinationPanel;
