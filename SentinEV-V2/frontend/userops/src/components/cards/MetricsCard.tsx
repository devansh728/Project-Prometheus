/**
 * MetricsCard - Displays computed health metrics
 * Shows anomaly score, RUL, failure probability, and overall health
 */
import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withSequence,
  withTiming,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import Svg, { Circle, G } from 'react-native-svg';
import { theme } from '../../theme';

interface MetricsData {
  anomalyScore: number;    // 0-1 (low is good)
  failureProbability: number;  // 0-100%
  remainingUsefulLife: number; // hours
  healthScore: number;     // 0-100
}

interface MetricsCardProps {
  metrics: MetricsData;
}

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

const ArcGauge: React.FC<{ value: number; color: string; size?: number }> = ({ 
  value, 
  color, 
  size = 100 
}) => {
  const animatedProgress = useSharedValue(0);
  const radius = (size - 10) / 2;
  const circumference = 2 * Math.PI * radius;
  
  useEffect(() => {
    animatedProgress.value = withSpring(value / 100, {
      damping: 15,
      stiffness: 80,
    });
  }, [value]);

  const animatedProps = useAnimatedStyle(() => {
    const strokeDashoffset = circumference * (1 - animatedProgress.value);
    return {
      strokeDashoffset,
    };
  });

  return (
    <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <G rotation="-90" origin={`${size / 2}, ${size / 2}`}>
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={8}
          fill="none"
        />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={8}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value / 100)}
          strokeLinecap="round"
        />
      </G>
    </Svg>
  );
};

const MetricItem: React.FC<{
  label: string;
  value: string;
  subtext?: string;
  status: 'good' | 'warning' | 'critical';
  icon: string;
}> = ({ label, value, subtext, status, icon }) => {
  const pulseScale = useSharedValue(1);
  
  useEffect(() => {
    pulseScale.value = withRepeat(
      withSequence(
        withTiming(1.02, { duration: 2000 }),
        withTiming(1, { duration: 2000 })
      ),
      -1,
      true
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  const statusColors = {
    good: theme.colors.success,
    warning: theme.colors.warning,
    critical: theme.colors.danger,
  };

  return (
    <Animated.View style={[styles.metricItem, animatedStyle]}>
      <View style={[styles.metricIconContainer, { backgroundColor: statusColors[status] + '20' }]}>
        <Text style={styles.metricIcon}>{icon}</Text>
      </View>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, { color: statusColors[status] }]}>{value}</Text>
      {subtext && <Text style={styles.metricSubtext}>{subtext}</Text>}
    </Animated.View>
  );
};

export const MetricsCard: React.FC<MetricsCardProps> = ({ metrics }) => {
  const getAnomalyStatus = (score: number) => {
    if (score < 0.1) return 'good';
    if (score < 0.3) return 'warning';
    return 'critical';
  };

  const getHealthStatus = (score: number) => {
    if (score >= 80) return 'good';
    if (score >= 60) return 'warning';
    return 'critical';
  };

  const getHealthColor = (score: number) => {
    if (score >= 80) return theme.colors.success;
    if (score >= 60) return theme.colors.warning;
    return theme.colors.danger;
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Health Metrics</Text>
      
      <LinearGradient
        colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
        style={styles.content}
      >
        {/* Main Health Score Arc */}
        <View style={styles.mainHealthContainer}>
          <View style={styles.arcContainer}>
            <ArcGauge 
              value={metrics.healthScore} 
              color={getHealthColor(metrics.healthScore)} 
              size={120}
            />
            <View style={styles.arcCenter}>
              <Text style={[styles.healthValue, { color: getHealthColor(metrics.healthScore) }]}>
                {metrics.healthScore.toFixed(0)}
              </Text>
              <Text style={styles.healthLabel}>Health</Text>
            </View>
          </View>
          <View style={styles.healthBadge}>
            <Text style={styles.healthBadgeText}>
              {metrics.healthScore >= 90 ? '✨ Excellent' : 
               metrics.healthScore >= 70 ? '👍 Good' : 
               metrics.healthScore >= 50 ? '⚠️ Fair' : '🔴 Poor'}
            </Text>
          </View>
        </View>

        {/* Metric Items Grid */}
        <View style={styles.metricsGrid}>
          <MetricItem
            label="Anomaly Score"
            value={(metrics.anomalyScore * 100).toFixed(1) + '%'}
            subtext={metrics.anomalyScore < 0.1 ? 'Normal' : 'Elevated'}
            status={getAnomalyStatus(metrics.anomalyScore)}
            icon="📊"
          />
          <MetricItem
            label="Failure Risk"
            value={metrics.failureProbability.toFixed(1) + '%'}
            subtext={metrics.failureProbability < 2 ? 'Minimal' : 'Monitor'}
            status={metrics.failureProbability < 2 ? 'good' : metrics.failureProbability < 10 ? 'warning' : 'critical'}
            icon="⚠️"
          />
          <MetricItem
            label="Est. RUL"
            value={metrics.remainingUsefulLife.toFixed(0)}
            subtext="hours"
            status={metrics.remainingUsefulLife > 2000 ? 'good' : metrics.remainingUsefulLife > 500 ? 'warning' : 'critical'}
            icon="⏱️"
          />
          <MetricItem
            label="Confidence"
            value="98.2%"
            subtext="ML Model"
            status="good"
            icon="🎯"
          />
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
  title: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    marginBottom: theme.spacing.md,
  },
  content: {
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  mainHealthContainer: {
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  arcContainer: {
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
  },
  arcCenter: {
    position: 'absolute',
    alignItems: 'center',
  },
  healthValue: {
    fontSize: 36,
    fontWeight: '700',
  },
  healthLabel: {
    fontSize: 12,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
  healthBadge: {
    marginTop: theme.spacing.sm,
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 20,
  },
  healthBadgeText: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.success,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricItem: {
    width: '48%',
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    alignItems: 'center',
  },
  metricIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  metricIcon: {
    fontSize: 18,
  },
  metricLabel: {
    fontSize: 11,
    color: theme.colors.textMuted,
    marginBottom: 4,
    textAlign: 'center',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: '700',
    textAlign: 'center',
  },
  metricSubtext: {
    fontSize: 10,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
});

export default MetricsCard;
