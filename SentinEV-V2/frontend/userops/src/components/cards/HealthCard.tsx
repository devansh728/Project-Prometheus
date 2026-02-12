/**
 * Animated Health Card Component
 * Premium glassmorphism design with real-time health visualization
 */
import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
  withSequence,
  Easing,
} from 'react-native-reanimated';
import { theme } from '../../theme';
import { Severity } from '../../types/api';

interface HealthCardProps {
  vehicleName: string;
  healthScore: number;
  severity: Severity | null;
  primaryConcern: string | null;
  onPress?: () => void;
}

const AnimatedLinearGradient = Animated.createAnimatedComponent(LinearGradient);

export const HealthCard: React.FC<HealthCardProps> = ({
  vehicleName,
  healthScore,
  severity,
  primaryConcern,
  onPress,
}) => {
  const pulseScale = useSharedValue(1);
  const glowOpacity = useSharedValue(0.3);

  React.useEffect(() => {
    // Pulse animation for critical states
    if (severity === Severity.CRITICAL || severity === Severity.EMERGENCY) {
      pulseScale.value = withRepeat(
        withSequence(
          withTiming(1.02, { duration: 800, easing: Easing.inOut(Easing.ease) }),
          withTiming(1, { duration: 800, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        true
      );
      glowOpacity.value = withRepeat(
        withSequence(
          withTiming(0.6, { duration: 800 }),
          withTiming(0.3, { duration: 800 })
        ),
        -1,
        true
      );
    }
  }, [severity]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  const getSeverityColor = () => {
    switch (severity) {
      case Severity.EMERGENCY:
        return theme.colors.severityEmergency;
      case Severity.CRITICAL:
        return theme.colors.severityCritical;
      case Severity.WARNING:
        return theme.colors.severityWarning;
      default:
        return theme.colors.success;
    }
  };

  const getGradient = (): readonly [string, string] => {
    switch (severity) {
      case Severity.EMERGENCY:
      case Severity.CRITICAL:
        return theme.colors.gradientDanger as unknown as readonly [string, string];
      case Severity.WARNING:
        return theme.colors.gradientWarning as unknown as readonly [string, string];
      default:
        return theme.colors.gradientSuccess as unknown as readonly [string, string];
    }
  };

  const getStatusText = () => {
    switch (severity) {
      case Severity.EMERGENCY:
        return 'EMERGENCY';
      case Severity.CRITICAL:
        return 'CRITICAL';
      case Severity.WARNING:
        return 'ATTENTION NEEDED';
      default:
        return 'EXCELLENT';
    }
  };

  return (
    <Pressable onPress={onPress}>
      <Animated.View style={[styles.container, animatedStyle]}>
        {/* Glow Effect */}
        <Animated.View
          style={[
            styles.glow,
            { backgroundColor: getSeverityColor() },
            glowStyle,
          ]}
        />

        <LinearGradient
          colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
          style={styles.gradient}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          {/* Header */}
          <View style={styles.header}>
            <View>
              <Text style={styles.vehicleName}>{vehicleName}</Text>
              <Text style={styles.subtitle}>Vehicle Health</Text>
            </View>
            <View style={[styles.statusBadge, { backgroundColor: getSeverityColor() + '20' }]}>
              <View style={[styles.statusDot, { backgroundColor: getSeverityColor() }]} />
              <Text style={[styles.statusText, { color: getSeverityColor() }]}>
                {getStatusText()}
              </Text>
            </View>
          </View>

          {/* Health Score Circle */}
          <View style={styles.scoreContainer}>
            <LinearGradient
              colors={getGradient()}
              style={styles.scoreCircle}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
            >
              <View style={styles.scoreInner}>
                <Text style={styles.scoreValue}>{healthScore}</Text>
                <Text style={styles.scoreLabel}>Health</Text>
              </View>
            </LinearGradient>

            {/* Primary Concern */}
            {primaryConcern && (
              <View style={styles.concernContainer}>
                <Text style={styles.concernLabel}>Primary Concern</Text>
                <Text style={styles.concernText}>
                  {primaryConcern.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </Text>
              </View>
            )}
          </View>

          {/* Quick Stats */}
          <View style={styles.statsRow}>
            <View style={styles.stat}>
              <Text style={styles.statValue}>23,450</Text>
              <Text style={styles.statLabel}>km driven</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.stat}>
              <Text style={styles.statValue}>5</Text>
              <Text style={styles.statLabel}>days to service</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.stat}>
              <Text style={styles.statValue}>98%</Text>
              <Text style={styles.statLabel}>battery</Text>
            </View>
          </View>
        </LinearGradient>
      </Animated.View>
    </Pressable>
  );
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: theme.spacing.md,
    marginVertical: theme.spacing.sm,
    borderRadius: theme.borderRadius.xl,
    overflow: 'hidden',
    ...theme.shadows.lg,
  },
  glow: {
    position: 'absolute',
    top: -50,
    left: -50,
    right: -50,
    bottom: -50,
    borderRadius: 100,
    zIndex: -1,
  },
  gradient: {
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    borderRadius: theme.borderRadius.xl,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.lg,
  },
  vehicleName: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
  },
  subtitle: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 6,
    borderRadius: theme.borderRadius.full,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  statusText: {
    ...theme.typography.caption,
    fontWeight: '600',
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  scoreCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreInner: {
    width: 84,
    height: 84,
    borderRadius: 42,
    backgroundColor: theme.colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreValue: {
    fontSize: 32,
    fontWeight: '700',
    color: theme.colors.textPrimary,
  },
  scoreLabel: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
  },
  concernContainer: {
    flex: 1,
    marginLeft: theme.spacing.lg,
  },
  concernLabel: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    marginBottom: 4,
  },
  concernText: {
    ...theme.typography.body,
    color: theme.colors.textPrimary,
    fontWeight: '500',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: theme.spacing.md,
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
  },
  stat: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
  },
  statLabel: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
  },
  statDivider: {
    width: 1,
    backgroundColor: theme.colors.glassBorder,
  },
});
