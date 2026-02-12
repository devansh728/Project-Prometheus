/**
 * LiveTelemetryPanel - Real-time animated telemetry gauges
 * Displays battery, temperature, brake, and motor metrics
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
  Easing,
  interpolate,
  interpolateColor,
} from 'react-native-reanimated';
import { theme } from '../../theme';

interface TelemetryReading {
  battery_voltage: number;
  engine_temp: number;
  brake_pressure: number;
  motor_rpm: number;
  coolant_level: number;
}

interface LiveTelemetryPanelProps {
  data: TelemetryReading;
  isActive: boolean;
}

interface GaugeProps {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  color: string;
  icon: string;
}

const AnimatedGauge: React.FC<GaugeProps> = ({ label, value, unit, min, max, color, icon }) => {
  const animatedValue = useSharedValue(0);
  const pulseScale = useSharedValue(1);
  
  const normalizedValue = Math.min(Math.max((value - min) / (max - min), 0), 1);
  
  useEffect(() => {
    animatedValue.value = withSpring(normalizedValue, {
      damping: 15,
      stiffness: 100,
    });
    
    // Pulse animation for active state
    pulseScale.value = withRepeat(
      withSequence(
        withTiming(1.05, { duration: 1000, easing: Easing.inOut(Easing.ease) }),
        withTiming(1, { duration: 1000, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );
  }, [normalizedValue]);

  const progressStyle = useAnimatedStyle(() => ({
    width: `${animatedValue.value * 100}%`,
  }));

  const containerStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  return (
    <Animated.View style={[styles.gaugeContainer, containerStyle]}>
      <View style={styles.gaugeHeader}>
        <Text style={styles.gaugeIcon}>{icon}</Text>
        <Text style={styles.gaugeLabel}>{label}</Text>
      </View>
      
      <View style={styles.gaugeValueRow}>
        <Text style={[styles.gaugeValue, { color }]}>{value.toFixed(1)}</Text>
        <Text style={styles.gaugeUnit}>{unit}</Text>
      </View>
      
      <View style={styles.progressBackground}>
        <Animated.View style={[styles.progressFill, progressStyle, { backgroundColor: color }]} />
      </View>
      
      <View style={styles.gaugeRange}>
        <Text style={styles.rangeText}>{min}</Text>
        <Text style={styles.rangeText}>{max}</Text>
      </View>
    </Animated.View>
  );
};

export const LiveTelemetryPanel: React.FC<LiveTelemetryPanelProps> = ({ data, isActive }) => {
  const pulseOpacity = useSharedValue(1);
  
  useEffect(() => {
    if (isActive) {
      pulseOpacity.value = withRepeat(
        withSequence(
          withTiming(0.6, { duration: 500 }),
          withTiming(1, { duration: 500 })
        ),
        -1,
        true
      );
    } else {
      pulseOpacity.value = withTiming(1);
    }
  }, [isActive]);

  const statusStyle = useAnimatedStyle(() => ({
    opacity: pulseOpacity.value,
  }));

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Live Telemetry</Text>
        <Animated.View style={[styles.statusIndicator, statusStyle]}>
          <View style={[styles.statusDot, isActive && styles.statusDotActive]} />
          <Text style={[styles.statusText, isActive && styles.statusTextActive]}>
            {isActive ? 'STREAMING' : 'INACTIVE'}
          </Text>
        </Animated.View>
      </View>
      
      <LinearGradient
        colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
        style={styles.content}
      >
        <View style={styles.gaugesGrid}>
          <AnimatedGauge
            label="Battery"
            value={data.battery_voltage}
            unit="V"
            min={11.0}
            max={13.5}
            color={theme.colors.success}
            icon="🔋"
          />
          <AnimatedGauge
            label="Motor Temp"
            value={data.engine_temp}
            unit="°C"
            min={60}
            max={120}
            color={data.engine_temp > 100 ? theme.colors.warning : theme.colors.info}
            icon="🌡️"
          />
          <AnimatedGauge
            label="Brake Pressure"
            value={data.brake_pressure}
            unit="PSI"
            min={30}
            max={60}
            color={theme.colors.primary}
            icon="🛞"
          />
          <AnimatedGauge
            label="Motor RPM"
            value={data.motor_rpm}
            unit="RPM"
            min={0}
            max={6000}
            color={theme.colors.accent}
            icon="⚡"
          />
        </View>

        {/* Coolant Level Bar */}
        <View style={styles.coolantContainer}>
          <View style={styles.coolantHeader}>
            <Text style={styles.coolantLabel}>💧 Coolant Level</Text>
            <Text style={styles.coolantValue}>{(data.coolant_level * 100).toFixed(0)}%</Text>
          </View>
          <View style={styles.coolantBar}>
            <Animated.View 
              style={[
                styles.coolantFill, 
                { width: `${data.coolant_level * 100}%` }
              ]} 
            />
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
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.surface,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.textMuted,
    marginRight: 6,
  },
  statusDotActive: {
    backgroundColor: theme.colors.success,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '700',
    color: theme.colors.textMuted,
    letterSpacing: 1,
  },
  statusTextActive: {
    color: theme.colors.success,
  },
  content: {
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  gaugesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  gaugeContainer: {
    width: '48%',
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  },
  gaugeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  gaugeIcon: {
    fontSize: 16,
    marginRight: 6,
  },
  gaugeLabel: {
    fontSize: 12,
    color: theme.colors.textMuted,
    fontWeight: '500',
  },
  gaugeValueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 8,
  },
  gaugeValue: {
    fontSize: 24,
    fontWeight: '700',
  },
  gaugeUnit: {
    fontSize: 12,
    color: theme.colors.textMuted,
    marginLeft: 4,
  },
  progressBackground: {
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  gaugeRange: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  rangeText: {
    fontSize: 9,
    color: theme.colors.textMuted,
  },
  coolantContainer: {
    marginTop: theme.spacing.sm,
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
  },
  coolantHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  coolantLabel: {
    fontSize: 14,
    color: theme.colors.textSecondary,
    fontWeight: '500',
  },
  coolantValue: {
    fontSize: 14,
    color: theme.colors.success,
    fontWeight: '700',
  },
  coolantBar: {
    height: 8,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  coolantFill: {
    height: '100%',
    backgroundColor: theme.colors.info,
    borderRadius: 4,
  },
});

export default LiveTelemetryPanel;
