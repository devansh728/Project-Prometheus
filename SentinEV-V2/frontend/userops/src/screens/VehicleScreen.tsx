/**
 * Vehicle Details Screen
 * Real-time telemetry visualization with animated gauges
 */
import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
  withSpring,
  Easing,
} from 'react-native-reanimated';
import { useRouter } from 'expo-router';

import { theme } from '../theme';
import { useStore } from '../store';
import { api, wsService } from '../services/api';
import { Severity } from '../types/api';

interface GaugeProps {
  label: string;
  value: number;
  maxValue: number;
  unit: string;
  status: 'normal' | 'warning' | 'critical';
  icon: string;
}

const Gauge: React.FC<GaugeProps> = ({ label, value, maxValue, unit, status, icon }) => {
  const progress = useSharedValue(0);

  useEffect(() => {
    progress.value = withTiming(value / maxValue, { 
      duration: 1000, 
      easing: Easing.out(Easing.cubic) 
    });
  }, [value]);

  const animatedStyle = useAnimatedStyle(() => ({
    width: `${progress.value * 100}%`,
  }));

  const getStatusColor = () => {
    switch (status) {
      case 'critical': return theme.colors.danger;
      case 'warning': return theme.colors.warning;
      default: return theme.colors.success;
    }
  };

  return (
    <View style={styles.gaugeContainer}>
      <View style={styles.gaugeHeader}>
        <Text style={styles.gaugeIcon}>{icon}</Text>
        <Text style={styles.gaugeLabel}>{label}</Text>
        <Text style={[styles.gaugeValue, { color: getStatusColor() }]}>
          {value.toFixed(1)} {unit}
        </Text>
      </View>
      <View style={styles.gaugeBar}>
        <Animated.View 
          style={[
            styles.gaugeFill, 
            { backgroundColor: getStatusColor() },
            animatedStyle
          ]} 
        />
      </View>
    </View>
  );
};

export default function VehicleScreen() {
  const router = useRouter();
  const { selectedVehicle, telemetry, setTelemetry, vehicleHealth } = useStore();
  const [isStreaming, setIsStreaming] = useState(false);

  // Mock telemetry data for demo
  const mockTelemetry = {
    engine_temp: vehicleHealth?.severity === Severity.WARNING ? 95 : 75,
    battery_voltage: 12.8,
    coolant_level: vehicleHealth?.severity === Severity.WARNING ? 65 : 92,
    brake_pressure: vehicleHealth?.severity === Severity.CRITICAL ? 45 : 88,
    vibration_amplitude: vehicleHealth?.severity === Severity.WARNING ? 0.8 : 0.2,
  };

  const gauges: GaugeProps[] = [
    {
      label: 'Engine Temperature',
      value: mockTelemetry.engine_temp,
      maxValue: 120,
      unit: '°C',
      status: mockTelemetry.engine_temp > 95 ? 'warning' : 'normal',
      icon: '🌡️',
    },
    {
      label: 'Battery Voltage',
      value: mockTelemetry.battery_voltage,
      maxValue: 15,
      unit: 'V',
      status: mockTelemetry.battery_voltage < 11.5 ? 'critical' : 'normal',
      icon: '🔋',
    },
    {
      label: 'Coolant Level',
      value: mockTelemetry.coolant_level,
      maxValue: 100,
      unit: '%',
      status: mockTelemetry.coolant_level < 70 ? 'warning' : 'normal',
      icon: '💧',
    },
    {
      label: 'Brake Pressure',
      value: mockTelemetry.brake_pressure,
      maxValue: 100,
      unit: '%',
      status: mockTelemetry.brake_pressure < 60 ? 'critical' : mockTelemetry.brake_pressure < 80 ? 'warning' : 'normal',
      icon: '🛞',
    },
    {
      label: 'Vibration Level',
      value: mockTelemetry.vibration_amplitude,
      maxValue: 2,
      unit: 'g',
      status: mockTelemetry.vibration_amplitude > 0.5 ? 'warning' : 'normal',
      icon: '📳',
    },
  ];

  const getSeverityInfo = () => {
    switch (vehicleHealth?.severity) {
      case Severity.EMERGENCY:
        return { color: theme.colors.severityEmergency, text: 'EMERGENCY', bg: theme.colors.dangerSoft };
      case Severity.CRITICAL:
        return { color: theme.colors.severityCritical, text: 'CRITICAL', bg: theme.colors.dangerSoft };
      case Severity.WARNING:
        return { color: theme.colors.severityWarning, text: 'WARNING', bg: theme.colors.warningSoft };
      default:
        return { color: theme.colors.success, text: 'NORMAL', bg: theme.colors.successSoft };
    }
  };

  const severityInfo = getSeverityInfo();

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea} edges={['top']}>
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Vehicle Header */}
          <Animated.View entering={FadeInDown.delay(100).springify()}>
            <LinearGradient
              colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
              style={styles.vehicleCard}
            >
              <View style={styles.vehicleHeader}>
                <View>
                  <Text style={styles.vehicleMake}>{selectedVehicle?.make}</Text>
                  <Text style={styles.vehicleModel}>{selectedVehicle?.model}</Text>
                </View>
                <View style={[styles.statusBadge, { backgroundColor: severityInfo.bg }]}>
                  <View style={[styles.statusDot, { backgroundColor: severityInfo.color }]} />
                  <Text style={[styles.statusText, { color: severityInfo.color }]}>
                    {severityInfo.text}
                  </Text>
                </View>
              </View>

              <View style={styles.vehicleStats}>
                <View style={styles.vehicleStat}>
                  <Text style={styles.statValue}>{selectedVehicle?.mileage?.toLocaleString()}</Text>
                  <Text style={styles.statLabel}>km driven</Text>
                </View>
                <View style={styles.vehicleStat}>
                  <Text style={styles.statValue}>{selectedVehicle?.year}</Text>
                  <Text style={styles.statLabel}>Year</Text>
                </View>
                <View style={styles.vehicleStat}>
                  <Text style={styles.statValue}>A+</Text>
                  <Text style={styles.statLabel}>Efficiency</Text>
                </View>
              </View>
            </LinearGradient>
          </Animated.View>

          {/* Recommended Action */}
          {vehicleHealth?.recommended_action && (
            <Animated.View entering={FadeInDown.delay(200).springify()}>
              <View style={[styles.recommendationCard, { borderColor: severityInfo.color }]}>
                <Text style={styles.recommendationTitle}>⚠️ Recommended Action</Text>
                <Text style={styles.recommendationText}>
                  {vehicleHealth.recommended_action}
                </Text>
                <Pressable 
                  style={styles.actionButton}
                  onPress={() => router.push('/booking')}
                >
                  <LinearGradient
                    colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
                    style={styles.actionButtonGradient}
                  >
                    <Text style={styles.actionButtonText}>Schedule Service</Text>
                  </LinearGradient>
                </Pressable>
              </View>
            </Animated.View>
          )}

          {/* Telemetry Section */}
          <Animated.View entering={FadeInDown.delay(300).springify()}>
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>Real-time Telemetry</Text>
                <View style={styles.liveIndicator}>
                  <View style={styles.liveDot} />
                  <Text style={styles.liveText}>LIVE</Text>
                </View>
              </View>

              {gauges.map((gauge, index) => (
                <Animated.View 
                  key={gauge.label}
                  entering={FadeInDown.delay(400 + index * 100).springify()}
                >
                  <Gauge {...gauge} />
                </Animated.View>
              ))}
            </View>
          </Animated.View>

          {/* Actions */}
          <Animated.View entering={FadeInDown.delay(800).springify()}>
            <View style={styles.actionsContainer}>
              <Pressable 
                style={styles.secondaryButton}
                onPress={() => router.push('/chat')}
              >
                <Text style={styles.secondaryButtonIcon}>💬</Text>
                <Text style={styles.secondaryButtonText}>Ask AI Assistant</Text>
              </Pressable>

              <Pressable 
                style={styles.secondaryButton}
                onPress={() => router.push('/booking')}
              >
                <Text style={styles.secondaryButtonIcon}>📅</Text>
                <Text style={styles.secondaryButtonText}>Book Service</Text>
              </Pressable>
            </View>
          </Animated.View>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  safeArea: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: theme.spacing.md,
    paddingBottom: 100,
  },
  vehicleCard: {
    borderRadius: theme.borderRadius.xl,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    ...theme.shadows.md,
  },
  vehicleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.lg,
  },
  vehicleMake: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  vehicleModel: {
    ...theme.typography.h2,
    color: theme.colors.textPrimary,
    marginTop: 4,
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
  vehicleStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: theme.spacing.md,
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
  },
  vehicleStat: {
    alignItems: 'center',
  },
  statValue: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
  },
  statLabel: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
  recommendationCard: {
    marginTop: theme.spacing.md,
    padding: theme.spacing.lg,
    backgroundColor: theme.colors.surfaceElevated,
    borderRadius: theme.borderRadius.lg,
    borderWidth: 1,
  },
  recommendationTitle: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    marginBottom: theme.spacing.sm,
  },
  recommendationText: {
    ...theme.typography.body,
    color: theme.colors.textSecondary,
    lineHeight: 22,
  },
  actionButton: {
    marginTop: theme.spacing.md,
  },
  actionButtonGradient: {
    paddingVertical: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
    alignItems: 'center',
  },
  actionButtonText: {
    ...theme.typography.body,
    color: '#0A0E1A',
    fontWeight: '600',
  },
  section: {
    marginTop: theme.spacing.xl,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  sectionTitle: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.success,
    marginRight: 6,
  },
  liveText: {
    ...theme.typography.caption,
    color: theme.colors.success,
    fontWeight: '600',
  },
  gaugeContainer: {
    backgroundColor: theme.colors.surfaceElevated,
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  gaugeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  gaugeIcon: {
    fontSize: 18,
    marginRight: theme.spacing.sm,
  },
  gaugeLabel: {
    ...theme.typography.bodySmall,
    color: theme.colors.textSecondary,
    flex: 1,
  },
  gaugeValue: {
    ...theme.typography.body,
    fontWeight: '600',
  },
  gaugeBar: {
    height: 6,
    backgroundColor: theme.colors.surface,
    borderRadius: 3,
    overflow: 'hidden',
  },
  gaugeFill: {
    height: '100%',
    borderRadius: 3,
  },
  actionsContainer: {
    flexDirection: 'row',
    marginTop: theme.spacing.xl,
    gap: theme.spacing.md,
  },
  secondaryButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.surfaceElevated,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  secondaryButtonIcon: {
    fontSize: 18,
    marginRight: theme.spacing.sm,
  },
  secondaryButtonText: {
    ...theme.typography.bodySmall,
    color: theme.colors.textPrimary,
    fontWeight: '500',
  },
});
