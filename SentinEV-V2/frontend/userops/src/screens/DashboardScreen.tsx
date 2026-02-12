/**
 * Dashboard Screen - Main Home View
 * Premium design with telemetry toggle, multi-agent coordination, and live metrics
 */
import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  StatusBar,
  RefreshControl,
  Pressable,
  Switch,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  FadeInDown,
  FadeInUp,
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { theme } from '../theme';
import { useStore } from '../store';
import { HealthCard } from '../components/cards/HealthCard';
import { QuickAction } from '../components/buttons/QuickAction';
import { IncomingCallOverlay } from '../components/overlays/IncomingCall';
import { LiveTelemetryPanel } from '../components/panels/LiveTelemetryPanel';
import { MetricsCard } from '../components/cards/MetricsCard';
import { GamificationCard } from '../components/cards/GamificationCard';
import { AgentCoordinationPanel } from '../components/panels/AgentCoordinationPanel';
import { AgentInsightsPanel } from '../components/panels/AgentInsightsPanel';
import { BrakeFadeVoiceCall } from '../components/overlays/BrakeFadeVoiceCall';
import { IncomingCallScreen } from '../components/overlays/IncomingCallScreen';
import { BrakeFadeNotification } from '../components/overlays/BrakeFadeNotification';
import { ServiceStatusCard } from '../components/cards/ServiceStatusCard';
import { Severity } from '../types/api';

// Simulate physics-based telemetry generation
const generateTelemetry = (prev: any, faultActive = false, degradationLevel = 0) => {
  const jitter = (base: number, range: number) => base + (Math.random() - 0.5) * range;
  const smoothTransition = (current: number, target: number, factor = 0.1) => 
    current + (target - current) * factor;

  // Base values - adjusted for aggressive profile
  let baseBattery = 12.5;
  let baseTemp = faultActive ? 92 + degradationLevel * 8 : 85; // Rising with fault
  let baseBrake = faultActive ? 48 - degradationLevel * 6 : 48; // Dropping with fault
  let baseRpm = 2800 + (Math.random() > 0.7 ? 800 : 0); // Aggressive bursts
  let baseCoolant = faultActive ? 0.90 - degradationLevel * 0.03 : 0.95;
  let baseVibration = faultActive ? 0.12 + degradationLevel * 0.08 : 0.08;

  return {
    battery_voltage: smoothTransition(prev?.battery_voltage || baseBattery, jitter(baseBattery, 0.4), 0.15),
    engine_temp: smoothTransition(prev?.engine_temp || baseTemp, jitter(baseTemp, faultActive ? 12 : 8), 0.1),
    brake_pressure: Math.max(30, smoothTransition(prev?.brake_pressure || baseBrake, jitter(baseBrake, faultActive ? 6 : 4), 0.12)),
    motor_rpm: smoothTransition(prev?.motor_rpm || baseRpm, jitter(baseRpm, 700), 0.08),
    coolant_level: smoothTransition(prev?.coolant_level || baseCoolant, jitter(baseCoolant, 0.03), 0.05),
    vibration: smoothTransition(prev?.vibration || baseVibration, jitter(baseVibration, 0.02), 0.1),
  };
};

// Simulate metrics computation with fault support
const generateMetrics = (faultActive = false, degradationLevel = 0) => {
  if (faultActive) {
    // Degradation increases anomaly and failure probability
    return {
      anomalyScore: Math.min(0.9, 0.15 + degradationLevel * 0.15 + Math.random() * 0.05),
      failureProbability: Math.min(95, 25 + degradationLevel * 15 + Math.random() * 5),
      remainingUsefulLife: Math.max(100, 2400 - degradationLevel * 350 - Math.random() * 50),
      healthScore: Math.max(55, 85 - degradationLevel * 8 - Math.random() * 2),
    };
  }
  return {
    anomalyScore: Math.random() * 0.08 + 0.01,
    failureProbability: Math.random() * 1.5 + 0.3,
    remainingUsefulLife: 2400 + Math.random() * 200,
    healthScore: 91 + Math.random() * 4,
  };
};

export default function DashboardScreen() {
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);
  const telemetryInterval = useRef<NodeJS.Timeout | null>(null);
  
  const {
    selectedVehicle,
    vehicleHealth,
    setVehicleHealth,
    incomingCall,
    setIncomingCall,
    isTelemetryActive,
    setTelemetryActive,
    gamification,
    addPoints,
    user,
    setUser,
    setSelectedVehicle,
  } = useStore();

  const [telemetryData, setTelemetryData] = useState({
    battery_voltage: 12.6,
    engine_temp: 85,
    brake_pressure: 48,
    motor_rpm: 2800,
    coolant_level: 0.95,
    vibration: 0.08,
  });

  const [metrics, setMetrics] = useState(generateMetrics());
  
  // Fault injection state
  const [isFaultInjected, setIsFaultInjected] = useState(false);
  const [degradationLevel, setDegradationLevel] = useState(0);
  const degradationRef = useRef(0);
  
  // Voice call state for brake fade scenario
  const [showIncomingCall, setShowIncomingCall] = useState(false);
  const [showVoiceCall, setShowVoiceCall] = useState(false);
  const [showNotification, setShowNotification] = useState(false);
  const voiceCallTriggeredRef = useRef(false);

  // Toggle glow animation
  const toggleGlow = useSharedValue(0);

  // Refs for interval access
  const isFaultInjectedRef = useRef(false);
  
  // Sync refs with state
  useEffect(() => {
    isFaultInjectedRef.current = isFaultInjected;
  }, [isFaultInjected]);
  
  // Real-time Service Status Polling
  const [activeServiceJob, setActiveServiceJob] = useState<any | null>(null);

  useEffect(() => {
    const checkServiceStatus = async () => {
      // In a real app, use user.id
      try {
        const apiUrl = process.env.EXPO_PUBLIC_API_URL || 'http://10.79.239.149:8000';
        console.log('Fetching active bookings from:', `${apiUrl}/api/v1/serviceops/bookings/active`);
        const response = await fetch(`${apiUrl}/api/v1/serviceops/bookings/active`);
        if (response.ok) {
          const data = await response.json();
          // Filter for current vehicle/user if needed, for demo take first active or fallback
          const targetVehicleId = selectedVehicle?.id || 'VH005';
          const userJob = data.bookings.find((b: any) => b.vehicle_id === targetVehicleId);
          setActiveServiceJob(userJob || null);
        }
      } catch (e) {
        console.log('Polling error', e);
      }
    };

    const interval = setInterval(checkServiceStatus, 5000); // Poll every 5s
    checkServiceStatus(); // Initial check
    
    return () => clearInterval(interval);
  }, [selectedVehicle]);
  
  const toggleGlowStyle = useAnimatedStyle(() => ({
    shadowOpacity: toggleGlow.value,
  }));

  // Start/stop telemetry streaming
  const handleTelemetryToggle = useCallback((value: boolean) => {
    setTelemetryActive(value);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    if (value) {
      toggleGlow.value = withRepeat(
        withSequence(
          withTiming(0.8, { duration: 1000 }),
          withTiming(0.3, { duration: 1000 })
        ),
        -1,
        true
      );

      // Start simulated telemetry stream
      telemetryInterval.current = setInterval(() => {
        const currentDegradation = degradationRef.current;
        const faultActive = isFaultInjectedRef.current;
        
        setTelemetryData(prev => generateTelemetry(prev, faultActive, currentDegradation));
        setMetrics(generateMetrics(faultActive, currentDegradation));
        
        // Increment degradation if fault is active
        if (faultActive && currentDegradation < 4) {
          degradationRef.current = currentDegradation + 0.2; // Faster degradation for demo
          setDegradationLevel(degradationRef.current);
        }
        
        // Award points
        if (Math.random() > (faultActive ? 0.9 : 0.7)) {
          addPoints(1);
        }
      }, 1000);
    } else {
      toggleGlow.value = withTiming(0);
      if (telemetryInterval.current) {
        clearInterval(telemetryInterval.current);
        telemetryInterval.current = null;
      }
    }
  }, [setTelemetryActive, addPoints]); // Removed isFaultInjected dependency as we use ref

  // Handle fault injection
  const handleInjectFault = useCallback(() => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    setIsFaultInjected(true);
    degradationRef.current = 0;
    setDegradationLevel(0);
  }, []);

  const handleResetFault = useCallback(() => {
    setIsFaultInjected(false);
    degradationRef.current = 0;
    setDegradationLevel(0);
    setMetrics(generateMetrics());
    voiceCallTriggeredRef.current = false;  // Allow voice call again on next fault
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, []);

  // Trigger incoming call screen when failure probability exceeds 65%
  useEffect(() => {
    if (isFaultInjected && metrics.failureProbability >= 65 && !voiceCallTriggeredRef.current) {
      voiceCallTriggeredRef.current = true;
      // Delay slightly to let user see the metrics first
      setTimeout(() => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
        setShowIncomingCall(true);
      }, 1500);
    }
  }, [isFaultInjected, metrics.failureProbability]);

  // Incoming call handlers
  const handleAcceptCall = useCallback(() => {
    setShowIncomingCall(false);
    setShowVoiceCall(true);
  }, []);

  const handleRejectCall = useCallback(() => {
    setShowIncomingCall(false);
    setShowNotification(true);
  }, []);

  const handleCallBack = useCallback(() => {
    setShowNotification(false);
    setShowIncomingCall(true);
  }, []);

  const handleChat = useCallback(() => {
    setShowNotification(false);
    // TODO: Open chatbot panel
    console.log('Opening chatbot...');
  }, []);

  const handleNotificationDismiss = useCallback(() => {
    setShowNotification(false);
  }, []);

  const handleVoiceCallClose = useCallback(() => {
    setShowVoiceCall(false);
  }, []);

  const handleBookingComplete = useCallback((details: { time: string; center: string }) => {
    console.log('Booking completed:', details);
    // Could add notification or navigate to booking confirmation
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (telemetryInterval.current) {
        clearInterval(telemetryInterval.current);
      }
    };
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    setMetrics(generateMetrics());
    setRefreshing(false);
  };

  const handleLogout = async () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    await AsyncStorage.multiRemove(['sentinev_user', 'sentinev_vehicle']);
    setUser(null);
    setSelectedVehicle(null);
  };

  const quickActions = [
    { icon: '📊', label: 'Diagnostics', route: '/vehicle' },
    { icon: '💬', label: 'Chat', route: '/chat' },
    { icon: '📅', label: 'Book Service', route: '/booking' },
    { icon: '📍', label: 'Find Center', route: '/centers' },
  ];

  // Get greeting based on time
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good Morning';
    if (hour < 17) return 'Good Afternoon';
    return 'Good Evening';
  };

  // Determine if healthy (no incoming call for healthy vehicles)
  const isHealthy = metrics.healthScore > 85 && metrics.anomalyScore < 0.1;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      <SafeAreaView style={styles.safeArea} edges={['top']}>
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={theme.colors.primary}
            />
          }
        >
          {/* Header */}
          <Animated.View 
            entering={FadeInDown.delay(100).springify()}
            style={styles.header}
          >
            <View>
              <Text style={styles.greeting}>{getGreeting()}</Text>
              <Text style={styles.userName}>{user?.name?.split(' ')[0] || 'Driver'}</Text>
            </View>
            <View style={styles.headerRight}>
              {isHealthy && (
                <View style={styles.healthyBadge}>
                  <Text style={styles.healthyText}>🟢 All Systems OK</Text>
                </View>
              )}
              <Pressable 
                onPress={handleLogout}
                style={{ marginLeft: 12, padding: 8, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 12 }}
              >
                <Text style={{ fontSize: 16 }}>🚪</Text>
              </Pressable>
            </View>
          </Animated.View>

          {/* Telemetry Toggle */}
          <Animated.View 
            entering={FadeInDown.delay(150).springify()}
            style={styles.toggleContainer}
          >
            <LinearGradient
              colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
              style={styles.toggleCard}
            >
              <View style={styles.toggleContent}>
                <View style={styles.toggleInfo}>
                  <Text style={styles.toggleIcon}>📡</Text>
                  <View>
                    <Text style={styles.toggleTitle}>Live Monitoring</Text>
                    <Text style={styles.toggleSubtitle}>
                      {isTelemetryActive ? 'Real-time data streaming...' : 'Tap to start monitoring'}
                    </Text>
                  </View>
                </View>
                <Animated.View style={[styles.toggleWrapper, toggleGlowStyle]}>
                  <Switch
                    value={isTelemetryActive}
                    onValueChange={handleTelemetryToggle}
                    trackColor={{ false: '#3E3E3E', true: theme.colors.primaryDark }}
                    thumbColor={isTelemetryActive ? theme.colors.primary : '#f4f3f4'}
                    ios_backgroundColor="#3E3E3E"
                  />
                </Animated.View>
              </View>
            </LinearGradient>
          </Animated.View>

          {/* Live Service Status - Highest Priority */}
          {activeServiceJob && (
            <Animated.View entering={FadeInDown.delay(50).springify()}>
              <ServiceStatusCard
                stage={activeServiceJob.current_stage}
                centerName={activeServiceJob.service_center_name}
                estimatedCompletion={activeServiceJob.estimated_completion} // Using mock time or real ETA
                serviceType={activeServiceJob.service_type}
              />
            </Animated.View>
          )}

          {/* Fault Injection Button (Demo Mode) - Only show if NO active service */}
          {selectedVehicle?.id === 'VH005' && !activeServiceJob && (
            <Animated.View entering={FadeInDown.delay(175).springify()}>
              <Pressable 
                onPress={isFaultInjected ? handleResetFault : handleInjectFault}
                style={styles.faultButton}
              >
                <LinearGradient
                  colors={isFaultInjected 
                    ? [theme.colors.danger, '#B91C1C'] 
                    : [theme.colors.warning, '#D97706']
                  }
                  style={styles.faultButtonGradient}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                >
                  <Text style={styles.faultButtonIcon}>
                    {isFaultInjected ? '🔧' : '⚠️'}
                  </Text>
                  <View style={styles.faultButtonContent}>
                    <Text style={styles.faultButtonTitle}>
                      {isFaultInjected ? 'Reset Scenario' : 'Inject Brake Fade'}
                    </Text>
                    <Text style={styles.faultButtonSubtitle}>
                      {isFaultInjected 
                        ? `Degradation: ${Math.round(degradationLevel * 25)}%`
                        : 'Demo: Simulate brake wear'
                      }
                    </Text>
                  </View>
                  {isFaultInjected && (
                    <View style={styles.faultActiveBadge}>
                      <Text style={styles.faultActiveBadgeText}>ACTIVE</Text>
                    </View>
                  )}
                </LinearGradient>
              </Pressable>
            </Animated.View>
          )}

          {/* Health Card */}
          <Animated.View entering={FadeInDown.delay(200).springify()}>
            <HealthCard
              vehicleName={`${selectedVehicle?.make || 'Tata'} ${selectedVehicle?.model || 'Nexon EV Max'}`}
              healthScore={Math.round(metrics.healthScore)}
              severity={isHealthy ? null : Severity.WARNING}
              primaryConcern={isHealthy ? null : 'Minor anomaly detected'}
              onPress={() => router.push('/vehicle')}
            />
          </Animated.View>

          {/* Live Telemetry Panel */}
          {isTelemetryActive && (
            <Animated.View entering={FadeInDown.springify()}>
              <LiveTelemetryPanel
                data={telemetryData}
                isActive={isTelemetryActive}
              />
            </Animated.View>
          )}

          {/* Agent Insights Panel (brake fade scenario) */}
          {isTelemetryActive && isFaultInjected && (
            <Animated.View entering={FadeInDown.springify()}>
              <AgentInsightsPanel
                isActive={isTelemetryActive}
                isFaultActive={isFaultInjected}
                degradationLevel={degradationLevel}
              />
            </Animated.View>
          )}

          {/* Metrics Card */}
          <Animated.View entering={FadeInDown.delay(300).springify()}>
            <MetricsCard metrics={metrics} />
          </Animated.View>

          {/* Agent Coordination Panel */}
          <Animated.View entering={FadeInDown.delay(350).springify()}>
            <AgentCoordinationPanel isActive={isTelemetryActive} />
          </Animated.View>

          {/* Quick Actions */}
          <Animated.View 
            entering={FadeInDown.delay(400).springify()}
            style={styles.quickActionsContainer}
          >
            <Text style={styles.sectionTitle}>Quick Actions</Text>
            <View style={styles.quickActions}>
              {quickActions.map((action, index) => (
                <QuickAction
                  key={action.label}
                  icon={action.icon}
                  label={action.label}
                  onPress={() => router.push(action.route as any)}
                  gradient={
                    index === 0
                      ? (theme.colors.gradientPrimary as unknown as readonly [string, string])
                      : index === 1
                      ? ['#7C3AED', '#5B21B6'] as const
                      : index === 2
                      ? (theme.colors.gradientSuccess as unknown as readonly [string, string])
                      : ['#F59E0B', '#D97706'] as const
                  }
                />
              ))}
            </View>
          </Animated.View>

          {/* Gamification Card */}
          <Animated.View entering={FadeInUp.delay(450).springify()}>
            <GamificationCard data={gamification} />
          </Animated.View>

          {/* Recent Activity - only for healthy vehicles */}
          {isHealthy && (
            <Animated.View 
              entering={FadeInDown.delay(500).springify()}
              style={styles.activityContainer}
            >
              <Text style={styles.sectionTitle}>Recent Activity</Text>
              
              <View style={styles.activityCard}>
                <LinearGradient
                  colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
                  style={styles.activityGradient}
                >
                  <View style={styles.activityItem}>
                    <View style={[styles.activityIcon, { backgroundColor: theme.colors.successSoft }]}>
                      <Text>✅</Text>
                    </View>
                    <View style={styles.activityContent}>
                      <Text style={styles.activityTitle}>System Check Complete</Text>
                      <Text style={styles.activityTime}>Just now</Text>
                    </View>
                  </View>

                  <View style={styles.activityDivider} />

                  <View style={styles.activityItem}>
                    <View style={[styles.activityIcon, { backgroundColor: theme.colors.primaryDark + '30' }]}>
                      <Text>📊</Text>
                    </View>
                    <View style={styles.activityContent}>
                      <Text style={styles.activityTitle}>Telemetry Logged</Text>
                      <Text style={styles.activityTime}>Continuous</Text>
                    </View>
                  </View>

                  <View style={styles.activityDivider} />

                  <View style={styles.activityItem}>
                    <View style={[styles.activityIcon, { backgroundColor: 'rgba(16, 185, 129, 0.2)' }]}>
                      <Text>🛡️</Text>
                    </View>
                    <View style={styles.activityContent}>
                      <Text style={styles.activityTitle}>No Anomalies Detected</Text>
                      <Text style={styles.activityTime}>Vehicle Healthy</Text>
                    </View>
                  </View>
                </LinearGradient>
              </View>
            </Animated.View>
          )}
        </ScrollView>
      </SafeAreaView>

      {/* Incoming Call Overlay - Only shown for non-healthy vehicles */}
      {/* Incoming Call Screen - Brake Fade Scenario */}
      <IncomingCallScreen
        visible={showIncomingCall}
        callerName="SentinEV Agent"
        phoneNumber="+0510-847-2931"
        callReason="Brake Fade Alert"
        onAccept={handleAcceptCall}
        onReject={handleRejectCall}
      />

      {/* Brake Fade Notification - Detailed Report on Reject */}
      <BrakeFadeNotification
        visible={showNotification}
        vehicleInfo={{
          make: selectedVehicle?.make || 'Kia',
          model: selectedVehicle?.model || 'EV6',
          id: selectedVehicle?.id || 'VH005'
        }}
        onCallBack={handleCallBack}
        onChat={handleChat}
        onDismiss={handleNotificationDismiss}
      />

      {/* Legacy Incoming Call Overlay - Keep if needed for other flows, but handleDeclineCall was removed. 
          Assuming this new flow replaces it for the demo. Commenting out for now.
      {!isHealthy && incomingCall && (
        <IncomingCallOverlay
          visible={incomingCall}
          callerName="SentinEV Alert"
          callerSubtitle="Issue Detected"
          onAccept={handleAcceptCall}
          // onDecline={handleDeclineCall} // Function removed
        />
      )}
      */}

      {/* Brake Fade Voice Call - Full-screen interactive voice agent */}
      <BrakeFadeVoiceCall
        visible={showVoiceCall}
        onClose={handleVoiceCallClose}
        onBookingComplete={handleBookingComplete}
      />
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
    paddingBottom: 120,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
  },
  greeting: {
    ...theme.typography.bodySmall,
    color: theme.colors.textMuted,
  },
  userName: {
    ...theme.typography.h2,
    color: theme.colors.textPrimary,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  healthyBadge: {
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  healthyText: {
    fontSize: 12,
    fontWeight: '600',
    color: theme.colors.success,
  },
  toggleContainer: {
    paddingHorizontal: theme.spacing.lg,
    marginBottom: theme.spacing.md,
  },
  toggleCard: {
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  toggleContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  toggleInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  toggleIcon: {
    fontSize: 28,
    marginRight: theme.spacing.md,
  },
  toggleTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.textPrimary,
  },
  toggleSubtitle: {
    fontSize: 12,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
  toggleWrapper: {
    shadowColor: theme.colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 10,
    elevation: 5,
  },
  quickActionsContainer: {
    paddingHorizontal: theme.spacing.lg,
    marginTop: theme.spacing.md,
  },
  sectionTitle: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    marginBottom: theme.spacing.md,
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  activityContainer: {
    paddingHorizontal: theme.spacing.lg,
    marginTop: theme.spacing.lg,
  },
  activityCard: {
    borderRadius: theme.borderRadius.lg,
    overflow: 'hidden',
    ...theme.shadows.md,
  },
  activityGradient: {
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    borderRadius: theme.borderRadius.lg,
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
  },
  activityIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  activityContent: {
    marginLeft: theme.spacing.md,
    flex: 1,
  },
  activityTitle: {
    ...theme.typography.bodySmall,
    color: theme.colors.textPrimary,
    fontWeight: '500',
  },
  activityTime: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
  activityDivider: {
    height: 1,
    backgroundColor: theme.colors.glassBorder,
    marginVertical: theme.spacing.xs,
  },
  // Fault injection button styles
  faultButton: {
    marginHorizontal: theme.spacing.lg,
    marginTop: theme.spacing.sm,
    borderRadius: theme.borderRadius.lg,
    overflow: 'hidden',
    ...theme.shadows.md,
  },
  faultButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.lg,
    borderRadius: theme.borderRadius.lg,
  },
  faultButtonIcon: {
    fontSize: 28,
    marginRight: theme.spacing.md,
  },
  faultButtonContent: {
    flex: 1,
  },
  faultButtonTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  faultButtonSubtitle: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.85)',
    marginTop: 2,
  },
  faultActiveBadge: {
    backgroundColor: 'rgba(255,255,255,0.25)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  faultActiveBadgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '700',
  },
});
