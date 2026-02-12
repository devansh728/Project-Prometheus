/**
 * Booking Screen - Service Appointment Flow
 * Calendar-style slot selection with animated confirmations
 */
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeInDown,
  FadeInUp,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { useRouter } from 'expo-router';

import { theme } from '../theme';
import { useStore } from '../store';
import { api } from '../services/api';
import { SchedulingSlot, Severity } from '../types/api';

export default function BookingScreen() {
  const router = useRouter();
  const { selectedVehicle, vehicleHealth, setActiveJob } = useStore();
  
  const [loading, setLoading] = useState(false);
  const [slots, setSlots] = useState<SchedulingSlot[]>([]);
  const [selectedSlot, setSelectedSlot] = useState<SchedulingSlot | null>(null);
  const [bookingSuccess, setBookingSuccess] = useState(false);
  const [bookedJobId, setBookedJobId] = useState<string | null>(null);

  const successScale = useSharedValue(0);

  // Mock slots for demo
  const mockSlots: SchedulingSlot[] = [
    {
      service_center: { id: 'SC001', name: 'EV Care Mumbai Central', distance_km: 2.3, quality_rating: 4.8 },
      mechanic: { id: 'M001', name: 'Rajesh Kumar', skill_match_score: 0.95 },
      available_at: new Date(Date.now() + 86400000).toISOString(), // Tomorrow
      estimated_duration_minutes: 120,
      parts_available: true,
      estimated_cost: 8500,
      recommendation_score: 0.92,
    },
    {
      service_center: { id: 'SC002', name: 'GreenDrive Service Hub', distance_km: 4.1, quality_rating: 4.6 },
      mechanic: { id: 'M003', name: 'Amit Sharma', skill_match_score: 0.88 },
      available_at: new Date(Date.now() + 86400000 * 2).toISOString(), // Day after
      estimated_duration_minutes: 90,
      parts_available: true,
      estimated_cost: 7200,
      recommendation_score: 0.85,
    },
    {
      service_center: { id: 'SC003', name: 'PowerEV Workshop', distance_km: 6.8, quality_rating: 4.9 },
      mechanic: { id: 'M005', name: 'Priya Patel', skill_match_score: 0.92 },
      available_at: new Date(Date.now() + 86400000 * 3).toISOString(),
      estimated_duration_minutes: 150,
      parts_available: true,
      estimated_cost: 9800,
      recommendation_score: 0.88,
    },
  ];

  useEffect(() => {
    // Use mock data for demo
    setSlots(mockSlots);
  }, []);

  useEffect(() => {
    if (bookingSuccess) {
      successScale.value = withSequence(
        withSpring(1.2),
        withSpring(1)
      );
    }
  }, [bookingSuccess]);

  const successStyle = useAnimatedStyle(() => ({
    transform: [{ scale: successScale.value }],
  }));

  const handleSlotSelect = (slot: SchedulingSlot) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setSelectedSlot(slot);
  };

  const handleBooking = async () => {
    if (!selectedSlot || !selectedVehicle) return;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setLoading(true);

    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setBookingSuccess(true);
      setBookedJobId(`JOB-${Date.now()}`);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    }, 1500);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-IN', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-IN', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  if (bookingSuccess) {
    return (
      <View style={styles.container}>
        <SafeAreaView style={styles.successContainer}>
          <Animated.View style={[styles.successContent, successStyle]}>
            <LinearGradient
              colors={theme.colors.gradientSuccess as unknown as readonly [string, string]}
              style={styles.successIcon}
            >
              <Text style={styles.successEmoji}>✓</Text>
            </LinearGradient>
            <Text style={styles.successTitle}>Booking Confirmed!</Text>
            <Text style={styles.successSubtitle}>
              Your service appointment has been scheduled
            </Text>

            <View style={styles.bookingDetails}>
              <View style={styles.bookingRow}>
                <Text style={styles.bookingLabel}>Service Center</Text>
                <Text style={styles.bookingValue}>{selectedSlot?.service_center.name}</Text>
              </View>
              <View style={styles.bookingRow}>
                <Text style={styles.bookingLabel}>Date & Time</Text>
                <Text style={styles.bookingValue}>
                  {formatDate(selectedSlot?.available_at || '')} at {formatTime(selectedSlot?.available_at || '')}
                </Text>
              </View>
              <View style={styles.bookingRow}>
                <Text style={styles.bookingLabel}>Technician</Text>
                <Text style={styles.bookingValue}>{selectedSlot?.mechanic.name}</Text>
              </View>
              <View style={styles.bookingRow}>
                <Text style={styles.bookingLabel}>Estimated Cost</Text>
                <Text style={styles.bookingValue}>₹{selectedSlot?.estimated_cost?.toLocaleString()}</Text>
              </View>
            </View>

            <Pressable 
              style={styles.doneButton}
              onPress={() => router.replace('/')}
            >
              <LinearGradient
                colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
                style={styles.doneButtonGradient}
              >
                <Text style={styles.doneButtonText}>Return to Dashboard</Text>
              </LinearGradient>
            </Pressable>
          </Animated.View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea} edges={['top']}>
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <Animated.View entering={FadeInDown.delay(100).springify()}>
            <Text style={styles.title}>Book Service</Text>
            <Text style={styles.subtitle}>
              Select a convenient time slot at a nearby service center
            </Text>
          </Animated.View>

          {/* Issue Summary */}
          {vehicleHealth?.primary_concern && (
            <Animated.View entering={FadeInDown.delay(200).springify()}>
              <View style={styles.issueCard}>
                <Text style={styles.issueTitle}>Service Needed</Text>
                <Text style={styles.issueText}>
                  {vehicleHealth.primary_concern.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </Text>
              </View>
            </Animated.View>
          )}

          {/* Available Slots */}
          <Animated.View entering={FadeInDown.delay(300).springify()}>
            <Text style={styles.sectionTitle}>Available Appointments</Text>
            
            {slots.map((slot, index) => {
              const isSelected = selectedSlot === slot;
              const isRecommended = index === 0;

              return (
                <Animated.View 
                  key={slot.service_center.id + slot.available_at}
                  entering={FadeInDown.delay(400 + index * 100).springify()}
                >
                  <Pressable
                    style={[
                      styles.slotCard,
                      isSelected && styles.slotCardSelected,
                    ]}
                    onPress={() => handleSlotSelect(slot)}
                  >
                    {isRecommended && (
                      <View style={styles.recommendedBadge}>
                        <Text style={styles.recommendedText}>⭐ Recommended</Text>
                      </View>
                    )}

                    <View style={styles.slotHeader}>
                      <View style={styles.slotInfo}>
                        <Text style={styles.centerName}>{slot.service_center.name}</Text>
                        <View style={styles.slotMeta}>
                          <Text style={styles.slotMetaText}>
                            📍 {slot.service_center.distance_km} km
                          </Text>
                          <Text style={styles.slotMetaText}>
                            ⭐ {slot.service_center.quality_rating}
                          </Text>
                        </View>
                      </View>
                      <View style={[
                        styles.radioOuter,
                        isSelected && styles.radioOuterSelected
                      ]}>
                        {isSelected && <View style={styles.radioInner} />}
                      </View>
                    </View>

                    <View style={styles.slotDetails}>
                      <View style={styles.slotDetail}>
                        <Text style={styles.detailLabel}>Date & Time</Text>
                        <Text style={styles.detailValue}>
                          {formatDate(slot.available_at)} • {formatTime(slot.available_at)}
                        </Text>
                      </View>
                      <View style={styles.slotDetail}>
                        <Text style={styles.detailLabel}>Technician</Text>
                        <Text style={styles.detailValue}>{slot.mechanic.name}</Text>
                      </View>
                      <View style={styles.slotDetail}>
                        <Text style={styles.detailLabel}>Duration</Text>
                        <Text style={styles.detailValue}>{slot.estimated_duration_minutes} mins</Text>
                      </View>
                      <View style={styles.slotDetail}>
                        <Text style={styles.detailLabel}>Estimated Cost</Text>
                        <Text style={[styles.detailValue, styles.costValue]}>
                          ₹{slot.estimated_cost.toLocaleString()}
                        </Text>
                      </View>
                    </View>

                    {slot.parts_available && (
                      <View style={styles.partsAvailable}>
                        <Text style={styles.partsText}>✓ All parts in stock</Text>
                      </View>
                    )}
                  </Pressable>
                </Animated.View>
              );
            })}
          </Animated.View>
        </ScrollView>

        {/* Book Button */}
        {selectedSlot && (
          <Animated.View 
            entering={FadeInUp.springify()}
            style={styles.bottomBar}
          >
            <View style={styles.bottomContent}>
              <View>
                <Text style={styles.totalLabel}>Total Estimate</Text>
                <Text style={styles.totalValue}>
                  ₹{selectedSlot.estimated_cost.toLocaleString()}
                </Text>
              </View>
              <Pressable
                style={styles.bookButton}
                onPress={handleBooking}
                disabled={loading}
              >
                <LinearGradient
                  colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
                  style={styles.bookButtonGradient}
                >
                  {loading ? (
                    <ActivityIndicator color="#0A0E1A" />
                  ) : (
                    <Text style={styles.bookButtonText}>Confirm Booking</Text>
                  )}
                </LinearGradient>
              </Pressable>
            </View>
          </Animated.View>
        )}
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
    padding: theme.spacing.lg,
    paddingBottom: 150,
  },
  title: {
    ...theme.typography.h1,
    color: theme.colors.textPrimary,
  },
  subtitle: {
    ...theme.typography.body,
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.xs,
  },
  issueCard: {
    marginTop: theme.spacing.lg,
    padding: theme.spacing.md,
    backgroundColor: theme.colors.warningSoft,
    borderRadius: theme.borderRadius.md,
    borderWidth: 1,
    borderColor: theme.colors.warning,
  },
  issueTitle: {
    ...theme.typography.caption,
    color: theme.colors.warning,
    marginBottom: 4,
  },
  issueText: {
    ...theme.typography.body,
    color: theme.colors.textPrimary,
    fontWeight: '500',
  },
  sectionTitle: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    marginTop: theme.spacing.xl,
    marginBottom: theme.spacing.md,
  },
  slotCard: {
    backgroundColor: theme.colors.surfaceElevated,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.md,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  slotCardSelected: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.primary + '10',
  },
  recommendedBadge: {
    position: 'absolute',
    top: -10,
    right: theme.spacing.md,
    backgroundColor: theme.colors.warning,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 4,
    borderRadius: theme.borderRadius.full,
  },
  recommendedText: {
    ...theme.typography.caption,
    color: '#0A0E1A',
    fontWeight: '600',
  },
  slotHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.md,
  },
  slotInfo: {
    flex: 1,
  },
  centerName: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    marginBottom: 4,
  },
  slotMeta: {
    flexDirection: 'row',
    gap: theme.spacing.md,
  },
  slotMetaText: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
  },
  radioOuter: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: theme.colors.textMuted,
    justifyContent: 'center',
    alignItems: 'center',
  },
  radioOuterSelected: {
    borderColor: theme.colors.primary,
  },
  radioInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: theme.colors.primary,
  },
  slotDetails: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.md,
    paddingTop: theme.spacing.md,
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
  },
  slotDetail: {
    width: '45%',
  },
  detailLabel: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    marginBottom: 2,
  },
  detailValue: {
    ...theme.typography.bodySmall,
    color: theme.colors.textPrimary,
    fontWeight: '500',
  },
  costValue: {
    color: theme.colors.success,
  },
  partsAvailable: {
    marginTop: theme.spacing.sm,
    paddingTop: theme.spacing.sm,
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
  },
  partsText: {
    ...theme.typography.caption,
    color: theme.colors.success,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: theme.colors.surface,
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    paddingBottom: 34, // Safe area
  },
  bottomContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  totalLabel: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
  },
  totalValue: {
    ...theme.typography.h2,
    color: theme.colors.textPrimary,
  },
  bookButton: {
    minWidth: 160,
  },
  bookButtonGradient: {
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.xl,
    borderRadius: theme.borderRadius.md,
    alignItems: 'center',
  },
  bookButtonText: {
    ...theme.typography.body,
    color: '#0A0E1A',
    fontWeight: '600',
  },
  successContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.lg,
  },
  successContent: {
    alignItems: 'center',
  },
  successIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
    ...theme.shadows.lg,
  },
  successEmoji: {
    fontSize: 48,
    color: '#FFFFFF',
  },
  successTitle: {
    ...theme.typography.h1,
    color: theme.colors.textPrimary,
    textAlign: 'center',
  },
  successSubtitle: {
    ...theme.typography.body,
    color: theme.colors.textSecondary,
    textAlign: 'center',
    marginTop: theme.spacing.sm,
  },
  bookingDetails: {
    marginTop: theme.spacing.xl,
    backgroundColor: theme.colors.surfaceElevated,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    width: '100%',
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  bookingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.glassBorder,
  },
  bookingLabel: {
    ...theme.typography.bodySmall,
    color: theme.colors.textMuted,
  },
  bookingValue: {
    ...theme.typography.bodySmall,
    color: theme.colors.textPrimary,
    fontWeight: '500',
    textAlign: 'right',
    flex: 1,
    marginLeft: theme.spacing.md,
  },
  doneButton: {
    marginTop: theme.spacing.xl,
    width: '100%',
  },
  doneButtonGradient: {
    paddingVertical: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
    alignItems: 'center',
  },
  doneButtonText: {
    ...theme.typography.body,
    color: '#0A0E1A',
    fontWeight: '600',
  },
});
