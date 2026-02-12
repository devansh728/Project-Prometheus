/**
 * Incoming Call Overlay Component
 * Full-screen animated overlay for simulated voice calls
 */
import React, { useEffect } from 'react';
import { View, Text, StyleSheet, Pressable, Modal, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  withSpring,
  Easing,
  runOnJS,
} from 'react-native-reanimated';
import { theme } from '../../theme';

const { width, height } = Dimensions.get('window');

interface IncomingCallOverlayProps {
  visible: boolean;
  callerName: string;
  callerSubtitle?: string;
  onAccept: () => void;
  onDecline: () => void;
}

export const IncomingCallOverlay: React.FC<IncomingCallOverlayProps> = ({
  visible,
  callerName,
  callerSubtitle = 'SentinEV Urgent Alert',
  onAccept,
  onDecline,
}) => {
  const ring1Scale = useSharedValue(1);
  const ring2Scale = useSharedValue(1);
  const ring3Scale = useSharedValue(1);
  const avatarScale = useSharedValue(1);
  const acceptScale = useSharedValue(1);
  const declineScale = useSharedValue(1);

  useEffect(() => {
    if (visible) {
      // Haptic vibration pattern
      const vibrate = () => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      };
      vibrate();
      const interval = setInterval(vibrate, 2000);

      // Ring animations
      ring1Scale.value = withRepeat(
        withSequence(
          withTiming(1.5, { duration: 1500, easing: Easing.out(Easing.ease) }),
          withTiming(1, { duration: 0 })
        ),
        -1
      );
      ring2Scale.value = withRepeat(
        withSequence(
          withTiming(1, { duration: 500 }),
          withTiming(1.5, { duration: 1500, easing: Easing.out(Easing.ease) }),
          withTiming(1, { duration: 0 })
        ),
        -1
      );
      ring3Scale.value = withRepeat(
        withSequence(
          withTiming(1, { duration: 1000 }),
          withTiming(1.5, { duration: 1500, easing: Easing.out(Easing.ease) }),
          withTiming(1, { duration: 0 })
        ),
        -1
      );
      avatarScale.value = withRepeat(
        withSequence(
          withTiming(1.05, { duration: 500 }),
          withTiming(1, { duration: 500 })
        ),
        -1,
        true
      );

      return () => clearInterval(interval);
    }
  }, [visible]);

  const ring1Style = useAnimatedStyle(() => ({
    transform: [{ scale: ring1Scale.value }],
    opacity: 2 - ring1Scale.value,
  }));

  const ring2Style = useAnimatedStyle(() => ({
    transform: [{ scale: ring2Scale.value }],
    opacity: 2 - ring2Scale.value,
  }));

  const ring3Style = useAnimatedStyle(() => ({
    transform: [{ scale: ring3Scale.value }],
    opacity: 2 - ring3Scale.value,
  }));

  const avatarStyle = useAnimatedStyle(() => ({
    transform: [{ scale: avatarScale.value }],
  }));

  const acceptBtnStyle = useAnimatedStyle(() => ({
    transform: [{ scale: acceptScale.value }],
  }));

  const declineBtnStyle = useAnimatedStyle(() => ({
    transform: [{ scale: declineScale.value }],
  }));

  const handleAccept = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    acceptScale.value = withSpring(0.9, {}, () => {
      acceptScale.value = withSpring(1);
      runOnJS(onAccept)();
    });
  };

  const handleDecline = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    declineScale.value = withSpring(0.9, {}, () => {
      declineScale.value = withSpring(1);
      runOnJS(onDecline)();
    });
  };

  return (
    <Modal visible={visible} transparent animationType="fade">
      <LinearGradient
        colors={['#1A1A2E', '#0A0E1A']}
        style={styles.container}
      >
        {/* Animated Rings */}
        <View style={styles.avatarContainer}>
          <Animated.View style={[styles.ring, ring1Style]} />
          <Animated.View style={[styles.ring, ring2Style]} />
          <Animated.View style={[styles.ring, ring3Style]} />
          
          <Animated.View style={avatarStyle}>
            <LinearGradient
              colors={theme.colors.gradientDanger as unknown as readonly [string, string]}
              style={styles.avatar}
            >
              <Text style={styles.avatarIcon}>🚨</Text>
            </LinearGradient>
          </Animated.View>
        </View>

        {/* Caller Info */}
        <View style={styles.callerInfo}>
          <Text style={styles.callerName}>{callerName}</Text>
          <Text style={styles.callerSubtitle}>{callerSubtitle}</Text>
          <Text style={styles.callingText}>Incoming Call...</Text>
        </View>

        {/* Action Buttons */}
        <View style={styles.actions}>
          <Animated.View style={declineBtnStyle}>
            <Pressable style={styles.declineBtn} onPress={handleDecline}>
              <LinearGradient
                colors={['#EF4444', '#DC2626']}
                style={styles.actionBtn}
              >
                <Text style={styles.actionIcon}>📞</Text>
              </LinearGradient>
              <Text style={styles.actionLabel}>Decline</Text>
            </Pressable>
          </Animated.View>

          <Animated.View style={acceptBtnStyle}>
            <Pressable style={styles.acceptBtn} onPress={handleAccept}>
              <LinearGradient
                colors={['#10B981', '#059669']}
                style={styles.actionBtn}
              >
                <Text style={styles.actionIcon}>📞</Text>
              </LinearGradient>
              <Text style={styles.actionLabel}>Accept</Text>
            </Pressable>
          </Animated.View>
        </View>
      </LinearGradient>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 80,
  },
  avatarContainer: {
    width: 200,
    height: 200,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 60,
  },
  ring: {
    position: 'absolute',
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 2,
    borderColor: theme.colors.danger,
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarIcon: {
    fontSize: 40,
  },
  callerInfo: {
    alignItems: 'center',
  },
  callerName: {
    ...theme.typography.h1,
    color: theme.colors.textPrimary,
    textAlign: 'center',
  },
  callerSubtitle: {
    ...theme.typography.body,
    color: theme.colors.textMuted,
    marginTop: theme.spacing.sm,
  },
  callingText: {
    ...theme.typography.bodySmall,
    color: theme.colors.primary,
    marginTop: theme.spacing.md,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    paddingHorizontal: theme.spacing.xxl,
  },
  acceptBtn: {
    alignItems: 'center',
  },
  declineBtn: {
    alignItems: 'center',
  },
  actionBtn: {
    width: 70,
    height: 70,
    borderRadius: 35,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
    ...theme.shadows.lg,
  },
  actionIcon: {
    fontSize: 28,
  },
  actionLabel: {
    ...theme.typography.bodySmall,
    color: theme.colors.textSecondary,
  },
});
