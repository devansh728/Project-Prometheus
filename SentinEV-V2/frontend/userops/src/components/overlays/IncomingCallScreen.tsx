/**
 * Incoming Call Screen - Full-screen incoming call UI
 * Displays before voice call interface with accept/reject options
 */
import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Animated,
  Dimensions,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { theme } from '../../theme';

const { width, height } = Dimensions.get('window');

interface IncomingCallScreenProps {
  visible: boolean;
  onAccept: () => void;
  onReject: () => void;
  callerName?: string;
  phoneNumber?: string;
  callReason?: string;
}

export const IncomingCallScreen: React.FC<IncomingCallScreenProps> = ({
  visible,
  onAccept,
  onReject,
  callerName = 'SentinEV',
  phoneNumber = '+0510-847-2931',
  callReason = 'Brake Fade Alert',
}) => {
  const [pulseAnim] = useState(new Animated.Value(1));
  const [rippleAnim] = useState(new Animated.Value(0));
  const [fadeAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    if (visible) {
      // Haptic feedback on mount
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);

      // Fade in animation
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }).start();

      // Pulse animation for avatar
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.15,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      ).start();

      // Ripple animation
      Animated.loop(
        Animated.sequence([
          Animated.timing(rippleAnim, {
            toValue: 1,
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(rippleAnim, {
            toValue: 0,
            duration: 0,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [visible]);

  const handleAccept = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    onAccept();
  };

  const handleReject = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    onReject();
  };

  if (!visible) return null;

  return (
    <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
      <LinearGradient
        colors={['#0A0E1A', '#1A1F3C', '#0A0E1A']}
        style={styles.gradient}
      >
        {/* Top Section - Caller Info */}
        <View style={styles.topSection}>
          <Text style={styles.incomingLabel}>Incoming Call</Text>

          {/* Avatar with Ripple Effect */}
          <View style={styles.avatarContainer}>
            {/* Ripple rings */}
            {[1, 2, 3].map((i) => (
              <Animated.View
                key={i}
                style={[
                  styles.rippleRing,
                  {
                    opacity: rippleAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0.4 - i * 0.1, 0],
                    }),
                    transform: [
                      {
                        scale: rippleAnim.interpolate({
                          inputRange: [0, 1],
                          outputRange: [1, 1.5 + i * 0.3],
                        }),
                      },
                    ],
                  },
                ]}
              />
            ))}

            {/* Main Avatar */}
            <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
              <LinearGradient
                colors={['#00D9FF', '#7C3AED']}
                style={styles.avatar}
              >
                <Text style={styles.avatarIcon}>🚗</Text>
              </LinearGradient>
            </Animated.View>
          </View>

          {/* Caller Details */}
          <Text style={styles.callerName}>{callerName}</Text>
          <Text style={styles.phoneNumber}>{phoneNumber}</Text>
          <View style={styles.reasonBadge}>
            <Text style={styles.reasonText}>⚠️ {callReason}</Text>
          </View>
        </View>

        {/* Bottom Section - Action Buttons */}
        <View style={styles.bottomSection}>
          <Text style={styles.swipeHint}>Swipe to answer or decline</Text>

          <View style={styles.actionsContainer}>
            {/* Reject Button */}
            <Pressable
              style={styles.actionButton}
              onPress={handleReject}
              android_ripple={{ color: 'rgba(239, 68, 68, 0.3)' }}
            >
              <View style={[styles.actionCircle, styles.rejectCircle]}>
                <Text style={styles.actionIcon}>✕</Text>
              </View>
              <Text style={styles.actionLabel}>Decline</Text>
            </Pressable>

            {/* Accept Button */}
            <Pressable
              style={styles.actionButton}
              onPress={handleAccept}
              android_ripple={{ color: 'rgba(16, 185, 129, 0.3)' }}
            >
              <View style={[styles.actionCircle, styles.acceptCircle]}>
                <Text style={styles.actionIcon}>📞</Text>
              </View>
              <Text style={styles.actionLabel}>Accept</Text>
            </Pressable>
          </View>
        </View>
      </LinearGradient>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 1000,
  },
  gradient: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topSection: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: Platform.OS === 'ios' ? 80 : 60,
  },
  incomingLabel: {
    fontSize: 16,
    color: '#A0AEC0',
    marginBottom: 40,
    fontWeight: '500',
  },
  avatarContainer: {
    width: 140,
    height: 140,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  rippleRing: {
    position: 'absolute',
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 2,
    borderColor: '#00D9FF',
  },
  avatar: {
    width: 120,
    height: 120,
    borderRadius: 60,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#00D9FF',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 10,
  },
  avatarIcon: {
    fontSize: 52,
  },
  callerName: {
    fontSize: 32,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  phoneNumber: {
    fontSize: 18,
    color: '#A0AEC0',
    marginBottom: 20,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  reasonBadge: {
    backgroundColor: 'rgba(251, 191, 36, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(251, 191, 36, 0.3)',
    borderRadius: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  reasonText: {
    color: '#FBBf24',
    fontSize: 15,
    fontWeight: '600',
  },
  bottomSection: {
    paddingBottom: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 40,
  },
  swipeHint: {
    textAlign: 'center',
    color: '#A0AEC0',
    fontSize: 14,
    marginBottom: 30,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  actionButton: {
    alignItems: 'center',
  },
  actionCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  rejectCircle: {
    backgroundColor: '#EF4444',
    shadowColor: '#EF4444',
  },
  acceptCircle: {
    backgroundColor: '#10B981',
    shadowColor: '#10B981',
  },
  actionIcon: {
    fontSize: 32,
  },
  actionLabel: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default IncomingCallScreen;
