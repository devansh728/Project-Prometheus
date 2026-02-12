/**
 * Voice Call Screen Component - Full-screen voice interaction UI
 */
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Pressable, Animated, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import Svg, { Circle } from 'react-native-svg';

const { width, height } = Dimensions.get('window');

interface VoiceCallScreenProps {
  visible: boolean;
  callerName?: string;
  callType?: 'incoming' | 'outgoing' | 'active';
  transcription?: string;
  onAccept?: () => void;
  onDecline?: () => void;
  onEndCall?: () => void;
}

export const VoiceCallScreen: React.FC<VoiceCallScreenProps> = ({
  visible,
  callerName = 'SentinEV Assistant',
  callType = 'incoming',
  transcription,
  onAccept,
  onDecline,
  onEndCall,
}) => {
  const [pulseAnim] = useState(new Animated.Value(1));
  const [waveAnim] = useState(new Animated.Value(0));
  const [dotsAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    if (visible) {
      // Pulse animation for avatar
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.1, duration: 800, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
        ])
      ).start();

      // Wave animation for voice visualization
      Animated.loop(
        Animated.timing(waveAnim, { toValue: 1, duration: 1500, useNativeDriver: true })
      ).start();

      // Dots animation for thinking
      Animated.loop(
        Animated.sequence([
          Animated.timing(dotsAnim, { toValue: 3, duration: 900, useNativeDriver: false }),
          Animated.timing(dotsAnim, { toValue: 0, duration: 0, useNativeDriver: false }),
        ])
      ).start();
    }
  }, [visible]);

  if (!visible) return null;

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#0A0E1A', '#1A1F3C', '#0A0E1A']}
        style={styles.gradient}
      >
        {/* Caller Info */}
        <View style={styles.callerSection}>
          <Animated.View style={[styles.avatarContainer, { transform: [{ scale: pulseAnim }] }]}>
            <LinearGradient
              colors={['#00D9FF', '#7C3AED']}
              style={styles.avatar}
            >
              <Text style={styles.avatarText}>🤖</Text>
            </LinearGradient>
            
            {/* Pulse rings */}
            {[1, 2, 3].map((i) => (
              <Animated.View
                key={i}
                style={[
                  styles.pulseRing,
                  {
                    opacity: waveAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0.3 - i * 0.08, 0],
                    }),
                    transform: [{
                      scale: waveAnim.interpolate({
                        inputRange: [0, 1],
                        outputRange: [1, 1 + i * 0.5],
                      }),
                    }],
                  },
                ]}
              />
            ))}
          </Animated.View>

          <Text style={styles.callerName}>{callerName}</Text>
          <Text style={styles.callStatus}>
            {callType === 'incoming' ? '📞 Incoming Call' : 
             callType === 'active' ? '🔊 Connected' : '📞 Calling...'}
          </Text>
        </View>

        {/* Voice Visualization */}
        {callType === 'active' && (
          <View style={styles.waveContainer}>
            {[...Array(12)].map((_, i) => (
              <Animated.View
                key={i}
                style={[
                  styles.waveBar,
                  {
                    height: waveAnim.interpolate({
                      inputRange: [0, 0.5, 1],
                      outputRange: [
                        10 + Math.random() * 20,
                        30 + Math.random() * 40,
                        10 + Math.random() * 20,
                      ],
                    }),
                  },
                ]}
              />
            ))}
          </View>
        )}

        {/* Transcription */}
        {transcription && (
          <View style={styles.transcriptionBox}>
            <Text style={styles.transcriptionLabel}>🎙️ AI Speaking:</Text>
            <Text style={styles.transcriptionText}>{transcription}</Text>
          </View>
        )}

        {/* Call Actions */}
        <View style={styles.actionsContainer}>
          {callType === 'incoming' ? (
            <>
              <Pressable style={styles.declineBtn} onPress={onDecline}>
                <Text style={styles.actionIcon}>📵</Text>
                <Text style={styles.actionLabel}>Decline</Text>
              </Pressable>
              
              <Pressable style={styles.acceptBtn} onPress={onAccept}>
                <Text style={styles.actionIcon}>📞</Text>
                <Text style={styles.actionLabel}>Accept</Text>
              </Pressable>
            </>
          ) : (
            <Pressable style={styles.endCallBtn} onPress={onEndCall}>
              <Text style={styles.endCallIcon}>📵</Text>
              <Text style={styles.endCallLabel}>End Call</Text>
            </Pressable>
          )}
        </View>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 999 },
  gradient: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32 },
  callerSection: { alignItems: 'center', marginBottom: 48 },
  avatarContainer: { width: 120, height: 120, marginBottom: 24, alignItems: 'center', justifyContent: 'center' },
  avatar: { width: 100, height: 100, borderRadius: 50, alignItems: 'center', justifyContent: 'center' },
  avatarText: { fontSize: 48 },
  pulseRing: { position: 'absolute', width: 100, height: 100, borderRadius: 50, borderWidth: 2, borderColor: '#00D9FF' },
  callerName: { fontSize: 24, fontWeight: '700', color: '#FFF', marginBottom: 8 },
  callStatus: { fontSize: 14, color: '#A0AEC0' },
  waveContainer: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 4, height: 60, marginBottom: 32 },
  waveBar: { width: 4, backgroundColor: '#00D9FF', borderRadius: 2 },
  transcriptionBox: { backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 16, padding: 20, marginBottom: 48, maxWidth: 320 },
  transcriptionLabel: { fontSize: 12, color: '#64748B', marginBottom: 8 },
  transcriptionText: { fontSize: 16, color: '#FFF', lineHeight: 24 },
  actionsContainer: { flexDirection: 'row', gap: 48 },
  declineBtn: { alignItems: 'center', backgroundColor: 'rgba(239,68,68,0.2)', width: 72, height: 72, borderRadius: 36, justifyContent: 'center' },
  acceptBtn: { alignItems: 'center', backgroundColor: 'rgba(16,185,129,0.2)', width: 72, height: 72, borderRadius: 36, justifyContent: 'center' },
  endCallBtn: { alignItems: 'center', backgroundColor: '#EF4444', width: 72, height: 72, borderRadius: 36, justifyContent: 'center' },
  actionIcon: { fontSize: 28 },
  actionLabel: { fontSize: 11, color: '#A0AEC0', marginTop: 4 },
  endCallIcon: { fontSize: 28 },
  endCallLabel: { fontSize: 11, color: '#FFF', marginTop: 4 },
});
