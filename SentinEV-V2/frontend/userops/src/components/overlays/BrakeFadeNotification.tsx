/**
 * Brake Fade Notification - Detailed notification shown when call is rejected
 * Provides full diagnostic report and action options
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  Animated,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { theme } from '../../theme';

interface BrakeFadeNotificationProps {
  visible: boolean;
  onCallBack: () => void;
  onChat: () => void;
  onDismiss: () => void;
  vehicleInfo?: {
    make: string;
    model: string;
    id: string;
  };
}

export const BrakeFadeNotification: React.FC<BrakeFadeNotificationProps> = ({
  visible,
  onCallBack,
  onChat,
  onDismiss,
  vehicleInfo = { make: 'Kia', model: 'EV6', id: 'VH005' },
}) => {
  const [expanded, setExpanded] = useState(false);
  const [slideAnim] = useState(new Animated.Value(300));

  React.useEffect(() => {
    if (visible) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      Animated.spring(slideAnim, {
        toValue: 0,
        tension: 50,
        friction: 8,
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(slideAnim, {
        toValue: 300,
        duration: 200,
        useNativeDriver: true,
      }).start();
    }
  }, [visible]);

  const handleCallBack = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    onCallBack();
  };

  const handleChat = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    onChat();
  };

  const handleDismiss = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    onDismiss();
  };

  const toggleExpanded = () => {
    Haptics.selectionAsync();
    setExpanded(!expanded);
  };

  if (!visible) return null;

  return (
    <View style={styles.overlay}>
      <Pressable style={styles.backdrop} onPress={handleDismiss} />
      
      <Animated.View
        style={[
          styles.container,
          { transform: [{ translateY: slideAnim }] },
        ]}
      >
        <LinearGradient
          colors={['#1A1F3C', '#0A0E1A']}
          style={styles.gradient}
        >
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Text style={styles.alertIcon}>⚠️</Text>
              <View>
                <Text style={styles.title}>Brake Fade Alert</Text>
                <Text style={styles.subtitle}>Action Required</Text>
              </View>
            </View>
            <Pressable onPress={handleDismiss} style={styles.closeButton}>
              <Text style={styles.closeIcon}>✕</Text>
            </Pressable>
          </View>

          {/* Vehicle Info */}
          <View style={styles.vehicleCard}>
            <Text style={styles.vehicleText}>
              {vehicleInfo.make} {vehicleInfo.model}
            </Text>
            <Text style={styles.vehicleId}>({vehicleInfo.id})</Text>
            <View style={styles.severityBadge}>
              <Text style={styles.severityText}>MEDIUM - Preventive</Text>
            </View>
          </View>

          {/* Issue Summary */}
          <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Issue Detected:</Text>
              <View style={styles.issueList}>
                <Text style={styles.issueItem}>• Brake temperature elevated</Text>
                <Text style={styles.issueItem}>• Braking efficiency reduced</Text>
                <Text style={styles.issueItem}>• 65% probability of brake fade in 6-7 days</Text>
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Recommendation:</Text>
              <Text style={styles.recommendationText}>
                Schedule brake service within 3-5 days to prevent safety issues and costlier repairs.
              </Text>
            </View>

            {/* Expandable Full Report */}
            <Pressable onPress={toggleExpanded} style={styles.expandButton}>
              <Text style={styles.expandText}>
                {expanded ? '▼' : '▶'} {expanded ? 'Hide' : 'View'} Full Report
              </Text>
            </Pressable>

            {expanded && (
              <View style={styles.fullReport}>
                <Text style={styles.reportTitle}>Detailed Diagnosis:</Text>
                <Text style={styles.reportText}>
                  • Brake pad wear: 68% (threshold: 70%)
                </Text>
                <Text style={styles.reportText}>
                  • Brake fluid temperature: 185°C (normal: 160°C)
                </Text>
                <Text style={styles.reportText}>
                  • Braking distance increase: 12%
                </Text>
                <Text style={styles.reportText}>
                  • Regenerative braking efficiency: 82% (normal: 95%)
                </Text>
                
                <Text style={styles.reportTitle}>Estimated Cost:</Text>
                <Text style={styles.reportText}>
                  ₹4,500 - ₹6,000 (covered under warranty)
                </Text>
                
                <Text style={styles.reportTitle}>Service Duration:</Text>
                <Text style={styles.reportText}>
                  Approximately 2 hours
                </Text>
              </View>
            )}
          </ScrollView>

          {/* Action Buttons */}
          <View style={styles.actions}>
            <Pressable style={styles.primaryButton} onPress={handleCallBack}>
              <LinearGradient
                colors={['#10B981', '#059669']}
                style={styles.buttonGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                <Text style={styles.primaryButtonText}>📞 Call Back</Text>
              </LinearGradient>
            </Pressable>

            <View style={styles.secondaryActions}>
              <Pressable style={styles.secondaryButton} onPress={handleChat}>
                <Text style={styles.secondaryButtonText}>💬 Chat with AI</Text>
              </Pressable>
              
              <Pressable style={styles.secondaryButton} onPress={handleDismiss}>
                <Text style={styles.secondaryButtonText}>Later</Text>
              </Pressable>
            </View>
          </View>
        </LinearGradient>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 999,
    justifyContent: 'flex-end',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  container: {
    maxHeight: '85%',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
  },
  gradient: {
    paddingTop: 20,
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  alertIcon: {
    fontSize: 32,
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  subtitle: {
    fontSize: 13,
    color: '#FBBf24',
    marginTop: 2,
  },
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeIcon: {
    color: '#A0AEC0',
    fontSize: 18,
  },
  vehicleCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    marginHorizontal: 20,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    marginBottom: 20,
  },
  vehicleText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  vehicleId: {
    fontSize: 14,
    color: '#A0AEC0',
    marginTop: 2,
    marginBottom: 8,
  },
  severityBadge: {
    backgroundColor: 'rgba(251, 191, 36, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(251, 191, 36, 0.3)',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    alignSelf: 'flex-start',
  },
  severityText: {
    color: '#FBBf24',
    fontSize: 12,
    fontWeight: '600',
  },
  content: {
    maxHeight: 300,
    paddingHorizontal: 20,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#A0AEC0',
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  issueList: {
    gap: 6,
  },
  issueItem: {
    fontSize: 15,
    color: '#E2E8F0',
    lineHeight: 22,
  },
  recommendationText: {
    fontSize: 15,
    color: '#E2E8F0',
    lineHeight: 22,
  },
  expandButton: {
    paddingVertical: 12,
    marginBottom: 10,
  },
  expandText: {
    color: theme.colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  fullReport: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  reportTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#A0AEC0',
    marginTop: 12,
    marginBottom: 6,
  },
  reportText: {
    fontSize: 14,
    color: '#E2E8F0',
    lineHeight: 20,
    marginBottom: 4,
  },
  actions: {
    paddingHorizontal: 20,
    paddingTop: 20,
    gap: 12,
  },
  primaryButton: {
    borderRadius: 12,
    overflow: 'hidden',
  },
  buttonGradient: {
    paddingVertical: 16,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  secondaryActions: {
    flexDirection: 'row',
    gap: 12,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#A0AEC0',
    fontSize: 14,
    fontWeight: '600',
  },
});

export default BrakeFadeNotification;
