import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Linking, Platform, Pressable } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { theme } from '../../theme';

interface ServiceStatusCardProps {
  stage: string;
  centerName: string;
  estimatedCompletion: string | null;
  serviceType: string;
}

const STAGES = ['BOOKED', 'CHECK_IN', 'DIAGNOSIS', 'REPAIR', 'READY'];

export const ServiceStatusCard: React.FC<ServiceStatusCardProps> = ({
  stage,
  centerName,
  estimatedCompletion,
  serviceType,
}) => {
  const currentStageIndex = STAGES.indexOf(stage);
  const progress = (currentStageIndex / (STAGES.length - 1)) * 100;

  const handleCallCenter = () => {
    Linking.openURL('tel:1800123456');
  };

  const handleDirections = () => {
    // Open maps with query
    const query = encodeURIComponent(centerName);
    const url = Platform.select({
      ios: `maps:0,0?q=${query}`,
      android: `geo:0,0?q=${query}`,
    });
    Linking.openURL(url || '');
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
        style={styles.card}
      >
        <View style={styles.header}>
          <Text style={styles.title}>Live Service Status</Text>
          <View style={styles.liveBadge}>
            <View style={styles.liveDot} />
            <Text style={styles.liveText}>LIVE</Text>
          </View>
        </View>

        {/* Vehicle Info */}
        <View style={styles.infoRow}>
          <Text style={styles.centerName}>{centerName}</Text>
          <Text style={styles.serviceType}>{serviceType.replace('_', ' ')}</Text>
        </View>

        {/* Progress Bar */}
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${progress}%` }]} />
          </View>
          <View style={styles.stagesRow}>
            {STAGES.map((s, idx) => {
              const isCompleted = idx <= currentStageIndex;
              const isCurrent = idx === currentStageIndex;
              return (
                <View key={s} style={styles.stageItem}>
                  <View
                    style={[
                      styles.stageDot,
                      isCompleted && styles.stageDotCompleted,
                      isCurrent && styles.stageDotCurrent,
                    ]}
                  />
                  {isCurrent && (
                    <Text style={styles.stageLabel}>{s.replace('_', ' ')}</Text>
                  )}
                </View>
              );
            })}
          </View>
        </View>

        {/* ETA & Actions */}
        <View style={styles.footer}>
          <View>
            <Text style={styles.etaLabel}>Estimated Completion</Text>
            <Text style={styles.etaValue}>
              {estimatedCompletion ? new Date(estimatedCompletion).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 'Calculating...'}
            </Text>
          </View>
          <View style={styles.actions}>
            <Pressable onPress={handleCallCenter} style={styles.actionButton}>
              <Text style={styles.actionIcon}>📞</Text>
            </Pressable>
            <Pressable onPress={handleDirections} style={styles.actionButton}>
              <Text style={styles.actionIcon}>🗺️</Text>
            </Pressable>
          </View>
        </View>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: theme.spacing.lg,
    marginBottom: theme.spacing.md,
  },
  card: {
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius.lg,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  title: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
    fontSize: 16,
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: theme.colors.danger,
    marginRight: 4,
  },
  liveText: {
    color: theme.colors.danger,
    fontSize: 10,
    fontWeight: '700',
  },
  infoRow: {
    marginBottom: theme.spacing.md,
  },
  centerName: {
    fontSize: 14,
    color: theme.colors.textPrimary,
    fontWeight: '600',
  },
  serviceType: {
    fontSize: 12,
    color: theme.colors.textMuted,
    textTransform: 'capitalize',
  },
  progressContainer: {
    marginBottom: theme.spacing.md,
  },
  progressBar: {
    height: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 2,
    marginBottom: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: theme.colors.primary,
  },
  stagesRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 0,
  },
  stageItem: {
    alignItems: 'center',
    width: 20,
    overflow: 'visible',
  },
  stageDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  stageDotCompleted: {
    backgroundColor: theme.colors.primary,
  },
  stageDotCurrent: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.background,
  },
  stageLabel: {
    position: 'absolute',
    top: 14,
    fontSize: 10,
    width: 60,
    textAlign: 'center',
    color: theme.colors.primary,
    fontWeight: '600',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
    paddingTop: theme.spacing.sm,
  },
  etaLabel: {
    fontSize: 10,
    color: theme.colors.textMuted,
  },
  etaValue: {
    fontSize: 14,
    color: theme.colors.textPrimary,
    fontWeight: '700',
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionIcon: {
    fontSize: 16,
  },
});
