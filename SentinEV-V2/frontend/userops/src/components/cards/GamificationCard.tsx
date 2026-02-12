/**
 * GamificationCard - Safe driving score, badges, and streak display
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
  FadeIn,
  SlideInRight,
} from 'react-native-reanimated';
import { theme } from '../../theme';

interface GamificationData {
  drivingScore: number;
  totalPoints: number;
  sessionPoints: number;
  streak: number;
  badges: string[];
  nextServiceDays: number;
}

interface GamificationCardProps {
  data: GamificationData;
}

const BadgeItem: React.FC<{ badge: string; index: number }> = ({ badge, index }) => {
  const bounceScale = useSharedValue(1);
  
  useEffect(() => {
    bounceScale.value = withRepeat(
      withSequence(
        withTiming(1.1, { duration: 1500 }),
        withTiming(1, { duration: 1500 })
      ),
      -1,
      true
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: bounceScale.value }],
  }));

  return (
    <Animated.View 
      entering={SlideInRight.delay(index * 100).springify()}
      style={[styles.badgeItem, animatedStyle]}
    >
      <Text style={styles.badgeText}>{badge}</Text>
    </Animated.View>
  );
};

export const GamificationCard: React.FC<GamificationCardProps> = ({ data }) => {
  const scoreScale = useSharedValue(0);
  const pointsValue = useSharedValue(0);
  
  useEffect(() => {
    scoreScale.value = withSpring(1, { damping: 8, stiffness: 100 });
    pointsValue.value = withTiming(data.sessionPoints, { duration: 2000 });
  }, [data.sessionPoints]);

  const scoreStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scoreScale.value }],
  }));

  const getScoreColor = (score: number) => {
    if (score >= 90) return '#10B981';
    if (score >= 70) return '#F59E0B';
    return '#EF4444';
  };

  const getScoreEmoji = (score: number) => {
    if (score >= 95) return '🏆';
    if (score >= 90) return '🥇';
    if (score >= 80) return '🥈';
    if (score >= 70) return '🥉';
    return '💪';
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Safe Driving</Text>
        <View style={styles.streakBadge}>
          <Text style={styles.streakEmoji}>🔥</Text>
          <Text style={styles.streakText}>{data.streak} day streak</Text>
        </View>
      </View>
      
      <LinearGradient
        colors={['#1E3A5F', '#0F172A']}
        style={styles.content}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        {/* Main Score Display */}
        <View style={styles.scoreSection}>
          <Animated.View style={[styles.scoreCircle, scoreStyle]}>
            <LinearGradient
              colors={[getScoreColor(data.drivingScore), getScoreColor(data.drivingScore) + '80']}
              style={styles.scoreGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
            >
              <Text style={styles.scoreValue}>{data.drivingScore}</Text>
              <Text style={styles.scoreLabel}>Score</Text>
            </LinearGradient>
          </Animated.View>
          
          <View style={styles.scoreInfo}>
            <Text style={styles.scoreRank}>
              {getScoreEmoji(data.drivingScore)} Top 5% of drivers
            </Text>
            <View style={styles.pointsContainer}>
              <View style={styles.pointItem}>
                <Text style={styles.pointValue}>{data.totalPoints.toLocaleString()}</Text>
                <Text style={styles.pointLabel}>Total Points</Text>
              </View>
              <View style={styles.pointDivider} />
              <View style={styles.pointItem}>
                <Text style={[styles.pointValue, styles.sessionPoints]}>
                  +{data.sessionPoints}
                </Text>
                <Text style={styles.pointLabel}>This Session</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Badges Section */}
        <View style={styles.badgesSection}>
          <Text style={styles.badgesTitle}>Achievements</Text>
          <View style={styles.badgesList}>
            {data.badges.map((badge, index) => (
              <BadgeItem key={badge} badge={badge} index={index} />
            ))}
          </View>
        </View>

        {/* Next Service */}
        <View style={styles.serviceSection}>
          <View style={styles.serviceIcon}>
            <Text style={styles.serviceEmoji}>🔧</Text>
          </View>
          <View style={styles.serviceInfo}>
            <Text style={styles.serviceTitle}>Next Service</Text>
            <Text style={styles.serviceValue}>In {data.nextServiceDays} days</Text>
          </View>
          <View style={styles.serviceStatus}>
            <Text style={styles.serviceStatusText}>🟢 On Track</Text>
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
  streakBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(245, 158, 11, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  streakEmoji: {
    fontSize: 14,
    marginRight: 4,
  },
  streakText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#F59E0B',
  },
  content: {
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  scoreSection: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  scoreCircle: {
    marginRight: theme.spacing.lg,
  },
  scoreGradient: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    ...theme.shadows.lg,
  },
  scoreValue: {
    fontSize: 36,
    fontWeight: '700',
    color: '#fff',
  },
  scoreLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
  },
  scoreInfo: {
    flex: 1,
  },
  scoreRank: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.textPrimary,
    marginBottom: theme.spacing.sm,
  },
  pointsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  pointItem: {
    alignItems: 'center',
  },
  pointValue: {
    fontSize: 18,
    fontWeight: '700',
    color: theme.colors.primary,
  },
  sessionPoints: {
    color: theme.colors.success,
  },
  pointLabel: {
    fontSize: 10,
    color: theme.colors.textMuted,
    marginTop: 2,
  },
  pointDivider: {
    width: 1,
    height: 30,
    backgroundColor: theme.colors.glassBorder,
    marginHorizontal: 16,
  },
  badgesSection: {
    marginBottom: theme.spacing.lg,
  },
  badgesTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: theme.colors.textMuted,
    marginBottom: theme.spacing.sm,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  badgesList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  badgeItem: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  badgeText: {
    fontSize: 12,
    fontWeight: '500',
    color: theme.colors.textPrimary,
  },
  serviceSection: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
  },
  serviceIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: theme.spacing.md,
  },
  serviceEmoji: {
    fontSize: 18,
  },
  serviceInfo: {
    flex: 1,
  },
  serviceTitle: {
    fontSize: 12,
    color: theme.colors.textMuted,
  },
  serviceValue: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.textPrimary,
  },
  serviceStatus: {
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  serviceStatusText: {
    fontSize: 11,
    color: theme.colors.success,
    fontWeight: '500',
  },
});

export default GamificationCard;
