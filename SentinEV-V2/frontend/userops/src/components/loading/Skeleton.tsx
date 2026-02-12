/**
 * Skeleton Loader - Shimmer effect for loading states
 */
import React, { useEffect } from 'react';
import { View, StyleSheet, Dimensions, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface SkeletonProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  style?: ViewStyle;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = 20,
  borderRadius = 8,
  style,
}) => {
  const shimmerX = useSharedValue(-1);

  useEffect(() => {
    shimmerX.value = withRepeat(
      withTiming(1, { duration: 1200 }),
      -1,
      false
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: interpolate(shimmerX.value, [-1, 1], [-SCREEN_WIDTH, SCREEN_WIDTH]) },
    ],
  }));

  return (
    <View
      style={[
        styles.skeleton,
        { width: width as any, height, borderRadius },
        style,
      ]}
    >
      <Animated.View style={[styles.shimmer, animatedStyle]}>
        <LinearGradient
          colors={['transparent', 'rgba(255,255,255,0.1)', 'transparent']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.gradient}
        />
      </Animated.View>
    </View>
  );
};

// Card skeleton preset
export const CardSkeleton: React.FC = () => (
  <View style={styles.cardSkeleton}>
    <View style={styles.cardHeader}>
      <Skeleton width={48} height={48} borderRadius={24} />
      <View style={styles.cardHeaderText}>
        <Skeleton width={120} height={14} />
        <Skeleton width={80} height={12} style={{ marginTop: 6 }} />
      </View>
    </View>
    <Skeleton height={60} style={{ marginTop: 16 }} />
    <View style={styles.cardFooter}>
      <Skeleton width={80} height={12} />
      <Skeleton width={60} height={12} />
    </View>
  </View>
);

// List item skeleton preset
export const ListItemSkeleton: React.FC = () => (
  <View style={styles.listItem}>
    <Skeleton width={40} height={40} borderRadius={8} />
    <View style={styles.listItemContent}>
      <Skeleton width="70%" height={14} />
      <Skeleton width="50%" height={12} style={{ marginTop: 6 }} />
    </View>
    <Skeleton width={24} height={24} borderRadius={12} />
  </View>
);

const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: '#1C2333',
    overflow: 'hidden',
  },
  shimmer: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: '100%',
  },
  gradient: {
    flex: 1,
    width: 100,
  },
  cardSkeleton: {
    backgroundColor: '#141925',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  cardHeaderText: {
    marginLeft: 12,
    flex: 1,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#141925',
    borderRadius: 12,
    marginBottom: 8,
  },
  listItemContent: {
    flex: 1,
    marginLeft: 12,
  },
});
