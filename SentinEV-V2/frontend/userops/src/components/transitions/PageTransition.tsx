/**
 * Page Transition Wrapper - Smooth screen transitions
 */
import React, { useEffect } from 'react';
import { StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withDelay,
  interpolate,
  FadeIn,
  FadeOut,
  SlideInRight,
  SlideOutLeft,
} from 'react-native-reanimated';

interface PageTransitionProps {
  children: React.ReactNode;
  style?: ViewStyle;
  delay?: number;
  type?: 'fade' | 'slide' | 'scale' | 'fadeUp';
}

export const PageTransition: React.FC<PageTransitionProps> = ({
  children,
  style,
  delay = 0,
  type = 'fadeUp',
}) => {
  const progress = useSharedValue(0);

  useEffect(() => {
    progress.value = withDelay(delay, withSpring(1, { damping: 20, stiffness: 150 }));
  }, []);

  const animatedStyle = useAnimatedStyle(() => {
    switch (type) {
      case 'fade':
        return { opacity: progress.value };
      case 'slide':
        return {
          opacity: progress.value,
          transform: [{ translateX: interpolate(progress.value, [0, 1], [100, 0]) }],
        };
      case 'scale':
        return {
          opacity: progress.value,
          transform: [{ scale: interpolate(progress.value, [0, 1], [0.95, 1]) }],
        };
      case 'fadeUp':
      default:
        return {
          opacity: progress.value,
          transform: [{ translateY: interpolate(progress.value, [0, 1], [30, 0]) }],
        };
    }
  });

  return (
    <Animated.View style={[styles.container, style, animatedStyle]}>
      {children}
    </Animated.View>
  );
};

// Staggered list item animation
interface StaggerItemProps {
  children: React.ReactNode;
  index: number;
  style?: ViewStyle;
}

export const StaggerItem: React.FC<StaggerItemProps> = ({ children, index, style }) => {
  const progress = useSharedValue(0);

  useEffect(() => {
    progress.value = withDelay(index * 60, withSpring(1, { damping: 18, stiffness: 120 }));
  }, [index]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: progress.value,
    transform: [
      { translateY: interpolate(progress.value, [0, 1], [20, 0]) },
      { scale: interpolate(progress.value, [0, 1], [0.97, 1]) },
    ],
  }));

  return (
    <Animated.View style={[style, animatedStyle]}>
      {children}
    </Animated.View>
  );
};

// Export pre-built entering/exiting animations
export const pageEntering = FadeIn.duration(300).springify();
export const pageExiting = FadeOut.duration(200);
export const slideEntering = SlideInRight.duration(300).springify();
export const slideExiting = SlideOutLeft.duration(200);

const styles = StyleSheet.create({
  container: { flex: 1 },
});
