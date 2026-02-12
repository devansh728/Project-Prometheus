/**
 * Animation Utilities - Shared animation presets for premium UI feel
 */
import Animated, {
  withSpring,
  withTiming,
  withSequence,
  withDelay,
  interpolate,
  Easing,
  SharedValue,
} from 'react-native-reanimated';

// Spring configurations for different feels
export const springConfigs = {
  // Snappy - quick response, minimal overshoot
  snappy: { damping: 20, stiffness: 300, mass: 0.8 },
  // Bouncy - playful, more overshoot
  bouncy: { damping: 12, stiffness: 180, mass: 1 },
  // Gentle - slow, smooth, elegant
  gentle: { damping: 25, stiffness: 120, mass: 1.2 },
  // Stiff - fast, no overshoot
  stiff: { damping: 30, stiffness: 400, mass: 0.6 },
};

// Timing configurations
export const timingConfigs = {
  fast: { duration: 150, easing: Easing.out(Easing.cubic) },
  normal: { duration: 250, easing: Easing.inOut(Easing.cubic) },
  slow: { duration: 400, easing: Easing.inOut(Easing.quad) },
  emphasis: { duration: 500, easing: Easing.bezier(0.34, 1.56, 0.64, 1) },
};

// Pre-built animation presets
export const animations = {
  // Fade in from bottom
  fadeInUp: (value: SharedValue<number>) => ({
    opacity: interpolate(value.value, [0, 1], [0, 1]),
    transform: [{ translateY: interpolate(value.value, [0, 1], [20, 0]) }],
  }),

  // Scale up with fade
  scaleIn: (value: SharedValue<number>) => ({
    opacity: interpolate(value.value, [0, 1], [0, 1]),
    transform: [{ scale: interpolate(value.value, [0, 1], [0.9, 1]) }],
  }),

  // Slide in from right
  slideInRight: (value: SharedValue<number>) => ({
    opacity: interpolate(value.value, [0, 1], [0, 1]),
    transform: [{ translateX: interpolate(value.value, [0, 1], [50, 0]) }],
  }),

  // Pulse effect
  pulse: (value: SharedValue<number>) => ({
    transform: [{ scale: interpolate(value.value, [0, 0.5, 1], [1, 1.05, 1]) }],
  }),
};

// Stagger delay helper
export const getStaggerDelay = (index: number, baseDelay = 50) => index * baseDelay;

// Entrance animation helper
export const createEntranceAnimation = (delay = 0) => {
  'worklet';
  return withDelay(delay, withSpring(1, springConfigs.gentle));
};

// Exit animation helper
export const createExitAnimation = () => {
  'worklet';
  return withTiming(0, timingConfigs.fast);
};

// Micro-interaction: Button press
export const buttonPressAnimation = (pressed: boolean) => {
  'worklet';
  return withSpring(pressed ? 0.96 : 1, springConfigs.snappy);
};

// Micro-interaction: Glow pulse
export const glowPulseAnimation = () => {
  'worklet';
  return withSequence(
    withTiming(1.2, { duration: 300 }),
    withTiming(1, { duration: 300 })
  );
};

// Card hover/focus state
export const cardElevationAnimation = (focused: boolean) => {
  'worklet';
  return {
    scale: withSpring(focused ? 1.02 : 1, springConfigs.snappy),
    shadowOpacity: withTiming(focused ? 0.3 : 0.1, timingConfigs.fast),
  };
};
