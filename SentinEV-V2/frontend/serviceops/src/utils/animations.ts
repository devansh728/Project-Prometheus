/**
 * Animation Utilities - Framer Motion presets for premium web UI
 */
import { Variants, Transition } from 'framer-motion';

// Transition presets
export const transitions = {
  snappy: { type: 'spring', stiffness: 400, damping: 30 } as Transition,
  bouncy: { type: 'spring', stiffness: 300, damping: 15 } as Transition,
  gentle: { type: 'spring', stiffness: 150, damping: 25 } as Transition,
  fast: { duration: 0.15, ease: [0.25, 0.1, 0.25, 1] } as Transition,
  normal: { duration: 0.25, ease: [0.25, 0.1, 0.25, 1] } as Transition,
  slow: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1] } as Transition,
};

// Fade up animation
export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

// Scale in animation
export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: { opacity: 1, scale: 1 },
};

// Slide in from right
export const slideInRight: Variants = {
  hidden: { opacity: 0, x: 30 },
  visible: { opacity: 1, x: 0 },
};

// Stagger container
export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.1,
    },
  },
};

// Stagger item
export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 15 },
  visible: {
    opacity: 1,
    y: 0,
    transition: transitions.gentle,
  },
};

// Card hover effect
export const cardHover = {
  rest: { scale: 1, y: 0 },
  hover: { scale: 1.02, y: -4 },
};

// Button press effect
export const buttonPress = {
  rest: { scale: 1 },
  pressed: { scale: 0.97 },
};

// Glow pulse for alerts
export const glowPulse: Variants = {
  pulse: {
    boxShadow: [
      '0 0 0 0 rgba(0, 217, 255, 0)',
      '0 0 20px 10px rgba(0, 217, 255, 0.3)',
      '0 0 0 0 rgba(0, 217, 255, 0)',
    ],
    transition: { duration: 2, repeat: Infinity },
  },
};

// Page transition
export const pageTransition: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
};

// Sidebar slide
export const sidebarSlide: Variants = {
  collapsed: { width: 72 },
  expanded: { width: 260 },
};

// List item entrance with index-based delay
export const createListItemVariant = (index: number): Variants => ({
  hidden: { opacity: 0, x: -20 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { delay: index * 0.05, ...transitions.gentle },
  },
});

// Number counter animation helper
export const animateNumber = (from: number, to: number, duration = 1000) => {
  return {
    initial: { value: from },
    animate: { value: to },
    transition: { duration: duration / 1000 },
  };
};
