/**
 * Animated Counter Component - Smooth number transitions
 */
import React, { useEffect, useState, useRef } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  className?: string;
}

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  value,
  duration = 1.2,
  prefix = '',
  suffix = '',
  decimals = 0,
  className,
}) => {
  const springValue = useSpring(0, { stiffness: 100, damping: 30 });
  const displayValue = useTransform(springValue, (v) => 
    `${prefix}${v.toFixed(decimals)}${suffix}`
  );
  const [display, setDisplay] = useState(`${prefix}0${suffix}`);

  useEffect(() => {
    springValue.set(value);
    const unsubscribe = displayValue.on('change', (v) => setDisplay(v));
    return unsubscribe;
  }, [value, springValue, displayValue]);

  return (
    <motion.span className={className}>
      {display}
    </motion.span>
  );
};

// Progress bar with animation
interface AnimatedProgressProps {
  value: number;
  max?: number;
  color?: string;
  height?: number;
  showLabel?: boolean;
  className?: string;
}

export const AnimatedProgress: React.FC<AnimatedProgressProps> = ({
  value,
  max = 100,
  color = 'var(--color-primary)',
  height = 8,
  showLabel = false,
  className,
}) => {
  const percentage = Math.min((value / max) * 100, 100);

  return (
    <div 
      className={className}
      style={{
        width: '100%',
        height,
        backgroundColor: 'var(--color-surface)',
        borderRadius: height / 2,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: 0.8, ease: [0.25, 0.1, 0.25, 1] }}
        style={{
          height: '100%',
          backgroundColor: color,
          borderRadius: height / 2,
        }}
      />
      {showLabel && (
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          style={{
            position: 'absolute',
            right: 8,
            top: '50%',
            transform: 'translateY(-50%)',
            fontSize: 10,
            fontWeight: 600,
            color: percentage > 50 ? '#0A0E1A' : 'var(--color-text-secondary)',
          }}
        >
          {Math.round(percentage)}%
        </motion.span>
      )}
    </div>
  );
};

// Shimmer skeleton loader
interface SkeletonProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = 20,
  borderRadius = 6,
  className,
}) => {
  return (
    <div
      className={className}
      style={{
        width,
        height,
        borderRadius,
        background: 'linear-gradient(90deg, var(--color-surface) 0%, var(--color-surface-elevated) 50%, var(--color-surface) 100%)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s infinite',
      }}
    />
  );
};

// Add shimmer keyframes via CSS injection
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
  `;
  document.head.appendChild(style);
}
