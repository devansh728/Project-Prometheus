/**
 * SentinEV Design System Tokens
 * Shared across all frontend applications
 */

export const colors = {
  // Primary Palette
  primary: {
    main: '#0066FF',      // Electric Blue - Trust & Technology
    light: '#3385FF',
    dark: '#0052CC',
    contrast: '#FFFFFF',
  },
  
  // Background
  background: {
    dark: '#1A1A2E',      // Dark Graphite
    card: '#16213E',
    surface: '#0F3460',
    overlay: 'rgba(0, 0, 0, 0.7)',
  },

  // Status Colors (Severity Mapping)
  status: {
    healthy: '#10B981',   // Emerald Green
    warning: '#F59E0B',   // Amber
    critical: '#EF4444',  // Red
    emergency: '#DC2626', // Deep Red
    info: '#3B82F6',      // Blue
  },

  // Text
  text: {
    primary: '#FFFFFF',
    secondary: '#9CA3AF',
    muted: '#6B7280',
    inverse: '#1A1A2E',
  },

  // Borders
  border: {
    default: '#374151',
    focus: '#0066FF',
  }
};

export const typography = {
  fontFamily: {
    sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', monospace",
  },
  
  fontSize: {
    xs: '12px',
    sm: '14px',
    base: '16px',
    lg: '18px',
    xl: '20px',
    '2xl': '24px',
    '3xl': '32px',
    '4xl': '40px',
  },

  fontWeight: {
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  lineHeight: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.75,
  }
};

export const spacing = {
  0: '0',
  1: '4px',
  2: '8px',
  3: '12px',
  4: '16px',
  5: '20px',
  6: '24px',
  8: '32px',
  10: '40px',
  12: '48px',
  16: '64px',
};

export const borderRadius = {
  none: '0',
  sm: '4px',
  md: '8px',
  lg: '16px',
  xl: '24px',
  full: '9999px',
};

export const shadows = {
  sm: '0 1px 2px rgba(0, 0, 0, 0.3)',
  md: '0 4px 6px rgba(0, 0, 0, 0.4)',
  lg: '0 10px 15px rgba(0, 0, 0, 0.5)',
  glow: {
    healthy: '0 0 20px rgba(16, 185, 129, 0.4)',
    warning: '0 0 20px rgba(245, 158, 11, 0.4)',
    critical: '0 0 20px rgba(239, 68, 68, 0.5)',
  }
};

export const animation = {
  duration: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms',
  },
  easing: {
    easeOut: 'cubic-bezier(0.16, 1, 0.3, 1)',
    easeIn: 'cubic-bezier(0.7, 0, 0.84, 0)',
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  }
};

// Severity-to-Style Mapping
export const severityStyles = {
  INFO: {
    color: colors.status.info,
    glow: shadows.glow.healthy,
    pulseSpeed: '3s',
  },
  WARNING: {
    color: colors.status.warning,
    glow: shadows.glow.warning,
    pulseSpeed: '2s',
  },
  CRITICAL: {
    color: colors.status.critical,
    glow: shadows.glow.critical,
    pulseSpeed: '1s',
  },
  EMERGENCY: {
    color: colors.status.emergency,
    glow: shadows.glow.critical,
    pulseSpeed: '0.5s',
  },
};
