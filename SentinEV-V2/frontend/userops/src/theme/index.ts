/**
 * SentinEV Theme - Premium Dark Mode with Vibrant Accents
 */

export const theme = {
  colors: {
    // Base
    background: '#0A0E1A',
    surface: '#141925',
    surfaceElevated: '#1C2333',
    card: '#1A1F2E',
    cardGradientStart: '#1E2640',
    cardGradientEnd: '#141925',

    // Text
    textPrimary: '#FFFFFF',
    textSecondary: '#A0AEC0',
    textMuted: '#64748B',

    // Brand / Accents
    primary: '#00D9FF',
    primaryDark: '#0099CC',
    secondary: '#7C3AED',
    accent: '#F97316', // Orange accent
    info: '#3B82F6', // Blue info color
    
    // Semantic
    success: '#10B981',
    successSoft: 'rgba(16, 185, 129, 0.15)',
    warning: '#F59E0B',
    warningSoft: 'rgba(245, 158, 11, 0.15)',
    danger: '#EF4444',
    dangerSoft: 'rgba(239, 68, 68, 0.15)',
    
    // Severity
    severityInfo: '#3B82F6',
    severityWarning: '#F59E0B',
    severityCritical: '#EF4444',
    severityEmergency: '#DC2626',

    // Gradients
    gradientPrimary: ['#00D9FF', '#7C3AED'],
    gradientSuccess: ['#10B981', '#059669'],
    gradientWarning: ['#F59E0B', '#D97706'],
    gradientDanger: ['#EF4444', '#DC2626'],
    gradientCard: ['#1E2640', '#141925'],
    
    // Glass
    glassBg: 'rgba(255, 255, 255, 0.05)',
    glassBorder: 'rgba(255, 255, 255, 0.1)',
  },

  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },

  borderRadius: {
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    full: 9999,
  },

  typography: {
    h1: {
      fontSize: 32,
      fontWeight: '700' as const,
      lineHeight: 40,
    },
    h2: {
      fontSize: 24,
      fontWeight: '600' as const,
      lineHeight: 32,
    },
    h3: {
      fontSize: 20,
      fontWeight: '600' as const,
      lineHeight: 28,
    },
    body: {
      fontSize: 16,
      fontWeight: '400' as const,
      lineHeight: 24,
    },
    bodySmall: {
      fontSize: 14,
      fontWeight: '400' as const,
      lineHeight: 20,
    },
    caption: {
      fontSize: 12,
      fontWeight: '500' as const,
      lineHeight: 16,
    },
  },

  shadows: {
    sm: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 2,
    },
    md: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 8,
      elevation: 4,
    },
    lg: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 0.2,
      shadowRadius: 16,
      elevation: 8,
    },
    glow: (color: string) => ({
      shadowColor: color,
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.5,
      shadowRadius: 16,
      elevation: 8,
    }),
  },
};

export type Theme = typeof theme;
export default theme;
