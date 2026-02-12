/**
 * Metric Card Component - Animated KPI display
 */
import React from 'react';
import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';
import styles from './MetricCard.module.css';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  color: 'primary' | 'success' | 'warning' | 'danger' | 'info';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color,
}) => {
  const colorMap = {
    primary: 'var(--color-primary)',
    success: 'var(--color-success)',
    warning: 'var(--color-warning)',
    danger: 'var(--color-danger)',
    info: 'var(--color-info)',
  };

  return (
    <motion.div
      className={styles.card}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      transition={{ duration: 0.3 }}
    >
      <div className={styles.header}>
        <span className={styles.title}>{title}</span>
        <div
          className={styles.iconContainer}
          style={{ backgroundColor: `${colorMap[color]}20` }}
        >
          <Icon size={18} style={{ color: colorMap[color] }} />
        </div>
      </div>

      <div className={styles.content}>
        <span className={styles.value}>{value}</span>
        {trend && (
          <span
            className={`${styles.trend} ${trend.isPositive ? styles.positive : styles.negative}`}
          >
            {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
          </span>
        )}
      </div>

      {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
    </motion.div>
  );
};
