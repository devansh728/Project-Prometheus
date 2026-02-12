/**
 * Job Card Component - Service job display with status badge
 */
import React from 'react';
import { motion } from 'framer-motion';
import { Clock, User, AlertTriangle, Wrench } from 'lucide-react';
import { ServiceJob, ServiceState } from '../../types/api';
import styles from './JobCard.module.css';

interface JobCardProps {
  job: ServiceJob;
  onClick?: () => void;
  compact?: boolean;
}

const stateColors: Record<ServiceState, string> = {
  [ServiceState.REQUESTED]: 'var(--color-requested)',
  [ServiceState.BOOKED]: 'var(--color-booked)',
  [ServiceState.CONFIRMED]: 'var(--color-confirmed)',
  [ServiceState.CHECK_IN]: 'var(--color-checkin)',
  [ServiceState.DIAGNOSIS]: 'var(--color-diagnosis)',
  [ServiceState.PARTS_ALLOCATED]: 'var(--color-parts)',
  [ServiceState.REPAIR_IN_PROGRESS]: 'var(--color-repair)',
  [ServiceState.QUALITY_CHECK]: 'var(--color-quality)',
  [ServiceState.READY]: 'var(--color-ready)',
  [ServiceState.COMPLETED]: 'var(--color-completed)',
  [ServiceState.CANCELLED]: 'var(--color-cancelled)',
};

const severityColors: Record<string, string> = {
  INFO: 'var(--color-info)',
  WARNING: 'var(--color-warning)',
  CRITICAL: 'var(--color-danger)',
  EMERGENCY: 'var(--color-danger)',
};

export const JobCard: React.FC<JobCardProps> = ({ job, onClick, compact }) => {
  const stateColor = stateColors[job.state] || 'var(--color-text-muted)';
  const severityColor = severityColors[job.severity] || 'var(--color-info)';

  const formatTime = (dateStr: string | null) => {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatFailureType = (type: string) => {
    return type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <motion.div
      className={`${styles.card} ${compact ? styles.compact : ''}`}
      onClick={onClick}
      whileHover={{ y: -2, boxShadow: 'var(--shadow-md)' }}
      whileTap={{ scale: 0.99 }}
      layout
    >
      {/* Header */}
      <div className={styles.header}>
        <span className={styles.jobId}>{job.job_id}</span>
        <div
          className={styles.stateBadge}
          style={{ backgroundColor: `${stateColor}20`, color: stateColor }}
        >
          {job.state.replace(/_/g, ' ')}
        </div>
      </div>

      {/* Vehicle Info */}
      <div className={styles.vehicleInfo}>
        <span className={styles.vehicleId}>{job.vehicle_id}</span>
        <div className={styles.severity} style={{ color: severityColor }}>
          <AlertTriangle size={14} />
          {job.severity}
        </div>
      </div>

      {/* Failure Type */}
      <div className={styles.failureType}>
        <Wrench size={14} />
        {formatFailureType(job.failure_type)}
      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <div className={styles.footerItem}>
          <Clock size={14} />
          <span>{formatTime(job.scheduled_at)}</span>
        </div>
        {job.mechanic_id && (
          <div className={styles.footerItem}>
            <User size={14} />
            <span>{job.mechanic_id}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};
