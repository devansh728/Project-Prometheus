/**
 * Priority Queue View - Shows urgency-ranked vehicles in queue
 */
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Clock, User, Car, Zap } from 'lucide-react';
import styles from './PriorityQueueView.module.css';

interface QueuedVehicle {
  request_id: string;
  vehicle_id: string;
  vehicle_name: string;
  customer_name: string;
  failure_type: string;
  severity: string;
  urgency_score: number;
  urgency_level: string;
  user_tier: string;
  max_diagnosis_days: number;
  status: string;
  created_at: string;
}

interface PriorityQueueViewProps {
  vehicles: QueuedVehicle[];
  onVehicleClick?: (requestId: string) => void;
  onSimulateBatch?: () => void;
  isLoading?: boolean;
}

const getUrgencyColor = (level: string): string => {
  switch (level.toLowerCase()) {
    case 'critical': return 'var(--color-error)';
    case 'high': return 'var(--color-warning)';
    case 'medium': return 'var(--color-primary)';
    default: return 'var(--color-success)';
  }
};

const getSeverityBadge = (severity: string) => {
  const colors: Record<string, string> = {
    critical: '#ff4757',
    high: '#ffa502',
    medium: '#3498db',
    low: '#2ed573',
  };
  return colors[severity.toLowerCase()] || '#95a5a6';
};

export const PriorityQueueView: React.FC<PriorityQueueViewProps> = ({
  vehicles,
  onVehicleClick,
  onSimulateBatch,
  isLoading,
}) => {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <Zap className={styles.icon} size={20} />
          <h3 className={styles.title}>Priority Queue</h3>
          <span className={styles.count}>{vehicles.length} vehicles</span>
        </div>
        {onSimulateBatch && (
          <button
            className={styles.simulateButton}
            onClick={onSimulateBatch}
            disabled={isLoading}
          >
            {isLoading ? 'Simulating...' : 'Simulate Batch'}
          </button>
        )}
      </div>

      <div className={styles.queueList}>
        <AnimatePresence mode="popLayout">
          {vehicles.map((vehicle, index) => (
            <motion.div
              key={vehicle.request_id}
              className={styles.queueItem}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onVehicleClick?.(vehicle.request_id)}
              style={{
                borderLeft: `4px solid ${getUrgencyColor(vehicle.urgency_level)}`,
              }}
            >
              <div className={styles.position}>#{index + 1}</div>
              
              <div className={styles.vehicleInfo}>
                <div className={styles.vehicleName}>
                  <Car size={14} />
                  {vehicle.vehicle_name}
                </div>
                <div className={styles.customerName}>
                  <User size={12} />
                  {vehicle.customer_name}
                </div>
              </div>

              <div className={styles.failureInfo}>
                <span
                  className={styles.severityBadge}
                  style={{ backgroundColor: getSeverityBadge(vehicle.severity) }}
                >
                  {vehicle.severity}
                </span>
                <span className={styles.failureType}>
                  {vehicle.failure_type.replace(/_/g, ' ')}
                </span>
              </div>

              <div className={styles.urgencySection}>
                <div
                  className={styles.urgencyScore}
                  style={{ color: getUrgencyColor(vehicle.urgency_level) }}
                >
                  {vehicle.urgency_score.toFixed(0)}
                </div>
                <div className={styles.urgencyLabel}>
                  {vehicle.urgency_level}
                </div>
              </div>

              <div className={styles.tierBadge} data-tier={vehicle.user_tier}>
                {vehicle.user_tier}
              </div>

              {vehicle.urgency_level === 'critical' && (
                <AlertTriangle className={styles.criticalIcon} size={16} />
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {vehicles.length === 0 && (
          <div className={styles.emptyState}>
            <Clock size={32} />
            <p>No vehicles in queue</p>
            <span>Click "Simulate Batch" to add demo vehicles</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PriorityQueueView;
