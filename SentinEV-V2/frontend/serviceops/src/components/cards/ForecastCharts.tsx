/**
 * Forecast Charts - Labour & Inventory forecasts
 */
import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Users, Package, AlertTriangle } from 'lucide-react';
import styles from './ForecastCharts.module.css';

interface UtilizationData {
  date: string;
  utilization: number;
  risk: string;
}

interface ReorderRecommendation {
  part: string;
  predicted_demand?: number;
  current_qty?: number;
  action: string;
  urgency: string;
  message: string;
}

interface ForecastChartsProps {
  centerId: string;
  labourForecast: UtilizationData[];
  inventoryRecommendations: ReorderRecommendation[];
}

const getRiskColor = (risk: string): string => {
  switch (risk.toLowerCase()) {
    case 'high': return 'var(--color-error)';
    case 'medium': return 'var(--color-warning)';
    default: return 'var(--color-success)';
  }
};

export const ForecastCharts: React.FC<ForecastChartsProps> = ({
  centerId,
  labourForecast,
  inventoryRecommendations,
}) => {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' });
  };

  const maxUtilization = Math.max(...labourForecast.map(d => d.utilization), 100);

  return (
    <div className={styles.container}>
      {/* Labour Forecast */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <Users size={18} className={styles.icon} />
          <h4 className={styles.sectionTitle}>7-Day Labour Forecast</h4>
        </div>

        <div className={styles.chartContainer}>
          {labourForecast.map((day, index) => (
            <motion.div
              key={day.date}
              className={styles.barColumn}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <div className={styles.barWrapper}>
                <motion.div
                  className={styles.bar}
                  style={{
                    backgroundColor: getRiskColor(day.risk),
                  }}
                  initial={{ height: 0 }}
                  animate={{ height: `${(day.utilization / maxUtilization) * 100}%` }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                />
                {day.risk === 'high' && (
                  <AlertTriangle className={styles.alertIcon} size={12} />
                )}
              </div>
              <span className={styles.utilValue}>{day.utilization.toFixed(0)}%</span>
              <span className={styles.dateLabel}>{formatDate(day.date)}</span>
            </motion.div>
          ))}
        </div>

        <div className={styles.legend}>
          <span className={styles.legendItem}>
            <span className={styles.dot} style={{ background: 'var(--color-success)' }} /> Low
          </span>
          <span className={styles.legendItem}>
            <span className={styles.dot} style={{ background: 'var(--color-warning)' }} /> Medium
          </span>
          <span className={styles.legendItem}>
            <span className={styles.dot} style={{ background: 'var(--color-error)' }} /> High Risk
          </span>
        </div>
      </div>

      {/* Inventory Recommendations */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <Package size={18} className={styles.icon} />
          <h4 className={styles.sectionTitle}>Inventory Recommendations</h4>
        </div>

        <div className={styles.recommendationsList}>
          {inventoryRecommendations.length > 0 ? (
            inventoryRecommendations.map((rec, index) => (
              <motion.div
                key={`${rec.part}-${index}`}
                className={styles.recommendationItem}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <div className={styles.recIcon}>
                  <Package size={14} />
                </div>
                <div className={styles.recContent}>
                  <span className={styles.partName}>{rec.part}</span>
                  <span className={styles.recMessage}>{rec.message}</span>
                </div>
                <span
                  className={styles.urgencyBadge}
                  data-urgency={rec.urgency}
                >
                  {rec.action.replace('_', ' ')}
                </span>
              </motion.div>
            ))
          ) : (
            <div className={styles.emptyRec}>
              <TrendingUp size={24} />
              <span>No immediate reorder needed</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ForecastCharts;
