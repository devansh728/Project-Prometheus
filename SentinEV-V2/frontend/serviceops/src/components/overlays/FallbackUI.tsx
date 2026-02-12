import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, AlertTriangle, X } from 'lucide-react';
import styles from './FallbackUI.module.css';

interface FallbackUIProps {
  isActive: boolean;
  reason?: 'timeout' | 'low_confidence' | 'simulated';
  onDismiss?: () => void;
}

export const FallbackUI: React.FC<FallbackUIProps> = ({
  isActive,
  reason = 'simulated',
  onDismiss
}) => {
  const getReasonText = () => {
    switch (reason) {
      case 'timeout':
        return 'Agent response timeout detected';
      case 'low_confidence':
        return 'Agent confidence below safety threshold';
      case 'simulated':
        return 'Simulated failure for demonstration';
      default:
        return 'Unknown fallback trigger';
    }
  };

  return (
    <AnimatePresence>
      {isActive && (
        <motion.div
          className={styles.banner}
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -100, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        >
          <div className={styles.content}>
            <div className={styles.iconSection}>
              <Shield size={24} />
            </div>
            
            <div className={styles.messageSection}>
              <div className={styles.title}>
                Switching to Safe Execution Mode
              </div>
              <div className={styles.reason}>
                <AlertTriangle size={14} />
                <span>{getReasonText()}</span>
              </div>
              <div className={styles.modeIndicator}>
                <span className={styles.crossed}>AI Agent Planning</span>
                <span className={styles.arrow}>→</span>
                <span className={styles.active}>Rule-Based Scheduler</span>
              </div>
            </div>

            {onDismiss && (
              <button onClick={onDismiss} className={styles.dismissBtn}>
                <X size={18} />
              </button>
            )}
          </div>

          <motion.div
            className={styles.progressBar}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
};
