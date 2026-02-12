/**
 * Simulation Controls - Demo scenario trigger buttons
 */
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, RefreshCw, Zap, AlertTriangle, Loader2 } from 'lucide-react';
import styles from './SimulationControls.module.css';

interface SimulationControlsProps {
  onRunScenario1: () => Promise<void>;
  onRunScenario2: () => Promise<void>;
  onReset: () => Promise<void>;
  currentScenario?: string;
  isRunning?: boolean;
  currentStep?: number;
  totalSteps?: number;
  stepDescription?: string;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  onRunScenario1,
  onRunScenario2,
  onReset,
  currentScenario,
  isRunning,
  currentStep,
  totalSteps,
  stepDescription,
}) => {
  const [loadingAction, setLoadingAction] = useState<string | null>(null);

  const handleAction = async (action: string, callback: () => Promise<void>) => {
    setLoadingAction(action);
    try {
      await callback();
    } finally {
      setLoadingAction(null);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <Play className={styles.icon} size={20} />
        <h3 className={styles.title}>Demo Simulation</h3>
        {isRunning && (
          <motion.div
            className={styles.runningBadge}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <Loader2 className={styles.spinner} size={12} />
            Running
          </motion.div>
        )}
      </div>

      <div className={styles.scenarios}>
        <motion.button
          className={`${styles.scenarioButton} ${styles.scenario1}`}
          onClick={() => handleAction('scenario1', onRunScenario1)}
          disabled={isRunning || loadingAction === 'scenario1'}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Zap size={24} />
          <div className={styles.scenarioInfo}>
            <span className={styles.scenarioTitle}>Scenario 1</span>
            <span className={styles.scenarioSubtitle}>Proactive Service at Scale</span>
          </div>
          {loadingAction === 'scenario1' && <Loader2 className={styles.buttonSpinner} />}
        </motion.button>

        <motion.button
          className={`${styles.scenarioButton} ${styles.scenario2}`}
          onClick={() => handleAction('scenario2', onRunScenario2)}
          disabled={isRunning || loadingAction === 'scenario2'}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <AlertTriangle size={24} />
          <div className={styles.scenarioInfo}>
            <span className={styles.scenarioTitle}>Scenario 2</span>
            <span className={styles.scenarioSubtitle}>Urgent Arrival + Stress</span>
          </div>
          {loadingAction === 'scenario2' && <Loader2 className={styles.buttonSpinner} />}
        </motion.button>
      </div>

      <motion.button
        className={styles.resetButton}
        onClick={() => handleAction('reset', onReset)}
        disabled={loadingAction === 'reset'}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <RefreshCw size={16} className={loadingAction === 'reset' ? styles.spinning : ''} />
        Reset Demo
      </motion.button>

      {isRunning && currentStep && totalSteps && (
        <motion.div
          className={styles.progress}
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <div className={styles.progressBar}>
            <motion.div
              className={styles.progressFill}
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <div className={styles.progressInfo}>
            <span>Step {currentStep} of {totalSteps}</span>
            {stepDescription && (
              <span className={styles.stepDesc}>{stepDescription}</span>
            )}
          </div>
        </motion.div>
      )}

      <div className={styles.instructions}>
        <h4>How to Demo:</h4>
        <ol>
          <li><strong>Scenario 1:</strong> Shows multi-vehicle queue, priority scoring, center bidding, and task planning</li>
          <li><strong>Scenario 2:</strong> Shows VIP preemption, fast-track processing, and diagnosis feedback</li>
          <li>Watch the <strong>Decision Feed</strong> for real-time agent reasoning</li>
        </ol>
      </div>
    </div>
  );
};

export default SimulationControls;
