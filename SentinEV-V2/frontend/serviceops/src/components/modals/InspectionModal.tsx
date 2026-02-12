import React, { useState } from 'react';
import { motion,AnimatePresence } from 'framer-motion';
import { 
  X, 
  Mic, 
  CheckCircle, 
  Clock, 
  ArrowRight,
  Activity
} from 'lucide-react';
import styles from './InspectionModal.module.css';
import type { VehicleCard } from '../boards/VehicleOperationsBoard';

interface InspectionModalProps {
  vehicle: VehicleCard | null;
  onClose: () => void;
  onSubmit: (actual: string) => Promise<{
    similarityScore: number;
    durationDelta: number;
    affectedTasks: number;
  }>;
}

export const InspectionModal: React.FC<InspectionModalProps> = ({
  vehicle,
  onClose,
  onSubmit
}) => {
  const [actualDiagnosis, setActualDiagnosis] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const handleSubmit = async () => {
    if (!actualDiagnosis.trim()) return;
    
    setProcessing(true);
    setIsSubmitted(true);
    
    // Simulate agent flow steps
    const steps = [
      { name: 'Verification Agent', duration: 1500 },
      { name: 'Planner Agent', duration: 1200 },
      { name: 'Rescheduling Agent', duration: 1800 },
      { name: 'Learning Signal', duration: 800 }
    ];

    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      await new Promise(resolve => setTimeout(resolve, steps[i].duration));
    }

    const response = await onSubmit(actualDiagnosis);
    setResult(response);
    setProcessing(false);
    setCurrentStep(steps.length);
  };

  if (!vehicle) return null;

  return (
    <AnimatePresence>
      <motion.div
        className={styles.overlay}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className={styles.modal}
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 20 }}
          onClick={e => e.stopPropagation()}
        >
          <div className={styles.header}>
            <div>
              <h2>Master Technician Inspection</h2>
              <p>{vehicle.vehicleId} - {vehicle.customerName}</p>
            </div>
            <button onClick={onClose} className={styles.closeBtn}>
              <X size={20} />
            </button>
          </div>

          <div className={styles.content}>
            {/* Predicted Diagnosis */}
            <div className={styles.section}>
              <h3>AI Predicted Diagnosis</h3>
              <div className={styles.predicted}>
                <div className={styles.diagnosisItem}>
                  <span className={styles.label}>Failure Type:</span>
                  <span className={styles.value}>{vehicle.failureType}</span>
                </div>
                <div className={styles.diagnosisItem}>
                  <span className={styles.label}>Severity:</span>
                  <span className={styles.value}>{vehicle.severity}</span>
                </div>
                <div className={styles.diagnosisItem}>
                  <span className={styles.label}>Required Parts:</span>
                  <span className={styles.value}>{vehicle.requiredParts.join(', ')}</span>
                </div>
              </div>
            </div>

            {/* Actual Findings Input */}
            <div className={styles.section}>
              <h3>Master Technician Findings</h3>
              <textarea
                className={styles.textarea}
                placeholder="Describe actual findings after physical inspection..."
                value={actualDiagnosis}
                onChange={e => setActualDiagnosis(e.target.value)}
                rows={4}
                disabled={isSubmitted}
              />
              <div className={styles.inputActions}>
                <button className={styles.voiceBtn} disabled={isSubmitted}>
                  <Mic size={16} />
                  <span>Voice Record</span>
                </button>
                <span className={styles.voiceNote}>(Mock - Visual Only)</span>
              </div>
            </div>

            {!isSubmitted ? (
              <button 
                className={styles.submitBtn}
                onClick={handleSubmit}
                disabled={!actualDiagnosis.trim()}
              >
                Submit Inspection Report
                <ArrowRight size={16} />
              </button>
            ) : (
              <div className={styles.agentFlow}>
                <h3>Agent Processing Flow</h3>
                
                {/* Step 1: Verification */}
                <div className={`${styles.flowStep} ${currentStep >= 0 ? styles.active : ''}`}>
                  <div className={styles.stepHeader}>
                    <div className={styles.stepIcon}>
                      {currentStep > 0 ? <CheckCircle size={20} /> : <Activity size={20} />}
                    </div>
                    <div className={styles.stepInfo}>
                      <h4>Verification Agent</h4>
                      <p>Comparing predicted vs actual diagnosis</p>
                    </div>
                  </div>
                  {currentStep > 0 && result && (
                    <motion.div 
                      className={styles.stepResult}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                    >
                      <div className={styles.similarityGauge}>
                        <div className={styles.gaugeLabel}>Similarity Score</div>
                        <motion.div 
                          className={styles.gaugeBar}
                          initial={{ width: 0 }}
                          animate={{ width: '100%' }}
                          transition={{ duration: 1 }}
                        >
                          <motion.div
                            className={styles.gaugeFill}
                            initial={{ width: 0 }}
                            animate={{ width: `${result.similarityScore * 100}%` }}
                            transition={{ duration: 1, delay: 0.3 }}
                            style={{
                              backgroundColor: result.similarityScore > 0.7 ? '#10b981' : 
                                              result.similarityScore > 0.4 ? '#f59e0b' : '#ef4444'
                            }}
                          />
                        </motion.div>
                        <div className={styles.gaugeValue}>
                          {(result.similarityScore * 100).toFixed(0)}%
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Step 2: Planner */}
                <div className={`${styles.flowStep} ${currentStep >= 1 ? styles.active : ''}`}>
                  <div className={styles.stepHeader}>
                    <div className={styles.stepIcon}>
                      {currentStep > 1 ? <CheckCircle size={20} /> : <Activity size={20} />}
                    </div>
                    <div className={styles.stepInfo}>
                      <h4>Planner Agent</h4>
                      <p>Detecting task duration changes</p>
                    </div>
                  </div>
                  {currentStep > 1 && result && (
                    <motion.div 
                      className={styles.stepResult}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                    >
                      <div className={styles.deltaInfo}>
                        <Clock size={16} />
                        <span>Duration delta: <strong>+{result.durationDelta.toFixed(1)} hours</strong></span>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Step 3: Rescheduling */}
                <div className={`${styles.flowStep} ${currentStep >= 2 ? styles.active : ''}`}>
                  <div className={styles.stepHeader}>
                    <div className={styles.stepIcon}>
                      {currentStep > 2 ? <CheckCircle size={20} /> : <Activity size={20} />}
                    </div>
                    <div className={styles.stepInfo}>
                      <h4>Rescheduling Agent</h4>
                      <p>Updating timetable and reassigning tasks</p>
                    </div>
                  </div>
                  {currentStep > 2 && result && (
                    <motion.div 
                      className={styles.stepResult}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                    >
                      <div className={styles.rippleEffect}>
                        <svg width="100%" height="40" viewBox="0 0 200 40">
                          <motion.circle
                            cx="100"
                            cy="20"
                            r="5"
                            fill="#3b82f6"
                            initial={{ scale: 0 }}
                            animate={{ scale: [0, 1, 1.5, 2] }}
                            transition={{ duration: 1.5, times: [0, 0.3, 0.6, 1] }}
                          />
                          <motion.circle
                            cx="100"
                            cy="20"
                            r="10"
                            fill="none"
                            stroke="#3b82f6"
                            strokeWidth="2"
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 3, opacity: [0, 1, 0] }}
                            transition={{ duration: 1.5 }}
                          />
                          <motion.circle
                            cx="100"
                            cy="20"
                            r="15"
                            fill="none"
                            stroke="#3b82f6"
                            strokeWidth="1"
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 2.5, opacity: [0, 0.7, 0] }}
                            transition={{ duration: 1.5, delay: 0.3 }}
                          />
                        </svg>
                        <span>{result.affectedTasks} tasks affected</span>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Step 4: Learning */}
                <div className={`${styles.flowStep} ${currentStep >= 3 ? styles.active : ''}`}>
                  <div className={styles.stepHeader}>
                    <div className={styles.stepIcon}>
                      {currentStep > 3 ? <CheckCircle size={20} /> : <Activity size={20} />}
                    </div>
                    <div className={styles.stepInfo}>
                      <h4>Learning Signal</h4>
                      <p>Sending feedback to SentinEV Global Master</p>
                    </div>
                  </div>
                  {currentStep > 3 && (
                    <motion.div 
                      className={styles.stepResult}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                    >
                      <div className={styles.successMessage}>
                        <CheckCircle size={16} color="#10b981" />
                        <span>Feedback successfully transmitted</span>
                      </div>
                    </motion.div>
                  )}
                </div>

                {currentStep >= 4 && (
                  <motion.button
                    className={styles.doneBtn}
                    onClick={onClose}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <CheckCircle size={16} />
                    Complete Inspection
                  </motion.button>
                )}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
