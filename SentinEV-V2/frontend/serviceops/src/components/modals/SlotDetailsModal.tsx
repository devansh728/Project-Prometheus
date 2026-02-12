import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Calendar, User, Wrench, Package, CheckSquare, AlertCircle } from 'lucide-react';
import styles from './SlotDetailsModal.module.css';

export interface SlotDetails {
  vehicleId: string;
  model: string;
  taskName: string;
  technician: string;
  masterTechnician?: string;
  requiredParts: string[];
  status: string;
  time: string;
}

interface SlotDetailsModalProps {
  details: SlotDetails | null;
  onClose: () => void;
}

export const SlotDetailsModal: React.FC<SlotDetailsModalProps> = ({ details, onClose }) => {
  if (!details) return null;

  return (
    <AnimatePresence>
      <div className={styles.overlay} onClick={onClose}>
        <motion.div 
          className={styles.modal}
          onClick={e => e.stopPropagation()}
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
        >
          <button className={styles.closeBtn} onClick={onClose}>
            <X size={20} />
          </button>

          <div className={styles.header}>
            <div className={styles.timeBadge}>
              <Calendar size={14} />
              <span>{details.time}</span>
            </div>
            <h2>{details.taskName}</h2>
            <div className={styles.vehicleInfo}>
              <span className={styles.vehicleId}>{details.vehicleId}</span>
              <span className={styles.model}>{details.model}</span>
            </div>
          </div>

          <div className={styles.content}>
            {/* Technicians */}
            <div className={styles.section}>
              <div className={styles.sectionTitle}>
                <User size={16} />
                <span>Assigned Team</span>
              </div>
              <div className={styles.card}>
                <div className={styles.row}>
                  <span className={styles.label}>Technician:</span>
                  <span className={styles.value}>{details.technician}</span>
                </div>
                {details.masterTechnician && (
                  <div className={styles.row}>
                    <span className={styles.label}>Master Tech:</span>
                    <span className={styles.highlight}>{details.masterTechnician}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Parts */}
            <div className={styles.section}>
              <div className={styles.sectionTitle}>
                <Package size={16} />
                <span>Parts Allocated</span>
              </div>
              <div className={styles.tags}>
                {details.requiredParts.map((part, i) => (
                  <span key={i} className={styles.tag}>{part}</span>
                ))}
              </div>
            </div>

            {/* To-Do List (Mock) */}
            <div className={styles.section}>
              <div className={styles.sectionTitle}>
                <CheckSquare size={16} />
                <span>Task Checklist</span>
              </div>
              <div className={styles.checklist}>
                <div className={`${styles.checkItem} ${styles.checked}`}>
                  <div className={styles.checkbox}>✓</div>
                  <span>Vehicle Inspection</span>
                </div>
                <div className={`${styles.checkItem} ${styles.checked}`}>
                  <div className={styles.checkbox}>✓</div>
                  <span>Diagnostic Scan</span>
                </div>
                <div className={styles.checkItem}>
                  <div className={styles.checkbox}>○</div>
                  <span>Part Replacement</span>
                </div>
                <div className={styles.checkItem}>
                  <div className={styles.checkbox}>○</div>
                  <span>Quality Check</span>
                </div>
              </div>
            </div>

            {/* Status Footer */}
            <div className={styles.footer} data-status={details.status}>
              <AlertCircle size={16} />
              <span>Status: {details.status}</span>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
