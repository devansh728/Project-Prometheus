/**
 * Vehicle Lifecycle Tracker - Shows service stages for vehicles
 * Stages: Booked → Check-in → Diagnosis → Repair → Ready
 */
import React from 'react';
import { motion } from 'framer-motion';
import { Car, CheckCircle, Search, Wrench, ThumbsUp, ChevronRight } from 'lucide-react';
import styles from './VehicleLifecycleTracker.module.css';

type ServiceStage = 'BOOKED' | 'CHECK_IN' | 'DIAGNOSIS' | 'REPAIR' | 'READY';

interface TrackedVehicle {
  vehicleId: string;
  vehicleName: string;
  customerName: string;
  currentStage: ServiceStage;
  estimatedCompletion: string;
  serviceType: string;
  bookingTime: string;
}

interface VehicleLifecycleTrackerProps {
  vehicles: TrackedVehicle[];
  onAdvanceStage?: (vehicleId: string, newStage: ServiceStage) => void;
  onSelectVehicle?: (vehicleId: string) => void;
  demoMode?: boolean;
}

const STAGES: { key: ServiceStage; label: string; icon: React.ElementType }[] = [
  { key: 'BOOKED', label: 'Booked', icon: Car },
  { key: 'CHECK_IN', label: 'Check-in', icon: CheckCircle },
  { key: 'DIAGNOSIS', label: 'Diagnosis', icon: Search },
  { key: 'REPAIR', label: 'Repair', icon: Wrench },
  { key: 'READY', label: 'Ready', icon: ThumbsUp },
];

const stageIndex = (stage: ServiceStage) => STAGES.findIndex(s => s.key === stage);

export const VehicleLifecycleTracker: React.FC<VehicleLifecycleTrackerProps> = ({
  vehicles,
  onAdvanceStage,
  onSelectVehicle,
  demoMode = false,
}) => {
  const getNextStage = (current: ServiceStage): ServiceStage | null => {
    const idx = stageIndex(current);
    if (idx < STAGES.length - 1) return STAGES[idx + 1].key;
    return null;
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3 className={styles.title}>Vehicle Service Lifecycle</h3>
        <span className={styles.count}>{vehicles.length} vehicles</span>
      </div>

      <div className={styles.vehicleList}>
        {vehicles.map((vehicle) => {
          const currentIdx = stageIndex(vehicle.currentStage);
          const nextStage = getNextStage(vehicle.currentStage);

          return (
            <motion.div
              key={vehicle.vehicleId}
              className={styles.vehicleCard}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              onClick={() => onSelectVehicle?.(vehicle.vehicleId)}
            >
              {/* Vehicle Info */}
              <div className={styles.vehicleInfo}>
                <div className={styles.vehicleIcon}>🚗</div>
                <div className={styles.vehicleDetails}>
                  <span className={styles.vehicleName}>{vehicle.vehicleName}</span>
                  <span className={styles.customerName}>{vehicle.customerName}</span>
                  <span className={styles.serviceType}>{vehicle.serviceType}</span>
                </div>
              </div>

              {/* Stage Progress */}
              <div className={styles.stageProgress}>
                {STAGES.map((stage, idx) => {
                  const Icon = stage.icon;
                  const isCompleted = idx < currentIdx;
                  const isCurrent = idx === currentIdx;
                  
                  return (
                    <React.Fragment key={stage.key}>
                      <div className={`${styles.stageItem} ${isCompleted ? styles.completed : ''} ${isCurrent ? styles.current : ''}`}>
                        <div className={styles.stageIconWrapper}>
                          <Icon size={16} />
                        </div>
                        <span className={styles.stageLabel}>{stage.label}</span>
                      </div>
                      {idx < STAGES.length - 1 && (
                        <div className={`${styles.connector} ${isCompleted ? styles.connectorCompleted : ''}`} />
                      )}
                    </React.Fragment>
                  );
                })}
              </div>

              {/* Footer */}
              <div className={styles.vehicleFooter}>
                <span className={styles.eta}>ETA: {vehicle.estimatedCompletion}</span>
                
                {/* Demo: Advance Stage Button */}
                {demoMode && nextStage && (
                  <button 
                    className={styles.advanceBtn}
                    onClick={(e) => {
                      e.stopPropagation();
                      onAdvanceStage?.(vehicle.vehicleId, nextStage);
                    }}
                  >
                    Advance to {STAGES.find(s => s.key === nextStage)?.label}
                    <ChevronRight size={14} />
                  </button>
                )}
              </div>
            </motion.div>
          );
        })}

        {vehicles.length === 0 && (
          <div className={styles.emptyState}>
            <span>No vehicles currently in service</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default VehicleLifecycleTracker;
