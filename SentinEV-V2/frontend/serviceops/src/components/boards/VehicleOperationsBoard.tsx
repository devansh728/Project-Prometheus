import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  MapPin, 
  Wrench, 
  Package,
  User,
  TrendingUp
} from 'lucide-react';
import styles from './VehicleOperationsBoard.module.css';

export type DecisionState = 'PENDING' | 'ROUTING' | 'BIDDING' | 'ASSIGNED';

export interface VehicleCard {
  vehicleId: string;
  customerId: string;
  customerName: string;
  location: { lat: number; lon: number; address: string };
  failureType: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  urgencyScore: number;
  rul: number; // Remaining Useful Life (days)
  failureProbability: number;
  maxDiagnosisDays: number;
  requiredSkills: string[];
  requiredParts: string[];
  userTier: 'STANDARD' | 'PREMIUM' | 'VIP';
  preferredDates?: string[];
  historicalCenter?: string;
  decisionState: DecisionState;
  agentNotes: string[];
  assignedCenterId?: string;
  assignedCenterName?: string;
  assignedTechnician?: string;
  masterTechnician?: string;
  bidHistory?: Array<{ center: string; score: number; cost: number; eta: string }>;
}

interface VehicleOperationsBoardProps {
  vehicles: VehicleCard[];
  animatingStep?: number;
  onVehicleClick?: (vehicle: VehicleCard) => void;
}

export const VehicleOperationsBoard: React.FC<VehicleOperationsBoardProps> = ({
  vehicles,
  animatingStep = -1,
  onVehicleClick
}) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return '#ef4444';
      case 'HIGH': return '#f59e0b';
      case 'MEDIUM': return '#eab308';
      case 'LOW': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getStateLabel = (state: DecisionState) => {
    switch (state) {
      case 'PENDING': return 'Awaiting Planning';
      case 'ROUTING': return 'Evaluating Centers';
      case 'BIDDING': return 'Running Auction';
      case 'ASSIGNED': return 'Scheduled';
      default: return 'Unknown';
    }
  };

  const getStateColor = (state: DecisionState) => {
    switch (state) {
      case 'PENDING': return '#6b7280';
      case 'ROUTING': return '#3b82f6';
      case 'BIDDING': return '#f59e0b';
      case 'ASSIGNED': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getTierBadgeColor = (tier: string) => {
    switch (tier) {
      case 'VIP': return '#8b5cf6';
      case 'PREMIUM': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  return (
    <div className={styles.board}>
      <div className={styles.header}>
        <h2>Vehicle Operations Board</h2>
        <div className={styles.stats}>
          <span>{vehicles.length} Active Requests</span>
          <span>{vehicles.filter(v => v.decisionState === 'ASSIGNED').length} Scheduled</span>
        </div>
      </div>

      <div className={styles.grid}>
        <AnimatePresence mode="popLayout">
          {vehicles.map((vehicle, index) => (
            <motion.div
              key={vehicle.vehicleId}
              className={styles.card}
              initial={{ opacity: 0, y: 20 }}
              animate={{ 
                opacity: 1, 
                y: 0,
                scale: animatingStep === 2 && vehicle.decisionState === 'BIDDING' ? 1.02 : 1
              }}
              exit={{ opacity: 0, scale: 0.95 }}
              layout
              transition={{ duration: 0.3, delay: index * 0.05 }}
              onClick={() => onVehicleClick?.(vehicle)}
              style={{
                borderLeft: `4px solid ${getSeverityColor(vehicle.severity)}`
              }}
            >
              {/* Status Strip */}
              <div 
                className={styles.statusStrip}
                style={{ backgroundColor: getSeverityColor(vehicle.severity) }}
              >
                <span>{vehicle.severity}</span>
              </div>

              {/* Header */}
              <div className={styles.cardHeader}>
                <div className={styles.vehicleInfo}>
                  <h3>{vehicle.vehicleId}</h3>
                  <div className={styles.owner}>
                    <User size={14} />
                    <span>{vehicle.customerName}</span>
                  </div>
                </div>
                <div 
                  className={styles.tierBadge}
                  style={{ backgroundColor: getTierBadgeColor(vehicle.userTier) }}
                >
                  {vehicle.userTier}
                </div>
              </div>

              {/* Location */}
              <div className={styles.location}>
                <MapPin size={14} />
                <span>{vehicle.location.address}</span>
              </div>

              {/* AI Summary */}
              <div className={styles.aiSummary}>
                <div className={styles.summaryTitle}>AI Analysis</div>
                <div className={styles.metrics}>
                  <div className={styles.metric}>
                    <Clock size={14} />
                    <span>RUL: {vehicle.rul} days</span>
                  </div>
                  <div className={styles.metric}>
                    <AlertTriangle size={14} />
                    <span>Risk: {(vehicle.failureProbability * 100).toFixed(0)}%</span>
                  </div>
                  <div className={styles.metric}>
                    <TrendingUp size={14} />
                    <span>Urgency: {vehicle.urgencyScore.toFixed(1)}</span>
                  </div>
                  <div className={styles.metric}>
                    <Clock size={14} />
                    <span>Max: {vehicle.maxDiagnosisDays}d</span>
                  </div>
                </div>
              </div>

              {/* Constraints */}
              <div className={styles.constraints}>
                <div className={styles.constraint}>
                  <Wrench size={14} />
                  <div className={styles.tags}>
                    {vehicle.requiredSkills.map(skill => (
                      <span key={skill} className={styles.tag}>{skill}</span>
                    ))}
                  </div>
                </div>
                <div className={styles.constraint}>
                  <Package size={14} />
                  <div className={styles.tags}>
                    {vehicle.requiredParts.map(part => (
                      <span key={part} className={styles.tag}>{part}</span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Agent Notes */}
              {vehicle.agentNotes.length > 0 && (
                <div className={styles.agentNotes}>
                  <div className={styles.notesTitle}>Agent Observations</div>
                  {vehicle.agentNotes.map((note, i) => (
                    <div key={i} className={styles.note}>
                      <span className={styles.noteDot}>•</span>
                      <span>{note}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Assignment Details */}
              {vehicle.decisionState === 'ASSIGNED' && (
                <motion.div 
                  className={styles.assignmentDetails}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.5 }}
                >
                  <div className={styles.assignmentRow}>
                    <span className={styles.label}>Center:</span>
                    <span className={styles.value}>{vehicle['assignedCenterName'] || 'Service Center 1'}</span>
                  </div>
                  <div className={styles.assignmentRow}>
                    <span className={styles.label}>Technician:</span>
                    <span className={styles.value}>{vehicle['assignedTechnician'] || 'Assigned'}</span>
                  </div>
                  {vehicle['masterTechnician'] && (
                    <div className={styles.assignmentRow}>
                      <span className={styles.label}>Master Tech:</span>
                      <span className={styles.value} style={{ color: '#a78bfa' }}>{vehicle['masterTechnician']}</span>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Decision State */}
              <motion.div 
                className={styles.decisionState}
                style={{ backgroundColor: getStateColor(vehicle.decisionState) }}
                animate={{
                  opacity: [1, 0.7, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: vehicle.decisionState !== 'ASSIGNED' ? Infinity : 0,
                  ease: "easeInOut"
                }}
              >
                {vehicle.decisionState === 'PENDING' && <Clock size={14} />}
                {vehicle.decisionState === 'ROUTING' && <MapPin size={14} />}
                {vehicle.decisionState === 'BIDDING' && <TrendingUp size={14} />}
                {vehicle.decisionState === 'ASSIGNED' && <CheckCircle size={14} />}
                <span>{getStateLabel(vehicle.decisionState)}</span>
              </motion.div>

              {/* Loading Indicator */}
              {vehicle.decisionState !== 'ASSIGNED' && (
                <motion.div 
                  className={styles.evaluating}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  ServiceOpsAI evaluating...
                </motion.div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};
