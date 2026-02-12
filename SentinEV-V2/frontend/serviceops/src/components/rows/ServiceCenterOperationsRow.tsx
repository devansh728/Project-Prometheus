import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  MapPin, 
  Star, 
  Users, 
  Package, 
  TrendingUp, 
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';
import styles from './ServiceCenterOperationsRow.module.css';
import { SlotDetailsModal, SlotDetails } from '../modals/SlotDetailsModal';

export interface TimeBlock {
  hour: number;
  date: string;
  status: 'AVAILABLE' | 'TENTATIVE' | 'RESERVED' | 'CONFIRMED';
  details?: SlotDetails;
}

export interface ServiceCenter {
  centerId: string;
  name: string;
  location: string;
  rating: number;
  status: 'STABLE' | 'OVERLOADED';
  currentLoad: number;
  forecastedLoad: number;
  maxCapacity: number;
  skilledTechnicians: {
    skill: string;
    count: number;
  }[];
  masterTechAvailable: boolean;
  inventory: {
    part: string;
    quantity: number;
    threshold: number;
  }[];
  timetable: TimeBlock[][];
  labourUtilization: number[];
}

interface ServiceCenterOperationsRowProps {
  centers: ServiceCenter[];
  animatingStep?: number;
}

export const ServiceCenterOperationsRow: React.FC<ServiceCenterOperationsRowProps> = ({
  centers,
  animatingStep = -1
}) => {
  const [selectedSlot, setSelectedSlot] = useState<SlotDetails | null>(null);

  const getStatusColor = (status: string) => {
    return status === 'STABLE' ? '#10b981' : '#ef4444';
  };

  const getBlockColor = (status: string) => {
    switch (status) {
      case 'AVAILABLE': return 'rgba(148, 163, 184, 0.2)';
      case 'TENTATIVE': return 'rgba(234, 179, 8, 0.5)';
      case 'RESERVED': return 'rgba(249, 115, 22, 0.6)';
      case 'CONFIRMED': return 'rgba(16, 185, 129, 0.6)';
      default: return 'rgba(148, 163, 184, 0.2)';
    }
  };

  const getLoadPercentage = (current: number, max: number) => {
    return Math.min((current / max) * 100, 100);
  };

  const handleSlotClick = (block: TimeBlock) => {
    if (block.details) {
      setSelectedSlot(block.details);
    }
  };

  return (
    <>
      <SlotDetailsModal 
        details={selectedSlot} 
        onClose={() => setSelectedSlot(null)} 
      />
      
      <div className={styles.row}>
        <div className={styles.header}>
          <h2>Service Center Operations</h2>
          <div className={styles.legend}>
            <div className={styles.legendItem}>
              <span className={styles.legendDot} style={{ backgroundColor: 'rgba(148, 163, 184, 0.3)' }} />
              <span>Available</span>
            </div>
            <div className={styles.legendItem}>
              <span className={styles.legendDot} style={{ backgroundColor: 'rgba(234, 179, 8, 0.5)' }} />
              <span>Tentative</span>
            </div>
            <div className={styles.legendItem}>
              <span className={styles.legendDot} style={{ backgroundColor: 'rgba(249, 115, 22, 0.6)' }} />
              <span>Reserved</span>
            </div>
            <div className={styles.legendItem}>
              <span className={styles.legendDot} style={{ backgroundColor: 'rgba(16, 185, 129, 0.6)' }} />
              <span>Confirmed</span>
            </div>
          </div>
        </div>

        <div className={styles.grid}>
          {centers.map((center, idx) => (
            <motion.div
              key={center.centerId}
              className={styles.card}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: idx * 0.1 }}
            >
              {/* Top Summary */}
              <div className={styles.topSummary}>
                <div className={styles.centerInfo}>
                  <h3>{center.name}</h3>
                  <div className={styles.location}>
                    <MapPin size={14} />
                    <span>{center.location}</span>
                  </div>
                  <div className={styles.rating}>
                    <Star size={14} fill="#f59e0b" color="#f59e0b" />
                    <span>{center.rating.toFixed(1)}</span>
                  </div>
                </div>
                <div 
                  className={styles.statusBadge}
                  style={{ backgroundColor: getStatusColor(center.status) }}
                >
                  {center.status === 'STABLE' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                  <span>{center.status}</span>
                </div>
              </div>

              {/* Load Bars */}
              <div className={styles.loadSection}>
                <div className={styles.loadHeader}>
                  <span>Workload</span>
                  <span>{center.currentLoad}/{center.maxCapacity} jobs</span>
                </div>
                <div className={styles.loadBar}>
                  <div className={styles.loadBg}>
                    <motion.div 
                      className={styles.loadFill}
                      style={{ backgroundColor: '#3b82f6' }}
                      initial={{ width: 0 }}
                      animate={{ 
                        width: `${getLoadPercentage(center.currentLoad, center.maxCapacity)}%` 
                      }}
                      transition={{ duration: 1, delay: 0.5 }}
                    />
                  </div>
                  <div className={styles.loadLabel}>Current</div>
                </div>
                <div className={styles.loadBar}>
                  <div className={styles.loadBg}>
                    <motion.div 
                      className={styles.loadFill}
                      style={{ backgroundColor: '#f59e0b' }}
                      initial={{ width: 0 }}
                      animate={{ 
                        width: `${getLoadPercentage(center.forecastedLoad, center.maxCapacity)}%` 
                      }}
                      transition={{ duration: 1, delay: animatingStep === 4 ? 0.2 : 0.7 }}
                    />
                  </div>
                  <div className={styles.loadLabel}>Forecasted</div>
                </div>
              </div>

              {/* Capabilities */}
              <div className={styles.capabilities}>
                <div className={styles.capTitle}>
                  <Users size={14} />
                  <span>Skilled Technicians</span>
                </div>
                <div className={styles.skillGrid}>
                  {center.skilledTechnicians.map(skill => (
                    <div key={skill.skill} className={styles.skillItem}>
                      <span className={styles.skillName}>{skill.skill}</span>
                      <span className={styles.skillCount}>{skill.count}</span>
                    </div>
                  ))}
                </div>
                <div className={styles.masterTech}>
                  {center.masterTechAvailable ? (
                    <>
                      <CheckCircle size={14} color="#10b981" />
                      <span>Master Technician Available</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle size={14} color="#ef4444" />
                      <span>Master Technician Unavailable</span>
                    </>
                  )}
                </div>
              </div>

              {/* Inventory Snapshot */}
              <div className={styles.inventory}>
                <div className={styles.capTitle}>
                  <Package size={14} />
                  <span>Inventory Snapshot</span>
                </div>
                <div className={styles.partsList}>
                  {center.inventory.slice(0, 3).map(item => (
                    <div key={item.part} className={styles.partItem}>
                      <span>{item.part}</span>
                      <span className={item.quantity <= item.threshold ? styles.lowStock : styles.normalStock}>
                        {item.quantity} units
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Timetable Grid */}
              <div className={styles.timetable}>
                <div className={styles.timetableHeader}>
                  <Clock size={14} />
                  <span>This Week</span>
                </div>
                <div className={styles.timetableGrid}>
                  {center.timetable.map((day, dayIdx) => (
                    <div key={dayIdx} className={styles.dayColumn}>
                      <div className={styles.dayLabel}>
                        {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dayIdx]}
                      </div>
                      <div className={styles.blocks}>
                        {day.map((block, blockIdx) => (
                          <motion.div
                            key={`${block.date}-${block.hour}`}
                            className={styles.block}
                            style={{ 
                              backgroundColor: getBlockColor(block.status),
                              cursor: block.details ? 'pointer' : 'default',
                              border: block.details ? '1px solid rgba(255,255,255,0.2)' : 'none'
                             }}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ 
                              duration: 0.2, 
                              delay: animatingStep === 3 ? (dayIdx * 0.05 + blockIdx * 0.02) : 0 
                            }}
                            title={`${block.hour}:00 - ${block.status}`}
                            onClick={() => handleSlotClick(block)}
                            whileHover={block.details ? { scale: 1.2, zIndex: 10 } : {}}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Mini Chart */}
              <div className={styles.miniChart}>
                <div className={styles.chartTitle}>
                  <TrendingUp size={14} />
                  <span>7-Day Load Forecast</span>
                </div>
                <div className={styles.chartBars}>
                  {center.labourUtilization.map((util, i) => (
                    <motion.div
                      key={i}
                      className={styles.chartBar}
                      initial={{ height: 0 }}
                      animate={{ height: `${util}%` }}
                      transition={{ duration: 0.5, delay: animatingStep === 4 ? i * 0.1 : 0 }}
                      style={{
                        backgroundColor: util > 85 ? '#ef4444' : util > 65 ? '#f59e0b' : '#10b981'
                      }}
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </>
  );
};
