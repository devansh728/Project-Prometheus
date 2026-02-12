/**
 * Service Center Card - Comprehensive card showing center details
 * Includes: slot timetable, workers, inventory, workload graph, free slots
 */
import React from 'react';
import { motion } from 'framer-motion';
import { MapPin, Star, Calendar, Users, Package, Clock, TrendingUp } from 'lucide-react';
import styles from './ServiceCenterCard.module.css';

interface Worker {
  id: string;
  name: string;
  certifications: string[];
  available: boolean;
  currentJob?: string;
}

interface ServiceCenter {
  id: string;
  name: string;
  address: string;
  rating: number;
  distance?: number;
  capabilities: string[];
  numBays: number;
  inventory: {
    brake_pads: number;
    brake_fluid: number;
    oil_filters: number;
    coolant: number;
  };
  workers: Worker[];
  slots: {
    [date: string]: number[]; // Array of available hours (9-18)
  };
  workload: number[]; // 7 days forecast percentages
  freeSlots: number;
}

interface ServiceCenterCardProps {
  center: ServiceCenter;
  isSelected?: boolean;
  onSelect?: (centerId: string) => void;
  compact?: boolean;
}

export const ServiceCenterCard: React.FC<ServiceCenterCardProps> = ({
  center,
  isSelected = false,
  onSelect,
  compact = false,
}) => {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const hours = [9, 10, 11, 12, 13, 14, 15, 16, 17];

  const getInventoryLevel = (current: number, max: number = 30) => {
    const percentage = (current / max) * 100;
    if (percentage > 60) return 'high';
    if (percentage > 30) return 'medium';
    return 'low';
  };

  return (
    <motion.div
      className={`${styles.card} ${isSelected ? styles.selected : ''} ${compact ? styles.compact : ''}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.01 }}
      onClick={() => onSelect?.(center.id)}
    >
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h3 className={styles.centerName}>{center.name}</h3>
          <div className={styles.addressRow}>
            <MapPin size={14} />
            <span>{center.address}</span>
            {center.distance && <span className={styles.distance}>{center.distance}km</span>}
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.rating}>
            <Star size={16} fill="#F59E0B" color="#F59E0B" />
            <span>{center.rating}</span>
          </div>
          <div className={styles.freeSlots}>
            <Calendar size={14} />
            <span>{center.freeSlots} slots</span>
          </div>
        </div>
      </div>

      {/* Capabilities Tags */}
      <div className={styles.capabilities}>
        {center.capabilities.map((cap) => (
          <span key={cap} className={`${styles.capTag} ${styles[cap.toLowerCase()]}`}>
            {cap}
          </span>
        ))}
      </div>

      {!compact && (
        <>
          {/* Slot Timetable */}
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <Clock size={16} />
              <span>Slot Availability</span>
            </div>
            <div className={styles.slotGrid}>
              <div className={styles.slotHeader}>
                <span></span>
                {hours.map(h => (
                  <span key={h} className={styles.hourLabel}>{h}</span>
                ))}
              </div>
              {days.map((day, dayIdx) => (
                <div key={day} className={styles.slotRow}>
                  <span className={styles.dayLabel}>{day}</span>
                  {hours.map((hour) => {
                    const isAvailable = center.slots[day]?.includes(hour) ?? Math.random() > 0.4;
                    return (
                      <div
                        key={hour}
                        className={`${styles.slot} ${isAvailable ? styles.available : styles.booked}`}
                        title={isAvailable ? 'Available' : 'Booked'}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          {/* Workers Grid */}
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <Users size={16} />
              <span>Technicians ({center.workers.length})</span>
            </div>
            <div className={styles.workersGrid}>
              {center.workers.map((worker) => (
                <div key={worker.id} className={`${styles.worker} ${worker.available ? '' : styles.busy}`}>
                  <div className={styles.workerAvatar}>
                    {worker.name.charAt(0)}
                    <span className={`${styles.statusDot} ${worker.available ? styles.online : styles.offline}`} />
                  </div>
                  <div className={styles.workerInfo}>
                    <span className={styles.workerName}>{worker.name.split(' ')[0]}</span>
                    <div className={styles.certs}>
                      {worker.certifications.slice(0, 2).map(cert => (
                        <span key={cert} className={styles.certBadge}>{cert}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Inventory Bars */}
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <Package size={16} />
              <span>Parts Inventory</span>
            </div>
            <div className={styles.inventoryGrid}>
              {Object.entries(center.inventory).map(([part, qty]) => (
                <div key={part} className={styles.inventoryItem}>
                  <div className={styles.inventoryHeader}>
                    <span>{part.replace('_', ' ')}</span>
                    <span className={styles.inventoryQty}>{qty}</span>
                  </div>
                  <div className={styles.inventoryBar}>
                    <div 
                      className={`${styles.inventoryFill} ${styles[getInventoryLevel(qty)]}`}
                      style={{ width: `${Math.min((qty / 30) * 100, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Workload Forecast */}
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <TrendingUp size={16} />
              <span>Labor Forecast</span>
            </div>
            <div className={styles.workloadChart}>
              {center.workload.map((load, idx) => (
                <div key={idx} className={styles.workloadDay}>
                  <div className={styles.workloadBarWrapper}>
                    <div 
                      className={`${styles.workloadBar} ${load > 80 ? styles.high : load > 50 ? styles.medium : styles.low}`}
                      style={{ height: `${load}%` }}
                    />
                  </div>
                  <span className={styles.workloadLabel}>{days[idx]}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Select Button */}
      {onSelect && (
        <button 
          className={`${styles.selectBtn} ${isSelected ? styles.selectedBtn : ''}`}
          onClick={(e) => { e.stopPropagation(); onSelect(center.id); }}
        >
          {isSelected ? '✓ Selected' : 'Select Center'}
        </button>
      )}
    </motion.div>
  );
};

export default ServiceCenterCard;
