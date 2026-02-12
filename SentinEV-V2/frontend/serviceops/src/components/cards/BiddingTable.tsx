/**
 * Bidding Table - Shows center bids with animated winner highlight
 */
import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Award, Building2, MapPin, Wrench, Package, TrendingUp, Check } from 'lucide-react';
import styles from './BiddingTable.module.css';

interface CenterBid {
  center_id: string;
  center_name: string;
  distance_km: number;
  estimated_cost: number;
  workload_score: number;
  skill_score: number;
  inventory_score: number;
  overall_bid_score: number;
  est_days_to_complete: number;
  is_historical: boolean;
  reasoning: string;
  available_mechanics: number;
  load_percentage: number;
  parts_available: boolean;
}

interface BiddingTableProps {
  requestId: string;
  vehicleName: string;
  bids: CenterBid[];
  winnerId?: string;
  isAnimating?: boolean;
  onSelectWinner?: (bid: CenterBid) => void;
}

export const BiddingTable: React.FC<BiddingTableProps> = ({
  requestId,
  vehicleName,
  bids,
  winnerId,
  isAnimating,
  onSelectWinner,
}) => {
  const [currentHighlight, setCurrentHighlight] = useState<number>(-1);
  const [showWinner, setShowWinner] = useState(false);

  // Animate through bids before showing winner
  useEffect(() => {
    if (isAnimating && bids.length > 0) {
      let index = 0;
      const interval = setInterval(() => {
        setCurrentHighlight(index);
        index++;
        if (index >= bids.length) {
          clearInterval(interval);
          setTimeout(() => {
            setShowWinner(true);
            setCurrentHighlight(-1);
          }, 300);
        }
      }, 200);
      return () => clearInterval(interval);
    } else {
      setShowWinner(true);
    }
  }, [isAnimating, bids.length]);

  const getScoreColor = (score: number): string => {
    if (score >= 0.8) return 'var(--color-success)';
    if (score >= 0.5) return 'var(--color-warning)';
    return 'var(--color-error)';
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <Award className={styles.icon} size={20} />
          <div>
            <h3 className={styles.title}>Center Bidding</h3>
            <span className={styles.subtitle}>for {vehicleName}</span>
          </div>
        </div>
        <span className={styles.bidCount}>{bids.length} bids received</span>
      </div>

      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Service Center</th>
              <th>Distance</th>
              <th>Load</th>
              <th>Skills</th>
              <th>Parts</th>
              <th>ETA</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence>
              {bids.map((bid, index) => {
                const isWinner = showWinner && bid.center_id === winnerId;
                const isHighlighted = currentHighlight === index;

                return (
                  <motion.tr
                    key={bid.center_id}
                    className={`${styles.row} ${isWinner ? styles.winner : ''} ${isHighlighted ? styles.highlighted : ''}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    onClick={() => onSelectWinner?.(bid)}
                  >
                    <td className={styles.centerCell}>
                      <Building2 size={14} />
                      <div>
                        <div className={styles.centerName}>{bid.center_name}</div>
                        {bid.is_historical && (
                          <span className={styles.historicalBadge}>Preferred</span>
                        )}
                      </div>
                      {isWinner && (
                        <motion.div
                          className={styles.winnerBadge}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: 'spring', stiffness: 500, damping: 25 }}
                        >
                          <Check size={12} /> Winner
                        </motion.div>
                      )}
                    </td>
                    <td>
                      <MapPin size={12} />
                      {bid.distance_km.toFixed(1)} km
                    </td>
                    <td>
                      <div className={styles.scoreBar}>
                        <div
                          className={styles.scoreBarFill}
                          style={{
                            width: `${100 - bid.load_percentage}%`,
                            backgroundColor: getScoreColor(bid.workload_score),
                          }}
                        />
                      </div>
                      <span>{bid.load_percentage.toFixed(0)}%</span>
                    </td>
                    <td>
                      <div className={styles.scoreBar}>
                        <div
                          className={styles.scoreBarFill}
                          style={{
                            width: `${bid.skill_score * 100}%`,
                            backgroundColor: getScoreColor(bid.skill_score),
                          }}
                        />
                      </div>
                    </td>
                    <td>
                      {bid.parts_available ? (
                        <span className={styles.available}>
                          <Package size={12} /> In Stock
                        </span>
                      ) : (
                        <span className={styles.unavailable}>
                          <Package size={12} /> Order
                        </span>
                      )}
                    </td>
                    <td className={styles.etaCell}>
                      {bid.est_days_to_complete} day{bid.est_days_to_complete > 1 ? 's' : ''}
                    </td>
                    <td>
                      <motion.div
                        className={styles.bidScore}
                        animate={{
                          scale: isWinner ? [1, 1.2, 1] : 1,
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {bid.overall_bid_score.toFixed(1)}
                      </motion.div>
                    </td>
                  </motion.tr>
                );
              })}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {showWinner && winnerId && (
        <motion.div
          className={styles.reasoningSection}
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <TrendingUp size={14} />
          <span>
            {bids.find(b => b.center_id === winnerId)?.reasoning}
          </span>
        </motion.div>
      )}
    </div>
  );
};

export default BiddingTable;
