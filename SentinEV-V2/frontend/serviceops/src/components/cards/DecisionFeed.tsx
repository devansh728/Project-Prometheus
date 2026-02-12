/**
 * Decision Feed - Live agent reasoning log
 */
import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, Inbox, Award, Wrench, Users, AlertTriangle,
  Package, FileText, Clock, RefreshCw
} from 'lucide-react';
import styles from './DecisionFeed.module.css';

interface Decision {
  id: string;
  event_type: string;
  timestamp: string;
  entity_id?: string;
  entity_type?: string;
  details: Record<string, any>;
  reasoning: string;
  impact?: string;
}

interface DecisionFeedProps {
  decisions: Decision[];
  maxItems?: number;
  autoScroll?: boolean;
}

const getEventIcon = (eventType: string) => {
  switch (eventType) {
    case 'QUEUE_ENTRY': return <Inbox size={14} />;
    case 'QUEUE_DEQUEUE': return <Inbox size={14} />;
    case 'BIDDING_START':
    case 'BIDDING_COMPLETE': return <Award size={14} />;
    case 'CENTER_SELECTED': return <Award size={14} />;
    case 'TASKS_CREATED': return <Wrench size={14} />;
    case 'TECHNICIAN_ASSIGNED': return <Users size={14} />;
    case 'PREEMPTION': return <AlertTriangle size={14} />;
    case 'REORDER_TRIGGERED': return <Package size={14} />;
    case 'DIAGNOSIS_FEEDBACK': return <FileText size={14} />;
    case 'SCHEDULE_UPDATED': return <RefreshCw size={14} />;
    default: return <Activity size={14} />;
  }
};

const getEventColor = (eventType: string): string => {
  switch (eventType) {
    case 'QUEUE_ENTRY': return 'var(--color-primary)';
    case 'BIDDING_COMPLETE':
    case 'CENTER_SELECTED': return 'var(--color-success)';
    case 'PREEMPTION': return 'var(--color-error)';
    case 'REORDER_TRIGGERED': return 'var(--color-warning)';
    case 'DIAGNOSIS_FEEDBACK': return 'var(--color-secondary)';
    default: return 'var(--color-text-secondary)';
  }
};

const formatTime = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

export const DecisionFeed: React.FC<DecisionFeedProps> = ({
  decisions,
  maxItems = 30,
  autoScroll = true,
}) => {
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [decisions.length, autoScroll]);

  const displayedDecisions = decisions.slice(0, maxItems);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <Activity className={styles.icon} size={20} />
          <h3 className={styles.title}>Agent Decisions</h3>
          <div className={styles.liveBadge}>
            <span className={styles.liveDot} />
            Live
          </div>
        </div>
        <span className={styles.count}>{decisions.length} total</span>
      </div>

      <div className={styles.feed} ref={feedRef}>
        <AnimatePresence mode="popLayout">
          {displayedDecisions.map((decision, index) => (
            <motion.div
              key={decision.id}
              className={styles.item}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.02 }}
              style={{
                borderLeft: `3px solid ${getEventColor(decision.event_type)}`,
              }}
            >
              <div
                className={styles.iconWrapper}
                style={{ color: getEventColor(decision.event_type) }}
              >
                {getEventIcon(decision.event_type)}
              </div>

              <div className={styles.content}>
                <div className={styles.eventType}>
                  {decision.event_type.replace(/_/g, ' ')}
                  {decision.entity_id && (
                    <span className={styles.entityId}>{decision.entity_id}</span>
                  )}
                </div>
                <div className={styles.reasoning}>{decision.reasoning}</div>
                {decision.impact && (
                  <div className={styles.impact}>
                    <span>Impact:</span> {decision.impact}
                  </div>
                )}
              </div>

              <div className={styles.timestamp}>
                <Clock size={10} />
                {formatTime(decision.timestamp)}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {decisions.length === 0 && (
          <div className={styles.emptyState}>
            <Activity size={32} />
            <p>No decisions yet</p>
            <span>Run a scenario to see agent reasoning</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default DecisionFeed;
