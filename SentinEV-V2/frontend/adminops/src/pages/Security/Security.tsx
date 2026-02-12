/**
 * UEBA Security Dashboard - Agent Behavior Monitoring
 */
import React from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertOctagon, Eye, Activity, User, Clock, Server } from 'lucide-react';
import { useStore } from '../../store';
import styles from './Security.module.css';

const severityColors = {
  low: '#3FB950',
  medium: '#D29922',
  high: '#F85149',
  critical: '#F85149',
};

const eventTypeLabels = {
  access_anomaly: 'Access Anomaly',
  volume_spike: 'Volume Spike',
  timing_anomaly: 'Timing Anomaly',
  pattern_deviation: 'Pattern Deviation',
};

export const SecurityDashboard: React.FC = () => {
  const { uebaEvents, agentProfiles } = useStore();
  const unresolvedEvents = uebaEvents.filter(e => !e.resolved);
  const criticalEvents = uebaEvents.filter(e => e.severity === 'critical' || e.severity === 'high');

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>UEBA Security Center</h1>
          <p className={styles.subtitle}>User & Entity Behavior Analytics</p>
        </div>
        <div className={styles.alertBadge}>
          <AlertOctagon size={16} />
          {unresolvedEvents.length} Active Alerts
        </div>
      </header>

      {/* Security Metrics */}
      <div className={styles.metricsGrid}>
        <motion.div className={styles.metricCard} whileHover={{ y: -2 }}>
          <Shield size={24} color="#58A6FF" />
          <div className={styles.metricContent}>
            <span className={styles.metricValue}>{agentProfiles.length}</span>
            <span className={styles.metricLabel}>Monitored Agents</span>
          </div>
        </motion.div>

        <motion.div className={styles.metricCard} whileHover={{ y: -2 }}>
          <Activity size={24} color="#3FB950" />
          <div className={styles.metricContent}>
            <span className={styles.metricValue}>2,847</span>
            <span className={styles.metricLabel}>Events Today</span>
          </div>
        </motion.div>

        <motion.div className={styles.metricCard} whileHover={{ y: -2 }}>
          <AlertOctagon size={24} color="#D29922" />
          <div className={styles.metricContent}>
            <span className={styles.metricValue}>{unresolvedEvents.length}</span>
            <span className={styles.metricLabel}>Unresolved Alerts</span>
          </div>
        </motion.div>

        <motion.div className={styles.metricCard} whileHover={{ y: -2 }}>
          <Eye size={24} color="#A371F7" />
          <div className={styles.metricContent}>
            <span className={styles.metricValue}>99.2%</span>
            <span className={styles.metricLabel}>Detection Rate</span>
          </div>
        </motion.div>
      </div>

      <div className={styles.contentGrid}>
        {/* Agent Profiles */}
        <motion.section className={styles.section} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h3 className={styles.sectionTitle}>
            <User size={16} /> Agent Baselines
          </h3>
          <div className={styles.agentList}>
            {agentProfiles.map(agent => (
              <div key={agent.agent_id} className={styles.agentCard}>
                <div className={styles.agentHeader}>
                  <Server size={16} color="#58A6FF" />
                  <span className={styles.agentName}>{agent.agent_name}</span>
                </div>
                <div className={styles.agentStats}>
                  <div className={styles.stat}>
                    <Clock size={12} />
                    <span>{agent.normal_access_hours}</span>
                  </div>
                  <div className={styles.stat}>
                    <Activity size={12} />
                    <span>{agent.avg_daily_requests} req/day</span>
                  </div>
                </div>
                <div className={styles.endpoints}>
                  {agent.typical_endpoints.slice(0, 2).map((ep, i) => (
                    <span key={i} className={styles.endpoint}>{ep}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </motion.section>

        {/* Event Feed */}
        <motion.section className={styles.section} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
          <h3 className={styles.sectionTitle}>
            <AlertOctagon size={16} /> Security Events
          </h3>
          <div className={styles.eventList}>
            {uebaEvents.map(event => (
              <div key={event.id} className={`${styles.eventCard} ${event.resolved ? styles.resolved : ''}`}>
                <div className={styles.eventHeader}>
                  <span className={styles.eventType} style={{ color: severityColors[event.severity] }}>
                    {eventTypeLabels[event.event_type]}
                  </span>
                  <span className={`${styles.severityBadge} ${styles[event.severity]}`}>
                    {event.severity.toUpperCase()}
                  </span>
                </div>
                <p className={styles.eventDesc}>{event.description}</p>
                <div className={styles.eventFooter}>
                  <span className={styles.eventAgent}>{event.agent_id}</span>
                  <span className={styles.eventTime}>
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.section>
      </div>

      {/* Audit Trail */}
      <motion.section className={styles.auditSection} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
        <h3 className={styles.sectionTitle}>📋 Recent Audit Trail</h3>
        <div className={styles.auditTable}>
          <div className={styles.auditRow}>
            <span className={styles.auditTime}>03:10:15</span>
            <span className={styles.auditAgent}>master_agent</span>
            <span className={styles.auditAction}>POST /api/v1/analyze</span>
            <span className={styles.auditStatus}>✓ 200</span>
          </div>
          <div className={styles.auditRow}>
            <span className={styles.auditTime}>03:10:12</span>
            <span className={styles.auditAgent}>diagnosis_agent</span>
            <span className={styles.auditAction}>POST /api/v1/rag/query</span>
            <span className={styles.auditStatus}>✓ 200</span>
          </div>
          <div className={styles.auditRow}>
            <span className={styles.auditTime}>03:10:08</span>
            <span className={styles.auditAgent}>scheduling_agent</span>
            <span className={styles.auditAction}>GET /api/v1/serviceops/slots</span>
            <span className={styles.auditStatus}>✓ 200</span>
          </div>
          <div className={`${styles.auditRow} ${styles.warning}`}>
            <span className={styles.auditTime}>03:09:55</span>
            <span className={styles.auditAgent}>scheduling_agent</span>
            <span className={styles.auditAction}>POST /api/v1/admin/capa</span>
            <span className={styles.auditStatus}>⚠ 403</span>
          </div>
        </div>
      </motion.section>
    </div>
  );
};
