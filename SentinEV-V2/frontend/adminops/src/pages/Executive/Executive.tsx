/**
 * Executive Dashboard - Fleet Health Overview with Charts
 */
import React from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';
import { Activity, AlertTriangle, CheckCircle, XCircle, TrendingUp, Truck } from 'lucide-react';
import { useStore } from '../../store';
import styles from './Executive.module.css';

const COLORS = { healthy: '#3FB950', warning: '#D29922', critical: '#F85149' };

export const ExecutiveDashboard: React.FC = () => {
  const { fleetHealth, failureTrends, capaPatterns, suppliers } = useStore();

  const pieData = [
    { name: 'Healthy', value: fleetHealth.healthy, color: COLORS.healthy },
    { name: 'Warning', value: fleetHealth.warning, color: COLORS.warning },
    { name: 'Critical', value: fleetHealth.critical, color: COLORS.critical },
  ];

  const openCAPAs = capaPatterns.filter(c => c.status !== 'resolved').length;
  const criticalCAPAs = capaPatterns.filter(c => c.severity === 'critical').length;
  const avgSupplierScore = Math.round(suppliers.reduce((a, s) => a + s.quality_score, 0) / suppliers.length);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Executive Dashboard</h1>
          <p className={styles.subtitle}>Fleet-wide health analytics and insights</p>
        </div>
        <div className={styles.liveIndicator}>
          <span className={styles.liveDot} />
          Live Updates
        </div>
      </header>

      {/* KPI Row */}
      <div className={styles.kpiGrid}>
        <motion.div className={styles.kpiCard} whileHover={{ y: -2 }}>
          <div className={styles.kpiIcon} style={{ background: 'rgba(88,166,255,0.15)' }}>
            <Truck size={20} color="#58A6FF" />
          </div>
          <div className={styles.kpiContent}>
            <span className={styles.kpiValue}>{fleetHealth.total_vehicles.toLocaleString()}</span>
            <span className={styles.kpiLabel}>Total Fleet</span>
          </div>
        </motion.div>

        <motion.div className={styles.kpiCard} whileHover={{ y: -2 }}>
          <div className={styles.kpiIcon} style={{ background: 'rgba(63,185,80,0.15)' }}>
            <CheckCircle size={20} color="#3FB950" />
          </div>
          <div className={styles.kpiContent}>
            <span className={styles.kpiValue}>{fleetHealth.avg_health_score}%</span>
            <span className={styles.kpiLabel}>Avg Health Score</span>
          </div>
        </motion.div>

        <motion.div className={styles.kpiCard} whileHover={{ y: -2 }}>
          <div className={styles.kpiIcon} style={{ background: 'rgba(248,81,73,0.15)' }}>
            <AlertTriangle size={20} color="#F85149" />
          </div>
          <div className={styles.kpiContent}>
            <span className={styles.kpiValue}>{openCAPAs}</span>
            <span className={styles.kpiLabel}>Open CAPAs</span>
          </div>
        </motion.div>

        <motion.div className={styles.kpiCard} whileHover={{ y: -2 }}>
          <div className={styles.kpiIcon} style={{ background: 'rgba(163,113,247,0.15)' }}>
            <TrendingUp size={20} color="#A371F7" />
          </div>
          <div className={styles.kpiContent}>
            <span className={styles.kpiValue}>{avgSupplierScore}%</span>
            <span className={styles.kpiLabel}>Avg Supplier Score</span>
          </div>
        </motion.div>
      </div>

      {/* Charts Row */}
      <div className={styles.chartsGrid}>
        {/* Fleet Health Pie */}
        <motion.div className={styles.chartCard} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h3 className={styles.chartTitle}>Fleet Health Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={4}>
                {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
              </Pie>
              <Tooltip contentStyle={{ background: '#161B22', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Failure Trends */}
        <motion.div className={styles.chartCard} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
          <h3 className={styles.chartTitle}>Failure Trends (6 Weeks)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={failureTrends}>
              <defs>
                <linearGradient id="colorBrake" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#F85149" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#F85149" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorBattery" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#D29922" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#D29922" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fill: '#8B949E', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#8B949E', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#161B22', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              <Area type="monotone" dataKey="brake" stroke="#F85149" fill="url(#colorBrake)" />
              <Area type="monotone" dataKey="battery" stroke="#D29922" fill="url(#colorBattery)" />
              <Area type="monotone" dataKey="coolant" stroke="#58A6FF" fill="transparent" />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* CAPA Summary */}
      <motion.div className={styles.capaSection} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
        <h3 className={styles.chartTitle}>Active CAPA Patterns</h3>
        <div className={styles.capaGrid}>
          {capaPatterns.filter(c => c.status !== 'resolved').map(capa => (
            <div key={capa.id} className={`${styles.capaCard} ${styles[capa.severity]}`}>
              <div className={styles.capaHeader}>
                <span className={styles.capaId}>{capa.id}</span>
                <span className={`${styles.capaBadge} ${styles[capa.severity]}`}>{capa.severity.toUpperCase()}</span>
              </div>
              <h4 className={styles.capaTitle}>{capa.title}</h4>
              <div className={styles.capaStats}>
                <span>{capa.occurrences} occurrences</span>
                <span>{capa.affected_vehicles} vehicles</span>
              </div>
              <span className={`${styles.capaStatus} ${styles[capa.status]}`}>{capa.status}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Supplier Scorecards */}
      <motion.div className={styles.supplierSection} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
        <h3 className={styles.chartTitle}>Supplier Quality Scorecards</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={suppliers} layout="vertical">
            <XAxis type="number" domain={[0, 100]} tick={{ fill: '#8B949E', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis dataKey="name" type="category" tick={{ fill: '#8B949E', fontSize: 11 }} axisLine={false} tickLine={false} width={120} />
            <Tooltip contentStyle={{ background: '#161B22', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
            <Bar dataKey="quality_score" fill="#58A6FF" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
};
