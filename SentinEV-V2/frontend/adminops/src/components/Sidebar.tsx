/**
 * AdminOps Sidebar Component
 */
import React from 'react';
import { motion } from 'framer-motion';
import { LayoutDashboard, FileWarning, Shield, BarChart3, Settings, Zap, ChevronLeft, ChevronRight } from 'lucide-react';
import { useStore } from '../../store';
import styles from './Sidebar.module.css';

const navItems = [
  { id: 'fleet', label: 'Fleet Overview', icon: <LayoutDashboard size={18} /> },
  { id: 'capa', label: 'RCA/CAPA', icon: <FileWarning size={18} /> },
  { id: 'security', label: 'UEBA Security', icon: <Shield size={18} /> },
  { id: 'analytics', label: 'Analytics', icon: <BarChart3 size={18} /> },
  { id: 'settings', label: 'Settings', icon: <Settings size={18} /> },
];

export const Sidebar: React.FC = () => {
  const { sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useStore();

  return (
    <motion.aside
      className={styles.sidebar}
      animate={{ width: sidebarCollapsed ? 64 : 240 }}
      transition={{ duration: 0.2 }}
    >
      <div className={styles.logo}>
        <div className={styles.logoIcon}><Zap size={20} /></div>
        {!sidebarCollapsed && <span className={styles.logoText}>AdminOps</span>}
      </div>

      <nav className={styles.nav}>
        {navItems.map(item => (
          <button
            key={item.id}
            className={`${styles.navItem} ${activeView === item.id ? styles.active : ''}`}
            onClick={() => setActiveView(item.id)}
          >
            <span className={styles.navIcon}>{item.icon}</span>
            {!sidebarCollapsed && <span className={styles.navLabel}>{item.label}</span>}
            {activeView === item.id && <div className={styles.activeBar} />}
          </button>
        ))}
      </nav>

      <button className={styles.toggle} onClick={toggleSidebar}>
        {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>
    </motion.aside>
  );
};
