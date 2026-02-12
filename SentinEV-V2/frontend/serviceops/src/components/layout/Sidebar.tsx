/**
 * Sidebar Component - Navigation with animated transitions
 */
import React from 'react';
import { motion } from 'framer-motion';
import {
  LayoutDashboard,
  ClipboardList,
  Calendar,
  Users,
  Package,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
  Bot,
} from 'lucide-react';
import { useStore } from '../../store';
import styles from './Sidebar.module.css';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Command Center', icon: <LayoutDashboard size={20} /> },
  { id: 'autonomy', label: 'AI Autonomy', icon: <Bot size={20} /> },
  { id: 'jobs', label: 'Active Jobs', icon: <ClipboardList size={20} /> },
  { id: 'schedule', label: 'Schedule', icon: <Calendar size={20} /> },
  { id: 'mechanics', label: 'Technicians', icon: <Users size={20} /> },
  { id: 'inventory', label: 'Inventory', icon: <Package size={20} /> },
  { id: 'settings', label: 'Settings', icon: <Settings size={20} /> },
];

export const Sidebar: React.FC = () => {
  const { sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useStore();

  return (
    <motion.aside
      className={styles.sidebar}
      animate={{ width: sidebarCollapsed ? 72 : 260 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
    >
      {/* Logo */}
      <div className={styles.logo}>
        <div className={styles.logoIcon}>
          <Zap size={24} />
        </div>
        <motion.span
          className={styles.logoText}
          animate={{ opacity: sidebarCollapsed ? 0 : 1 }}
          transition={{ duration: 0.2 }}
        >
          SentinEV
        </motion.span>
      </div>

      {/* Navigation */}
      <nav className={styles.nav}>
        {navItems.map((item) => (
          <motion.button
            key={item.id}
            className={`${styles.navItem} ${activeView === item.id ? styles.active : ''}`}
            onClick={() => setActiveView(item.id)}
            whileHover={{ x: 4 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className={styles.navIcon}>{item.icon}</span>
            <motion.span
              className={styles.navLabel}
              animate={{ opacity: sidebarCollapsed ? 0 : 1 }}
              transition={{ duration: 0.2 }}
            >
              {item.label}
            </motion.span>
            {activeView === item.id && (
              <motion.div
                className={styles.activeIndicator}
                layoutId="activeIndicator"
                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
              />
            )}
          </motion.button>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <button className={styles.collapseBtn} onClick={toggleSidebar}>
        {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
      </button>
    </motion.aside>
  );
};
