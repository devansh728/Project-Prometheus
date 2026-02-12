/**
 * Header Component - Top bar with search and notifications
 */
import React from 'react';
import { motion } from 'framer-motion';
import { Search, Bell, User } from 'lucide-react';
import styles from './Header.module.css';

interface HeaderProps {
  title: string;
}

export const Header: React.FC<HeaderProps> = ({ title }) => {
  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <h1 className={styles.title}>{title}</h1>
      </div>

      <div className={styles.center}>
        <div className={styles.searchContainer}>
          <Search size={18} className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search jobs, vehicles, technicians..."
            className={styles.searchInput}
          />
        </div>
      </div>

      <div className={styles.right}>
        <motion.button
          className={styles.iconBtn}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Bell size={20} />
          <span className={styles.badge}>3</span>
        </motion.button>

        <div className={styles.userInfo}>
          <div className={styles.userDetails}>
            <span className={styles.userName}>Service Manager</span>
            <span className={styles.userRole}>EV Care Mumbai</span>
          </div>
          <div className={styles.avatar}>
            <User size={20} />
          </div>
        </div>
      </div>
    </header>
  );
};
