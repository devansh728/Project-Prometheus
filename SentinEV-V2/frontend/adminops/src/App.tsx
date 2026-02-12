/**
 * AdminOps App Component
 */
import React from 'react';
import { motion } from 'framer-motion';
import { Sidebar } from './components/Sidebar';
import { ExecutiveDashboard } from './pages/Executive';
import { SecurityDashboard } from './pages/Security';
import { useStore } from './store';
import './index.css';

const PlaceholderView: React.FC<{ title: string }> = ({ title }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontSize: 24, color: 'var(--color-text-secondary)' }}>
    {title} — Coming Soon
  </div>
);

function App() {
  const { activeView, sidebarCollapsed } = useStore();

  const renderView = () => {
    switch (activeView) {
      case 'fleet': return <ExecutiveDashboard />;
      case 'capa': return <PlaceholderView title="RCA/CAPA Management" />;
      case 'security': return <SecurityDashboard />;
      case 'analytics': return <PlaceholderView title="Analytics Center" />;
      case 'settings': return <PlaceholderView title="Settings" />;
      default: return <ExecutiveDashboard />;
    }
  };

  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <motion.main
        style={{ flex: 1, minHeight: '100vh' }}
        animate={{ marginLeft: sidebarCollapsed ? 64 : 240 }}
        transition={{ duration: 0.2 }}
      >
        {renderView()}
      </motion.main>
    </div>
  );
}

export default App;
