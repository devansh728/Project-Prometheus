/**
 * App Component - Main layout with sidebar and routing
 */
import React from 'react';
import { motion } from 'framer-motion';
import { Sidebar } from './components/layout/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { AutonomyDashboard } from './pages/AutonomyDashboard/AutonomyDashboard';
import { useStore } from './store';
import './index.css';

function App() {
  const { sidebarCollapsed, activeView } = useStore();

  // Render the active view
  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard />;
      case 'autonomy':
        return <AutonomyDashboard />;
      case 'jobs':
        return <PlaceholderView title="Active Jobs" description="Detailed job management coming soon" />;
      case 'schedule':
        return <PlaceholderView title="Schedule" description="Drag-and-drop calendar coming soon" />;
      case 'mechanics':
        return <PlaceholderView title="Technicians" description="Mechanic roster management coming soon" />;
      case 'inventory':
        return <PlaceholderView title="Inventory" description="Parts and stock management coming soon" />;
      case 'settings':
        return <PlaceholderView title="Settings" description="Configuration options coming soon" />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="app">
      <Sidebar />
      <motion.main
        className="main-content"
        animate={{
          marginLeft: sidebarCollapsed ? 72 : 260,
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {renderView()}
      </motion.main>
    </div>
  );
}

// Placeholder for other views
const PlaceholderView: React.FC<{ title: string; description: string }> = ({
  title,
  description,
}) => (
  <div style={{
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    color: 'var(--color-text-secondary)',
  }}>
    <h1 style={{ fontSize: 32, marginBottom: 8, color: 'var(--color-text-primary)' }}>
      {title}
    </h1>
    <p style={{ fontSize: 16 }}>{description}</p>
  </div>
);

export default App;

