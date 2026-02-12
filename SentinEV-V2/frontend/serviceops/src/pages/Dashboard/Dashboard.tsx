/**
 * Dashboard Page - Command Center with metrics and Kanban
 */
import React from 'react';
import { motion } from 'framer-motion';
import {
  ClipboardList,
  Clock,
  Users,
  Package,
  TrendingUp,
  Star,
} from 'lucide-react';
import { useStore } from '../../store';
import { Header } from '../../components/layout/Header';
import { MetricCard } from '../../components/cards/MetricCard';
import { KanbanBoard } from '../../components/kanban/KanbanBoard';
import { JobCard } from '../../components/cards/JobCard';
import { ServiceCenterCard } from '../../components/cards/ServiceCenterCard';
import { VehicleLifecycleTracker } from '../../components/cards/VehicleLifecycleTracker';
import styles from './Dashboard.module.css';

// Mock service center data
const MOCK_SERVICE_CENTERS = [
  {
    id: 'SC001',
    name: 'EV Care Mumbai Central',
    address: '12 Phoenix Mills, Lower Parel, Mumbai 400013',
    rating: 4.8,
    capabilities: ['EV Battery', 'Brake', 'General'],
    numBays: 6,
    inventory: { brake_pads: 18, brake_fluid: 24, oil_filters: 30, coolant: 22 },
    workers: [
      { id: 'W1', name: 'Rajesh Kumar', certifications: ['EV Certified', 'Brake Specialist'], available: true },
      { id: 'W2', name: 'Amit Sharma', certifications: ['General', 'Battery'], available: true },
      { id: 'W3', name: 'Suresh Patel', certifications: ['EV Certified'], available: false },
    ],
    slots: { Mon: [9,10,11,14,15], Tue: [10,11,12,15,16], Wed: [9,11,14,16,17], Thu: [10,13,14,15], Fri: [9,10,11,12], Sat: [10,11], Sun: [] },
    workload: [65, 78, 82, 70, 88, 45, 20],
    freeSlots: 24,
  },
  {
    id: 'SC002',
    name: 'GreenDrive Pune',
    address: '45 Koregaon Park, Pune 411001',
    rating: 4.6,
    capabilities: ['EV Battery', 'Suspension', 'General'],
    numBays: 5,
    inventory: { brake_pads: 12, brake_fluid: 18, oil_filters: 25, coolant: 15 },
    workers: [
      { id: 'W4', name: 'Vikram Singh', certifications: ['EV Certified', 'Suspension'], available: true },
      { id: 'W5', name: 'Rohit Jain', certifications: ['General'], available: true },
    ],
    slots: { Mon: [9,10,14,15], Tue: [11,12,15], Wed: [9,10,11,14], Thu: [10,11,14,15,16], Fri: [9,10,11], Sat: [10], Sun: [] },
    workload: [55, 68, 72, 65, 78, 35, 15],
    freeSlots: 18,
  },
  {
    id: 'SC003',
    name: 'ElectriCare Bangalore',
    address: '78 Indiranagar, Bangalore 560038',
    rating: 4.9,
    capabilities: ['EV Battery', 'Brake', 'Motor', 'AC'],
    numBays: 8,
    inventory: { brake_pads: 25, brake_fluid: 30, oil_filters: 35, coolant: 28 },
    workers: [
      { id: 'W6', name: 'Karan Mehta', certifications: ['EV Master', 'Motor Specialist'], available: true },
      { id: 'W7', name: 'Neha Reddy', certifications: ['EV Certified', 'AC'], available: true },
      { id: 'W8', name: 'Pradeep Kumar', certifications: ['Brake Specialist'], available: true },
      { id: 'W9', name: 'Sanjay Rao', certifications: ['General'], available: false },
    ],
    slots: { Mon: [9,10,11,12,14,15,16], Tue: [9,10,11,14,15,16], Wed: [10,11,12,14,15], Thu: [9,11,14,15,16,17], Fri: [9,10,11,14], Sat: [10,11,12], Sun: [11] },
    workload: [72, 85, 90, 78, 92, 55, 30],
    freeSlots: 35,
  },
  {
    id: 'SC004',
    name: 'PowerEV Delhi',
    address: '23 Connaught Place, New Delhi 110001',
    rating: 4.7,
    capabilities: ['EV Battery', 'Brake', 'General'],
    numBays: 6,
    inventory: { brake_pads: 15, brake_fluid: 20, oil_filters: 28, coolant: 18 },
    workers: [
      { id: 'W10', name: 'Arjun Kapoor', certifications: ['EV Certified'], available: true },
      { id: 'W11', name: 'Ravi Verma', certifications: ['Brake', 'General'], available: true },
    ],
    slots: { Mon: [9,10,11,14], Tue: [10,11,15,16], Wed: [9,14,15,16], Thu: [10,11,14,15], Fri: [9,10,11], Sat: [10,11], Sun: [] },
    workload: [60, 72, 68, 75, 80, 40, 10],
    freeSlots: 21,
  },
  {
    id: 'SC005',
    name: 'ChargePoint Hyderabad',
    address: '56 Hitech City, Hyderabad 500081',
    rating: 4.5,
    capabilities: ['EV Battery', 'Charging', 'General'],
    numBays: 4,
    inventory: { brake_pads: 10, brake_fluid: 15, oil_filters: 20, coolant: 12 },
    workers: [
      { id: 'W12', name: 'Manoj Reddy', certifications: ['EV Certified', 'Charging'], available: true },
      { id: 'W13', name: 'Deepak Sharma', certifications: ['General'], available: false },
    ],
    slots: { Mon: [9,10,14], Tue: [10,11,15], Wed: [9,11,14], Thu: [10,14,15], Fri: [9,10], Sat: [10], Sun: [] },
    workload: [50, 62, 58, 65, 70, 30, 5],
    freeSlots: 15,
  },
];

// Lifecycle stages
type ServiceStage = 'BOOKED' | 'CHECK_IN' | 'DIAGNOSIS' | 'REPAIR' | 'READY';

interface ActiveBooking {
  booking_id: string;
  vehicle_id: string;
  vehicle_name: string;
  customer_name: string;
  current_stage: string;
  service_type: string;
  estimated_completion: string;
}

export const Dashboard: React.FC = () => {
  const { jobs, mechanics, parts, setSelectedJob } = useStore();
  const [activeBookings, setActiveBookings] = React.useState<ActiveBooking[]>([]);

  // Fetch active bookings
  React.useEffect(() => {
    const fetchBookings = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/serviceops/bookings/active`);
        if (response.ok) {
          const data = await response.json();
          setActiveBookings(data.bookings);
        }
      } catch (error) {
        console.error('Failed to fetch bookings:', error);
      }
    };

    fetchBookings();
    const interval = setInterval(fetchBookings, 5000); // Polling every 5s for real-time updates
    return () => clearInterval(interval);
  }, []);

  const handleAdvanceStage = async (vehicleId: string, newStage: string) => {
    const booking = activeBookings.find(b => b.vehicle_id === vehicleId);
    if (!booking) return;

    try {
      await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/serviceops/booking/${booking.booking_id}/lifecycle`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: newStage })
      });
      
      // Optimistic update
      setActiveBookings(prev => prev.map(b => 
        b.vehicle_id === vehicleId ? { ...b, current_stage: newStage } : b
      ));
    } catch (e) {
      console.error('Failed to update stage', e);
    }
  };

  // Calculate metrics
  const activeJobs = jobs.filter((j) => j.state !== 'COMPLETED' && j.state !== 'CANCELLED');
  const criticalJobs = jobs.filter((j) => j.severity === 'CRITICAL' || j.severity === 'EMERGENCY');
  const mechanicUtilization = Math.round(
    (mechanics.reduce((acc, m) => acc + m.current_load, 0) /
      mechanics.reduce((acc, m) => acc + m.max_daily_capacity, 0)) *
      100
  );
  const lowStockParts = parts.filter((p) => p.quantity <= p.reorder_point);

  return (
    <div className={styles.page}>
      <Header title="Command Center" />

      <div className={styles.content}>
        {/* Metrics Grid */}
        <motion.section
          className={styles.metricsGrid}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ staggerChildren: 0.1 }}
        >
          <MetricCard
            title="Active Jobs"
            value={activeJobs.length}
            subtitle={`${criticalJobs.length} critical`}
            icon={ClipboardList}
            color="primary"
            trend={{ value: 12, isPositive: true }}
          />
          <MetricCard
            title="Avg. Completion"
            value="4.2h"
            subtitle="Target: 5h"
            icon={Clock}
            color="success"
            trend={{ value: 8, isPositive: true }}
          />
          <MetricCard
            title="Tech Utilization"
            value={`${mechanicUtilization}%`}
            subtitle={`${mechanics.length} technicians`}
            icon={Users}
            color="info"
          />
          <MetricCard
            title="Low Stock Alerts"
            value={lowStockParts.length}
            subtitle="Parts below threshold"
            icon={Package}
            color={lowStockParts.length > 0 ? 'warning' : 'success'}
          />
          <MetricCard
            title="Today's Revenue"
            value="₹85.4K"
            subtitle="23 jobs completed"
            icon={TrendingUp}
            color="success"
            trend={{ value: 15, isPositive: true }}
          />
          <MetricCard
            title="Customer Rating"
            value="4.8"
            subtitle="Based on 156 reviews"
            icon={Star}
            color="primary"
          />
        </motion.section>

        {/* Service Centers Grid - 5 Centers */}
        <section className={styles.serviceCentersSection}>
          <div className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>Service Centers (5)</h2>
            <button className={styles.viewAllBtn}>Manage Centers</button>
          </div>
          <div className={styles.centersGrid}>
            {MOCK_SERVICE_CENTERS.map((center) => (
              <ServiceCenterCard
                key={center.id}
                center={center}
                compact={true}
                onSelect={(id) => console.log('Selected center:', id)}
              />
            ))}
          </div>
        </section>

        {/* Vehicle Lifecycle Tracker */}
        <section className={styles.lifecycleSection}>
          <div className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>Vehicles in Service</h2>
            <span className={styles.liveIndicator}>● LIVE</span>
          </div>
          <VehicleLifecycleTracker
            vehicles={activeBookings.map(b => ({
              vehicleId: b.vehicle_id,
              vehicleName: b.vehicle_name,
              customerName: b.customer_name,
              currentStage: b.current_stage as ServiceStage,
              estimatedCompletion: b.estimated_completion,
              serviceType: b.service_type,
              bookingTime: '' // Not needed for display
            }))}
            demoMode={true}
            onAdvanceStage={handleAdvanceStage}
            onSelectVehicle={(id) => console.log('Selected vehicle:', id)}
          />
        </section>

        {/* Kanban Board */}
        <section className={styles.kanbanSection}>
          <div className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>Active Jobs Pipeline</h2>
            <button className={styles.viewAllBtn}>View All</button>
          </div>
          <KanbanBoard jobs={jobs} onJobClick={setSelectedJob} />
        </section>

        {/* Bottom Grid */}
        <div className={styles.bottomGrid}>
          {/* Recent Critical Jobs */}
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Critical Jobs</h2>
            </div>
            <div className={styles.jobsList}>
              {criticalJobs.slice(0, 3).map((job) => (
                <JobCard
                  key={job.job_id}
                  job={job}
                  onClick={() => setSelectedJob(job)}
                />
              ))}
              {criticalJobs.length === 0 && (
                <div className={styles.emptyState}>No critical jobs 🎉</div>
              )}
            </div>
          </section>

          {/* Technician Status */}
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Technician Status</h2>
            </div>
            <div className={styles.techList}>
              {mechanics.map((mech) => (
                <div key={mech.id} className={styles.techItem}>
                  <div className={styles.techAvatar}>
                    {mech.name.charAt(0)}
                  </div>
                  <div className={styles.techInfo}>
                    <span className={styles.techName}>{mech.name}</span>
                    <span className={styles.techRole}>{mech.certification_level}</span>
                  </div>
                  <div className={styles.techLoad}>
                    <div className={styles.loadBar}>
                      <div
                        className={styles.loadFill}
                        style={{
                          width: `${(mech.current_load / mech.max_daily_capacity) * 100}%`,
                          backgroundColor:
                            mech.current_load >= mech.max_daily_capacity
                              ? 'var(--color-danger)'
                              : 'var(--color-success)',
                        }}
                      />
                    </div>
                    <span className={styles.loadText}>
                      {mech.current_load}/{mech.max_daily_capacity}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
