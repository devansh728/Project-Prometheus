/**
 * Autonomy Dashboard - Enhanced Agentic UI for ServiceOpsAI
 * Features: Vehicle Operations Board, Service Center Operations, Inspection Modal, Fallback UI
 */
import React, { useState, useEffect, useCallback } from 'react';
import styles from './AutonomyDashboard.module.css';
import { VehicleOperationsBoard, VehicleCard } from '../../components/boards/VehicleOperationsBoard';
import { ServiceCenterOperationsRow, ServiceCenter } from '../../components/rows/ServiceCenterOperationsRow';
import { InspectionModal } from '../../components/modals/InspectionModal';
import { FallbackUI } from '../../components/overlays/FallbackUI';
import { DecisionFeed } from '../../components/cards/DecisionFeed';
import { SimulationControls } from '../../components/cards/SimulationControls';
import { ChatbotPanel } from '../../components/cards/ChatbotPanel';
import { BidScoreChart, WorkloadPieChart, RiskTrendChart } from '../../components/charts/AgentInsightsCharts';

const API_BASE = 'http://localhost:8000/api/v1/serviceops';

export const AutonomyDashboard: React.FC = () => {
  // State
  const [vehicles, setVehicles] = useState<VehicleCard[]>([]);
  const [centers, setCenters] = useState<ServiceCenter[]>([]);
  const [decisions, setDecisions] = useState<any[]>([]);
  const [simulationState, setSimulationState] = useState<any>({});
  const [isLoading, setIsLoading] = useState(false);
  const [fallbackActive, setFallbackActive] = useState(false);
  const [selectedVehicle, setSelectedVehicle] = useState<VehicleCard | null>(null);
  const [animatingStep, setAnimatingStep] = useState(-1);

  // Fetch demo vehicles
  const fetchVehicles = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/demo/vehicles`);
      const data = await res.json();
      setVehicles(data.vehicles || []);
    } catch (err) {
      console.error('Failed to fetch vehicles:', err);
    }
  }, []);

  // Fetch demo centers
  const fetchCenters = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/demo/centers`);
      const data = await res.json();
      setCenters(data.centers || []);
    } catch (err) {
      console.error('Failed to fetch centers:', err);
    }
  }, []);

  // Fetch decisions
  const fetchDecisions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/decisions/log?limit=50`);
      const data = await res.json();
      setDecisions(data.decisions || []);
    } catch (err) {
      console.error('Failed to fetch decisions:', err);
    }
  }, []);

  // Fetch demo state
  const fetchDemoState = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/demo/state`);
      const data = await res.json();
      setSimulationState(data.simulation || {});
      setDecisions(data.decisions || []);
    } catch (err) {
      console.error('Failed to fetch demo state:', err);
    }
  }, []);

  // Poll for updates
  useEffect(() => {
    fetchVehicles();
    fetchCenters();
    fetchDecisions();

    const interval = setInterval(() => {
      fetchDemoState();
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchVehicles, fetchCenters, fetchDecisions, fetchDemoState]);

  // Handlers
  const handleRunPlanningCycle = async () => {
    setIsLoading(true);
    try {
      // Run step-by-step animation
      const steps = [0, 1, 2, 3, 4, 5];
      for (const step of steps) {
        setAnimatingStep(step);
        await new Promise(resolve => setTimeout(resolve, step === 0 ? 2000 : step === 1 ? 3000 : step === 2 ? 4000 : step === 3 ? 3000 : step === 4 ? 2000 : 2000));
        
        // Update vehicle states progressively
        if (step === 1) {
          setVehicles(prev => prev.map((v, i) => ({ ...v, decisionState: i < 5 ? 'ROUTING' as any : v.decisionState })));
        } else if (step === 2) {
          setVehicles(prev => prev.map((v, i) => ({ ...v, decisionState: i < 5 ? 'BIDDING' as any : v.decisionState })));
        } else if (step === 3) {
          setVehicles(prev => prev.map((v, i) => ({ ...v, decisionState: i < 5 ? 'ASSIGNED' as any : v.decisionState })));
        }
      }
      
      // Run the actual scenario
      const res = await fetch(`${API_BASE}/demo/scenario-1`, { method: 'POST' });
      const data = await res.json();
      setSimulationState(data.final_state?.simulation || {});
      setDecisions(data.final_state?.decisions || []);
      await fetchVehicles();
      await fetchCenters();
    } finally {
      setIsLoading(false);
      setAnimatingStep(-1);
    }
  };

  const handleRunScenario2 = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/demo/scenario-2`, { method: 'POST' });
      const data = await res.json();
      setSimulationState(data.final_state?.simulation || {});
      setDecisions(data.final_state?.decisions || []);
      await fetchVehicles();
      await fetchCenters();
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_BASE}/demo/reset`, { method: 'POST' });
      await fetchVehicles();
      await fetchCenters();
      setDecisions([]);
      setSimulationState({});
      setFallbackActive(false);
    } catch (err) {
      console.error('Failed to reset:', err);
    }
  };

  const handleTriggerFallback = async () => {
    try {
      await fetch(`${API_BASE}/demo/trigger-fallback`, { method: 'POST' });
      setFallbackActive(true);
      await fetchDecisions();
    } catch (err) {
      console.error('Failed to trigger fallback:', err);
    }
  };

  const handleVehicleClick = (vehicle: VehicleCard) => {
    if (vehicle.decisionState === 'ASSIGNED') {
      setSelectedVehicle(vehicle);
    }
  };

  const handleInspectionSubmit = async (actualDiagnosis: string) => {
    if (!selectedVehicle) return { similarityScore: 0, durationDelta: 0, affectedTasks: 0 };

    try {
      const res = await fetch(`${API_BASE}/inspection/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vehicle_id: selectedVehicle.vehicleId,
          predicted_diagnosis: {
            failure_type: selectedVehicle.failureType,
            severity: selectedVehicle.severity,
          },
          actual_diagnosis: actualDiagnosis,
        }),
      });
      const data = await res.json();
      await fetchDecisions();
      return {
        similarityScore: data.similarityScore,
        durationDelta: data.durationDelta,
        affectedTasks: data.affectedTasks,
      };
    } catch (err) {
      console.error('Failed to submit inspection:', err);
      return { similarityScore: 0, durationDelta: 0, affectedTasks: 0 };
    }
  };

  const handleChatbotQuery = async (query: string, centerId: string) => {
    const res = await fetch(`${API_BASE}/chatbot/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, center_id: centerId }),
    });
    return await res.json();
  };

  return (
    <div className={styles.container}>
      {/* Fallback UI */}
      <FallbackUI
        isActive={fallbackActive}
        reason="simulated"
        onDismiss={() => setFallbackActive(false)}
      />

      <div className={styles.header}>
        <h1 className={styles.title}>🤖 ServiceOpsAI Autonomy Dashboard</h1>
        <p className={styles.subtitle}>
          Autonomous multi-vehicle, multi-center operations with progressive decision reveal
        </p>
      </div>

      {/* Simulation Controls */}
      <div className={styles.controlsSection}>
        <SimulationControls
          onRunScenario1={handleRunPlanningCycle}
          onRunScenario2={handleRunScenario2}
          onReset={handleReset}
          currentScenario={simulationState.scenario}
          isRunning={simulationState.running || isLoading}
          currentStep={animatingStep >= 0 ? animatingStep : simulationState.current_step}
          totalSteps={6}
          stepDescription={animatingStep >= 0 ? ['Intake Agent', 'Routing Agent', 'Bidding Agent', 'Scheduling Agent', 'Labour Agent', 'Supply Agent'][animatingStep] : simulationState.step_description}
        />
        <button
          className={styles.fallbackBtn}
          onClick={handleTriggerFallback}
          disabled={fallbackActive}
        >
          Simulate Fallback
        </button>
      </div>

      {/* Vehicle Operations Board */}
      <div className={styles.section}>
        <VehicleOperationsBoard
          vehicles={vehicles}
          animatingStep={animatingStep}
          onVehicleClick={handleVehicleClick}
        />
      </div>

      {/* Service Center Operations Row */}
      <div className={styles.section}>
        <ServiceCenterOperationsRow
          centers={centers}
          animatingStep={animatingStep}
        />
      </div>

      {/* Real-time Agent Insights */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2>Real-time Agent Insights</h2>
        </div>
        <div className={styles.chartsGrid}>
          <div className={styles.chartWrapper}>
            <BidScoreChart 
              title="Bidding Landscape (Live Auction)" 
              data={selectedVehicle?.bidHistory || [
                { center: 'Service Center 1', score: 85 },
                { center: 'Service Center 2', score: 72 },
                { center: 'Service Center 3', score: 64 },
              ]} 
            />
          </div>
          <div className={styles.chartWrapper}>
            <WorkloadPieChart 
              title="Global Workload Distribution" 
              data={centers.map(c => ({ name: c.name, value: c.currentLoad }))} 
            />
          </div>
          <div className={styles.chartWrapper}>
            <RiskTrendChart 
              title="Fleet Risk Reduction Trend" 
              data={[
                { day: 'Day 1', risk: 85, optimized: 85 },
                { day: 'Day 2', risk: 78, optimized: 65 },
                { day: 'Day 3', risk: 72, optimized: 45 },
                { day: 'Day 4', risk: 68, optimized: 30 },
                { day: 'Day 5', risk: 65, optimized: 20 },
                { day: 'Day 6', risk: 62, optimized: 15 },
                { day: 'Day 7', risk: 60, optimized: 10 },
              ]} 
            />
          </div>
        </div>
      </div>

      {/* Bottom Panel */}
      <div className={styles.bottomPanel}>
        <div className={styles.decisionColumn}>
          <DecisionFeed
            decisions={decisions}
            maxItems={30}
            autoScroll={true}
          />
        </div>
        <div className={styles.chatColumn}>
          <ChatbotPanel
            centerId="SC001"
            onQuery={handleChatbotQuery}
          />
        </div>
      </div>

      {/* Inspection Modal */}
      <InspectionModal
        vehicle={selectedVehicle}
        onClose={() => setSelectedVehicle(null)}
        onSubmit={handleInspectionSubmit}
      />
    </div>
  );
};

export default AutonomyDashboard;
