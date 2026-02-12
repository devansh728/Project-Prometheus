/**
 * Kanban Board Component - Job workflow visualization
 */
import React from 'react';
import { motion } from 'framer-motion';
import { ServiceJob, ServiceState } from '../../types/api';
import { JobCard } from '../cards/JobCard';
import styles from './KanbanBoard.module.css';

interface KanbanBoardProps {
  jobs: ServiceJob[];
  onJobClick?: (job: ServiceJob) => void;
}

interface Column {
  id: ServiceState;
  title: string;
  color: string;
}

const columns: Column[] = [
  { id: ServiceState.REQUESTED, title: 'Requested', color: 'var(--color-requested)' },
  { id: ServiceState.CHECK_IN, title: 'Check-In', color: 'var(--color-checkin)' },
  { id: ServiceState.DIAGNOSIS, title: 'Diagnosis', color: 'var(--color-diagnosis)' },
  { id: ServiceState.REPAIR_IN_PROGRESS, title: 'In Progress', color: 'var(--color-repair)' },
  { id: ServiceState.QUALITY_CHECK, title: 'QC', color: 'var(--color-quality)' },
];

export const KanbanBoard: React.FC<KanbanBoardProps> = ({ jobs, onJobClick }) => {
  const getJobsForColumn = (state: ServiceState) => {
    return jobs.filter((job) => job.state === state);
  };

  return (
    <div className={styles.board}>
      {columns.map((column, index) => {
        const columnJobs = getJobsForColumn(column.id);
        
        return (
          <motion.div
            key={column.id}
            className={styles.column}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className={styles.columnHeader}>
              <div className={styles.columnTitle}>
                <span
                  className={styles.columnDot}
                  style={{ backgroundColor: column.color }}
                />
                {column.title}
              </div>
              <span className={styles.columnCount}>{columnJobs.length}</span>
            </div>

            <div className={styles.columnContent}>
              {columnJobs.map((job) => (
                <JobCard
                  key={job.job_id}
                  job={job}
                  onClick={() => onJobClick?.(job)}
                  compact
                />
              ))}
              
              {columnJobs.length === 0 && (
                <div className={styles.emptyColumn}>
                  No jobs
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};
