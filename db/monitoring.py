"""
SentinEV - Monitoring Database Extension
=========================================
Extends main database with tables for inference logs, agent actions, and drift metrics.
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Phase 13: UEBA and monitoring utilities
from db.monitoring_utils import (
    UEBA_RULES,
    SLA_THRESHOLDS,
    check_frequency_anomaly,
    check_off_hours_access,
    check_sensitive_resource,
    check_cross_tenant,
    check_sla_status,
    generate_fingerprint,
    compare_fingerprints,
    SLAStatus,
    DailyReport,
)


class MonitoringDB:
    """
    Monitoring database extension for SentinEV.

    Handles:
    - Inference logs (predictions, latencies)
    - Agent actions (UEBA monitoring)
    - Drift metrics
    - Alerts log
    """

    def __init__(self, db_path: str = "data/sentinev.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_monitoring_tables()

    def _get_connection(self):
        """Get database connection with optimizations."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_monitoring_tables(self):
        """Initialize monitoring tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Inference logs
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                vehicle_id TEXT NOT NULL,
                failure_prob REAL,
                severity TEXT,
                anomaly_score REAL,
                is_anomaly INTEGER,
                latency_ms REAL,
                window_type TEXT,
                feature_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Agent actions for UEBA
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                agent_name TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_resource TEXT,
                details TEXT,
                ueba_flagged INTEGER DEFAULT 0,
                ueba_reason TEXT,
                risk_score REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Drift metrics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                feature_name TEXT NOT NULL,
                baseline_value REAL,
                current_value REAL,
                drift_pct REAL,
                drift_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Alerts log
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                vehicle_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                probability REAL,
                message TEXT,
                action_taken TEXT,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Fleet/Tenant table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fleet_tenants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT UNIQUE NOT NULL,
                tenant_name TEXT NOT NULL,
                vehicle_ids TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_inference_vehicle ON inference_logs(vehicle_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_inference_time ON inference_logs(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_actions_agent ON agent_actions(agent_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_vehicle ON alerts_log(vehicle_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts_log(severity)"
        )

        conn.commit()
        conn.close()

    # =========================================================================
    # Inference Logging
    # =========================================================================

    def log_inference(
        self,
        vehicle_id: str,
        failure_prob: float,
        severity: str,
        anomaly_score: float = 0,
        is_anomaly: bool = False,
        latency_ms: float = 0,
        window_type: str = "medium",
        feature_count: int = 0,
    ) -> int:
        """Log an inference result."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO inference_logs 
            (timestamp, vehicle_id, failure_prob, severity, anomaly_score, 
             is_anomaly, latency_ms, window_type, feature_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                vehicle_id,
                failure_prob,
                severity,
                anomaly_score,
                1 if is_anomaly else 0,
                latency_ms,
                window_type,
                feature_count,
            ),
        )

        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id

    def get_inference_stats(self, hours: int = 24) -> Dict:
        """Get inference statistics for the last N hours."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = time.time() - (hours * 3600)

        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_inferences,
                AVG(failure_prob) as avg_failure_prob,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count,
                COUNT(DISTINCT vehicle_id) as unique_vehicles
            FROM inference_logs
            WHERE timestamp > ?
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else {}

    # =========================================================================
    # Agent Action Logging (UEBA)
    # =========================================================================

    def log_agent_action(
        self,
        agent_name: str,
        action_type: str,
        target_resource: str = None,
        details: Dict = None,
        ueba_flagged: bool = False,
        ueba_reason: str = None,
        risk_score: float = 0,
    ) -> int:
        """Log an agent action for UEBA monitoring."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO agent_actions
            (timestamp, agent_name, action_type, target_resource, details,
             ueba_flagged, ueba_reason, risk_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                agent_name,
                action_type,
                target_resource,
                json.dumps(details) if details else None,
                1 if ueba_flagged else 0,
                ueba_reason,
                risk_score,
            ),
        )

        action_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return action_id

    def get_ueba_alerts(self, limit: int = 100) -> List[Dict]:
        """Get recent UEBA flagged actions."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM agent_actions
            WHERE ueba_flagged = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Alert Logging
    # =========================================================================

    def log_alert(
        self,
        vehicle_id: str,
        alert_type: str,
        severity: str,
        probability: float = 0,
        message: str = None,
        action_taken: str = None,
    ) -> int:
        """Log an alert."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO alerts_log
            (timestamp, vehicle_id, alert_type, severity, probability, message, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                vehicle_id,
                alert_type,
                severity,
                probability,
                message,
                action_taken,
            ),
        )

        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return alert_id

    def get_alerts(
        self, vehicle_id: str = None, severity: str = None, limit: int = 100
    ) -> List[Dict]:
        """Get alerts with optional filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM alerts_log WHERE 1=1"
        params = []

        if vehicle_id:
            query += " AND vehicle_id = ?"
            params.append(vehicle_id)
        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Drift Metrics
    # =========================================================================

    def log_drift(
        self,
        feature_name: str,
        baseline_value: float,
        current_value: float,
        drift_type: str = "data_drift",
    ) -> int:
        """Log drift metric."""
        drift_pct = (
            abs(current_value - baseline_value) / max(abs(baseline_value), 0.001) * 100
        )

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO drift_metrics
            (timestamp, feature_name, baseline_value, current_value, drift_pct, drift_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                feature_name,
                baseline_value,
                current_value,
                drift_pct,
                drift_type,
            ),
        )

        drift_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return drift_id

    def get_drift_summary(self, hours: int = 24) -> List[Dict]:
        """Get drift summary for the last N hours."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = time.time() - (hours * 3600)

        cursor.execute(
            """
            SELECT 
                feature_name,
                AVG(baseline_value) as avg_baseline,
                AVG(current_value) as avg_current,
                AVG(drift_pct) as avg_drift_pct,
                MAX(drift_pct) as max_drift_pct
            FROM drift_metrics
            WHERE timestamp > ?
            GROUP BY feature_name
            HAVING avg_drift_pct > 10
            ORDER BY avg_drift_pct DESC
        """,
            (cutoff,),
        )

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Phase 13: UEBA Rule Checking
    # =========================================================================

    def check_ueba_rules(
        self,
        agent_name: str,
        action_type: str,
        target_resource: str = None,
        agent_tenant: str = "default",
        resource_tenant: str = "default",
    ) -> Dict[str, Any]:
        """Check all UEBA rules for an action."""
        violations = []
        total_risk = 0.0

        # Rule 1: Check frequency
        action_count = self._get_recent_action_count(agent_name, 60)
        is_freq_violation, freq_reason = check_frequency_anomaly(
            agent_name, action_count
        )
        if is_freq_violation:
            violations.append({"rule": "UEBA-001", "reason": freq_reason})
            total_risk += 6.0

        # Rule 2: Check off-hours
        is_offhours, offhours_reason = check_off_hours_access()
        if is_offhours:
            violations.append({"rule": "UEBA-002", "reason": offhours_reason})
            total_risk += 4.0

        # Rule 3: Check sensitive resource
        if target_resource:
            is_sensitive, sensitive_reason = check_sensitive_resource(target_resource)
            if is_sensitive:
                violations.append({"rule": "UEBA-003", "reason": sensitive_reason})
                total_risk += 8.0

        # Rule 4: Check cross-tenant
        is_cross, cross_reason = check_cross_tenant(agent_tenant, resource_tenant)
        if is_cross:
            violations.append({"rule": "UEBA-004", "reason": cross_reason})
            total_risk += 9.0

        return {
            "flagged": len(violations) > 0,
            "violations": violations,
            "total_risk_score": min(total_risk, 10.0),
            "action_count_1min": action_count,
        }

    def _get_recent_action_count(self, agent_name: str, seconds: int) -> int:
        """Get action count for agent in time window."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = time.time() - seconds
        cursor.execute(
            "SELECT COUNT(*) FROM agent_actions WHERE agent_name = ? AND timestamp > ?",
            (agent_name, cutoff),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # =========================================================================
    # Phase 13: Agent Fingerprinting
    # =========================================================================

    def log_fingerprint(
        self,
        agent_id: str,
        ip_address: str = "127.0.0.1",
        user_agent: str = "SentinEV-Agent/1.0",
        session_id: str = None,
    ) -> Dict[str, Any]:
        """Log agent fingerprint and detect changes."""
        # Generate current fingerprint
        current_fp = generate_fingerprint(agent_id, ip_address, user_agent, session_id)

        # Check for existing fingerprint in DB (simplified - store in agent_actions)
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT details FROM agent_actions WHERE agent_name = ? AND action_type = 'fingerprint' ORDER BY timestamp DESC LIMIT 1",
            (agent_id,),
        )
        row = cursor.fetchone()

        is_new = True
        is_changed = False
        changes = []

        if row and row[0]:
            stored_data = json.loads(row[0])
            if stored_data.get("fingerprint_hash") != current_fp.fingerprint_hash:
                is_changed = True
                changes = ["fingerprint_hash"]
            if stored_data.get("ip_hash") != current_fp.ip_hash:
                is_changed = True
                changes.append("ip_hash")
            is_new = False

        # Log new fingerprint
        self.log_agent_action(
            agent_name=agent_id,
            action_type="fingerprint",
            details={
                "fingerprint_hash": current_fp.fingerprint_hash,
                "ip_hash": current_fp.ip_hash,
                "user_agent": current_fp.user_agent,
                "session_id": current_fp.session_id,
            },
            ueba_flagged=is_changed,
            ueba_reason=(
                "Fingerprint changed: " + ", ".join(changes) if is_changed else None
            ),
            risk_score=5.0 if is_changed else 0,
        )

        conn.close()

        return {
            "agent_id": agent_id,
            "fingerprint_hash": current_fp.fingerprint_hash,
            "is_new": is_new,
            "is_changed": is_changed,
            "changes": changes,
            "alert": is_changed,
        }

    # =========================================================================
    # Phase 13: SLA Enforcement
    # =========================================================================

    def log_sla_check(
        self,
        operation: str,
        latency_ms: float,
        vehicle_id: str = None,
    ) -> Dict[str, Any]:
        """Check SLA and log result."""
        status, message = check_sla_status(operation, latency_ms)

        if status != SLAStatus.COMPLIANT:
            # Log as alert
            self.log_alert(
                vehicle_id=vehicle_id or "system",
                alert_type="sla_" + status.value,
                severity="high" if status == SLAStatus.VIOLATION else "medium",
                probability=latency_ms / 1000,
                message=message,
                action_taken="alert_logged",
            )

        return {
            "operation": operation,
            "latency_ms": latency_ms,
            "status": status.value,
            "message": message,
            "alert": status != SLAStatus.COMPLIANT,
        }

    def get_sla_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get SLA compliance summary."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = time.time() - (hours * 3600)

        # Get violation counts
        cursor.execute(
            "SELECT COUNT(*) FROM alerts_log WHERE alert_type LIKE 'sla_%' AND timestamp > ?",
            (cutoff,),
        )
        total_issues = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM alerts_log WHERE alert_type = 'sla_violation' AND timestamp > ?",
            (cutoff,),
        )
        violations = cursor.fetchone()[0]

        conn.close()

        return {
            "period_hours": hours,
            "total_issues": total_issues,
            "violations": violations,
            "warnings": total_issues - violations,
            "compliance_rate": (
                1.0 if total_issues == 0 else max(0, 1 - violations / 100)
            ),
        }

    # =========================================================================
    # Phase 13: Daily Operational Reports
    # =========================================================================

    def generate_daily_report(self, report_date: str = None) -> Dict[str, Any]:
        """Generate comprehensive daily operational report."""
        if not report_date:
            report_date = datetime.now().strftime("%Y-%m-%d")

        # Inference stats
        inference_stats = self.get_inference_stats(hours=24)

        # Alert summary
        alerts = self.get_alerts(limit=1000)
        alert_summary = {
            "total": len(alerts),
            "by_severity": {},
        }
        for alert in alerts:
            sev = alert.get("severity", "unknown")
            alert_summary["by_severity"][sev] = (
                alert_summary["by_severity"].get(sev, 0) + 1
            )

        # UEBA events
        ueba_events = self.get_ueba_alerts(limit=50)

        # Drift status
        drift_status = {
            "features_with_drift": self.get_drift_summary(hours=24),
            "retrain_recommended": len(self.get_drift_summary()) > 3,
        }

        # SLA compliance
        sla_compliance = self.get_sla_summary(hours=24)

        # Top vehicles by alerts
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = time.time() - 86400
        cursor.execute(
            "SELECT vehicle_id, COUNT(*) as alert_count FROM alerts_log WHERE timestamp > ? GROUP BY vehicle_id ORDER BY alert_count DESC LIMIT 10",
            (cutoff,),
        )
        top_vehicles = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Recommendations
        recommendations = []
        if drift_status["retrain_recommended"]:
            recommendations.append("Model retraining recommended due to data drift")
        if sla_compliance["violations"] > 0:
            recommendations.append(
                f"Address {sla_compliance['violations']} SLA violations"
            )
        if len(ueba_events) > 10:
            recommendations.append("Review elevated UEBA activity")

        return {
            "success": True,
            "report_date": report_date,
            "generated_at": datetime.now().isoformat(),
            "inference_stats": inference_stats,
            "alert_summary": alert_summary,
            "ueba_events_count": len(ueba_events),
            "ueba_events": ueba_events[:5],
            "drift_status": drift_status,
            "sla_compliance": sla_compliance,
            "top_vehicles": top_vehicles,
            "recommendations": recommendations,
        }


# Singleton instance
_monitoring_db: Optional[MonitoringDB] = None


def get_monitoring_db() -> MonitoringDB:
    """Get or create monitoring DB instance."""
    global _monitoring_db
    if _monitoring_db is None:
        _monitoring_db = MonitoringDB()
    return _monitoring_db


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Monitoring DB")
    print("=" * 60)

    db = get_monitoring_db()

    # Test inference logging
    log_id = db.log_inference(
        vehicle_id="TEST_001",
        failure_prob=0.75,
        severity="high",
        anomaly_score=0.8,
        latency_ms=15.5,
    )
    print(f"✅ Logged inference: {log_id}")

    # Test agent action logging
    action_id = db.log_agent_action(
        agent_name="scheduling_agent",
        action_type="book_appointment",
        target_resource="service_center_1",
        details={"slot_id": "slot_123"},
    )
    print(f"✅ Logged agent action: {action_id}")

    # Test alert logging
    alert_id = db.log_alert(
        vehicle_id="TEST_001",
        alert_type="failure_prediction",
        severity="high",
        probability=0.75,
        message="Motor bearing wear detected",
    )
    print(f"✅ Logged alert: {alert_id}")

    # Get stats
    stats = db.get_inference_stats()
    print(f"✅ Inference stats: {stats}")

    print("\n✅ Monitoring DB test complete")
