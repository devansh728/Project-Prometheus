"""
SentinEV - Database Schema and Models
SQLite database for appointments, service centers, and feedback
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import threading


class AppointmentStatus(Enum):
    """Status of a service appointment."""

    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ServiceUrgency(Enum):
    """Urgency level for service."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ServiceCenter:
    """Service center data model."""

    id: str
    name: str
    location: str
    phone: str
    specialties: List[str]
    capacity_per_day: int
    operating_hours: Dict[str, str]
    rating: float


@dataclass
class TimeSlot:
    """Available time slot."""

    id: str
    center_id: str
    date: str
    start_time: str
    end_time: str
    available: bool
    component_type: str


@dataclass
class Appointment:
    """Service appointment."""

    id: str
    vehicle_id: str
    center_id: str
    slot_id: str
    component: str
    diagnosis_summary: str
    estimated_cost: str
    urgency: str
    status: str
    created_at: str
    scheduled_date: str
    scheduled_time: str
    completed_at: Optional[str]
    notes: str


@dataclass
class ServiceFeedback:
    """Customer feedback for service."""

    id: str
    appointment_id: str
    vehicle_id: str
    rating: int
    comments: str
    submitted_at: str


class Database:
    """
    SQLite database manager for SentinEV.
    Handles service centers, appointments, and feedback.
    """

    def __init__(self, db_path: str = "data/sentinev.db"):
        self.db_path = db_path
        self._write_lock = threading.RLock()  # Reentrant lock for write operations
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self._seed_data_if_empty()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory, timeout, and WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Service Centers table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS service_centers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                phone TEXT,
                specialties TEXT,
                capacity_per_day INTEGER DEFAULT 10,
                operating_hours TEXT,
                rating REAL DEFAULT 4.5
            )
        """
        )

        # Time Slots table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS time_slots (
                id TEXT PRIMARY KEY,
                center_id TEXT NOT NULL,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                available INTEGER DEFAULT 1,
                component_type TEXT DEFAULT 'general',
                FOREIGN KEY (center_id) REFERENCES service_centers(id)
            )
        """
        )

        # Appointments table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                vehicle_id TEXT NOT NULL,
                center_id TEXT NOT NULL,
                slot_id TEXT,
                component TEXT NOT NULL,
                diagnosis_summary TEXT,
                estimated_cost TEXT,
                urgency TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'scheduled',
                created_at TEXT NOT NULL,
                scheduled_date TEXT,
                scheduled_time TEXT,
                completed_at TEXT,
                notes TEXT,
                stage TEXT DEFAULT 'INTAKE',
                ticket_id TEXT,
                booked_via TEXT DEFAULT 'web',
                FOREIGN KEY (center_id) REFERENCES service_centers(id),
                FOREIGN KEY (slot_id) REFERENCES time_slots(id)
            )
        """
        )

        # Feedback table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                appointment_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comments TEXT,
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            )
        """
        )

        # Maintenance History table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS maintenance_history (
                id TEXT PRIMARY KEY,
                vehicle_id TEXT NOT NULL,
                appointment_id TEXT,
                service_type TEXT NOT NULL,
                component TEXT NOT NULL,
                description TEXT,
                cost TEXT,
                performed_at TEXT NOT NULL,
                next_service_due TEXT,
                mileage INTEGER
            )
        """
        )

        # Service Demand Log (for forecasting)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS demand_log (
                id TEXT PRIMARY KEY,
                center_id TEXT NOT NULL,
                date TEXT NOT NULL,
                hour INTEGER NOT NULL,
                component_type TEXT,
                appointment_count INTEGER DEFAULT 0,
                FOREIGN KEY (center_id) REFERENCES service_centers(id)
            )
        """
        )

        # Service Tickets table (for lifecycle tracking)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS service_tickets (
                id TEXT PRIMARY KEY,
                appointment_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                status TEXT DEFAULT 'INTAKE',
                stage_log TEXT,
                estimated_completion TEXT,
                technician_id TEXT,
                technician_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            )
        """
        )

        # User Profiles table (for personalized feedback)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                vehicle_id TEXT PRIMARY KEY,
                driving_style TEXT DEFAULT 'normal',
                past_feedback TEXT,
                preferences TEXT,
                total_services INTEGER DEFAULT 0,
                avg_satisfaction REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()

    def _seed_data_if_empty(self):
        """Seed initial data if database is empty."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if centers exist
        cursor.execute("SELECT COUNT(*) FROM service_centers")
        if cursor.fetchone()[0] == 0:
            self._seed_service_centers(cursor)
            self._seed_time_slots(cursor)
            self._seed_demand_history(cursor)
            # Note: Dummy appointments disabled - scheduler starts empty
            # self._seed_dummy_appointments(cursor)
            conn.commit()

        conn.close()

    def _seed_service_centers(self, cursor):
        """Seed service center data."""
        centers = [
            {
                "id": "SC-001",
                "name": "EV Master Service Center",
                "location": "123 Electric Ave, Downtown",
                "phone": "+1-555-0101",
                "specialties": json.dumps(["battery", "motor", "general"]),
                "capacity_per_day": 12,
                "operating_hours": json.dumps(
                    {"weekdays": "8:00-18:00", "saturday": "9:00-14:00"}
                ),
                "rating": 4.8,
            },
            {
                "id": "SC-002",
                "name": "QuickCharge Auto",
                "location": "456 Battery Blvd, Midtown",
                "phone": "+1-555-0102",
                "specialties": json.dumps(["battery", "charging", "thermal"]),
                "capacity_per_day": 8,
                "operating_hours": json.dumps(
                    {"weekdays": "7:00-19:00", "saturday": "8:00-16:00"}
                ),
                "rating": 4.6,
            },
            {
                "id": "SC-003",
                "name": "BrakesPro EV",
                "location": "789 Safety Lane, Westside",
                "phone": "+1-555-0103",
                "specialties": json.dumps(["brakes", "suspension", "tires"]),
                "capacity_per_day": 10,
                "operating_hours": json.dumps(
                    {"weekdays": "8:00-17:00", "saturday": "9:00-13:00"}
                ),
                "rating": 4.7,
            },
            {
                "id": "SC-004",
                "name": "PowerTrain Experts",
                "location": "321 Motor Way, Industrial",
                "phone": "+1-555-0104",
                "specialties": json.dumps(["motor", "inverter", "drivetrain"]),
                "capacity_per_day": 6,
                "operating_hours": json.dumps(
                    {"weekdays": "9:00-18:00", "saturday": "closed"}
                ),
                "rating": 4.9,
            },
            {
                "id": "SC-005",
                "name": "AllEV Service Hub",
                "location": "555 Universal Dr, Eastside",
                "phone": "+1-555-0105",
                "specialties": json.dumps(["general", "battery", "brakes", "motor"]),
                "capacity_per_day": 15,
                "operating_hours": json.dumps(
                    {
                        "weekdays": "7:00-20:00",
                        "saturday": "8:00-18:00",
                        "sunday": "10:00-14:00",
                    }
                ),
                "rating": 4.5,
            },
        ]

        for center in centers:
            cursor.execute(
                """
                INSERT INTO service_centers 
                (id, name, location, phone, specialties, capacity_per_day, operating_hours, rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    center["id"],
                    center["name"],
                    center["location"],
                    center["phone"],
                    center["specialties"],
                    center["capacity_per_day"],
                    center["operating_hours"],
                    center["rating"],
                ),
            )

    def _seed_time_slots(self, cursor):
        """Generate time slots for next 7 days."""
        centers = ["SC-001", "SC-002", "SC-003", "SC-004", "SC-005"]
        components = ["general", "battery", "brakes", "motor"]

        for center_id in centers:
            for day_offset in range(7):
                date = (datetime.now() + timedelta(days=day_offset)).strftime(
                    "%Y-%m-%d"
                )

                # Generate slots from 8 AM to 5 PM
                for hour in range(8, 17):
                    slot_id = f"{center_id}-{date}-{hour:02d}"
                    start_time = f"{hour:02d}:00"
                    end_time = f"{hour+1:02d}:00"
                    component = components[hour % len(components)]

                    cursor.execute(
                        """
                        INSERT INTO time_slots 
                        (id, center_id, date, start_time, end_time, available, component_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (slot_id, center_id, date, start_time, end_time, 1, component),
                    )

    def _seed_demand_history(self, cursor):
        """Seed historical demand data for forecasting."""
        centers = ["SC-001", "SC-002", "SC-003", "SC-004", "SC-005"]
        components = ["general", "battery", "brakes", "motor"]

        import random

        # Generate 30 days of historical data
        for day_offset in range(-30, 0):
            date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")

            for center_id in centers:
                for hour in range(8, 18):
                    # Random appointment count with realistic patterns
                    base_count = 2 if 10 <= hour <= 14 else 1  # Peak at midday
                    count = base_count + random.randint(0, 3)
                    component = components[random.randint(0, len(components) - 1)]

                    demand_id = f"demand-{center_id}-{date}-{hour}"
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO demand_log 
                        (id, center_id, date, hour, component_type, appointment_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (demand_id, center_id, date, hour, component, count),
                    )

    def _seed_dummy_appointments(self, cursor):
        """Seed dummy appointments for realistic forecasting."""
        import random

        centers = ["SC-001", "SC-002", "SC-003", "SC-004", "SC-005"]
        center_names = {
            "SC-001": "EV Master Service Center",
            "SC-002": "QuickCharge Auto",
            "SC-003": "BrakesPro EV",
            "SC-004": "PowerTrain Experts",
            "SC-005": "AllEV Service Hub",
        }
        components = ["battery", "brakes", "motor", "thermal", "general"]
        urgencies = ["low", "medium", "high", "critical"]
        vehicles = [f"VIN-{str(i).zfill(3)}" for i in range(1, 21)]  # 20 vehicles

        diagnoses = {
            "battery": [
                "Battery health below optimal. Cell balancing required.",
                "Charging efficiency reduced. Diagnostic recommended.",
                "Battery temperature regulation needs calibration.",
            ],
            "brakes": [
                "Brake pad wear detected. Replacement recommended.",
                "Regenerative braking needs recalibration.",
                "Brake fluid level low. Service required.",
            ],
            "motor": [
                "Motor efficiency below spec. Inspection needed.",
                "Unusual motor vibration detected.",
                "Motor bearing wear detected.",
            ],
            "thermal": [
                "Cooling system efficiency reduced.",
                "HVAC compressor needs service.",
                "Thermal paste replacement due.",
            ],
            "general": [
                "Routine maintenance due.",
                "Software update required.",
                "General inspection recommended.",
            ],
        }

        costs = {
            "battery": ["$200 - $500", "$350 - $800", "$150 - $300"],
            "brakes": ["$100 - $250", "$200 - $400", "$150 - $350"],
            "motor": ["$300 - $700", "$500 - $1200", "$250 - $600"],
            "thermal": ["$150 - $350", "$200 - $450", "$100 - $250"],
            "general": ["$50 - $100", "$75 - $150", "$100 - $200"],
        }

        # Generate past appointments (completed)
        for day_offset in range(-30, -1):
            date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")

            # 3-8 appointments per day
            num_appointments = random.randint(3, 8)

            for _ in range(num_appointments):
                center_id = random.choice(centers)
                component = random.choice(components)
                vehicle = random.choice(vehicles)
                urgency = random.choice(urgencies)
                hour = random.randint(8, 16)

                appt_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
                slot_id = f"{center_id}-{date}-{hour:02d}"

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO appointments 
                    (id, vehicle_id, center_id, slot_id, component, diagnosis_summary, 
                     estimated_cost, urgency, status, created_at, scheduled_date, scheduled_time, completed_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?)
                """,
                    (
                        appt_id,
                        vehicle,
                        center_id,
                        slot_id,
                        component,
                        random.choice(diagnoses[component]),
                        random.choice(costs[component]),
                        urgency,
                        (datetime.now() + timedelta(days=day_offset - 1)).isoformat(),
                        date,
                        f"{hour:02d}:00",
                        (datetime.now() + timedelta(days=day_offset)).isoformat(),
                        f"Completed by technician. Service center: {center_names[center_id]}",
                    ),
                )

        # Generate upcoming scheduled appointments (for next 7 days)
        for day_offset in range(0, 7):
            date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")

            # 2-5 scheduled appointments per day
            num_appointments = random.randint(2, 5)
            used_hours = set()

            for _ in range(num_appointments):
                center_id = random.choice(centers)
                component = random.choice(components)
                vehicle = random.choice(vehicles)
                urgency = random.choice(urgencies[:3])  # No critical for scheduled

                # Find available hour
                hour = random.randint(8, 16)
                attempts = 0
                while hour in used_hours and attempts < 10:
                    hour = random.randint(8, 16)
                    attempts += 1
                used_hours.add(hour)

                appt_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
                slot_id = f"{center_id}-{date}-{hour:02d}"

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO appointments 
                    (id, vehicle_id, center_id, slot_id, component, diagnosis_summary, 
                     estimated_cost, urgency, status, created_at, scheduled_date, scheduled_time, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)
                """,
                    (
                        appt_id,
                        vehicle,
                        center_id,
                        slot_id,
                        component,
                        random.choice(diagnoses[component]),
                        random.choice(costs[component]),
                        urgency,
                        datetime.now().isoformat(),
                        date,
                        f"{hour:02d}:00",
                        "Appointment scheduled via SentinEV system",
                    ),
                )

                # Mark slot as booked
                cursor.execute(
                    "UPDATE time_slots SET available = 0 WHERE id = ?", (slot_id,)
                )

    # ==================== Service Centers ====================

    def get_service_centers(self) -> List[Dict]:
        """Get all service centers."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM service_centers")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "location": row["location"],
                "phone": row["phone"],
                "specialties": json.loads(row["specialties"]),
                "capacity_per_day": row["capacity_per_day"],
                "operating_hours": json.loads(row["operating_hours"]),
                "rating": row["rating"],
            }
            for row in rows
        ]

    def get_center_by_specialty(self, component: str) -> List[Dict]:
        """Get centers that specialize in a component."""
        centers = self.get_service_centers()
        return [
            c
            for c in centers
            if component in c["specialties"] or "general" in c["specialties"]
        ]

    # ==================== Time Slots ====================

    def get_available_slots(
        self,
        center_id: str = None,
        date: str = None,
        component: str = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Get available time slots with filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM time_slots WHERE available = 1"
        params = []

        if center_id:
            query += " AND center_id = ?"
            params.append(center_id)
        if date:
            query += " AND date = ?"
            params.append(date)
        if component:
            query += " AND (component_type = ? OR component_type = 'general')"
            params.append(component)

        query += " ORDER BY date, start_time LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def book_slot(self, slot_id: str) -> bool:
        """Mark a slot as booked."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE time_slots SET available = 0 WHERE id = ?", (slot_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def release_slot(self, slot_id: str) -> bool:
        """Mark a slot as available again."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE time_slots SET available = 1 WHERE id = ?", (slot_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    # ==================== Appointments ====================

    def create_appointment(
        self,
        vehicle_id: str,
        center_id: str,
        slot_id: str,
        component: str,
        diagnosis_summary: str,
        estimated_cost: str,
        urgency: str = "medium",
        notes: str = "",
    ) -> Dict:
        """Create a new appointment with thread-safe locking."""
        with self._write_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                # Get slot details
                cursor.execute(
                    "SELECT date, start_time FROM time_slots WHERE id = ?", (slot_id,)
                )
                slot = cursor.fetchone()

                if not slot:
                    conn.close()
                    return {"error": "Slot not found"}

                appointment_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
                now = datetime.now().isoformat()

                cursor.execute(
                    """
                    INSERT INTO appointments 
                    (id, vehicle_id, center_id, slot_id, component, diagnosis_summary, 
                     estimated_cost, urgency, status, created_at, scheduled_date, scheduled_time, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)
                """,
                    (
                        appointment_id,
                        vehicle_id,
                        center_id,
                        slot_id,
                        component,
                        diagnosis_summary,
                        estimated_cost,
                        urgency,
                        now,
                        slot["date"],
                        slot["start_time"],
                        notes,
                    ),
                )

                # Mark slot as booked (inline to avoid nested connection)
                cursor.execute(
                    "UPDATE time_slots SET available = 0 WHERE id = ?", (slot_id,)
                )

                conn.commit()
                conn.close()

                return {
                    "id": appointment_id,
                    "vehicle_id": vehicle_id,
                    "center_id": center_id,
                    "slot_id": slot_id,
                    "component": component,
                    "scheduled_date": slot["date"],
                    "scheduled_time": slot["start_time"],
                    "status": "scheduled",
                }
            except Exception as e:
                conn.close()
                raise e

    def get_appointments(
        self, vehicle_id: str = None, status: str = None
    ) -> List[Dict]:
        """Get appointments with optional filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT a.*, c.name as center_name FROM appointments a JOIN service_centers c ON a.center_id = c.id WHERE 1=1"
        params = []

        if vehicle_id:
            query += " AND a.vehicle_id = ?"
            params.append(vehicle_id)
        if status:
            query += " AND a.status = ?"
            params.append(status)

        query += " ORDER BY a.scheduled_date, a.scheduled_time"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_appointment_status(self, appointment_id: str, status: str) -> bool:
        """Update appointment status."""
        conn = self._get_connection()
        cursor = conn.cursor()

        update_fields = "status = ?"
        params = [status]

        if status == "completed":
            update_fields += ", completed_at = ?"
            params.append(datetime.now().isoformat())

        params.append(appointment_id)
        cursor.execute(f"UPDATE appointments SET {update_fields} WHERE id = ?", params)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancel an appointment and release the slot."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get slot_id first
        cursor.execute(
            "SELECT slot_id FROM appointments WHERE id = ?", (appointment_id,)
        )
        row = cursor.fetchone()

        if row and row["slot_id"]:
            self.release_slot(row["slot_id"])

        cursor.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?",
            (appointment_id,),
        )
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    # ==================== Feedback ====================

    def submit_feedback(
        self, appointment_id: str, vehicle_id: str, rating: int, comments: str = ""
    ) -> Dict:
        """Submit service feedback."""
        conn = self._get_connection()
        cursor = conn.cursor()

        feedback_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO feedback (id, appointment_id, vehicle_id, rating, comments, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (feedback_id, appointment_id, vehicle_id, rating, comments, now),
        )

        conn.commit()
        conn.close()

        return {
            "id": feedback_id,
            "appointment_id": appointment_id,
            "rating": rating,
            "submitted_at": now,
        }

    def get_feedback(self, vehicle_id: str = None) -> List[Dict]:
        """Get feedback records."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if vehicle_id:
            cursor.execute(
                "SELECT * FROM feedback WHERE vehicle_id = ? ORDER BY submitted_at DESC",
                (vehicle_id,),
            )
        else:
            cursor.execute("SELECT * FROM feedback ORDER BY submitted_at DESC")

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ==================== Maintenance History ====================

    def add_maintenance_record(
        self,
        vehicle_id: str,
        appointment_id: str,
        service_type: str,
        component: str,
        description: str,
        cost: str,
        next_service_due: str = None,
        mileage: int = None,
    ) -> Dict:
        """Add a maintenance history record."""
        conn = self._get_connection()
        cursor = conn.cursor()

        record_id = f"MH-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO maintenance_history 
            (id, vehicle_id, appointment_id, service_type, component, description, cost, performed_at, next_service_due, mileage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record_id,
                vehicle_id,
                appointment_id,
                service_type,
                component,
                description,
                cost,
                now,
                next_service_due,
                mileage,
            ),
        )

        conn.commit()
        conn.close()

        return {"id": record_id, "performed_at": now}

    def get_maintenance_history(self, vehicle_id: str) -> List[Dict]:
        """Get maintenance history for a vehicle."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM maintenance_history WHERE vehicle_id = ? ORDER BY performed_at DESC",
            (vehicle_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ==================== Demand Forecasting ====================

    def get_demand_history(self, center_id: str, days: int = 30) -> List[Dict]:
        """Get historical demand data for forecasting."""
        conn = self._get_connection()
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT date, hour, SUM(appointment_count) as total
            FROM demand_log
            WHERE center_id = ? AND date >= ?
            GROUP BY date, hour
            ORDER BY date, hour
        """,
            (center_id, start_date),
        )

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def log_demand(self, center_id: str, component: str):
        """Log a demand event for forecasting."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        hour = now.hour
        demand_id = f"demand-{center_id}-{date}-{hour}"

        cursor.execute(
            """
            INSERT INTO demand_log (id, center_id, date, hour, component_type, appointment_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(id) DO UPDATE SET appointment_count = appointment_count + 1
        """,
            (demand_id, center_id, date, hour, component),
        )

        conn.commit()
        conn.close()

    # ==================== Service Tickets ====================

    def create_service_ticket(
        self,
        appointment_id: str,
        vehicle_id: str,
        technician_id: str = None,
        estimated_completion: str = None,
    ) -> Dict:
        """Create a service ticket when appointment is confirmed."""
        conn = self._get_connection()
        cursor = conn.cursor()

        ticket_id = f"SR-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()

        initial_stage = {
            "timestamp": now,
            "stage": "INTAKE",
            "note": "Vehicle received at service center",
        }

        cursor.execute(
            """
            INSERT INTO service_tickets 
            (id, appointment_id, vehicle_id, status, stage_log, estimated_completion, 
             technician_id, created_at, updated_at)
            VALUES (?, ?, ?, 'INTAKE', ?, ?, ?, ?, ?)
        """,
            (
                ticket_id,
                appointment_id,
                vehicle_id,
                json.dumps([initial_stage]),
                estimated_completion,
                technician_id,
                now,
                now,
            ),
        )

        conn.commit()
        conn.close()

        return {
            "ticket_id": ticket_id,
            "appointment_id": appointment_id,
            "vehicle_id": vehicle_id,
            "status": "INTAKE",
            "created_at": now,
        }

    def update_ticket_status(
        self,
        ticket_id: str,
        new_status: str,
        note: str = "",
        technician_notes: str = None,
        estimated_completion: str = None,
    ) -> Dict:
        """Update service ticket status (webhook handler)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get current ticket
        cursor.execute("SELECT * FROM service_tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return {"error": "Ticket not found"}

        ticket = dict(row)
        now = datetime.now().isoformat()

        # Add to stage log
        stage_log = json.loads(ticket.get("stage_log") or "[]")
        stage_log.append({"timestamp": now, "stage": new_status, "note": note})

        # Update ticket
        update_query = """
            UPDATE service_tickets 
            SET status = ?, stage_log = ?, updated_at = ?
        """
        params = [new_status, json.dumps(stage_log), now]

        if technician_notes:
            update_query += ", technician_notes = ?"
            params.append(technician_notes)

        if estimated_completion:
            update_query += ", estimated_completion = ?"
            params.append(estimated_completion)

        update_query += " WHERE id = ?"
        params.append(ticket_id)

        cursor.execute(update_query, params)
        conn.commit()
        conn.close()

        return {
            "ticket_id": ticket_id,
            "status": new_status,
            "stage_log": stage_log,
            "updated_at": now,
        }

    def get_ticket_by_id(self, ticket_id: str) -> Optional[Dict]:
        """Get service ticket by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM service_tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            ticket = dict(row)
            ticket["stage_log"] = json.loads(ticket.get("stage_log") or "[]")
            return ticket
        return None

    def get_ticket_by_vehicle(self, vehicle_id: str) -> Optional[Dict]:
        """Get active service ticket for a vehicle."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM service_tickets 
            WHERE vehicle_id = ? AND status != 'PICKED_UP'
            ORDER BY created_at DESC LIMIT 1
        """,
            (vehicle_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            ticket = dict(row)
            ticket["stage_log"] = json.loads(ticket.get("stage_log") or "[]")
            return ticket
        return None

    def get_all_tickets(self, status: str = None) -> List[Dict]:
        """Get all service tickets with optional status filter."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                "SELECT * FROM service_tickets WHERE status = ? ORDER BY updated_at DESC",
                (status,),
            )
        else:
            cursor.execute("SELECT * FROM service_tickets ORDER BY updated_at DESC")

        rows = cursor.fetchall()
        conn.close()

        tickets = []
        for row in rows:
            ticket = dict(row)
            ticket["stage_log"] = json.loads(ticket.get("stage_log") or "[]")
            tickets.append(ticket)
        return tickets

    # ==================== User Profiles ====================

    def get_user_profile(self, vehicle_id: str) -> Optional[Dict]:
        """Get user profile for personalized feedback."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_profiles WHERE vehicle_id = ?", (vehicle_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            profile = dict(row)
            profile["past_feedback"] = json.loads(profile.get("past_feedback") or "[]")
            profile["preferences"] = json.loads(profile.get("preferences") or "{}")
            return profile
        return None

    def create_or_update_user_profile(
        self,
        vehicle_id: str,
        driving_style: str = None,
        preferences: Dict = None,
    ) -> Dict:
        """Create or update user profile."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Check if exists
        cursor.execute(
            "SELECT * FROM user_profiles WHERE vehicle_id = ?", (vehicle_id,)
        )
        existing = cursor.fetchone()

        if existing:
            # Update
            updates = ["updated_at = ?"]
            params = [now]

            if driving_style:
                updates.append("driving_style = ?")
                params.append(driving_style)
            if preferences:
                updates.append("preferences = ?")
                params.append(json.dumps(preferences))

            params.append(vehicle_id)
            cursor.execute(
                f"UPDATE user_profiles SET {', '.join(updates)} WHERE vehicle_id = ?",
                params,
            )
        else:
            # Create
            cursor.execute(
                """
                INSERT INTO user_profiles 
                (vehicle_id, driving_style, past_feedback, preferences, created_at, updated_at)
                VALUES (?, ?, '[]', ?, ?, ?)
            """,
                (
                    vehicle_id,
                    driving_style or "normal",
                    json.dumps(preferences or {}),
                    now,
                    now,
                ),
            )

        conn.commit()
        conn.close()

        return {"vehicle_id": vehicle_id, "updated_at": now}

    def add_feedback_to_profile(
        self,
        vehicle_id: str,
        sentiment_score: float,
        pain_points: List[str],
        positive_points: List[str],
        service_type: str,
    ) -> Dict:
        """Add feedback analysis to user profile."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Get or create profile
        cursor.execute(
            "SELECT * FROM user_profiles WHERE vehicle_id = ?", (vehicle_id,)
        )
        row = cursor.fetchone()

        if row:
            profile = dict(row)
            past_feedback = json.loads(profile.get("past_feedback") or "[]")
            total_services = profile.get("total_services", 0)
            avg_satisfaction = profile.get("avg_satisfaction", 0.0)
        else:
            past_feedback = []
            total_services = 0
            avg_satisfaction = 0.0

        # Add new feedback
        past_feedback.append(
            {
                "date": now[:10],
                "sentiment_score": sentiment_score,
                "pain_points": pain_points,
                "positive_points": positive_points,
                "service_type": service_type,
            }
        )

        # Update averages
        new_total = total_services + 1
        new_avg = ((avg_satisfaction * total_services) + sentiment_score) / new_total

        if row:
            cursor.execute(
                """
                UPDATE user_profiles 
                SET past_feedback = ?, total_services = ?, avg_satisfaction = ?, updated_at = ?
                WHERE vehicle_id = ?
            """,
                (json.dumps(past_feedback), new_total, new_avg, now, vehicle_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO user_profiles 
                (vehicle_id, driving_style, past_feedback, preferences, total_services, 
                 avg_satisfaction, created_at, updated_at)
                VALUES (?, 'normal', ?, '{}', ?, ?, ?, ?)
            """,
                (vehicle_id, json.dumps(past_feedback), new_total, new_avg, now, now),
            )

        conn.commit()
        conn.close()

        return {
            "vehicle_id": vehicle_id,
            "feedback_added": True,
            "total_services": new_total,
            "avg_satisfaction": round(new_avg, 2),
        }


# Singleton instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get or create Database singleton."""
    global _database
    if _database is None:
        _database = Database()
    return _database
