"""
SQLAlchemy ORM Models for SentinEV V2
"""

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    Enum,
    JSON,
    Boolean,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from shared.db.database import Base


# ================== ENUMS ==================


class VehicleCategory(str, enum.Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class DrivingProfile(str, enum.Enum):
    ECO = "eco"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


class SeverityLevel(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class ServiceStatusEnum(str, enum.Enum):
    REQUESTED = "REQUESTED"
    BOOKED = "BOOKED"
    CONFIRMED = "CONFIRMED"
    CHECK_IN = "CHECK_IN"
    DIAGNOSIS = "DIAGNOSIS"
    PARTS_ALLOCATED = "PARTS_ALLOCATED"
    REPAIR_IN_PROGRESS = "REPAIR_IN_PROGRESS"
    QUALITY_CHECK = "QUALITY_CHECK"
    READY = "READY"
    COMPLETED = "COMPLETED"


# ================== VEHICLE ==================


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(String, primary_key=True)
    vin = Column(String(17), unique=True, nullable=False)
    make = Column(String(50), nullable=False)
    model = Column(String(50), nullable=False)
    year = Column(Integer, nullable=False)
    owner_id = Column(String, ForeignKey("customers.id"), nullable=True)

    # Dynamic state
    mileage = Column(Integer, default=0)
    health_score = Column(Float, default=100.0)
    category = Column(Enum(VehicleCategory), default=VehicleCategory.NORMAL)
    driving_profile = Column(Enum(DrivingProfile), default=DrivingProfile.NORMAL)

    # Location
    last_lat = Column(Float, nullable=True)
    last_lon = Column(Float, nullable=True)

    # Timestamps
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_service_at = Column(DateTime, nullable=True)
    last_telemetry_at = Column(DateTime, nullable=True)

    # Config for simulation
    baseline_config = Column(JSON, default={})
    degradation_config = Column(JSON, default={})

    # Relationships
    owner = relationship("Customer", back_populates="vehicles")
    telemetry_records = relationship("TelemetryRecord", back_populates="vehicle")
    maintenance_records = relationship("MaintenanceRecord", back_populates="vehicle")
    appointments = relationship("Appointment", back_populates="vehicle")


# ================== CUSTOMER ==================


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20), nullable=True)

    # Preferences
    notification_enabled = Column(Boolean, default=True)
    voice_calls_enabled = Column(Boolean, default=True)
    preferred_contact_time = Column(String(20), default="anytime")

    # Gamification
    driving_score = Column(Float, default=80.0)
    points = Column(Integer, default=0)
    badges = Column(JSON, default=[])

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    vehicles = relationship("Vehicle", back_populates="owner")


# ================== TELEMETRY ==================


class TelemetryRecord(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String, ForeignKey("vehicles.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Sensor readings
    engine_temp = Column(Float)
    battery_voltage = Column(Float)
    coolant_level = Column(Float)
    brake_pressure = Column(Float)
    vibration_amplitude = Column(Float)
    rpm = Column(Integer)

    # Computed
    anomaly_score = Column(Float, default=0.0)

    # GPS
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="telemetry_records")


# ================== MAINTENANCE HISTORY ==================


class MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String, ForeignKey("vehicles.id"), nullable=False)
    service_center_id = Column(String, ForeignKey("service_centers.id"), nullable=True)

    service_type = Column(String(50))  # oil_change, brake_service, etc.
    description = Column(String(500))
    cost = Column(Float, nullable=True)
    mileage_at_service = Column(Integer)

    # DTC codes if any
    dtc_codes = Column(JSON, default=[])

    performed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="maintenance_records")
    service_center = relationship("ServiceCenter", back_populates="maintenance_records")


# ================== SERVICE CENTER ==================


class ServiceCenter(Base):
    __tablename__ = "service_centers"

    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    address = Column(String(200))
    lat = Column(Float)
    lon = Column(Float)

    # Capabilities
    capabilities = Column(JSON, default=[])  # ["brake", "electrical", "ev", "general"]
    quality_rating = Column(Float, default=4.0)

    # Capacity
    num_bays = Column(Integer, default=4)
    operating_hours = Column(JSON, default={"start": "08:00", "end": "18:00"})

    # Relationships
    mechanics = relationship("Mechanic", back_populates="service_center")
    appointments = relationship("Appointment", back_populates="service_center")
    inventory = relationship("PartInventory", back_populates="service_center")
    maintenance_records = relationship(
        "MaintenanceRecord", back_populates="service_center"
    )


# ================== MECHANIC ==================


class Mechanic(Base):
    __tablename__ = "mechanics"

    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    service_center_id = Column(String, ForeignKey("service_centers.id"), nullable=False)

    # Skills
    certifications = Column(
        JSON, default=[]
    )  # ["general", "brake", "ev", "electrical"]
    experience_years = Column(Integer, default=1)
    efficiency_rating = Column(Float, default=1.0)

    # Relationships
    service_center = relationship("ServiceCenter", back_populates="mechanics")


# ================== APPOINTMENT ==================


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(String, primary_key=True)
    vehicle_id = Column(String, ForeignKey("vehicles.id"), nullable=False)
    service_center_id = Column(String, ForeignKey("service_centers.id"), nullable=False)
    mechanic_id = Column(String, ForeignKey("mechanics.id"), nullable=True)

    # Status
    status = Column(Enum(ServiceStatusEnum), default=ServiceStatusEnum.REQUESTED)

    # Details
    fault_type = Column(String(50))
    severity = Column(Enum(SeverityLevel), default=SeverityLevel.INFO)
    priority_score = Column(Float, default=0.0)

    # Timing
    scheduled_at = Column(DateTime)
    checked_in_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_duration_hours = Column(Float, default=1.0)

    # AI metadata
    confidence_score = Column(Float, default=0.0)
    triggered_by = Column(
        String(50), default="manual"
    )  # ai_prediction, manual, scheduled

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="appointments")
    service_center = relationship("ServiceCenter", back_populates="appointments")


# ================== PARTS INVENTORY ==================


class PartInventory(Base):
    __tablename__ = "parts_inventory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service_center_id = Column(String, ForeignKey("service_centers.id"), nullable=False)

    part_number = Column(String(50))
    description = Column(String(200))
    category = Column(String(50))  # brake, electrical, cooling
    quantity = Column(Integer, default=0)
    reorder_point = Column(Integer, default=5)
    unit_cost = Column(Float)

    # Relationships
    service_center = relationship("ServiceCenter", back_populates="inventory")


# ================== AGENT EVENT LOG (FOR UEBA) ==================


class AgentEventLog(Base):
    __tablename__ = "agent_event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    agent_id = Column(String(50), nullable=False)
    action = Column(String(100), nullable=False)
    target_resource = Column(String(200))
    parameters = Column(JSON, default={})

    success = Column(Boolean, default=True)
    duration_ms = Column(Integer, nullable=True)
    risk_score = Column(Float, default=0.0)

    correlation_id = Column(String(50), nullable=True)
