"""
Database Seeder - Populate database with initial data
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime

from shared.db.database import async_session, init_db
from shared.db.models import (
    Vehicle,
    Customer,
    ServiceCenter,
    Mechanic,
    MaintenanceRecord,
    VehicleCategory,
    DrivingProfile,
)


async def seed_database():
    """Load seed data from JSON files and populate database."""

    # Initialize tables
    await init_db()

    data_dir = Path(__file__).parent.parent.parent.parent / "data"

    async with async_session() as session:
        # Load fleet data
        fleet_path = data_dir / "fleet_seed.json"
        if fleet_path.exists():
            with open(fleet_path, "r") as f:
                fleet_data = json.load(f)

            # Seed customers first (foreign key dependency)
            for cust_data in fleet_data.get("customers", []):
                customer = Customer(
                    id=cust_data["id"],
                    name=cust_data["name"],
                    email=cust_data["email"],
                    phone=cust_data.get("phone"),
                    driving_score=cust_data.get("driving_score", 80.0),
                )
                session.add(customer)

            # Seed service centers
            for sc_data in fleet_data.get("service_centers", []):
                service_center = ServiceCenter(
                    id=sc_data["id"],
                    name=sc_data["name"],
                    address=sc_data.get("address"),
                    lat=sc_data.get("lat"),
                    lon=sc_data.get("lon"),
                    capabilities=sc_data.get("capabilities", []),
                    quality_rating=sc_data.get("quality_rating", 4.0),
                    num_bays=sc_data.get("num_bays", 4),
                )
                session.add(service_center)

            # Seed mechanics
            for mech_data in fleet_data.get("mechanics", []):
                mechanic = Mechanic(
                    id=mech_data["id"],
                    name=mech_data["name"],
                    service_center_id=mech_data["service_center_id"],
                    certifications=mech_data.get("certifications", []),
                    experience_years=mech_data.get("experience_years", 1),
                    efficiency_rating=mech_data.get("efficiency_rating", 1.0),
                )
                session.add(mechanic)

            # Seed vehicles
            for veh_data in fleet_data.get("vehicles", []):
                category = VehicleCategory(veh_data.get("category", "normal"))
                profile = DrivingProfile(veh_data.get("driving_profile", "normal"))

                vehicle = Vehicle(
                    id=veh_data["id"],
                    vin=veh_data["vin"],
                    make=veh_data["make"],
                    model=veh_data["model"],
                    year=veh_data["year"],
                    owner_id=veh_data.get("owner_id"),
                    mileage=veh_data.get("mileage", 0),
                    health_score=veh_data.get("health_score", 100.0),
                    category=category,
                    driving_profile=profile,
                    baseline_config=veh_data.get("baseline_config", {}),
                    degradation_config=veh_data.get("degradation_config", {}),
                )
                session.add(vehicle)

            await session.commit()
            print(
                f"✓ Seeded fleet: {len(fleet_data.get('vehicles', []))} vehicles, "
                f"{len(fleet_data.get('customers', []))} customers, "
                f"{len(fleet_data.get('service_centers', []))} service centers, "
                f"{len(fleet_data.get('mechanics', []))} mechanics"
            )

        # Load maintenance history
        maint_path = data_dir / "maintenance_history.json"
        if maint_path.exists():
            with open(maint_path, "r") as f:
                maint_data = json.load(f)

            for record in maint_data:
                maintenance = MaintenanceRecord(
                    vehicle_id=record["vehicle_id"],
                    service_center_id=record.get("service_center_id"),
                    service_type=record["service_type"],
                    description=record["description"],
                    cost=record.get("cost"),
                    mileage_at_service=record["mileage_at_service"],
                    dtc_codes=record.get("dtc_codes", []),
                    performed_at=datetime.fromisoformat(record["performed_at"]),
                )
                session.add(maintenance)

            await session.commit()
            print(f"✓ Seeded {len(maint_data)} maintenance records")


if __name__ == "__main__":
    asyncio.run(seed_database())
