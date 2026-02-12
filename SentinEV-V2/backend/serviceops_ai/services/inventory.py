"""
Inventory Management for ServiceOps AI
Handles parts tracking, allocation, and reorder alerts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PartStatus(str, Enum):
    """Part availability status."""

    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    ON_ORDER = "on_order"


@dataclass
class PartInfo:
    """Part with availability info."""

    part_number: str
    description: str
    category: str
    quantity: int
    reorder_point: int
    unit_cost: float
    status: PartStatus
    is_sufficient: bool  # For a specific job requirement


class InventoryManager:
    """
    Manages parts inventory across service centers.
    """

    # Standard parts needed by failure type
    PARTS_REQUIREMENTS = {
        "brake_degradation": [
            {
                "part_number": "BP-2024-A",
                "description": "Brake Pad Set (Front)",
                "category": "brake",
                "quantity_needed": 1,
            },
            {
                "part_number": "BF-2024-A",
                "description": "Brake Fluid DOT 4",
                "category": "brake",
                "quantity_needed": 1,
            },
        ],
        "battery_degradation": [
            {
                "part_number": "BC-2024-A",
                "description": "Battery Cell Module",
                "category": "ev_battery",
                "quantity_needed": 1,
            },
            {
                "part_number": "BMS-2024-A",
                "description": "BMS Controller",
                "category": "ev_battery",
                "quantity_needed": 1,
            },
        ],
        "cooling_degradation": [
            {
                "part_number": "CL-2024-A",
                "description": "EV Coolant 1L",
                "category": "cooling",
                "quantity_needed": 2,
            },
            {
                "part_number": "TH-2024-A",
                "description": "Thermostat Assembly",
                "category": "cooling",
                "quantity_needed": 1,
            },
        ],
    }

    # Simulated inventory per service center
    # In production, this would query the database
    INVENTORY = {
        "SC001": {
            "BP-2024-A": {"quantity": 15, "reorder_point": 5, "unit_cost": 2500},
            "BF-2024-A": {"quantity": 20, "reorder_point": 10, "unit_cost": 500},
            "BC-2024-A": {"quantity": 3, "reorder_point": 2, "unit_cost": 45000},
            "BMS-2024-A": {"quantity": 2, "reorder_point": 1, "unit_cost": 12000},
            "CL-2024-A": {"quantity": 25, "reorder_point": 10, "unit_cost": 800},
            "TH-2024-A": {"quantity": 8, "reorder_point": 3, "unit_cost": 3500},
        },
        "SC002": {
            "BP-2024-A": {"quantity": 8, "reorder_point": 5, "unit_cost": 2500},
            "BF-2024-A": {"quantity": 12, "reorder_point": 10, "unit_cost": 500},
            "BC-2024-A": {"quantity": 1, "reorder_point": 2, "unit_cost": 45000},
            "CL-2024-A": {"quantity": 15, "reorder_point": 10, "unit_cost": 800},
            "TH-2024-A": {"quantity": 4, "reorder_point": 3, "unit_cost": 3500},
        },
        "SC003": {
            "BP-2024-A": {"quantity": 20, "reorder_point": 5, "unit_cost": 2500},
            "BF-2024-A": {"quantity": 30, "reorder_point": 10, "unit_cost": 500},
            "BC-2024-A": {"quantity": 5, "reorder_point": 2, "unit_cost": 45000},
            "BMS-2024-A": {"quantity": 4, "reorder_point": 1, "unit_cost": 12000},
            "CL-2024-A": {"quantity": 40, "reorder_point": 10, "unit_cost": 800},
            "TH-2024-A": {"quantity": 12, "reorder_point": 3, "unit_cost": 3500},
        },
    }

    def check_parts_availability(
        self, service_center_id: str, failure_type: str
    ) -> Dict[str, Any]:
        """
        Check if all required parts are available for a repair job.
        """
        required_parts = self.PARTS_REQUIREMENTS.get(failure_type, [])
        center_inventory = self.INVENTORY.get(service_center_id, {})

        parts_status = []
        all_available = True
        total_cost = 0.0

        for req in required_parts:
            part_num = req["part_number"]
            needed = req["quantity_needed"]
            inv = center_inventory.get(
                part_num, {"quantity": 0, "reorder_point": 0, "unit_cost": 0}
            )

            quantity = inv.get("quantity", 0)
            reorder_point = inv.get("reorder_point", 0)
            unit_cost = inv.get("unit_cost", 0)

            # Determine status
            if quantity < needed:
                status = PartStatus.OUT_OF_STOCK
                all_available = False
            elif quantity <= reorder_point:
                status = PartStatus.LOW_STOCK
            else:
                status = PartStatus.IN_STOCK

            parts_status.append(
                PartInfo(
                    part_number=part_num,
                    description=req["description"],
                    category=req["category"],
                    quantity=quantity,
                    reorder_point=reorder_point,
                    unit_cost=unit_cost,
                    status=status,
                    is_sufficient=quantity >= needed,
                )
            )

            if quantity >= needed:
                total_cost += unit_cost * needed

        return {
            "service_center_id": service_center_id,
            "failure_type": failure_type,
            "all_parts_available": all_available,
            "parts": [vars(p) for p in parts_status],
            "estimated_parts_cost": total_cost,
        }

    def allocate_parts(
        self, service_center_id: str, failure_type: str, job_id: str
    ) -> bool:
        """
        Allocate (reserve) parts for a job. Returns True if successful.
        """
        required_parts = self.PARTS_REQUIREMENTS.get(failure_type, [])
        center_inventory = self.INVENTORY.get(service_center_id, {})

        # Check all available first
        for req in required_parts:
            part_num = req["part_number"]
            needed = req["quantity_needed"]
            available = center_inventory.get(part_num, {}).get("quantity", 0)
            if available < needed:
                return False

        # Deduct inventory
        for req in required_parts:
            part_num = req["part_number"]
            needed = req["quantity_needed"]
            if part_num in center_inventory:
                center_inventory[part_num]["quantity"] -= needed

        return True

    def get_reorder_alerts(self, service_center_id: str) -> List[Dict[str, Any]]:
        """Get list of parts that need reordering."""
        center_inventory = self.INVENTORY.get(service_center_id, {})
        alerts = []

        for part_num, inv in center_inventory.items():
            if inv["quantity"] <= inv["reorder_point"]:
                alerts.append(
                    {
                        "part_number": part_num,
                        "current_quantity": inv["quantity"],
                        "reorder_point": inv["reorder_point"],
                        "severity": "critical" if inv["quantity"] == 0 else "warning",
                    }
                )

        return alerts


# Singleton
inventory_manager = InventoryManager()
