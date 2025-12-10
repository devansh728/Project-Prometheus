"""
SentinEV - Fleet Manager
========================
Multi-tenant fleet simulation for enterprise customers.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FleetTenant:
    """Represents a fleet tenant/organization."""

    tenant_id: str
    tenant_name: str
    vehicle_ids: List[str] = field(default_factory=list)
    contact_email: str = ""
    tier: str = "standard"  # standard, premium, enterprise
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FleetHealthSummary:
    """Aggregated health summary for a fleet."""

    tenant_id: str
    tenant_name: str
    total_vehicles: int
    healthy_vehicles: int
    warning_vehicles: int
    critical_vehicles: int
    avg_failure_prob: float
    top_issues: List[Dict]
    last_updated: str


class FleetManager:
    """
    Multi-tenant fleet manager.

    Manages vehicle-to-tenant mapping and provides
    fleet-level health summaries.
    """

    def __init__(self):
        self.tenants: Dict[str, FleetTenant] = {}
        self.vehicle_to_tenant: Dict[str, str] = {}
        self._seed_demo_fleets()

    def _seed_demo_fleets(self):
        """Seed demo fleet data."""
        # Demo tenants
        demo_tenants = [
            FleetTenant(
                tenant_id="fleet_acme",
                tenant_name="Acme Logistics",
                vehicle_ids=["EV_001", "EV_002", "EV_003"],
                contact_email="fleet@acme.com",
                tier="enterprise",
            ),
            FleetTenant(
                tenant_id="fleet_metro",
                tenant_name="Metro Transport",
                vehicle_ids=["EV_004", "EV_005"],
                contact_email="ops@metrotransport.com",
                tier="premium",
            ),
            FleetTenant(
                tenant_id="fleet_green",
                tenant_name="Green Delivery Co",
                vehicle_ids=["EV_006", "EV_007", "EV_008", "EV_009", "EV_010"],
                contact_email="manager@greendelivery.com",
                tier="enterprise",
            ),
        ]

        for tenant in demo_tenants:
            self.tenants[tenant.tenant_id] = tenant
            for vid in tenant.vehicle_ids:
                self.vehicle_to_tenant[vid] = tenant.tenant_id

    def get_tenant(self, tenant_id: str) -> Optional[FleetTenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def get_tenant_for_vehicle(self, vehicle_id: str) -> Optional[FleetTenant]:
        """Get tenant that owns a vehicle."""
        tenant_id = self.vehicle_to_tenant.get(vehicle_id)
        if tenant_id:
            return self.tenants.get(tenant_id)
        return None

    def get_fleet_vehicles(self, tenant_id: str) -> List[str]:
        """Get all vehicles in a fleet."""
        tenant = self.tenants.get(tenant_id)
        return tenant.vehicle_ids if tenant else []

    def add_vehicle_to_fleet(self, tenant_id: str, vehicle_id: str):
        """Add a vehicle to a fleet."""
        if tenant_id in self.tenants:
            if vehicle_id not in self.tenants[tenant_id].vehicle_ids:
                self.tenants[tenant_id].vehicle_ids.append(vehicle_id)
            self.vehicle_to_tenant[vehicle_id] = tenant_id

    def create_tenant(self, tenant: FleetTenant):
        """Create a new tenant."""
        self.tenants[tenant.tenant_id] = tenant
        for vid in tenant.vehicle_ids:
            self.vehicle_to_tenant[vid] = tenant.tenant_id

    def get_fleet_health_summary(
        self, tenant_id: str, vehicle_health: Dict[str, Dict] = None
    ) -> Optional[FleetHealthSummary]:
        """
        Get aggregated health summary for a fleet.

        Args:
            tenant_id: Fleet tenant ID
            vehicle_health: Dict of vehicle_id -> health data
                           {failure_prob, severity, issues}
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None

        vehicle_health = vehicle_health or {}

        healthy = 0
        warning = 0
        critical = 0
        total_prob = 0
        issues_count: Dict[str, int] = {}

        for vid in tenant.vehicle_ids:
            health = vehicle_health.get(vid, {})
            prob = health.get("failure_prob", 0)
            severity = health.get("severity", "low")

            total_prob += prob

            if severity in ["critical"]:
                critical += 1
            elif severity in ["high", "medium"]:
                warning += 1
            else:
                healthy += 1

            for issue in health.get("issues", []):
                issues_count[issue] = issues_count.get(issue, 0) + 1

        # Top issues
        top_issues = sorted(
            [{"issue": k, "count": v} for k, v in issues_count.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        return FleetHealthSummary(
            tenant_id=tenant_id,
            tenant_name=tenant.tenant_name,
            total_vehicles=len(tenant.vehicle_ids),
            healthy_vehicles=healthy,
            warning_vehicles=warning,
            critical_vehicles=critical,
            avg_failure_prob=total_prob / max(len(tenant.vehicle_ids), 1),
            top_issues=top_issues,
            last_updated=datetime.now().isoformat(),
        )

    def list_all_tenants(self) -> List[FleetTenant]:
        """List all tenants."""
        return list(self.tenants.values())

    def to_dict(self, obj) -> Dict:
        """Convert dataclass to dict."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        return obj

    # ==================== Phase 10: Voice-Friendly Summaries ====================

    def generate_voice_summary(
        self, tenant_id: str, vehicle_health: Dict[str, Dict] = None
    ) -> str:
        """
        Generate a natural language summary suitable for voice output.

        Args:
            tenant_id: Fleet tenant ID
            vehicle_health: Dict of vehicle_id -> health data

        Returns:
            Voice-friendly summary string
        """
        summary = self.get_fleet_health_summary(tenant_id, vehicle_health)
        if not summary:
            return f"I don't have information for fleet {tenant_id}."

        # Build natural language summary
        parts = [f"Here's the fleet health update for {summary.tenant_name}."]

        # Overall status
        if summary.critical_vehicles > 0:
            parts.append(
                f"Urgent: {summary.critical_vehicles} vehicle{'s' if summary.critical_vehicles > 1 else ''} require{'s' if summary.critical_vehicles == 1 else ''} immediate attention."
            )

        if summary.warning_vehicles > 0:
            parts.append(
                f"{summary.warning_vehicles} vehicle{'s' if summary.warning_vehicles > 1 else ''} {'have' if summary.warning_vehicles > 1 else 'has'} minor warnings."
            )

        parts.append(
            f"{summary.healthy_vehicles} of {summary.total_vehicles} vehicles are in good condition."
        )

        # Top issues
        if summary.top_issues:
            top_issue = summary.top_issues[0]
            issue_name = top_issue["issue"].replace("_", " ")
            parts.append(
                f"The most common issue is {issue_name}, affecting {top_issue['count']} vehicle{'s' if top_issue['count'] > 1 else ''}."
            )

        # Average risk
        if summary.avg_failure_prob > 0.5:
            parts.append(
                "Overall fleet risk is elevated. Consider scheduling preventive maintenance."
            )
        elif summary.avg_failure_prob > 0.3:
            parts.append("Fleet risk is moderate. Keep monitoring key vehicles.")
        else:
            parts.append("Fleet is performing well overall.")

        return " ".join(parts)

    def get_priority_vehicles(
        self, tenant_id: str, vehicle_health: Dict[str, Dict] = None
    ) -> List[Dict]:
        """
        Get list of vehicles needing priority attention.

        Args:
            tenant_id: Fleet tenant ID
            vehicle_health: Dict of vehicle_id -> health data

        Returns:
            List of priority vehicles with details
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return []

        vehicle_health = vehicle_health or {}
        priority_list = []

        for vid in tenant.vehicle_ids:
            health = vehicle_health.get(vid, {})
            severity = health.get("severity", "low")
            if severity in ["critical", "high"]:
                priority_list.append(
                    {
                        "vehicle_id": vid,
                        "severity": severity,
                        "failure_prob": health.get("failure_prob", 0),
                        "issues": health.get("issues", []),
                    }
                )

        # Sort by severity (critical first) and failure probability
        def sort_key(x):
            sev_order = {"critical": 0, "high": 1}
            return (sev_order.get(x["severity"], 2), -x["failure_prob"])

        return sorted(priority_list, key=sort_key)


# Singleton
_fleet_manager: Optional[FleetManager] = None


def get_fleet_manager() -> FleetManager:
    """Get or create fleet manager."""
    global _fleet_manager
    if _fleet_manager is None:
        _fleet_manager = FleetManager()
    return _fleet_manager


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fleet Manager")
    print("=" * 60)

    fm = get_fleet_manager()

    print(f"\n📊 Tenants: {len(fm.tenants)}")
    for tenant in fm.list_all_tenants():
        print(
            f"  - {tenant.tenant_name}: {len(tenant.vehicle_ids)} vehicles ({tenant.tier})"
        )

    # Test vehicle lookup
    vehicle = "EV_001"
    tenant = fm.get_tenant_for_vehicle(vehicle)
    print(
        f"\n🚗 Vehicle {vehicle} belongs to: {tenant.tenant_name if tenant else 'Unknown'}"
    )

    # Test fleet summary
    summary = fm.get_fleet_health_summary(
        "fleet_acme",
        vehicle_health={
            "EV_001": {
                "failure_prob": 0.7,
                "severity": "high",
                "issues": ["motor_wear"],
            },
            "EV_002": {"failure_prob": 0.2, "severity": "low", "issues": []},
            "EV_003": {
                "failure_prob": 0.9,
                "severity": "critical",
                "issues": ["battery_degradation"],
            },
        },
    )
    print(f"\n📈 Fleet Summary for {summary.tenant_name}:")
    print(
        f"   Healthy: {summary.healthy_vehicles}, Warning: {summary.warning_vehicles}, Critical: {summary.critical_vehicles}"
    )
    print(f"   Avg failure prob: {summary.avg_failure_prob:.2f}")

    print("\n✅ Fleet Manager test complete")
