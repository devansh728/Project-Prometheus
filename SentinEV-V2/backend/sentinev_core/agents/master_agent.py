"""
Master Agent - Orchestrates all worker agents for SentinEV Core.
Uses a simple state machine pattern (compatible with LangGraph integration).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json

from sentinev_core.agents.worker_agents import (
    DataAnalysisAgent,
    DiagnosisAgent,
    EngagementAgent,
    SchedulingAgent,
    AgentContext,
    WORKER_AGENTS,
)


class WorkflowState(str, Enum):
    """States in the master agent workflow."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    DIAGNOSING = "diagnosing"
    DECIDING = "deciding"
    SCHEDULING = "scheduling"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class WorkflowResult:
    """Final result of a master agent workflow."""

    vehicle_id: str
    state: WorkflowState
    severity: Optional[str] = None
    primary_concern: Optional[str] = None
    recommended_action: Optional[str] = None
    engagement_action: Optional[str] = None
    service_request: Optional[Dict] = None
    voice_script: Optional[Dict] = None
    actions_log: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MasterAgent:
    """
    Orchestrates the SentinEV analysis pipeline.

    Pipeline:
    1. DataAnalysisAgent: Process telemetry → anomaly detection
    2. DiagnosisAgent: Generate diagnosis with RAG
    3. EngagementAgent: Decide on customer action
    4. SchedulingAgent: Prepare service request
    """

    def __init__(self):
        self.state = WorkflowState.IDLE
        self.agents = WORKER_AGENTS

    def run(
        self,
        vehicle_id: str,
        sensors: Dict[str, float],
        baseline: Dict[str, Dict],
        degradation_config: Dict[str, Any],
        customer_id: Optional[str] = None,
        mileage: int = 0,
        metadata: Optional[Dict] = None,
    ) -> WorkflowResult:
        """
        Execute the full agent pipeline synchronously.
        """
        # Initialize context
        context = AgentContext(
            vehicle_id=vehicle_id,
            customer_id=customer_id,
            sensors=sensors,
            baseline=baseline,
            degradation_config=degradation_config,
            mileage=mileage,
            metadata=metadata or {},
        )

        try:
            # Stage 1: Data Analysis
            self.state = WorkflowState.ANALYZING
            context = self.agents["data_analysis"].process(context)

            # Stage 2: Diagnosis
            self.state = WorkflowState.DIAGNOSING
            context = self.agents["diagnosis"].process(context)

            # Stage 3: Engagement Decision
            self.state = WorkflowState.DECIDING
            context = self.agents["engagement"].process(context)

            # Stage 4: Scheduling
            self.state = WorkflowState.SCHEDULING
            context = self.agents["scheduling"].process(context)

            # Complete
            self.state = WorkflowState.COMPLETE

            return WorkflowResult(
                vehicle_id=vehicle_id,
                state=self.state,
                severity=(
                    context.diagnosis.severity.value if context.diagnosis else None
                ),
                primary_concern=(
                    context.diagnosis.primary_concern if context.diagnosis else None
                ),
                recommended_action=(
                    context.diagnosis.recommended_action if context.diagnosis else None
                ),
                engagement_action=context.metadata.get("engagement_action"),
                service_request=context.metadata.get("service_request"),
                voice_script=context.metadata.get("voice_script"),
                actions_log=context.actions_taken,
            )

        except Exception as e:
            self.state = WorkflowState.ERROR
            return WorkflowResult(
                vehicle_id=vehicle_id,
                state=self.state,
                actions_log=context.actions_taken,
                error=str(e),
            )

    async def run_async(
        self,
        vehicle_id: str,
        sensors: Dict[str, float],
        baseline: Dict[str, Dict],
        degradation_config: Dict[str, Any],
        **kwargs
    ):
        """
        Async version for WebSocket streaming.
        Yields state updates as the pipeline progresses.
        """
        context = AgentContext(
            vehicle_id=vehicle_id,
            sensors=sensors,
            baseline=baseline,
            degradation_config=degradation_config,
            **kwargs
        )

        stages = [
            (WorkflowState.ANALYZING, "data_analysis"),
            (WorkflowState.DIAGNOSING, "diagnosis"),
            (WorkflowState.DECIDING, "engagement"),
            (WorkflowState.SCHEDULING, "scheduling"),
        ]

        for state, agent_name in stages:
            self.state = state
            yield {
                "event": "state_change",
                "state": state.value,
                "agent": agent_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

            context = self.agents[agent_name].process(context)

            yield {
                "event": "agent_complete",
                "agent": agent_name,
                "actions": context.actions_taken[-1] if context.actions_taken else "",
                "timestamp": datetime.utcnow().isoformat(),
            }

        self.state = WorkflowState.COMPLETE
        yield {
            "event": "workflow_complete",
            "state": self.state.value,
            "result": {
                "severity": (
                    context.diagnosis.severity.value if context.diagnosis else None
                ),
                "engagement_action": context.metadata.get("engagement_action"),
                "primary_concern": (
                    context.diagnosis.primary_concern if context.diagnosis else None
                ),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


# Singleton instance
master_agent = MasterAgent()
