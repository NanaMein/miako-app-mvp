from fastapi import HTTPException, status
from typing import Any
from multi_agent_workflow.crewai_crew.crew import MultiAgentWorkflow as AgentsWorkflow



agents = AgentsWorkflow()

def workflow_orchestrator(inputs: dict[str, Any]):
    try:
        crewai = agents.crew().kickoff(inputs=inputs)
        return crewai
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request")

