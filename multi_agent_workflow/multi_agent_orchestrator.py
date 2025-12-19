from fastapi import HTTPException, status
from typing import Any
from multi_agent_workflow.crewai_crew.crew import MultiAgentWorkflow as AgentsWorkflow



agents = AgentsWorkflow()

def workflow_orchestrator(inputs: dict[str, Any]):
    try:
        crewai = agents.crew().kickoff(inputs=inputs)
        return crewai.raw
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected Error Handling: {ex}")



if __name__ == "__main__":
    inputs = {
        "topic":"Asahina Mafuyu"
    }
    result = workflow_orchestrator(inputs=inputs)
    print(result)