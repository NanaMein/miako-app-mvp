import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from multi_agent_workflow.llms.custom_llm_for_crewai import GroqLLM
from typing import List


def oss_llm():
    return GroqLLM(
        model="oss-20b",
        max_tokens=60000
    )

def groq_llm():
    return GroqLLM(
        model="maverick",
        max_tokens=8000,
    )

@CrewBase
class MultiAgentWorkflow:
    """MultiAgentWorkflow crew"""

    base_path = Path(__file__).parent.parent

    agents_config = str(base_path / "config" / "agents.yaml")
    tasks_config = str(base_path / "config" / "tasks.yaml")

    def __init__(self):
        self.maverick = groq_llm()
        self.oss_llm = oss_llm()

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            llm=self.oss_llm,
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            llm=self.maverick,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            output_file="task_report1.md"
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='task_report2.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
