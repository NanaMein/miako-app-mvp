import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from multi_agent_workflow.llms.custom_llm_for_crewai import GroqLLM
from typing import List


def llm_sample_selector(select: int = 0):
    if select == 1:
        return "meta-llama/llama-4-scout-17b-16e-instruct"
    elif select == 2:
        return "openai/gpt-oss-120b"
    elif select == 3:
        return "qwen/qwen3-32b"
    else:
        return "meta-llama/llama-4-maverick-17b-128e-instruct"


def groq_llm():
    return GroqLLM(
        model=llm_sample_selector(3),
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=8000,
    )

@CrewBase
class MultiAgentWorkflow:
    """MultiAgentWorkflow crew"""

    base_path = Path(__file__).parent.parent

    agents_config = str(base_path / "config" / "agents.yaml")
    tasks_config = str(base_path / "config" / "tasks.yaml")

    def __init__(self):
        self.llm = groq_llm()

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            llm=self.llm,
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            llm=self.llm,
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
