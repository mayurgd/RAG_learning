from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()
from tools.custom_tool import *
from crewai_tools import NL2SQLTool

nl2sql = NL2SQLTool(
    db_uri="postgresql://postgres:Mayur%40356@localhost:5432/DVDRentals"
)


@CrewBase
class SqlAgent:
    """SqlAgent crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def database_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["database_developer"],
            verbose=True,
            tools=[list_tables, tables_schema, nl2sql, execute_sql],
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(config=self.agents_config["data_analyst"], verbose=True)

    @agent
    def report_writer(self) -> Agent:
        return Agent(config=self.agents_config["report_writer"], verbose=True)

    @task
    def query_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_creation_task"],
        )

    @task
    def analysis_task(self) -> Task:
        return Task(config=self.tasks_config["analysis_task"])

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        """Creates the SqlAgent crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
