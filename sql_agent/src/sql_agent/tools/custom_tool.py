from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."


from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.tools import BaseTool
from pydantic import Field
from sqlalchemy import create_engine


class QueryTool(BaseTool):
    name: str = "query_tool"
    description: str = "Execute a SQL query against the database. Returns the result"
    db: str

    def __init__(self):  # Constructor to initialize the database
        super().__init__()
        self.db = create_engine("sqlite:///company.db")
        self.query_tool = QuerySQLDataBaseTool(db=self.db)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.query_tool.invoke(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"
