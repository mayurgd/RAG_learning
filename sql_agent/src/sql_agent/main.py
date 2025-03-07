#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import SqlAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {"query": "Get names of different deparments"}

    try:
        SqlAgent().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


run()
