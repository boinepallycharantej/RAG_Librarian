"""
Orchestrator Agent — routes tasks to Research or Data agents.
"""
import anthropic
import json
from typing import Literal

client = anthropic.Anthropic()

ROUTE_TOOL = {
    "name": "route_task",
    "description": "Determine what type of task the user request is and how to handle it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": ["research", "data_analysis", "combined"],
                "description": (
                    "research: user wants to research a topic via web search. "
                    "data_analysis: user provides a CSV file to analyze. "
                    "combined: user wants both research AND data analysis together."
                ),
            },
            "research_topic": {
                "type": "string",
                "description": "The topic to research (if task_type is research or combined).",
            },
            "csv_path": {
                "type": "string",
                "description": "Path to the CSV file (if task_type is data_analysis or combined).",
            },
            "report_title": {
                "type": "string",
                "description": "A concise title for the output report.",
            },
        },
        "required": ["task_type", "report_title"],
    },
}


def route(user_request: str) -> dict:
    """
    Ask Claude to classify the user request and extract task parameters.
    Returns a dict with task_type, optional topic/csv_path, and report_title.
    """
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        tools=[ROUTE_TOOL],
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a task router. Analyze the user's request and call route_task "
                    "with the appropriate parameters.\n\n"
                    f"User request: {user_request}"
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "route_task":
            return block.input

    raise ValueError("Orchestrator failed to classify the request.")
