"""
Research Agent — searches the web and reads pages to synthesize findings on a topic.
"""
from agents.base_agent import BaseAgent
from tools.search import web_search, SEARCH_TOOL_SCHEMA
from tools.scraper import read_page, SCRAPER_TOOL_SCHEMA

SYSTEM_PROMPT = """You are a research agent. Your job is to thoroughly research a topic and produce a detailed synthesis.

Follow this process:
1. Run 2-3 targeted web searches using different angles on the topic.
2. Pick the 3-4 most promising URLs from the results and read each page.
3. After reading, synthesize what you learned into a comprehensive, well-structured report.

Your final response (after all tool calls) must be a detailed markdown report with:
- ## Overview
- ## Key Findings (bullet points with specifics)
- ## Notable Trends or Developments
- ## Key Sources (list the URLs you read)

Be specific. Include names, numbers, dates, and examples wherever available. Do not pad with vague generalities."""


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL_SCHEMA, SCRAPER_TOOL_SCHEMA],
            tool_handlers={
                "web_search": web_search,
                "read_page": read_page,
            },
        )

    def run(self, topic: str) -> str:
        return super().run(f"Research this topic thoroughly: {topic}")
