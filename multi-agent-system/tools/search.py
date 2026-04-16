"""
Web search tool using DuckDuckGo (no API key required).
Returns a list of {title, url, snippet} results.
"""
import json
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for a query. Returns JSON list of results with title, url, snippet.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return json.dumps(results, indent=2)


SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for information on a topic. "
        "Returns a list of results with title, URL, and a short snippet. "
        "Use this to discover relevant pages before reading them in full."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
